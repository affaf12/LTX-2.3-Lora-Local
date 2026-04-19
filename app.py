
import os
import subprocess
import sys

# Disable torch.compile / dynamo before any torch import
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

# Install xformers for memory-efficient attention
subprocess.run([sys.executable, "-m", "pip", "install", "xformers==0.0.32.post2", "--no-build-isolation"], check=False)

# Clone LTX-2 repo and install packages
LTX_REPO_URL = "https://github.com/Lightricks/LTX-2.git"
LTX_REPO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "LTX-2")

LTX_COMMIT = "ae855f8538843825f9015a419cf4ba5edaf5eec2"  # known working commit with decode_video

if not os.path.exists(LTX_REPO_DIR):
    print(f"Cloning {LTX_REPO_URL}...")
    subprocess.run(["git", "clone", LTX_REPO_URL, LTX_REPO_DIR], check=True)
    subprocess.run(["git", "checkout", LTX_COMMIT], cwd=LTX_REPO_DIR, check=True)

print("Installing ltx-core and ltx-pipelines from cloned repo...")
subprocess.run(
    [sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-deps", "-e",
     os.path.join(LTX_REPO_DIR, "packages", "ltx-core"),
     "-e", os.path.join(LTX_REPO_DIR, "packages", "ltx-pipelines")],
    check=True,
)

sys.path.insert(0, os.path.join(LTX_REPO_DIR, "packages", "ltx-pipelines", "src"))
sys.path.insert(0, os.path.join(LTX_REPO_DIR, "packages", "ltx-core", "src"))

import logging
import random
import tempfile
from pathlib import Path
import gc
import hashlib

import torch
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True

import spaces
import gradio as gr
import numpy as np
from huggingface_hub import hf_hub_download, snapshot_download

from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.model.audio_vae import encode_audio as vae_encode_audio
from ltx_core.model.upsampler import upsample_video
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number, decode_video as vae_decode_video
from ltx_core.quantization import QuantizationPolicy
from ltx_core.types import Audio, AudioLatentShape, VideoPixelShape
from ltx_pipelines.distilled import DistilledPipeline
from ltx_pipelines.utils import euler_denoising_loop
from ltx_pipelines.utils.args import ImageConditioningInput
from ltx_pipelines.utils.constants import DISTILLED_SIGMA_VALUES, STAGE_2_DISTILLED_SIGMA_VALUES
from ltx_pipelines.utils.helpers import (
    cleanup_memory,
    combined_image_conditionings,
    denoise_video_only,
    encode_prompts,
    simple_denoising_func,
)
from ltx_pipelines.utils.media_io import decode_audio_from_file, encode_video
from ltx_core.loader.primitives import LoraPathStrengthAndSDOps
from ltx_core.loader.sd_ops import LTXV_LORA_COMFY_RENAMING_MAP

# Force-patch xformers attention into the LTX attention module.
from ltx_core.model.transformer import attention as _attn_mod
print(f"[ATTN] Before patch: memory_efficient_attention={_attn_mod.memory_efficient_attention}")
try:
    from xformers.ops import memory_efficient_attention as _mea
    _attn_mod.memory_efficient_attention = _mea
    print(f"[ATTN] After patch: memory_efficient_attention={_attn_mod.memory_efficient_attention}")
except Exception as e:
    print(f"[ATTN] xformers patch FAILED: {type(e).__name__}: {e}")

logging.getLogger().setLevel(logging.INFO)

MAX_SEED = np.iinfo(np.int32).max
DEFAULT_PROMPT = (
    "An astronaut hatches from a fragile egg on the surface of the Moon, "
    "the shell cracking and peeling apart in gentle low-gravity motion. "
    "Fine lunar dust lifts and drifts outward with each movement, floating "
    "in slow arcs before settling back onto the ground."
)
DEFAULT_FRAME_RATE = 24.0

# Resolution presets: (width, height)
RESOLUTIONS = {
    "high": {"16:9": (1536, 1024), "9:16": (1024, 1536), "1:1": (1024, 1024), "9:7": (1408, 1088), "7:9": (1088, 1408), "19:13": (1472, 1008), "13:19": (1008, 1472)},
    "low": {"16:9": (768, 512), "9:16": (512, 768), "1:1": (768, 768), "9:7": (704, 544), "7:9": (544, 704), "19:13": (736, 504), "13:19": (504, 736)},
}


class LTX23DistilledA2VPipeline(DistilledPipeline):
    """DistilledPipeline with optional audio conditioning."""

    def __call__(
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images: list[ImageConditioningInput],
        audio_path: str | None = None,
        tiling_config: TilingConfig | None = None,
        enhance_prompt: bool = False,
    ):
        # Standard path when no audio input is provided.
        print(prompt)
        if audio_path is None:
            return super().__call__(
                prompt=prompt,
                seed=seed,
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=frame_rate,
                images=images,
                tiling_config=tiling_config,
                enhance_prompt=enhance_prompt,
            )

        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        stepper = EulerDiffusionStep()
        dtype = torch.bfloat16

        (ctx_p,) = encode_prompts(
            [prompt],
            self.model_ledger,
            enhance_first_prompt=enhance_prompt,
            enhance_prompt_image=images[0].path if len(images) > 0 else None,
        )
        video_context, audio_context = ctx_p.video_encoding, ctx_p.audio_encoding

        video_duration = num_frames / frame_rate
        decoded_audio = decode_audio_from_file(audio_path, self.device, 0.0, video_duration)
        if decoded_audio is None:
            raise ValueError(f"Could not extract audio stream from {audio_path}")

        encoded_audio_latent = vae_encode_audio(decoded_audio, self.model_ledger.audio_encoder())
        audio_shape = AudioLatentShape.from_duration(batch=1, duration=video_duration, channels=8, mel_bins=16)
        expected_frames = audio_shape.frames
        actual_frames = encoded_audio_latent.shape[2]

        if actual_frames > expected_frames:
            encoded_audio_latent = encoded_audio_latent[:, :, :expected_frames, :]
        elif actual_frames < expected_frames:
            pad = torch.zeros(
                encoded_audio_latent.shape[0],
                encoded_audio_latent.shape[1],
                expected_frames - actual_frames,
                encoded_audio_latent.shape[3],
                device=encoded_audio_latent.device,
                dtype=encoded_audio_latent.dtype,
            )
            encoded_audio_latent = torch.cat([encoded_audio_latent, pad], dim=2)

        video_encoder = self.model_ledger.video_encoder()
        transformer = self.model_ledger.transformer()
        stage_1_sigmas = torch.tensor(DISTILLED_SIGMA_VALUES, device=self.device)

        def denoising_loop(sigmas, video_state, audio_state, stepper):
            return euler_denoising_loop(
                sigmas=sigmas,
                video_state=video_state,
                audio_state=audio_state,
                stepper=stepper,
                denoise_fn=simple_denoising_func(
                    video_context=video_context,
                    audio_context=audio_context,
                    transformer=transformer,
                ),
            )

        stage_1_output_shape = VideoPixelShape(
            batch=1,
            frames=num_frames,
            width=width // 2,
            height=height // 2,
            fps=frame_rate,
        )
        stage_1_conditionings = combined_image_conditionings(
            images=images,
            height=stage_1_output_shape.height,
            width=stage_1_output_shape.width,
            video_encoder=video_encoder,
            dtype=dtype,
            device=self.device,
        )
        video_state = denoise_video_only(
            output_shape=stage_1_output_shape,
            conditionings=stage_1_conditionings,
            noiser=noiser,
            sigmas=stage_1_sigmas,
            stepper=stepper,
            denoising_loop_fn=denoising_loop,
            components=self.pipeline_components,
            dtype=dtype,
            device=self.device,
            initial_audio_latent=encoded_audio_latent,
        )

        torch.cuda.synchronize()
        cleanup_memory()

        upscaled_video_latent = upsample_video(
            latent=video_state.latent[:1],
            video_encoder=video_encoder,
            upsampler=self.model_ledger.spatial_upsampler(),
        )
        stage_2_sigmas = torch.tensor(STAGE_2_DISTILLED_SIGMA_VALUES, device=self.device)
        stage_2_output_shape = VideoPixelShape(batch=1, frames=num_frames, width=width, height=height, fps=frame_rate)
        stage_2_conditionings = combined_image_conditionings(
            images=images,
            height=stage_2_output_shape.height,
            width=stage_2_output_shape.width,
            video_encoder=video_encoder,
            dtype=dtype,
            device=self.device,
        )
        video_state = denoise_video_only(
            output_shape=stage_2_output_shape,
            conditionings=stage_2_conditionings,
            noiser=noiser,
            sigmas=stage_2_sigmas,
            stepper=stepper,
            denoising_loop_fn=denoising_loop,
            components=self.pipeline_components,
            dtype=dtype,
            device=self.device,
            noise_scale=stage_2_sigmas[0],
            initial_video_latent=upscaled_video_latent,
            initial_audio_latent=encoded_audio_latent,
        )

        torch.cuda.synchronize()
        del transformer
        del video_encoder
        cleanup_memory()

        decoded_video = vae_decode_video(
            video_state.latent,
            self.model_ledger.video_decoder(),
            tiling_config,
            generator,
        )
        original_audio = Audio(
            waveform=decoded_audio.waveform.squeeze(0),
            sampling_rate=decoded_audio.sampling_rate,
        )
        return decoded_video, original_audio


# Model repos
LTX_MODEL_REPO = "Lightricks/LTX-2.3"
GEMMA_REPO ="Lightricks/gemma-3-12b-it-qat-q4_0-unquantized"


# Download model checkpoints
print("=" * 80)
print("Downloading LTX-2.3 distilled model + Gemma...")
print("=" * 80)

# LoRA cache directory and currently-applied key
LORA_CACHE_DIR = Path("lora_cache")
LORA_CACHE_DIR.mkdir(exist_ok=True)
current_lora_key: str | None = None

PENDING_LORA_KEY: str | None = None
PENDING_LORA_STATE: dict[str, torch.Tensor] | None = None
PENDING_LORA_STATUS: str = "No LoRA state prepared yet."

weights_dir = Path("weights")
weights_dir.mkdir(exist_ok=True)
checkpoint_path = hf_hub_download(
    repo_id=LTX_MODEL_REPO,
    filename="ltx-2.3-22b-distilled.safetensors",
    local_dir=str(weights_dir),
    local_dir_use_symlinks=False,
)
spatial_upsampler_path = hf_hub_download(repo_id=LTX_MODEL_REPO, filename="ltx-2.3-spatial-upscaler-x2-1.0.safetensors")
gemma_root = snapshot_download(repo_id=GEMMA_REPO)

# ---- Insert block (LoRA downloads) between lines 268 and 269 ----
# LoRA repo + download the requested LoRA adapters
LORA_REPO = "rahul7star/Ltx-2-3-Lora-Collection"

print("=" * 80)
print("Downloading LoRA adapters from dagloop5/LoRA...")
print("=" * 80)
paste_lora_path = hf_hub_download(repo_id=LORA_REPO, filename="ltx23-kabapaste.safetensors")
ruri_lora_path = hf_hub_download(repo_id=LORA_REPO, filename="LTX23-ruri.safetensors")
transit_lora_path = hf_hub_download(repo_id="valiantcat/LTX-2.3-Transition-LORA", filename="ltx2.3-transition.safetensors")
pose_lora_path = hf_hub_download(repo_id=LORA_REPO, filename="LTX2_3_NSFW_furry_concat_v2.safetensors")
general_lora_path = hf_hub_download(repo_id=LORA_REPO, filename="LTX2.3_Reasoning_V1.safetensors")
motion_lora_path = hf_hub_download(repo_id=LORA_REPO, filename="motion_helper.safetensors")
dreamlay_lora_path = hf_hub_download(repo_id=LORA_REPO, filename="DR34ML4Y_LTXXX_PREVIEW_RC1.safetensors") # m15510n4ry, bl0wj0b, d0ubl3_bj, d0gg1e, c0wg1rl
mself_lora_path = hf_hub_download(repo_id=LORA_REPO, filename="Furry Hyper Masturbation - LTX-2 I2V v1.safetensors") # Hyperfap
dramatic_lora_path = hf_hub_download(repo_id=LORA_REPO, filename="LTX-2.3 - Orgasm.safetensors") # "[He | She] is having am orgasm." (am or an?)
fluid_lora_path = hf_hub_download(repo_id=LORA_REPO, filename="cr3ampi3_animation_i2v_ltx2_v1.0.safetensors") # cr3ampi3 animation., missionary animation, doggystyle bouncy animation, double penetration animation
liquid_lora_path = hf_hub_download(repo_id=LORA_REPO, filename="liquid_wet_dr1pp_ltx2_v1.0_scaled.safetensors") # wet dr1pp
demopose_lora_path = hf_hub_download(repo_id=LORA_REPO, filename="ltx23__demopose_d3m0p0s3.safetensors") # d3m0p0s3

print(f"Paste LoRA: {paste_lora_path }")
print(f"Ruri LoRA: {ruri_lora_path }")
print(f"Transit LoRA: {transit_lora_path }")
print(f"Pose LoRA: {pose_lora_path}")
print(f"General LoRA: {general_lora_path}")
print(f"Motion LoRA: {motion_lora_path}")
print(f"Dreamlay LoRA: {dreamlay_lora_path}")
print(f"Mself LoRA: {mself_lora_path}")
print(f"Dramatic LoRA: {dramatic_lora_path}")
print(f"Fluid LoRA: {fluid_lora_path}")
print(f"Liquid LoRA: {liquid_lora_path}")
print(f"Demopose LoRA: {demopose_lora_path}")
# ----------------------------------------------------------------

print(f"Checkpoint: {checkpoint_path}")
print(f"Spatial upsampler: {spatial_upsampler_path}")
print(f"Gemma root: {gemma_root}")

# Initialize pipeline WITH text encoder and optional audio support
# ---- Replace block (pipeline init) lines 275-281 ----
pipeline = LTX23DistilledA2VPipeline(
    distilled_checkpoint_path=checkpoint_path,
    spatial_upsampler_path=spatial_upsampler_path,
    gemma_root=gemma_root,
    loras=[],
    quantization=QuantizationPolicy.fp8_cast(),  # keep FP8 quantization unchanged
)
# ----------------------------------------------------------------

def _make_lora_key(paste_strength: float,ruri_strength: float,transit_strength: float,pose_strength: float, general_strength: float, motion_strength: float, dreamlay_strength: float, mself_strength: float, dramatic_strength: float, fluid_strength: float, liquid_strength: float, demopose_strength: float) -> tuple[str, str]:
    ps = round(float(paste_strength), 2)
    rr = round(float(ruri_strength), 2)
    rt = round(float(transit_strength), 2)
    rp = round(float(pose_strength), 2)
    rg = round(float(general_strength), 2)
    rm = round(float(motion_strength), 2)
    rd = round(float(dreamlay_strength), 2)
    rs = round(float(mself_strength), 2)
    rr = round(float(dramatic_strength), 2)
    rf = round(float(fluid_strength), 2)
    rl = round(float(liquid_strength), 2)
    ro = round(float(demopose_strength), 2)
    key_str = f"{paste_lora_path}:{ps}|{ruri_lora_path}:{rr}|{transit_lora_path}:{rt}|{pose_lora_path}:{rp}|{general_lora_path}:{rg}|{motion_lora_path}:{rm}|{dreamlay_lora_path}:{rd}|{mself_lora_path}:{rs}|{dramatic_lora_path}:{rr}|{fluid_lora_path}:{rf}|{liquid_lora_path}:{rl}|{demopose_lora_path}:{ro}"
    key = hashlib.sha256(key_str.encode("utf-8")).hexdigest()
    return key, key_str


def prepare_lora_cache(
    paste_strength: float,
    ruri_strength: float,
    transit_strength: float,
    pose_strength: float,
    general_strength: float,
    motion_strength: float,
    dreamlay_strength: float,
    mself_strength: float,
    dramatic_strength: float,
    fluid_strength: float,
    liquid_strength: float,
    demopose_strength: float,
    progress=gr.Progress(track_tqdm=True),
):
    """
    CPU-only step:
    - checks cache
    - loads cached fused transformer state_dict, or
    - builds fused transformer on CPU and saves it
    The resulting state_dict is stored in memory and can be applied later.
    """
    global PENDING_LORA_KEY, PENDING_LORA_STATE, PENDING_LORA_STATUS

    ledger = pipeline.model_ledger
    key, _ = _make_lora_key(paste_strength,ruri_strength,transit_strength,pose_strength, general_strength, motion_strength, dreamlay_strength, mself_strength, dramatic_strength, fluid_strength, liquid_strength, demopose_strength)
    cache_path = LORA_CACHE_DIR / f"{key}.pt"

    progress(0.05, desc="Preparing LoRA state")
    if cache_path.exists():
        try:
            progress(0.20, desc="Loading cached fused state")
            state = torch.load(cache_path, map_location="cpu")
            PENDING_LORA_KEY = key
            PENDING_LORA_STATE = state
            PENDING_LORA_STATUS = f"Loaded cached LoRA state: {cache_path.name}"
            return PENDING_LORA_STATUS
        except Exception as e:
            print(f"[LoRA] Cache load failed: {type(e).__name__}: {e}")

    entries = [
        (paste_lora_path, round(float(paste_strength), 2)),
        (ruri_lora_path, round(float(ruri_strength), 2)),
        (transit_lora_path, round(float(transit_strength), 2)),
        (pose_lora_path, round(float(pose_strength), 2)),
        (general_lora_path, round(float(general_strength), 2)),
        (motion_lora_path, round(float(motion_strength), 2)),
        (dreamlay_lora_path, round(float(dreamlay_strength), 2)),
        (mself_lora_path, round(float(mself_strength), 2)),
        (dramatic_lora_path, round(float(dramatic_strength), 2)),
        (fluid_lora_path, round(float(fluid_strength), 2)),
        (liquid_lora_path, round(float(liquid_strength), 2)),
        (demopose_lora_path, round(float(demopose_strength), 2)),
    ]
    loras_for_builder = [
        LoraPathStrengthAndSDOps(path, strength, LTXV_LORA_COMFY_RENAMING_MAP)
        for path, strength in entries
        if path is not None and float(strength) != 0.0
    ]

    if not loras_for_builder:
        PENDING_LORA_KEY = None
        PENDING_LORA_STATE = None
        PENDING_LORA_STATUS = "No non-zero LoRA strengths selected; nothing to prepare."
        return PENDING_LORA_STATUS

    tmp_ledger = None
    new_transformer_cpu = None
    try:
        progress(0.35, desc="Building fused CPU transformer")
        tmp_ledger = pipeline.model_ledger.__class__(
            dtype=ledger.dtype,
            device=torch.device("cpu"),
            checkpoint_path=str(checkpoint_path),
            spatial_upsampler_path=str(spatial_upsampler_path),
            gemma_root_path=str(gemma_root),
            loras=tuple(loras_for_builder),
            quantization=getattr(ledger, "quantization", None),
        )
        new_transformer_cpu = tmp_ledger.transformer()

        progress(0.70, desc="Extracting fused state_dict")
        state = new_transformer_cpu.state_dict()
        torch.save(state, cache_path)

        PENDING_LORA_KEY = key
        PENDING_LORA_STATE = state
        PENDING_LORA_STATUS = f"Built and cached LoRA state: {cache_path.name}"
        return PENDING_LORA_STATUS

    except Exception as e:
        import traceback
        print(f"[LoRA] Prepare failed: {type(e).__name__}: {e}")
        print(traceback.format_exc())
        PENDING_LORA_KEY = None
        PENDING_LORA_STATE = None
        PENDING_LORA_STATUS = f"LoRA prepare failed: {type(e).__name__}: {e}"
        return PENDING_LORA_STATUS

    finally:
        try:
            del new_transformer_cpu
        except Exception:
            pass
        try:
            del tmp_ledger
        except Exception:
            pass
        gc.collect()


def apply_prepared_lora_state_to_pipeline():
    """
    Fast step: copy the already prepared CPU state into the live transformer.
    This is the only part that should remain near generation time.
    """
    global current_lora_key, PENDING_LORA_KEY, PENDING_LORA_STATE

    if PENDING_LORA_STATE is None or PENDING_LORA_KEY is None:
        print("[LoRA] No prepared LoRA state available; skipping.")
        return False

    if current_lora_key == PENDING_LORA_KEY:
        print("[LoRA] Prepared LoRA state already active; skipping.")
        return True

    existing_transformer = _transformer
    existing_params = {name: param for name, param in existing_transformer.named_parameters()}
    existing_buffers = {name: buf for name, buf in existing_transformer.named_buffers()}

    with torch.no_grad():
        for k, v in PENDING_LORA_STATE.items():
            if k in existing_params:
                existing_params[k].data.copy_(v.to(existing_params[k].device))
            elif k in existing_buffers:
                existing_buffers[k].data.copy_(v.to(existing_buffers[k].device))

    current_lora_key = PENDING_LORA_KEY
    print("[LoRA] Prepared LoRA state applied to the pipeline."+current_lora_key)
    return True

# ---- REPLACE PRELOAD BLOCK START ----
# Preload all models for ZeroGPU tensor packing.
print("Preloading all models (including Gemma and audio components)...")
ledger = pipeline.model_ledger

# Save the original factory methods so we can rebuild individual components later.
# These are bound callables on ledger that will call the builder when invoked.
_orig_transformer_factory = ledger.transformer
_orig_video_encoder_factory = ledger.video_encoder
_orig_video_decoder_factory = ledger.video_decoder
_orig_audio_encoder_factory = ledger.audio_encoder
_orig_audio_decoder_factory = ledger.audio_decoder
_orig_vocoder_factory = ledger.vocoder
_orig_spatial_upsampler_factory = ledger.spatial_upsampler
_orig_text_encoder_factory = ledger.text_encoder
_orig_gemma_embeddings_factory = ledger.gemma_embeddings_processor

# Call the original factories once to create the cached instances we will serve by default.
_transformer = _orig_transformer_factory()
_video_encoder = _orig_video_encoder_factory()
_video_decoder = _orig_video_decoder_factory()
_audio_encoder = _orig_audio_encoder_factory()
_audio_decoder = _orig_audio_decoder_factory()
_vocoder = _orig_vocoder_factory()
_spatial_upsampler = _orig_spatial_upsampler_factory()
_text_encoder = _orig_text_encoder_factory()
_embeddings_processor = _orig_gemma_embeddings_factory()

# Replace ledger methods with lightweight lambdas that return the cached instances.
# We keep the original factories above so we can call them later to rebuild components.
ledger.transformer = lambda: _transformer
ledger.video_encoder = lambda: _video_encoder
ledger.video_decoder = lambda: _video_decoder
ledger.audio_encoder = lambda: _audio_encoder
ledger.audio_decoder = lambda: _audio_decoder
ledger.vocoder = lambda: _vocoder
ledger.spatial_upsampler = lambda: _spatial_upsampler
ledger.text_encoder = lambda: _text_encoder
ledger.gemma_embeddings_processor = lambda: _embeddings_processor

print("All models preloaded (including Gemma text encoder and audio encoder)!")
# ---- REPLACE PRELOAD BLOCK END ----

print("=" * 80)
print("Pipeline ready!")
print("=" * 80)


def log_memory(tag: str):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        peak = torch.cuda.max_memory_allocated() / 1024**3
        free, total = torch.cuda.mem_get_info()
        print(f"[VRAM {tag}] allocated={allocated:.2f}GB peak={peak:.2f}GB free={free / 1024**3:.2f}GB total={total / 1024**3:.2f}GB")


def detect_aspect_ratio(image) -> str:
    if image is None:
        return "16:9"
    if hasattr(image, "size"):
        w, h = image.size
    elif hasattr(image, "shape"):
        h, w = image.shape[:2]
    else:
        return "16:9"
    ratio = w / h
    candidates = {"16:9": 16 / 9, "9:16": 9 / 16, "1:1": 1.0}
    return min(candidates, key=lambda k: abs(ratio - candidates[k]))


def on_image_upload(first_image, last_image, high_res):
    ref_image = first_image if first_image is not None else last_image
    aspect = detect_aspect_ratio(ref_image)
    tier = "high" if high_res else "low"
    w, h = RESOLUTIONS[tier][aspect]
    return gr.update(value=w), gr.update(value=h)


def on_highres_toggle(first_image, last_image, high_res):
    ref_image = first_image if first_image is not None else last_image
    aspect = detect_aspect_ratio(ref_image)
    tier = "high" if high_res else "low"
    w, h = RESOLUTIONS[tier][aspect]
    return gr.update(value=w), gr.update(value=h)


def get_gpu_duration(
    first_image,
    last_image,
    input_audio,
    prompt: str,
    duration: float,
    gpu_duration: float,
    enhance_prompt: bool = True,
    seed: int = 42,
    randomize_seed: bool = True,
    height: int = 1024,
    width: int = 1536,
    paste_strength: float = 0.0,
    ruri_strength: float = 0.0,
    transit_strength: float = 0.0,
    pose_strength: float = 0.0,
    general_strength: float = 0.0,
    motion_strength: float = 0.0,
    dreamlay_strength: float = 0.0,
    mself_strength: float = 0.0,
    dramatic_strength: float = 0.0,
    fluid_strength: float = 0.0,
    liquid_strength: float = 0.0,
    demopose_strength: float = 0.0,
    progress=None,
):
    return int(gpu_duration)

@spaces.GPU(duration=get_gpu_duration)
@torch.inference_mode()
def generate_video(
    first_image,
    last_image,
    input_audio,
    prompt: str,
    duration: float,
    gpu_duration: float,
    enhance_prompt: bool = True,
    seed: int = 42,
    randomize_seed: bool = True,
    height: int = 1024,
    width: int = 1536,
    paste_strength: float = 0.0,
    ruri_strength: float = 0.0,
    transit_strength: float = 0.0,
    pose_strength: float = 0.0,
    general_strength: float = 0.0,
    motion_strength: float = 0.0,
    dreamlay_strength: float = 0.0,
    mself_strength: float = 0.0,
    dramatic_strength: float = 0.0,
    fluid_strength: float = 0.0,
    liquid_strength: float = 0.0,
    demopose_strength: float = 0.0,
    progress=gr.Progress(track_tqdm=True),
):
    try:
        torch.cuda.reset_peak_memory_stats()
        log_memory("start")

        current_seed = random.randint(0, MAX_SEED) if randomize_seed else int(seed)

        frame_rate = DEFAULT_FRAME_RATE
        num_frames = int(duration * frame_rate) + 1
        num_frames = ((num_frames - 1 + 7) // 8) * 8 + 1

        print(f"Prompt:{prompt},Generating: {height}x{width}, {num_frames} frames ({duration}s), seed={current_seed},")

        images = []
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)

        if first_image is not None:
            temp_first_path = output_dir / f"temp_first_{current_seed}.jpg"
            if hasattr(first_image, "save"):
                first_image.save(temp_first_path)
            else:
                temp_first_path = Path(first_image)
            images.append(ImageConditioningInput(path=str(temp_first_path), frame_idx=0, strength=1.0))

        if last_image is not None:
            temp_last_path = output_dir / f"temp_last_{current_seed}.jpg"
            if hasattr(last_image, "save"):
                last_image.save(temp_last_path)
            else:
                temp_last_path = Path(last_image)
            images.append(ImageConditioningInput(path=str(temp_last_path), frame_idx=num_frames - 1, strength=1.0))

        tiling_config = TilingConfig.default()
        video_chunks_number = get_video_chunks_number(num_frames, tiling_config)

        log_memory("before pipeline call")

        apply_prepared_lora_state_to_pipeline()

        video, audio = pipeline(
            prompt=prompt,
            seed=current_seed,
            height=int(height),
            width=int(width),
            num_frames=num_frames,
            frame_rate=frame_rate,
            images=images,
            audio_path=input_audio,
            tiling_config=tiling_config,
            enhance_prompt=enhance_prompt,
        )

        log_memory("after pipeline call")

        output_path = tempfile.mktemp(suffix=".mp4")
        encode_video(
            video=video,
            fps=frame_rate,
            audio=audio,
            output_path=output_path,
            video_chunks_number=video_chunks_number,
        )

        log_memory("after encode_video")
        return str(output_path), current_seed

    except Exception as e:
        import traceback
        log_memory("on error")
        print(f"Error: {str(e)}\n{traceback.format_exc()}")
        return None, current_seed
with gr.Blocks(title="LTX-2.3 Distilled") as demo:
    gr.Markdown("# LTX-2.3 F2LF with Fast Audio-Video Generation and Lora support")
    

    with gr.Row():
        with gr.Column():
            with gr.Row():
                first_image = gr.Image(label="First Frame (Optional)", type="pil")
                last_image = gr.Image(label="Last Frame (Optional)", type="pil")
            input_audio = gr.Audio(label="Audio Input (Optional)", type="filepath")
            prompt = gr.Textbox(
                label="Prompt",
                info="for best results - make it as elaborate as possible",
                value="Make this image come alive with cinematic motion, smooth animation",
                lines=3,
                placeholder="Describe the motion and animation you want...",
            )
            duration = gr.Slider(label="Duration (seconds)", minimum=1.0, maximum=30.0, value=10.0, step=0.1)
                

            generate_btn = gr.Button("Generate Video", variant="primary", size="lg")

            with gr.Accordion("Advanced Settings", open=True):
                seed = gr.Slider(label="Seed", minimum=0, maximum=MAX_SEED, value=10, step=1)
                randomize_seed = gr.Checkbox(label="Randomize Seed", value=True)
                with gr.Row():
                    width = gr.Number(label="Width", value=1536, precision=0)
                    height = gr.Number(label="Height", value=1024, precision=0)
                with gr.Row():
                    enhance_prompt = gr.Checkbox(label="Enhance Prompt", value=False)
                    high_res = gr.Checkbox(label="High Resolution", value=True)
                with gr.Column():
                    gr.Markdown("### LoRA adapter strengths (set to 0 to disable)")
                    paste_strength = gr.Slider(label="Paste anything Lora strength", minimum=0.0, maximum=2.0, value=0.0, step=0.01)
                    ruri_strength = gr.Slider(label="Ruri lady Lora strength", minimum=0.0, maximum=2.0, value=0.0, step=0.01)
                    transit_strength = gr.Slider(label="Transition Lora strength", minimum=0.0, maximum=2.0, value=0.0, step=0.01)
                    pose_strength = gr.Slider(label="Anthro Enhancer strength", minimum=0.0, maximum=2.0, value=0.0, step=0.01)
                    general_strength = gr.Slider(label="Reasoning Enhancer strength", minimum=0.0, maximum=2.0, value=0.0, step=0.01)
                    motion_strength = gr.Slider(label="Anthro Posing Helper strength", minimum=0.0, maximum=2.0, value=0.0, step=0.01)
                    dreamlay_strength = gr.Slider(label="Dreamfly strength", minimum=0.0, maximum=2.0, value=0.0, step=0.01)
                    mself_strength = gr.Slider(label="Mself strength", minimum=0.0, maximum=2.0, value=0.0, step=0.01)
                    dramatic_strength = gr.Slider(label="Dramatic strength", minimum=0.0, maximum=2.0, value=0.0, step=0.01)
                    fluid_strength = gr.Slider(label="Fluid Helper strength", minimum=0.0, maximum=2.0, value=0.0, step=0.01)
                    liquid_strength = gr.Slider(label="Liquid Helper strength", minimum=0.0, maximum=2.0, value=0.0, step=0.01)
                    demopose_strength = gr.Slider(label="Demopose Helper strength", minimum=0.0, maximum=2.0, value=0.0, step=0.01)

                prepare_lora_btn = gr.Button("Prepare / Load LoRA Cache", variant="secondary")
                lora_status = gr.Textbox(
                    label="LoRA Cache Status",
                    value="No LoRA state prepared yet.",
                    interactive=False,
                )

        with gr.Column():
            output_video = gr.Video(label="Generated Video", autoplay=False)
            gpu_duration = gr.Slider(
                label="ZeroGPU duration (seconds)",
                minimum=40.0,
                maximum=240.0,
                value=85.0,
                step=1.0,
            )

    gr.Examples(
        examples=[
            [
                "asc.jpg",
                None,
                None,
                "A low-angle wide shot establishes a winding, wet asphalt road flanked by a dense, dark forest...",
                3.0,
                80.0,
                False,
                42,
                True,
                1024,
                1024,
                1.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            [None, "asc.jpg", "KABAPASTE The girl is squeezed from a tube like paste...", "paste lora"]
        ],
        inputs=[
            first_image, last_image, input_audio, prompt, duration, gpu_duration,
            enhance_prompt, seed, randomize_seed, height, width,
            paste_strength, pose_strength, general_strength, motion_strength,
            dreamlay_strength, mself_strength, dramatic_strength,
            fluid_strength, liquid_strength, demopose_strength,
        ],
    )

    # ✅ FIXED EXAMPLE OUTPUT SECTION (ONLY CHANGE)
    with gr.Column():
        gr.Markdown("## 🎬 Example Output")

        prompt_preview = gr.Textbox(
            label="Prompt",
            value="KABAPASTE The girl is squeezed from a tube like paste. The video is silent with the sound of rain and squeezing sounds. After the girl forms, her umbrella opens up.",
            interactive=False
        )

        video_preview = gr.Video(
            value="abc.mp4",   
            autoplay=True,
            interactive=False,
            width=150,
            height=150
             
   
   
        )

    first_image.change(
        fn=on_image_upload,
        inputs=[first_image, last_image, high_res],
        outputs=[width, height],
    )

    last_image.change(
        fn=on_image_upload,
        inputs=[first_image, last_image, high_res],
        outputs=[width, height],
    )

    high_res.change(
        fn=on_highres_toggle,
        inputs=[first_image, last_image, high_res],
        outputs=[width, height],
    )

    prepare_lora_btn.click(
        fn=prepare_lora_cache,
        inputs=[paste_strength, ruri_strength, transit_strength, pose_strength, general_strength, motion_strength, dreamlay_strength, mself_strength, dramatic_strength, fluid_strength, liquid_strength, demopose_strength],
        outputs=[lora_status],
    )
    
    generate_btn.click(
        fn=generate_video,
        inputs=[
            first_image, last_image, input_audio, prompt, duration, gpu_duration, enhance_prompt,
            seed, randomize_seed, height, width,
            paste_strength, ruri_strength, transit_strength, pose_strength, general_strength, motion_strength, dreamlay_strength, mself_strength, dramatic_strength, fluid_strength, liquid_strength, demopose_strength,
        ],
        outputs=[output_video, seed],
    )

    
css = """
.fillable{max-width: 1200px !important}
"""

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Citrus(), css=css)
