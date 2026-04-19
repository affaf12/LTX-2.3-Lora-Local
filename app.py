import streamlit as st
import random
from pathlib import Path
import tempfile
import torch

# Import your pipeline and helpers from the existing code
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_pipelines.utils.args import ImageConditioningInput
from ltx_pipelines.utils.media_io import encode_video
from your_pipeline_file import pipeline, apply_prepared_lora_state_to_pipeline, MAX_SEED, DEFAULT_FRAME_RATE

st.title("🎬 LTX-2.3 Distilled - Streamlit UI")

# Inputs
first_image = st.file_uploader("Upload First Frame (Optional)", type=["jpg","png"])
last_image = st.file_uploader("Upload Last Frame (Optional)", type=["jpg","png"])
input_audio = st.file_uploader("Upload Audio (Optional)", type=["wav","mp3"])
prompt = st.text_area("Prompt", "Make this image come alive with cinematic motion, smooth animation")
duration = st.slider("Duration (seconds)", 1.0, 30.0, 10.0, 0.1)
seed = st.slider("Seed", 0, MAX_SEED, 42, 1)
enhance_prompt = st.checkbox("Enhance Prompt", True)

if st.button("Generate Video"):
    st.write("🚀 Running pipeline...")
    current_seed = random.randint(0, MAX_SEED) if seed == 0 else seed
    frame_rate = DEFAULT_FRAME_RATE
    num_frames = int(duration * frame_rate) + 1
    num_frames = ((num_frames - 1 + 7) // 8) * 8 + 1

    images = []
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    if first_image is not None:
        temp_first_path = output_dir / f"temp_first_{current_seed}.jpg"
        with open(temp_first_path, "wb") as f:
            f.write(first_image.read())
        images.append(ImageConditioningInput(path=str(temp_first_path), frame_idx=0, strength=1.0))

    if last_image is not None:
        temp_last_path = output_dir / f"temp_last_{current_seed}.jpg"
        with open(temp_last_path, "wb") as f:
            f.write(last_image.read())
        images.append(ImageConditioningInput(path=str(temp_last_path), frame_idx=num_frames - 1, strength=1.0))

    tiling_config = TilingConfig.default()
    video_chunks_number = get_video_chunks_number(num_frames, tiling_config)

    apply_prepared_lora_state_to_pipeline()

    video, audio = pipeline(
        prompt=prompt,
        seed=current_seed,
        height=1024,
        width=1536,
        num_frames=num_frames,
        frame_rate=frame_rate,
        images=images,
        audio_path=None if input_audio is None else input_audio.name,
        tiling_config=tiling_config,
        enhance_prompt=enhance_prompt,
    )

    output_path = tempfile.mktemp(suffix=".mp4")
    encode_video(video=video, fps=frame_rate, audio=audio, output_path=output_path, video_chunks_number=video_chunks_number)

    st.video(output_path)
