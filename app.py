from pathlib import Path

import streamlit as st
import torch
from PIL import Image, ImageOps, UnidentifiedImageError

from model_utils import create_transform, install_checkpoint, load_trained_model
from training_job import get_training_status, read_training_log, start_training


st.set_page_config(page_title="Plant Disease Classifier", layout="centered")
st.title("Plant disease classifier")
st.caption("Upload a leaf photo for a model classification. Results are a research aid, not a confirmed diagnosis.")

checkpoint_path = Path(__file__).resolve().with_name("plant_disease_resnet50.pth")


@st.fragment(run_every="5s")
def model_setup(reload_when_ready=True):
    if reload_when_ready and checkpoint_path.is_file():
        st.rerun()
    status = get_training_status()
    running = status == "running"
    if running:
        st.info("Training is running. Keep this computer on; the browser can be closed.", title="Preparing model")
    elif status in {"failed", "interrupted"}:
        st.error("Training did not finish. Check the training log, resolve the error, and retry.")
    elif status == "succeeded":
        st.warning("Training finished, but a usable checkpoint is still required.")

    st.subheader("Train locally")
    st.caption("First run: approximately 2.18 GB download, at least 5 GB free disk space. CPU training may take hours.")
    if st.button("Download data and train", icon=":material/model_training:", disabled=running):
        try:
            start_training()
        except (OSError, RuntimeError) as error:
            st.error(str(error))
        else:
            st.info("Training requested. Waiting for the training process to start.")

    log = read_training_log()
    if log:
        with st.expander("Training log", expanded=running or status in {"failed", "interrupted"}):
            st.code(log, language="text")

    st.subheader("Import trained model")
    st.caption("Only import a trusted checkpoint created by this project. No training dataset is needed for predictions.")
    uploaded_checkpoint = st.file_uploader("Model checkpoint", type=["pth"], disabled=running)
    if st.button("Validate and install", icon=":material/upload_file:", disabled=running or uploaded_checkpoint is None):
        try:
            with st.spinner("Validating model weights..."):
                install_checkpoint(uploaded_checkpoint, checkpoint_path)
        except (OSError, RuntimeError, ValueError) as error:
            st.error(f"The checkpoint was not installed: {error}")
        else:
            st.rerun()


if not checkpoint_path.is_file():
    st.warning("No trained model is installed yet.", title="Model setup required")
    model_setup()
    st.stop()


@st.cache_resource(max_entries=1)
def load_diagnostic_model(path: str, modified_ns: int, size: int):
    checkpoint_path = Path(path)
    current_stat = checkpoint_path.stat()
    if (current_stat.st_mtime_ns, current_stat.st_size) != (modified_ns, size):
        raise RuntimeError("The checkpoint changed while it was loading. Refresh the page and retry.")

    return load_trained_model(checkpoint_path)


try:
    checkpoint_stat = checkpoint_path.stat()
    model, checkpoint = load_diagnostic_model(
        str(checkpoint_path), checkpoint_stat.st_mtime_ns, checkpoint_stat.st_size
    )
except (OSError, RuntimeError, ValueError) as error:
    st.error("The model checkpoint is invalid. Retrain the model to create a compatible checkpoint.", title="Model unavailable")
    st.caption(str(error))
    model_setup(reload_when_ready=False)
    st.stop()

if checkpoint.get("validation_accuracy") is not None:
    st.caption(f"Saved validation accuracy: {checkpoint['validation_accuracy']:.1%}")

uploaded_file = st.file_uploader("Choose a leaf photograph", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    if uploaded_file.size > 10 * 1024**2:
        st.error("Choose an image smaller than 10 MB.")
        st.stop()
    try:
        with Image.open(uploaded_file) as uploaded_image:
            image = ImageOps.exif_transpose(uploaded_image).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        st.error("This file could not be opened as an image. Choose a valid JPG or PNG file.")
        st.stop()

    st.image(image, caption="Uploaded leaf", alt="Uploaded crop leaf", width="stretch")
    input_tensor = create_transform()(image).unsqueeze(0)
    with torch.inference_mode():
        probabilities = torch.softmax(model(input_tensor), dim=1)[0]
        if not torch.isfinite(probabilities).all():
            st.error("The model returned invalid scores. Replace or retrain the checkpoint.")
            st.stop()
        confidence, class_index = torch.max(probabilities, dim=0)

    class_name = checkpoint["class_names"][class_index.item()]
    display_name = class_name.replace("___", " - ").replace("_", " ")
    st.success(f"Plant disease classification: {display_name}")
    st.metric("Model confidence", f"{confidence.item():.1%}")
