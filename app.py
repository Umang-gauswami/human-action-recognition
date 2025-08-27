import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# -------------------------
# Page config
# -------------------------
st.set_page_config(
    page_title="Human Action Recognition",
    page_icon="🤖",
    layout="wide"
)

# -------------------------
# Sidebar
# -------------------------
st.sidebar.title("Human Action Recognition")
st.sidebar.markdown(
    """
This app predicts the action of a human in a given image.
- Upload an image of a human performing an action.
- Supported actions: calling, clapping, cycling, dancing, drinking, eating, fighting, hugging, laughing, listening_to_music, running, sitting, sleeping, texting, using_laptop.
"""
)

# -------------------------
# Title
# -------------------------
st.title("🕺 Human Action Recognition")
st.markdown("Upload an image and see the model prediction on the right.")

# -------------------------
# Load Model
# -------------------------
@st.cache_resource
def load_har_model():
    model_path = "trained-model.h5"
    model = load_model(model_path)
    return model

model = load_har_model()

# -------------------------
# Class names
# -------------------------
class_names = [
    'calling', 'clapping', 'cycling', 'dancing', 'drinking', 'eating',
    'fighting', 'hugging', 'laughing', 'listening_to_music', 'running',
    'sitting', 'sleeping', 'texting', 'using_laptop'
]

# -------------------------
# File uploader
# -------------------------
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")

    # Preprocess image
    img_resized = img.resize((128, 128))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Prediction
    pred = model.predict(img_array)
    pred_class_idx = np.argmax(pred, axis=1)[0]
    pred_label = class_names[pred_class_idx]

    # -------------------------
    # Show image + prediction side by side
    # -------------------------
    col1, col2 = st.columns(2)

    with col1:
        st.image(img, caption="Uploaded Image", use_container_width=True)

    with col2:
        st.markdown("### Prediction Result")
        st.markdown(f"**Class Index:** {pred_class_idx}")
        st.markdown(f"**Predicted Action:** 🏷️ {pred_label}")
        st.success(f"The model predicts: {pred_label}")

    st.markdown("---")