"""
Banana Leaf Disease Classification — Streamlit app
Model: VGG16 + MobileNetV2 hybrid (frozen backbones, GAP -> concat -> Dense)
Weights: vgg16_mobilenet_hybrid_weights.h5  (weights-only, so architecture is rebuilt below)
"""

import os
import numpy as np
import cv2
from PIL import Image
import streamlit as st
import tensorflow as tf
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Concatenate, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16, MobileNetV2

# ----------------------------- config -----------------------------
IMG_SIZE = (224, 224)
WEIGHTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vgg16_mobilenet_hybrid_weights.h5")

CLASS_NAMES = [
    "Banana Healthy Leaf",
    "Banana Insect Pest Disease",
    "Banana Moko Disease or dead",
    "Banana Yellow Sigatoka Disease",
    "Black Sigotika",
    "Fusarium Wilt Panama",
]

st.set_page_config(page_title="Banana Leaf Disease Classifier", page_icon="🍌", layout="centered")


# ----------------------------- model -----------------------------
def build_model(num_classes: int = 6):
    inp = Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3), name="input_layer")

    vgg = VGG16(weights=None, include_top=False, input_tensor=inp)
    mob = MobileNetV2(weights=None, include_top=False, input_tensor=inp)

    for layer in vgg.layers:
        layer.trainable = False
    for layer in mob.layers:
        layer.trainable = False

    v = GlobalAveragePooling2D()(vgg.output)
    m = GlobalAveragePooling2D()(mob.output)

    x = Concatenate()([v, m])
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.1)(x)
    out = Dense(num_classes, activation="softmax")(x)

    return Model(inputs=inp, outputs=out, name="vgg16_mobilenet_hybrid")


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    model = build_model(len(CLASS_NAMES))
    if not os.path.exists(WEIGHTS_PATH):
        st.error(f"Weights file not found: {WEIGHTS_PATH}")
        st.stop()
    model.load_weights(WEIGHTS_PATH)
    return model


# ----------------------------- preprocessing -----------------------------
def preprocess(pil_img: Image.Image) -> np.ndarray:
    """RGB -> LAB -> CLAHE on L -> back to RGB -> Gaussian blur -> resize -> /255."""
    img = np.array(pil_img.convert("RGB"))

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype("float32") / 255.0

    return np.expand_dims(img, axis=0)


# ----------------------------- UI -----------------------------
st.title("🍌 Banana Leaf Disease Classification")
st.caption("VGG16 + MobileNetV2 hybrid model — upload a banana leaf image to get a prediction.")

uploaded = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png", "bmp", "webp"])

col_a, col_b = st.columns(2)

if uploaded is not None:
    pil_img = Image.open(uploaded)

    with col_a:
        st.image(pil_img, caption="Input image", use_container_width=True)

    model = load_model()
    batch = preprocess(pil_img)
    probs = model.predict(batch, verbose=0)[0]

    top_idx = int(np.argmax(probs))

    with col_b:
        st.image((batch[0] * 255).astype("uint8"), caption="After CLAHE preprocessing",
                 use_container_width=True)

    st.success(f"**Prediction: {CLASS_NAMES[top_idx]}**  \nConfidence: {probs[top_idx] * 100:.2f}%")

    st.subheader("Class probabilities")
    order = np.argsort(probs)[::-1]
    for i in order:
        st.write(f"{CLASS_NAMES[i]} — {probs[i] * 100:.2f}%")
        st.progress(float(probs[i]))
else:
    st.info("Upload an image to run the classifier.")

with st.expander("About this model"):
    st.markdown(
        """
- **Architecture:** VGG16 and MobileNetV2 (both frozen) → GlobalAveragePooling on each branch →
  Concatenate → Dense(128, ReLU) → Dropout(0.1) → Dense(6, Softmax)
- **Input:** 224×224 RGB
- **Preprocessing:** RGB→LAB, CLAHE (clipLimit 2.0, 8×8 tiles) on the L channel,
  3×3 Gaussian blur, back to RGB, rescale 1/255
- **Classes:** 6 banana leaf conditions
        """
    )
