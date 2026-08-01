"""
Banana Leaf Disease Classifier — VGG16 + MobileNetV2 hybrid
Streamlit Community Cloud deployment.
"""

import os
import numpy as np
import cv2
from PIL import Image
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, GlobalAveragePooling2D, Concatenate
)
from tensorflow.keras.applications import VGG16, MobileNetV2

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------

# Explicit location, if you keep the weights somewhere specific. Relative paths
# are tried against both app.py's folder and the working directory. Leave as
# None to rely on the automatic search below.
WEIGHTS_PATH = "banana-leaf-disease-app/vgg16_mobilenet_hybrid_weights.h5"

# Fallback filenames, searched when WEIGHTS_PATH doesn't resolve.
WEIGHTS_CANDIDATES = [
    "vgg16_mobilenet_hybrid_weights.weights.h5",
    "vgg16_mobilenet_hybrid_weights.h5",
]
APP_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_SIZE = (224, 224)

# Order must match ImageDataGenerator.class_indices from training (alphabetical)
CLASS_NAMES = [
    "Banana Healthy Leaf",
    "Banana Insect Pest Disease",
    "Banana Moko Disease or dead",
    "Banana Yellow Sigatoka Disease",
    "Black Sigotika",
    "Fusarium Wilt Panama",
]

CLASS_NOTES = {
    "Banana Healthy Leaf": "No disease signs detected. Keep monitoring during wet season.",
    "Banana Insect Pest Disease": "Feeding damage present. Inspect the underside of leaves and the pseudostem.",
    "Banana Moko Disease or dead": "Bacterial wilt (Ralstonia). Isolate the mat and disinfect tools before reuse.",
    "Banana Yellow Sigatoka Disease": "Fungal leaf spot. Remove affected leaves and improve field drainage.",
    "Black Sigotika": "Black Sigatoka. More aggressive than the yellow form; treat early.",
    "Fusarium Wilt Panama": "Soil-borne fungus. Avoid moving soil or planting material off the affected plot.",
}

st.set_page_config(
    page_title="Banana Leaf Disease Classifier",
    page_icon="🍌",
    layout="wide",
)

# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------


def resolve_weights_path() -> str:
    """Return a path Keras will load with the correct backend.

    Keras 3 chooses its loader from the filename suffix alone: `.weights.h5`
    uses the native format, plain `.h5` uses the legacy HDF5 reader. This
    checkpoint is native format, so if it is sitting under a plain `.h5` name
    the legacy reader sees zero layers. Detect that and expose the file under a
    `.weights.h5` alias.
    """
    import glob

    import h5py  # ships with TensorFlow

    search_dirs = [
        APP_DIR,
        os.getcwd(),
        os.path.join(APP_DIR, "banana-leaf-disease-app"),
        os.path.join(os.getcwd(), "banana-leaf-disease-app"),
        os.path.join(APP_DIR, "weights"),
    ]
    seen = set()

    path = None

    # 1. Explicit setting, resolved against app dir and CWD
    if WEIGHTS_PATH:
        for base in (APP_DIR, os.getcwd(), ""):
            candidate = (
                WEIGHTS_PATH
                if os.path.isabs(WEIGHTS_PATH) or not base
                else os.path.join(base, WEIGHTS_PATH)
            )
            if os.path.exists(candidate):
                path = candidate
                break

    # 2. Known filenames in likely directories
    if path is None:
        for d in search_dirs:
            if d in seen or not os.path.isdir(d):
                continue
            seen.add(d)
            for name in WEIGHTS_CANDIDATES:
                candidate = os.path.join(d, name)
                if os.path.exists(candidate):
                    path = candidate
                    break
            if path:
                break

    # 3. Any .h5 in those directories, or one level below them
    if path is None:
        for d in seen:
            found = sorted(
                glob.glob(os.path.join(d, "*.h5"))
                + glob.glob(os.path.join(d, "*", "*.h5"))
            )
            if found:
                path = found[0]
                break

    if path is None:
        listing = []
        for d in seen:
            entries = sorted(os.listdir(d))[:25]
            listing.append(f"{d} -> {entries if entries else 'empty'}")
        raise FileNotFoundError(
            "No .h5 weights file found. Set WEIGHTS_PATH at the top of app.py "
            "to the file's location, or put "
            f"'{WEIGHTS_CANDIDATES[0]}' next to app.py.\n\n"
            "Searched:\n" + "\n".join(listing)
        )

    if path.endswith(".weights.h5"):
        return path

    with h5py.File(path, "r") as f:
        is_native = "layers" in f.keys()

    if not is_native:
        return path  # genuine legacy HDF5 checkpoint

    alias = "/tmp/vgg16_mobilenet_hybrid_weights.weights.h5"
    if not os.path.exists(alias):
        try:
            os.symlink(os.path.abspath(path), alias)
        except OSError:
            import shutil

            shutil.copyfile(path, alias)
    return alias


@st.cache_resource(show_spinner="Loading the hybrid model…")
def load_model():
    """Rebuild the training architecture exactly, then load the trained weights.

    weights=None on both backbones: every layer is overwritten by the
    checkpoint, so downloading ImageNet weights at startup is wasted time.
    """
    inp = Input(shape=(224, 224, 3), name="hybrid1_input")

    vgg_base = VGG16(weights=None, include_top=False, input_tensor=inp)
    mob_base = MobileNetV2(weights=None, include_top=False, input_tensor=inp)

    for layer in vgg_base.layers:
        layer.trainable = False
    for layer in mob_base.layers:
        layer.trainable = False

    vgg_feat = GlobalAveragePooling2D(name="gap_vgg16_h1")(vgg_base.output)
    mob_feat = GlobalAveragePooling2D(name="gap_mobilenetv2_h1")(mob_base.output)

    merged = Concatenate(name="concat_vgg_mobilenet_h1")([vgg_feat, mob_feat])
    x = Dense(128, activation="relu")(merged)
    x = Dropout(0.1)(x)
    out = Dense(len(CLASS_NAMES), activation="softmax")(x)

    model = Model(inputs=inp, outputs=out, name="VGG16_MobileNetV2_Hybrid")

    model.load_weights(resolve_weights_path())
    return model


# ----------------------------------------------------------------------------
# Preprocessing — identical to the training pipeline
# ----------------------------------------------------------------------------

_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def preprocess_banana_disease(image: np.ndarray) -> np.ndarray:
    """RGB -> LAB, CLAHE on L, Gaussian blur, back to RGB."""
    img = np.clip(image, 0, 255).astype(np.uint8)

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    l_enhanced = _clahe.apply(l)
    l_denoised = cv2.GaussianBlur(l_enhanced, (3, 3), 0)

    lab_out = cv2.merge([l_denoised, a, b])
    rgb_out = cv2.cvtColor(lab_out, cv2.COLOR_LAB2RGB)
    return rgb_out.astype(np.float32)


def prepare_input(pil_img: Image.Image):
    """Returns (batch for the model, enhanced image for display)."""
    # Keras flow_from_dataframe resizes with 'nearest' by default
    pil_img = pil_img.convert("RGB").resize(IMG_SIZE, Image.NEAREST)
    arr = np.asarray(pil_img, dtype=np.float32)

    enhanced = preprocess_banana_disease(arr)      # preprocessing_function
    batch = np.expand_dims(enhanced / 255.0, 0)    # rescale=1./255
    return batch, enhanced.astype(np.uint8)


# ----------------------------------------------------------------------------
# Grad-CAM (VGG16 branch, block5_conv3)
# ----------------------------------------------------------------------------


def grad_cam(model, batch, class_index, layer_name="block5_conv3"):
    grad_model = Model(
        inputs=model.inputs,
        outputs=[model.get_layer(layer_name).output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(batch)
        loss = preds[:, class_index]

    grads = tape.gradient(loss, conv_out)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.reduce_sum(conv_out[0] * weights, axis=-1)
    cam = tf.maximum(cam, 0)

    cam = cam.numpy()
    if cam.max() > 0:
        cam = cam / cam.max()
    return cv2.resize(cam, IMG_SIZE)


def overlay_heatmap(base_rgb: np.ndarray, cam: np.ndarray, alpha=0.4):
    heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(base_rgb, 1 - alpha, heat, alpha, 0)


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------

st.title("🍌 Banana Leaf Disease Classifier")
st.caption(
    "VGG16 + MobileNetV2 hybrid · CLAHE-enhanced input · six disease classes"
)

with st.sidebar:
    st.header("About the model")
    st.markdown(
        """
Two frozen ImageNet backbones share one input. Each produces a feature
vector through global average pooling; the two are concatenated and passed
to a 128-unit dense layer with dropout before the softmax head.

**Input pipeline**
1. Resize to 224 × 224
2. RGB → LAB, CLAHE on the L-channel (clip 2.0, 8 × 8 tiles)
3. 3 × 3 Gaussian blur, back to RGB
4. Rescale to [0, 1]
"""
    )
    show_cam = st.checkbox("Show Grad-CAM (VGG16 branch)", value=True)
    st.divider()
    st.caption("Research demo. Not a substitute for field diagnosis by an agronomist.")

uploaded = st.file_uploader(
    "Upload a banana leaf photo", type=["jpg", "jpeg", "png", "bmp", "webp"]
)

if uploaded is None:
    st.info("Upload a single leaf image to run the classifier. JPG, PNG, BMP or WebP.")
    st.stop()

image = Image.open(uploaded)

try:
    model = load_model()
except Exception as exc:  # noqa: BLE001
    st.error(f"The model could not be loaded: {exc}")
    st.stop()

batch, enhanced = prepare_input(image)

with st.spinner("Classifying…"):
    probs = model.predict(batch, verbose=0)[0]

top_idx = int(np.argmax(probs))
top_label = CLASS_NAMES[top_idx]
confidence = float(probs[top_idx])

col_a, col_b, col_c = st.columns(3)
with col_a:
    st.image(image, caption="Uploaded image", use_container_width=True)
with col_b:
    st.image(enhanced, caption="After CLAHE enhancement", use_container_width=True)
with col_c:
    if show_cam:
        cam = grad_cam(model, batch, top_idx)
        st.image(
            overlay_heatmap(enhanced, cam),
            caption="Grad-CAM · VGG16 block5_conv3",
            use_container_width=True,
        )
    else:
        st.empty()

st.subheader(top_label)
st.progress(confidence, text=f"Confidence {confidence:.1%}")
st.write(CLASS_NOTES[top_label])

if confidence < 0.60:
    st.warning(
        "Confidence is low. Retake the photo with the leaf filling the frame "
        "in even daylight, or try a second leaf from the same plant."
    )

with st.expander("All class probabilities"):
    ranked = sorted(zip(CLASS_NAMES, probs), key=lambda p: p[1], reverse=True)
    for name, p in ranked:
        st.write(f"**{name}** — {p:.2%}")
        st.progress(float(p))
