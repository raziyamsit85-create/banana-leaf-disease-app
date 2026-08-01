"""
model_utils.py
--------------
Rebuilds the VGG16 + MobileNetV2 hybrid EXACTLY as it was built in Banana_V2.ipynb
and reproduces the same preprocessing pipeline used during training.

IMPORTANT: the layer order here must stay identical to the training notebook,
because the Keras 3 `.weights.h5` format stores weights positionally
(conv2d, conv2d_1, batch_normalization, ...), not by custom layer name.
"""

import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, GlobalAveragePooling2D, Concatenate, Dense, Dropout
)
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16, MobileNetV2

IMG_SIZE = (224, 224)

# Order comes from flow_from_dataframe(), which sorts folder names alphabetically.
CLASS_NAMES = [
    "Banana Healthy Leaf",
    "Banana Insect Pest Disease",
    "Banana Moko Disease or dead",
    "Banana Yellow Sigatoka Disease",
    "Black Sigotika",
    "Fusarium Wilt Panama",
]

WEIGHTS_PATH = os.environ.get(
    "WEIGHTS_PATH", "vgg16_mobilenet_hybrid_weights.h5"
)

# --------------------------------------------------------------------------- #
# Preprocessing (identical to preprocess_banana_disease() in the notebook)
# --------------------------------------------------------------------------- #
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def preprocess_banana_disease(image):
    """CLAHE on the L channel of LAB + light Gaussian denoising. Input/output RGB."""
    img = np.clip(image, 0, 255).astype(np.uint8)

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    l_enhanced = _clahe.apply(l)
    l_denoised = cv2.GaussianBlur(l_enhanced, (3, 3), 0)

    lab_out = cv2.merge([l_denoised, a, b])
    rgb_out = cv2.cvtColor(lab_out, cv2.COLOR_LAB2RGB)

    return rgb_out.astype(np.float32)


def prepare_image(pil_image):
    """
    PIL.Image -> (1, 224, 224, 3) float32 batch, matching the training generator:
    load_img(target_size=(224,224), interpolation='nearest')
      -> preprocessing_function
      -> rescale 1./255
    Returns (batch, display_rgb_uint8).
    """
    pil_image = pil_image.convert("RGB").resize(IMG_SIZE, resample=0)  # 0 = NEAREST
    arr = np.asarray(pil_image, dtype=np.float32)

    processed = preprocess_banana_disease(arr)
    processed = processed * (1.0 / 255.0)

    batch = np.expand_dims(processed, axis=0)
    display_rgb = np.asarray(pil_image, dtype=np.uint8)
    return batch, display_rgb


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def build_hybrid_model(load_imagenet=False):
    """
    VGG16 + MobileNetV2 dual-branch hybrid.

    load_imagenet=False keeps the Space light and fast: we overwrite every weight
    with the fine-tuned checkpoint anyway, so downloading ImageNet weights is
    pure waste on startup.
    """
    weights = "imagenet" if load_imagenet else None

    input_layer = Input(shape=(224, 224, 3), name="hybrid1_input")

    vgg_base = VGG16(weights=weights, include_top=False, input_tensor=input_layer)
    mobilenet_base = MobileNetV2(
        weights=weights, include_top=False, input_tensor=input_layer
    )

    for layer in vgg_base.layers:
        layer.trainable = False
    for layer in mobilenet_base.layers:
        layer.trainable = False

    vgg_feat = GlobalAveragePooling2D(name="gap_vgg16_h1")(vgg_base.output)
    mobilenet_feat = GlobalAveragePooling2D(name="gap_mobilenetv2_h1")(
        mobilenet_base.output
    )

    merged = Concatenate(name="concat_vgg_mobilenet_h1")([vgg_feat, mobilenet_feat])
    x = Dense(128, activation="relu")(merged)
    x = Dropout(0.1)(x)
    x = Dense(6, activation="softmax")(x)

    model = Model(inputs=input_layer, outputs=x, name="VGG16_MobileNetV2_Hybrid")
    # No compile(): this is inference only. Compiling would make load_weights
    # try to restore the Adam slot variables and emit a mismatch warning.
    return model


def _keras3_weights_path(path):
    """
    Keras picks its loader from the file extension: anything ending in plain
    `.h5` goes to the legacy HDF5 reader, which cannot read this checkpoint
    (it was written by Keras 3 `save_weights`, so the layers live under a
    `layers/` group). Give Keras a `.weights.h5` name via a symlink in /tmp.
    """
    if path.endswith(".weights.h5"):
        return path

    import tempfile

    link_dir = os.path.join(tempfile.gettempdir(), "banana_weights")
    os.makedirs(link_dir, exist_ok=True)
    link = os.path.join(link_dir, "hybrid.weights.h5")

    if os.path.islink(link) or os.path.exists(link):
        os.remove(link)
    try:
        os.symlink(os.path.abspath(path), link)
    except OSError:  # filesystems without symlink support
        import shutil

        shutil.copyfile(path, link)
    return link


_model = None


def get_model():
    """Lazy singleton so gunicorn workers only build the graph once."""
    global _model
    if _model is None:
        if not os.path.exists(WEIGHTS_PATH):
            raise FileNotFoundError(
                f"Weights not found at '{WEIGHTS_PATH}'. "
                "Upload vgg16_mobilenet_hybrid_weights.h5 to the Space root "
                "(use Git LFS - the file is ~71 MB)."
            )
        print("[model] building hybrid architecture ...", flush=True)
        model = build_hybrid_model(load_imagenet=False)
        print(f"[model] loading weights from {WEIGHTS_PATH} ...", flush=True)
        model.load_weights(_keras3_weights_path(WEIGHTS_PATH))
        # Warm-up so the first real request is not slow.
        model.predict(np.zeros((1, 224, 224, 3), dtype=np.float32), verbose=0)
        _model = model
        print("[model] ready.", flush=True)
    return _model


# --------------------------------------------------------------------------- #
# Grad-CAM (VGG16 branch)
# --------------------------------------------------------------------------- #
_grad_models = {}


def _get_grad_model(model, conv_layer_name):
    """Build the gradient sub-model once and reuse it across requests."""
    if conv_layer_name not in _grad_models:
        conv_layer = model.get_layer(conv_layer_name)
        _grad_models[conv_layer_name] = Model(
            inputs=model.input, outputs=[conv_layer.output, model.output]
        )
    return _grad_models[conv_layer_name]


def gradcam_heatmap(model, batch, class_index, conv_layer_name="block5_conv3"):
    """Returns a (H, W) heatmap normalised to [0, 1], or None on failure."""
    try:
        grad_model = _get_grad_model(model, conv_layer_name)

        with tf.GradientTape() as tape:
            inputs = tf.convert_to_tensor(batch)
            tape.watch(inputs)
            conv_out, preds = grad_model(inputs, training=False)
            loss = preds[:, class_index]

        grads = tape.gradient(loss, conv_out)
        if grads is None:
            return None

        pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_out = conv_out[0]
        heatmap = tf.reduce_sum(conv_out * pooled, axis=-1)
        heatmap = tf.maximum(heatmap, 0)
        denom = tf.reduce_max(heatmap)
        if float(denom) == 0.0:
            return None
        heatmap = (heatmap / denom).numpy()
        return heatmap
    except Exception as exc:  # never let XAI break a prediction
        print(f"[gradcam] skipped: {exc}", flush=True)
        return None


def overlay_heatmap(display_rgb, heatmap, alpha=0.4):
    """Blend a Grad-CAM heatmap over the resized input image. Returns RGB uint8."""
    heatmap = cv2.resize(heatmap, (display_rgb.shape[1], display_rgb.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap)
    colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    blended = cv2.addWeighted(display_rgb, 1 - alpha, colored, alpha, 0)
    return blended.astype(np.uint8)
