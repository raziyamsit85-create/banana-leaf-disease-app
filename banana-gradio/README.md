---
title: Banana Leaf Disease Diagnosis
emoji: 🍌
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 6.20.0
app_file: app.py
pinned: false
license: mit
---

# Banana Leaf Disease Diagnosis — VGG16 + MobileNetV2 Hybrid

Gradio build of the banana leaf classifier. Same model, same preprocessing and
same Grad-CAM as the Flask version — only the UI layer changed, so no Docker
SDK is needed.

Dual-branch hybrid: frozen VGG16 + frozen MobileNetV2 → global average pooling
on each branch → concatenate (1792) → Dense 128 → Dropout 0.1 → Softmax 6.

## Classes

| # | Class | Bangla |
|---|-------|--------|
| 0 | Banana Healthy Leaf | সুস্থ কলা পাতা |
| 1 | Banana Insect Pest Disease | পোকামাকড়ের আক্রমণ |
| 2 | Banana Moko Disease or dead | মোকো রোগ |
| 3 | Banana Yellow Sigatoka Disease | ইয়েলো সিগাটোকা |
| 4 | Black Sigotika | ব্ল্যাক সিগাটোকা |
| 5 | Fusarium Wilt Panama | পানামা রোগ |

This is the alphabetical order `flow_from_dataframe()` produced during
training. Do not reorder it.

## Files

```
app.py             Gradio interface
model_utils.py     model rebuild, preprocessing, Grad-CAM
disease_info.py    per-class descriptions and management advice
requirements.txt   pinned dependencies
packages.txt       apt package needed by OpenCV
.gitattributes     Git LFS rule for *.h5
```

Add `vgg16_mobilenet_hybrid_weights.h5` to the project root yourself — it is
not in this zip.

Optional: create an `examples/` folder and drop a few leaf images in it. The
app picks them up automatically and shows them as clickable samples.

## Deploy to Hugging Face Spaces

1. **New Space → SDK: Gradio → Blank.** Leave the hardware on CPU Basic.

2. Clone and copy the files in:

   ```bash
   git clone https://huggingface.co/spaces/<your-username>/<space-name>
   cd <space-name>
   # copy everything from this zip in, including .gitattributes
   ```

3. Add the weights through Git LFS — the file is ~71 MB, over the plain-git
   limit:

   ```bash
   git lfs install
   cp /path/to/vgg16_mobilenet_hybrid_weights.h5 .
   git lfs track "*.h5"
   git add .gitattributes vgg16_mobilenet_hybrid_weights.h5
   ```

4. Push:

   ```bash
   git add .
   git commit -m "Banana leaf disease Gradio app"
   git push
   ```

First build takes 5–10 minutes; TensorFlow is a large wheel. Watch the **Logs**
tab — the app is serving once you see `[model] ready.`

You can also upload every file through **Files → Add file** in the Space's web
UI, which handles LFS for you.

### If the Space fails to build

* `ModuleNotFoundError: cv2` or a libGL error → check that `packages.txt` was
  committed.
* Version conflict on `keras` → the pins in `requirements.txt` are tested
  together (TF 2.17.1 + Keras 3.15.0 + NumPy 1.26.4). Don't loosen them
  casually; NumPy 2 in particular will break this TensorFlow build.
* `sdk_version` mismatch → the version in the README front matter must be one
  Spaces actually offers. If 6.20.0 is rejected, set it to whatever the Space
  settings page lists and change the `gradio==` pin to match.

## Run locally

```bash
pip install -r requirements.txt
# put vgg16_mobilenet_hybrid_weights.h5 next to app.py
python app.py
# http://localhost:7860
```

## Notes on correctness

* **Preprocessing must match training.** `preprocess_banana_disease()` is
  copied verbatim from the notebook: RGB → LAB, CLAHE (clipLimit 2.0, 8×8) on
  the L channel, 3×3 Gaussian blur, back to RGB, then `× 1/255`. Resizing uses
  nearest-neighbour, matching `flow_from_dataframe`. Any change here quietly
  costs accuracy.

* **Layer order must match training.** The checkpoint uses the Keras 3
  `.weights.h5` layout, which stores tensors by position in the layer traversal
  (`conv2d`, `conv2d_1`, `batch_normalization`, …) rather than by layer name.
  VGG16 is built first, then MobileNetV2 — keep that order.

* **The `.h5` filename is handled for you.** Keras routes plain `.h5` files to
  its legacy HDF5 reader, which cannot read this checkpoint and fails with
  `Layer count mismatch ... found 0 saved layers`. `model_utils.py` symlinks
  the file to a `.weights.h5` name in `/tmp` before loading.

* **ImageNet weights are not downloaded** at startup (`weights=None`). The
  checkpoint overwrites every weight anyway.

* If loading ever fails on shapes, the architecture drifted. The concatenated
  feature vector must be 1792 wide (512 from VGG16 + 1280 from MobileNetV2).

## Performance

On CPU Basic, expect roughly 0.7–1.5 s per image, plus about the same again
with Grad-CAM enabled. Uncheck the Grad-CAM box for faster responses. The
Space sleeps when idle, so the first request after a nap includes a cold start
of a minute or two while the model rebuilds.
