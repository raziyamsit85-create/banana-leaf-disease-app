"""
Banana Leaf Disease Diagnosis — Gradio Space.

Same model and preprocessing as the Flask build; only the UI layer changed.
Run locally:  python app.py   ->  http://localhost:7860
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import gradio as gr
import numpy as np
from PIL import Image

from disease_info import get_info
from model_utils import (
    CLASS_NAMES,
    get_model,
    gradcam_heatmap,
    overlay_heatmap,
    prepare_image,
)

SEVERITY_COLOR = {
    "healthy": "#3e7c4a",
    "moderate": "#c98a1e",
    "severe": "#8e2f3f",
    "unknown": "#4c5a50",
}


def diagnose(image, show_gradcam):
    """image: PIL.Image from gr.Image(type='pil'). Returns (verdict_md, probs, slider, advice_md)."""
    if image is None:
        raise gr.Error("Load a leaf photo first.")

    model = get_model()
    batch, display_rgb = prepare_image(image)

    probs = model.predict(batch, verbose=0)[0].astype(float)
    top_index = int(np.argmax(probs))
    top_label = CLASS_NAMES[top_index]
    info = get_info(top_label)

    label_output = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}

    colour = SEVERITY_COLOR.get(info["severity"], SEVERITY_COLOR["unknown"])
    verdict = (
        f'<div class="verdict">'
        f'<span class="sev" style="background:{colour}">{info["severity"]}</span>'
        f"<h3>{top_label}</h3>"
        f'<p class="bn">{info["bn"]}</p>'
        f'<p class="score"><b>{probs[top_index] * 100:.2f}%</b> confidence</p>'
        f'<p class="sum">{info["summary"]}</p>'
        f"</div>"
    )

    advice = "### What to do next\n" + "\n".join(f"- {a}" for a in info["advice"])

    if show_gradcam:
        heatmap = gradcam_heatmap(model, batch, top_index)
        if heatmap is not None:
            cam = overlay_heatmap(display_rgb, heatmap)
            slider = gr.update(
                value=(Image.fromarray(display_rgb), Image.fromarray(cam)),
                visible=True,
            )
        else:
            slider = gr.update(visible=False)
    else:
        slider = gr.update(visible=False)

    return verdict, label_output, slider, advice


CSS = """
.gradio-container{max-width:1100px!important}
#title h1{font-size:1.7rem;margin-bottom:.15rem;letter-spacing:-.02em}
#title p{color:#4c5a50;font-size:.9rem;margin:0}
.verdict{border-left:3px solid #d6ddcb;padding:2px 0 2px 16px}
.verdict h3{margin:.5rem 0 .1rem;font-size:1.5rem;line-height:1.15}
.verdict .sev{display:inline-block;color:#fff;font-size:.66rem;letter-spacing:.14em;
  text-transform:uppercase;padding:3px 8px;border-radius:2px}
.verdict .bn{margin:0;color:#4c5a50}
.verdict .score{margin:.6rem 0 .2rem;font-size:1.05rem}
.verdict .sum{margin:.4rem 0 0;font-size:.92rem;line-height:1.55;color:#33403a}
footer{display:none!important}
"""

THEME = gr.themes.Soft(primary_hue="green", neutral_hue="stone")

with gr.Blocks(title="Banana Leaf Disease Diagnosis") as demo:

    gr.HTML(
        '<div id="title">'
        "<h1>Banana leaf disease diagnosis</h1>"
        "<p>VGG16 + MobileNetV2 hybrid · 6 classes · 224×224 · Grad-CAM on the VGG16 branch</p>"
        "</div>"
    )

    with gr.Row():
        with gr.Column(scale=1):
            image_in = gr.Image(type="pil", label="Leaf photo", height=320)
            gradcam_in = gr.Checkbox(value=True, label="Show Grad-CAM attention")
            run_btn = gr.Button("Diagnose leaf", variant="primary")
            gr.Markdown(
                "The image is enhanced with CLAHE on the LAB lightness channel "
                "before it reaches the network — the same pipeline used during training."
            )

        with gr.Column(scale=1):
            verdict_out = gr.HTML()
            cam_out = gr.ImageSlider(
                label="Preprocessed  ↔  Grad-CAM",
                visible=False,
                height=300,
            )
            probs_out = gr.Label(num_top_classes=6, label="Class probabilities")
            advice_out = gr.Markdown()

    if os.path.isdir("examples"):
        example_files = sorted(
            os.path.join("examples", f)
            for f in os.listdir("examples")
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
        )
        if example_files:
            gr.Examples(examples=example_files, inputs=image_in, label="Sample leaves")

    gr.Markdown(
        "Research prototype. Confirm any severe diagnosis with a plant pathologist "
        "before treating a field."
    )

    run_btn.click(
        diagnose,
        inputs=[image_in, gradcam_in],
        outputs=[verdict_out, probs_out, cam_out, advice_out],
    )


demo.queue(max_size=12)

# Launched at import time so it works both with `python app.py` locally and
# with the way Hugging Face Spaces starts a Gradio app.
demo.launch(
    theme=THEME,
    css=CSS,
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7860)),
)
