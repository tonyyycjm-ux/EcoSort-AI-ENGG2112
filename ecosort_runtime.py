"""English local interface for the completed expanded model and verified rules."""
import os
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
import json
import hashlib
from pathlib import Path
import threading
import time
import urllib.error
import urllib.request
import numpy as np
from PIL import Image, ImageOps
from scipy.special import softmax
import tensorflow as tf
import gradio as gr
from sydney_rules import ITEM_TYPES, disposal_guidance

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "trained_expanded"
DISPLAY = {"battery":"Battery", "cardboard":"Cardboard", "ceramics":"Ceramics", "ewaste":"Electronic waste", "food_organics":"Food organics", "glass":"Glass", "metal":"Metal", "paper":"Paper", "plastic":"Plastic", "textile_trash":"Textiles", "trash":"Other waste", "vegetation":"Vegetation", "wood":"Wood", "mixed_items":"Mixed / unspecified material", "household_problem_waste":"Household problem waste", "light_bulbs":"Light bulbs", "medicines":"Medicines"}

def application_running():
    try:
        with urllib.request.urlopen("http://127.0.0.1:7861/config", timeout=1) as response:
            config = json.load(response)
        return isinstance(config, dict) and config.get("title") == "EcoSort AI"
    except (urllib.error.URLError, OSError, ValueError):
        return False

class EcoSort:
    def __init__(self, results=None):
        if results is None:
            active = json.loads((BASE / "active_model.json").read_text())["results"] if (BASE / "active_model.json").exists() else "trained_expanded"
            assert active in {"trained_expanded", "trained_semantic"}, "Unknown active model."
            results = BASE / active
        assert results.name in {"trained_expanded", "trained_semantic"}, "Archived experiments are not deployable."
        completion = json.loads((results / "training_completed.json").read_text())
        assert completion["status"] == "completed", "Complete expanded training first."
        self.settings = json.loads((results / "app_settings.json").read_text())
        self.classes = self.settings["classes"]
        mobile_settings = json.loads((RESULTS / "app_settings.json").read_text())
        assert self.classes == mobile_settings["classes"], "Classifier label order mismatch."
        self.mobile_temperature = mobile_settings["temperature"]
        self.model = tf.keras.models.load_model(RESULTS / "ecosort_model.keras", compile=False)
        assert self.model.output_shape[-1] == len(self.classes)
        self.lock = threading.Lock()
        self.semantic = None
        if results.name == "trained_semantic":
            assert self.settings["mobile_model_sha256"] == hashlib.sha256((RESULTS / "ecosort_model.keras").read_bytes()).hexdigest(), "Ensemble base model changed."
            from semantic_model import SemanticPredictor
            self.semantic = SemanticPredictor(self.settings)

    def predict(self, image):
        if image is None:
            raise gr.Error("Upload an image or capture a photo first.")
        if not isinstance(image, Image.Image):
            raise gr.Error("The input must be an image.")
        rgb = ImageOps.exif_transpose(image).convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
        pixels = np.asarray(rgb, dtype=np.float32)[None, ...]
        with self.lock:
            raw = self.model(pixels, training=False).numpy()[0]
            if self.semantic is not None:
                from semantic_model import combine
                linear, zero = self.semantic.logits(image)
                mobile = softmax(np.log(np.clip(raw, 1e-8, 1)) / self.mobile_temperature)
                scores = combine(mobile, linear, zero, self.settings)
            else:
                scores = softmax(np.log(np.clip(raw, 1e-8, 1)) / self.settings["temperature"])
        if scores.shape != (len(self.classes),) or not np.isfinite(scores).all():
            raise gr.Error("Invalid model output. Please try another image.")
        order = np.argsort(scores)
        label = self.classes[int(order[-1])]
        confidence = float(scores[order[-1]])
        uncertain = confidence < self.settings["review_threshold"] or float(scores[order[-1]] - scores[order[-2]]) < self.settings["minimum_top_two_margin"] or label in self.settings["manual_review_classes"]
        return label, confidence, uncertain, dict(zip(self.classes, map(float, scores)))

    def classify(self, image, item_type="Not confirmed", food_service=False):
        item_type = item_type or "Not confirmed"
        label, confidence, uncertain, scores = self.predict(image)
        route, guidance = disposal_guidance(label, item_type, food_service, uncertain)
        status = "Unable to identify reliably. Top candidate" if uncertain else "Model suggestion"
        decision = f"{status}: {DISPLAY[label]} ({confidence:.1%}) | {route}"
        return {DISPLAY[key]: value for key, value in scores.items()}, decision, guidance

def launch(prevent_thread_lock=False, port=7861):
    engine = EcoSort()
    # shortcut: the pinned Gradio frontend export controls built-in translations; recheck after upgrading.
    core = next((Path(gr.__file__).parent / "templates/frontend/assets").glob("core-*.js")).name
    english = f"<script type='module'>import {{setupi18n, changeLocale}} from '/assets/{core}'; await setupi18n({{}}, 'en'); changeLocale('en'); document.documentElement.lang = 'en';</script>"
    with gr.Blocks(title="EcoSort AI", analytics_enabled=False) as demo:
        gr.Markdown("# EcoSort AI")
        with gr.Row():
            with gr.Column():
                image = gr.Image(sources=["upload", "webcam"], type="pil", label="Waste item", height=450)
                item_type = gr.Dropdown(ITEM_TYPES, value="Not confirmed", label="Confirmed item type")
                food_service = gr.Checkbox(value=False, label="I have access to a City of Sydney food scraps service")
                with gr.Row():
                    submit = gr.Button("Identify item", variant="primary")
                    clear = gr.ClearButton([image, item_type], value="Clear")
            with gr.Column():
                decision = gr.Textbox(label="Identification and disposal route", lines=3, interactive=False)
                scores = gr.Label(num_top_classes=5, label="Candidate scores (not a confirmed identification)")
                guidance = gr.Markdown()
        clear.add([food_service, scores, decision, guidance])
        submit.click(engine.classify, [image, item_type, food_service], [scores, decision, guidance])
        image.change(lambda: ("Not confirmed", {}, "", ""), outputs=[item_type, scores, decision, guidance])
        item_type.change(lambda: ("", ""), outputs=[decision, guidance])
        food_service.change(lambda: ("", ""), outputs=[decision, guidance])
    demo.launch(share=False, server_name="127.0.0.1", server_port=port, inbrowser=False, prevent_thread_lock=prevent_thread_lock, show_error=True, footer_links=[], head=english)
    return demo

if __name__ == "__main__":
    launch()
