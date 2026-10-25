"""English local interface for the project-trained MobileNetV3 classifier."""
import os
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
import hashlib
import json
from pathlib import Path
import threading
import urllib.error
import urllib.request
import webbrowser

import gradio as gr
from PIL import Image
from classifier import TrainedPredictor
from disposal_rules import ITEM_TYPES, disposal_guidance

BASE = Path(__file__).resolve().parent
PORT = 7862
DISPLAY = {"battery":"Battery", "cardboard":"Cardboard", "ceramics":"Ceramics", "ewaste":"Electronic waste", "food_organics":"Food organics", "glass":"Glass", "metal":"Metal", "paper":"Paper", "plastic":"Plastic", "textile_trash":"Textiles", "trash":"Other waste", "vegetation":"Vegetation", "wood":"Wood", "mixed_items":"Mixed / unspecified material", "household_problem_waste":"Household problem waste", "light_bulbs":"Light bulbs", "medicines":"Medicines"}


class EcoSort:
    def __init__(self):
        self.classifier = TrainedPredictor()
        self.lock = threading.Lock()

    def classify(self, image, item_type="Not confirmed", food_service=False):
        if not isinstance(image, Image.Image):
            raise gr.Error("Upload an image or capture a photo first.")
        item_type = item_type or "Not confirmed"
        if item_type not in ITEM_TYPES:
            raise gr.Error("Unknown item confirmation.")
        with self.lock:
            result = self.classifier.identify(image, food_service)
        route, guidance = result["route"], result["guidance"]
        if item_type != "Not confirmed":
            route, guidance = disposal_guidance(result["label"], item_type, food_service)
        status = "Unable to identify reliably. Top candidate" if result["uncertain"] else "Model suggestion"
        decision = f"{status}: {result['object']} | {DISPLAY[result['label']]} | {route}"
        return {DISPLAY[k]:v for k,v in result["scores"].items()}, decision, guidance


def verify_release():
    record = json.loads((BASE/"evidence/trained_model_verified.json").read_text())
    expected = {"app.py", "classifier.py", "disposal_rules.py", "evaluate.py", "train.py"}
    assert set(record["code_sha256"]) == expected
    assert record["model"] == "MobileNetV3Large" and record["training_images"] == 117893
    for name, digest in record["code_sha256"].items():
        assert hashlib.sha256((BASE/name).read_bytes()).hexdigest() == digest, "Code changed; run evaluate.py again."
    for name, digest in record["artifact_sha256"].items():
        assert name in {"ecosort_model.keras", "app_settings.json", "training_completed.json"}
        assert hashlib.sha256((BASE/"trained_expanded"/name).read_bytes()).hexdigest() == digest


def launch(prevent_thread_lock=False):
    verify_release()
    url = f"http://127.0.0.1:{PORT}"
    try:
        with urllib.request.urlopen(url+"/config", timeout=1) as response:
            config = json.load(response)
        if config.get("title") != "EcoSort AI - Demo":
            raise RuntimeError(f"Port {PORT} belongs to another application.")
        webbrowser.open(url)
        return None
    except urllib.error.URLError:
        pass
    engine = EcoSort()
    # shortcut: pinned Gradio frontend exports enforce English; recheck after upgrading.
    core = next((Path(gr.__file__).parent/"templates/frontend/assets").glob("core-*.js")).name
    english = f"<script type='module'>import {{setupi18n, changeLocale}} from '/assets/{core}'; await setupi18n({{}}, 'en'); changeLocale('en'); document.documentElement.lang = 'en';</script>"
    with gr.Blocks(title="EcoSort AI - Demo", analytics_enabled=False) as demo:
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
                scores = gr.Label(num_top_classes=5, label="Relative candidate scores (not calibrated probabilities)")
                guidance = gr.Markdown()
        clear.add([food_service, scores, decision, guidance])
        submit.click(engine.classify, [image, item_type, food_service], [scores, decision, guidance])
        image.change(lambda: ("Not confirmed", {}, "", ""), outputs=[item_type, scores, decision, guidance])
        item_type.change(lambda: ("", ""), outputs=[decision, guidance])
        food_service.change(lambda: ("", ""), outputs=[decision, guidance])
    demo.launch(share=False, server_name="127.0.0.1", server_port=PORT, inbrowser=True,
                prevent_thread_lock=prevent_thread_lock, show_error=True, footer_links=[], head=english)
    return demo


if __name__ == "__main__":
    launch()
