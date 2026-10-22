"""Local frozen OpenCLIP features; source-pinned weights, no remote inference."""
from pathlib import Path
import hashlib
import numpy as np
from PIL import Image, ImageOps
import torch
import open_clip
from torch.utils.data import Dataset
from scipy.special import softmax
from download_expanded import download_file

BASE = Path(__file__).resolve().parent
WEIGHTS = BASE / "models/open_clip_model.safetensors"
SHA256 = "ac4f8c4b88af6d963118cbf40ad93176d092abbedfcb752601ae1866352656e6"
REVISION = "1a25a446712ba5ee05982a381eed697ef9b435cf"
PREPROCESS_VERSION = "full-object-letterbox-v2"
URL = f"https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K/resolve/{REVISION}/open_clip_model.safetensors"

ITEMS = {
    "battery": ["household battery", "rechargeable battery", "button cell battery"],
    "cardboard": ["corrugated cardboard box", "cardboard packaging", "piece of cardboard"],
    "ceramics": ["ceramic mug", "porcelain plate", "broken ceramic crockery", "ceramic vase"],
    "ewaste": ["electronic device", "computer keyboard", "mobile phone", "electrical appliance", "circuit board", "electrical cable"],
    "food_organics": ["food scraps", "fruit peel", "vegetable leftovers", "discarded meat or fish"],
    "glass": ["glass bottle", "glass jar", "broken glass", "glass container"],
    "household_problem_waste": ["household chemical container", "paint can", "gas cylinder", "fire extinguisher"],
    "light_bulbs": ["light bulb", "fluorescent light tube", "LED light bulb"],
    "medicines": ["medicine tablets", "pharmaceutical blister pack", "medicine bottle"],
    "metal": ["aluminium beverage can", "steel food tin", "metal cutlery", "scrap metal"],
    "mixed_items": ["mixed-material household object", "assortment of household items made from different materials"],
    "paper": ["newspaper", "sheet of printed paper", "paper bag", "book", "magazine", "paper ticket"],
    "plastic": ["plastic drink bottle", "plastic food container", "plastic bag", "plastic packaging", "plastic household item"],
    "textile_trash": ["cotton clothing", "fabric garment", "towel", "piece of textile"],
    "trash": ["used tissue", "disposable nappy", "cigarette butt", "disposable face mask", "miscellaneous residual rubbish"],
    "vegetation": ["garden leaves", "grass clippings", "plant prunings"],
    "wood": ["wooden object", "wooden utensil", "piece of timber"],
}

class FullImageTransform:
    def __init__(self, transform):
        self.transform = transform
        self.fill = tuple(round(value*255) for value in transform.transforms[-1].mean)

    def __call__(self, image):
        # Center-cropping narrow objects can remove their identifying shape.
        image = ImageOps.pad(image, (224,224), method=Image.Resampling.BICUBIC, color=self.fill)
        return self.transform(image)

class Images(Dataset):
    """Importable dataset for macOS spawn workers, including Jupyter runs."""
    def __init__(self, paths, transform):
        self.paths, self.transform = paths, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            return self.transform(ImageOps.exif_transpose(image).convert("RGB"))

def load_encoder(download=False):
    if download:
        download_file(URL, WEIGHTS, 605143316, SHA256)
    assert WEIGHTS.is_file(), "Download the pinned OpenCLIP weights first."
    with WEIGHTS.open("rb") as source:
        assert hashlib.file_digest(source, "sha256").hexdigest() == SHA256
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.set_num_threads(8)
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained=str(WEIGHTS), device=device)
    model.eval()
    if device == "mps":
        model.half()
    return model, FullImageTransform(preprocess), device

def image_features(model, batch, device):
    with torch.inference_mode():
        values = model.encode_image(batch.to(device=device, dtype=next(model.parameters()).dtype))
        values = values.float()
        values /= values.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return values.cpu().numpy()

def text_features(model, classes, device):
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    templates = ["a photo of {}.", "a close-up photo of {}.", "a photo of discarded {}."]
    vectors = []
    with torch.inference_mode():
        for label in classes:
            tokens = tokenizer([template.format(item) for item in ITEMS[label] for template in templates]).to(device)
            values = model.encode_text(tokens).float()
            values /= values.norm(dim=-1, keepdim=True)
            mean = values.mean(dim=0)
            vectors.append((mean / mean.norm()).cpu().numpy())
    return np.stack(vectors)

def combine(mobile, linear, zero, settings):
    weights = settings["weights"]
    result = weights[0] * mobile + weights[1] * softmax(linear / settings["linear_temperature"], axis=-1) + weights[2] * softmax(zero / settings["zero_temperature"], axis=-1)
    assert np.isfinite(result).all() and np.allclose(result.sum(axis=-1), 1, atol=1e-5)
    return result

class SemanticPredictor:
    def __init__(self, settings):
        self.settings = settings
        assert settings["preprocess_version"] == PREPROCESS_VERSION, "Classifier preprocessing mismatch."
        self.model, self.preprocess, self.device = load_encoder()
        with np.load(BASE / "trained_semantic/classifier.npz") as archive:
            self.coef = archive["coef"]
            self.intercept = archive["intercept"]
            self.text = archive["text"]
        assert self.coef.shape == self.text.shape == (len(settings["classes"]),512)
        assert self.intercept.shape == (len(settings["classes"]),)
        assert np.isclose(sum(settings["weights"]),1) and min(settings["weights"]) >= 0

    def logits(self, image):
        pixels = self.preprocess(ImageOps.exif_transpose(image).convert("RGB")).unsqueeze(0)
        features = image_features(self.model, pixels, self.device)
        return (features @ self.coef.T + self.intercept)[0], (features @ self.text.T * self.settings["zero_logit_scale"])[0]

if __name__ == "__main__":
    encoder, preprocess, device = load_encoder(download=True)
    print("Loaded verified OpenCLIP encoder on", device, flush=True)
