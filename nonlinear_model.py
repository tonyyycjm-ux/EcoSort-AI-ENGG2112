"""Train-only nonlinear and visual-neighbor classifiers over frozen features."""
import numpy as np
import torch
from torch import nn


class FeatureHead(nn.Module):
    def __init__(self, classes, hidden=768, inputs=512):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(inputs, hidden), nn.LayerNorm(hidden), nn.GELU(),
                                    nn.Dropout(.3), nn.Linear(hidden, classes))

    def forward(self, features):
        return self.layers(features)


def neighbor_probabilities(queries, references, labels, classes, settings, device):
    assert references.shape[0] == len(labels) and references.shape[1] == queries.shape[1]
    assert np.isfinite(queries).all() and np.isfinite(references).all()
    assert max(s[0] for s in settings) <= len(references)
    reference = torch.as_tensor(np.asarray(references).copy(), device=device)
    reference = nn.functional.normalize(reference, dim=1)
    labels = torch.tensor(np.asarray(labels).copy(), dtype=torch.long, device=device)
    results = {str((k, temperature)): [] for k, temperature in settings}
    with torch.inference_mode():
        for start in range(0, len(queries), 128):
            query = torch.as_tensor(np.asarray(queries[start:start+128]).copy(), device=device)
            similarities = nn.functional.normalize(query, dim=1) @ reference.T
            values, indices = similarities.topk(max(k for k, _ in settings), dim=1)
            for k, temperature in settings:
                weights = ((values[:, :k] - values[:, :1]) / temperature).exp()
                probability = torch.zeros((len(query), classes), device=device)
                probability.scatter_add_(1, labels[indices[:, :k]], weights)
                probability /= probability.sum(dim=1, keepdim=True)
                results[str((k, temperature))].append(probability.cpu().numpy())
    return {key: np.concatenate(values) for key, values in results.items()}


if __name__ == "__main__":
    x = np.eye(4, dtype=np.float32)
    result = neighbor_probabilities(x, x, np.arange(4), 4, [(1,.07), (3,.07)], "cpu")
    assert all(np.array_equal(value.argmax(1), np.arange(4)) and np.allclose(value.sum(1),1) for value in result.values())
    assert FeatureHead(17)(torch.zeros(2,512)).shape == (2,17)
    print("Feature head and neighbor checks passed.")


class VisualPredictor:
    """Load completed, fixed-split candidates for local inference."""
    def __init__(self, results):
        import hashlib
        import json
        from pathlib import Path
        import pandas as pd
        from semantic_model import load_encoder
        self.base = Path(__file__).resolve().parent
        self.settings = json.loads((results/"app_settings.json").read_text())
        source = self.base/"data/expanded/split_manifest.csv"
        assert hashlib.sha256(source.read_bytes()).hexdigest() == self.settings["manifest_sha256"]
        assert self.settings["preprocess_version"] == "full-object-letterbox-v2"
        self.encoder,self.transform,self.device = load_encoder()
        self.dino = None
        if self.settings["feature_source"] == "features_clip_dino":
            from extract_dino_features import load_dino
            self.dino,self.dino_transform,_ = load_dino()
        with np.load(results/"normalization.npz") as archive:
            self.mean,self.std = archive["mean"],archive["std"]
        self.heads = {}
        selected = self.settings["selected"]
        names = [selected["first"],selected["second"]] if "first" in selected else [selected["name"]]
        expanded = set(names)
        if "average_fine" in names:
            expanded.update(self.settings["fine_members"])
            expanded.remove("average_fine")
        for name in expanded:
            if name.startswith("knn:"):
                continue
            assert name == "material" or name in self.settings.get("fine_members",["fine_label"]), "Unsupported inference component."
            checkpoint = torch.load(results/f"{name}.pt",map_location="cpu",weights_only=True)
            model = FeatureHead(checkpoint["outputs"],inputs=checkpoint.get("inputs",512)).to(self.device).eval()
            model.load_state_dict(checkpoint["state_dict"])
            self.heads[name] = (model,checkpoint["mapping"])
        self.references = None
        if any(name.startswith("knn:") for name in names):
            manifest = pd.read_csv(source)
            frame = manifest[manifest.split=="train"].sort_values(["label","path"])
            self.reference_labels = frame.label.map(dict(zip(self.settings["classes"],range(len(self.settings["classes"]))))).to_numpy()
            directory = self.base/self.settings["feature_source"]
            marker = json.loads((directory/"features_train.json").read_text())
            assert marker["completed"] and marker["manifest_sha256"] == self.settings["manifest_sha256"]
            self.references = np.load(directory/"features_train.npy",mmap_mode="r")
            assert self.references.shape == (len(self.reference_labels),len(self.mean))

    def from_features(self,features):
        import ast
        from scipy.special import softmax
        assert features.ndim == 2 and features.shape[1] == len(self.mean) and np.isfinite(features).all()
        def predict(name):
            if name == "average_fine":
                return np.mean([predict(member) for member in self.settings["fine_members"]],axis=0)
            if name.startswith("knn:"):
                pair = tuple(ast.literal_eval(name[4:]))
                return neighbor_probabilities(features,self.references,self.reference_labels,len(self.settings["classes"]),[pair],self.device)[str(pair)]
            model,mapping = self.heads[name]
            with torch.inference_mode():
                raw = model(torch.tensor((features-self.mean)/self.std,device=self.device)).softmax(1).cpu().numpy()
            p = np.zeros((len(features),len(self.settings["classes"])),dtype=np.float32)
            for i,target in enumerate(mapping):
                p[:,target] += raw[:,i]
            return p
        selected = self.settings["selected"]
        p = (selected["weight"]*predict(selected["first"])+(1-selected["weight"])*predict(selected["second"])
             if "first" in selected else predict(selected["name"]))
        result = softmax(np.log(np.clip(p,1e-8,1))/self.settings["temperature"],axis=1)
        assert np.isfinite(result).all() and np.allclose(result.sum(1),1,atol=1e-5)
        return result

    def scores(self,image):
        from PIL import ImageOps
        from semantic_model import image_features
        image = ImageOps.exif_transpose(image).convert("RGB")
        features = image_features(self.encoder,self.transform(image).unsqueeze(0),self.device)
        if self.dino is not None:
            from extract_dino_features import dino_features
            features = np.concatenate([features,dino_features(self.dino,self.dino_transform(image).unsqueeze(0),self.device)],axis=1)
        return self.from_features(features)[0]
