"""Add complementary frozen DINOv2 shape/texture features without changing splits."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader
from torchvision import transforms

from download_expanded import download_file
from semantic_model import Images

BASE = Path(__file__).resolve().parent
OUT = BASE / "features_clip_dino"
WEIGHTS = BASE / "models/dinov2_small.safetensors"
REVISION = "4610ca143709d58a633b6397a74412c2c3842454"
DIGEST = "04d27f3400d059fc0cfd7d17dd1909a75bf3ea8fb3eeb48b97cb99e57ee20081"


class DinoTransform:
    def __init__(self):
        self.tensor = transforms.Compose([transforms.ToTensor(),transforms.Normalize((.485,.456,.406),(.229,.224,.225))])

    def __call__(self,image):
        return self.tensor(ImageOps.pad(image,(224,224),method=Image.Resampling.BICUBIC,color=(124,116,104)))


def load_dino(download=False):
    if download:
        download_file(f"https://huggingface.co/timm/vit_small_patch14_dinov2.lvd142m/resolve/{REVISION}/model.safetensors",WEIGHTS,88240510,DIGEST)
    with WEIGHTS.open("rb") as source:
        assert hashlib.file_digest(source,"sha256").hexdigest() == DIGEST
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.set_num_threads(8)
    model = timm.create_model("vit_small_patch14_dinov2.lvd142m",pretrained=True,
                              pretrained_cfg_overlay={"file":str(WEIGHTS)},img_size=224,num_classes=0).to(device).eval()
    if device == "mps":
        model.half()
    return model,DinoTransform(),device


def dino_features(model,batch,device):
    with torch.inference_mode():
        tokens = model.forward_features(batch.to(device=device,dtype=next(model.parameters()).dtype)).float()
        cls = torch.nn.functional.normalize(tokens[:,0],dim=1)
        patches = torch.nn.functional.normalize(tokens[:,1:].mean(1),dim=1)
        result = torch.cat([cls,patches],dim=1).cpu().numpy()
    assert result.shape == (len(batch),768) and np.isfinite(result).all()
    return result


def main():
    OUT.mkdir(exist_ok=True)
    source = BASE / "data/expanded/split_manifest.csv"
    signature = hashlib.sha256(source.read_bytes()).hexdigest()
    frame = pd.read_csv(source)
    model,transform,device = load_dino(download=True)
    for split in ("train","validation","test"):
        rows = frame[frame.split == split].sort_values(["label","path"]).reset_index(drop=True)
        clip = np.load(BASE/f"trained_semantic/features_{split}.npy",mmap_mode="r")
        marker = OUT/f"features_{split}.json"
        metadata = json.loads(marker.read_text()) if marker.exists() else {}
        assert not metadata or metadata["manifest_sha256"] == signature
        if metadata.get("completed"):
            continue
        path = OUT/f"features_{split}.npy"
        values = np.lib.format.open_memmap(path,mode="r+" if path.exists() else "w+",dtype=np.float32,shape=(len(rows),1280))
        start = position = metadata.get("rows",0)
        began = time.perf_counter()
        loader = DataLoader(Images(rows.path.iloc[start:].to_list(),transform),batch_size=64,num_workers=2,persistent_workers=True)
        for index,batch in enumerate(loader):
            dino = dino_features(model,batch,device)
            values[position:position+len(batch)] = np.concatenate([clip[position:position+len(batch)],dino],axis=1)
            position += len(batch)
            if index % 50 == 0 or position == len(rows):
                values.flush()
                marker.write_text(json.dumps(dict(manifest_sha256=signature,preprocess_version="full-object-letterbox-v2",dino_preprocess="224-letterbox-cls-patchmean-v1",weights_sha256=DIGEST,weights_revision=REVISION,rows=position,completed=position==len(rows))))
                rate = (position-start)/max(time.perf_counter()-began,.001)
                print(f"DINO {split} {position}/{len(rows)}, {rate:.1f} images/s, remaining {(len(rows)-position)/max(rate,1)/60:.1f} min",flush=True)
        assert position == len(rows)


if __name__ == "__main__":
    main()
