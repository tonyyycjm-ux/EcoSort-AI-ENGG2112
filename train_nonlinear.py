"""Validation-selected feature-head experiment; fixed splits, no test fitting."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from scipy.optimize import minimize_scalar
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch import nn

from nonlinear_model import FeatureHead, neighbor_probabilities

BASE = Path(__file__).resolve().parent
OUT = BASE / "trained_nonlinear"


def metrics(p, truth):
    return dict(accuracy=float(accuracy_score(truth,p.argmax(1))),
                macro_f1=float(f1_score(truth,p.argmax(1),average="macro",zero_division=0)))


def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="trained_semantic")
    parser.add_argument("--output", default="trained_nonlinear")
    parser.add_argument("--seeds",type=int,default=1)
    args = parser.parse_args()
    assert 1 <= args.seeds <= 4
    feature_dir = BASE / args.features
    OUT = BASE / args.output
    assert feature_dir.parent == BASE and OUT.parent == BASE
    started = time.perf_counter()
    OUT.mkdir(exist_ok=True)
    assert not (OUT / "training_completed.json").exists(), "Preserve completed experiment evidence."
    torch.manual_seed(20261022)
    torch.set_num_threads(8)
    rng = np.random.default_rng(20261022)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    source = BASE / "data/expanded/split_manifest.csv"
    signature = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = pd.read_csv(source)
    classes = json.loads((BASE / "trained_semantic/app_settings.json").read_text())["classes"]
    assert manifest.sha256.is_unique and manifest.groupby("group_id").split.nunique().max() == 1
    frames = {split:manifest[manifest.split == split].sort_values(["label","path"]).reset_index(drop=True)
              for split in ("train","validation","test")}
    for split in frames:
        marker = json.loads((feature_dir / f"features_{split}.json").read_text())
        assert marker["completed"] and marker["manifest_sha256"] == signature
        assert marker["preprocess_version"] == "full-object-letterbox-v2"
    truth = {split:f.label.map(dict(zip(classes,range(len(classes))))).to_numpy() for split,f in frames.items()}
    features = {split:np.load(feature_dir / f"features_{split}.npy",mmap_mode="r") for split in frames}
    inputs = features["train"].shape[1]
    mean = np.asarray(features["train"]).mean(0)
    std = np.maximum(np.asarray(features["train"]).std(0),1e-3)
    np.savez(OUT / "normalization.npz",mean=mean,std=std)
    x = torch.tensor((features["train"]-mean)/std,device=device)
    v = torch.tensor((features["validation"]-mean)/std,device=device)
    y = torch.tensor(truth["train"],device=device,dtype=torch.long)
    candidates = []
    fine_names = ["fine_label"] + [f"fine_label_seed{i}" for i in range(1,args.seeds)]
    for mode in ["material"] + fine_names:
        if mode.startswith("fine_label"):
            fine = frames["train"].fine_label.astype(str) + "|" + frames["train"].label
            names = sorted(fine.unique())
            index = {name:i for i,name in enumerate(names)}
            targets = torch.tensor(fine.map(index).to_numpy(),device=device,dtype=torch.long)
            mapping = np.array([classes.index(name.rsplit("|",1)[1]) for name in names])
        else:
            targets, mapping = y, np.arange(len(classes))
        head = FeatureHead(len(mapping),inputs=inputs).to(device)
        optimizer = torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.01)
        best, stale, history = -1, 0, []
        for epoch in range(100):
            head.train()
            order = rng.permutation(len(x))
            loss_sum = 0
            for start in range(0,len(x),1024):
                batch = torch.tensor(order[start:start+1024],device=device)
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.cross_entropy(head(x[batch]),targets[batch],label_smoothing=.03)
                loss.backward()
                nn.utils.clip_grad_norm_(head.parameters(),5)
                optimizer.step()
                loss_sum += float(loss.detach().cpu()) * len(batch)
            head.eval()
            with torch.inference_mode():
                raw = torch.cat([head(batch).softmax(1).cpu() for batch in v.split(1024)]).numpy()
            p = np.zeros((len(v),len(classes)),dtype=np.float32)
            for i, target in enumerate(mapping):
                p[:,target] += raw[:,i]
            score = metrics(p,truth["validation"])
            row = dict(mode=mode,epoch=epoch+1,loss=loss_sum/len(x),**score)
            history.append(row)
            print(json.dumps(row),flush=True)
            if score["accuracy"] > best + .00005:
                best, stale = score["accuracy"], 0
                saved = {key:value.detach().cpu().clone() for key,value in head.state_dict().items()}
                best_p = p.copy()
                best_epoch = epoch+1
            else:
                stale += 1
            if stale >= 12:
                break
        torch.save(dict(state_dict=saved,mapping=mapping.tolist(),outputs=len(mapping),inputs=inputs,epoch=best_epoch),OUT/f"{mode}.pt")
        pd.DataFrame(history).to_csv(OUT/f"{mode}_history.csv",index=False)
        np.save(OUT/f"{mode}_validation.npy",best_p)
        candidates.append((mode,best_p))
    if len(fine_names)>1:
        candidates.append(("average_fine",np.mean([dict(candidates)[name] for name in fine_names],axis=0)))
    neighbor_settings = [(k,t) for k in (1,5,15,35) for t in (.03,.07)]
    neighbors = neighbor_probabilities(features["validation"],features["train"],truth["train"],len(classes),neighbor_settings,device)
    candidates.extend(("knn:"+key,value) for key,value in neighbors.items())
    old = json.loads((BASE / "trained_semantic/app_settings.json").read_text())
    with np.load(BASE / "trained_semantic/classifier.npz") as archive:
        linear = np.load(BASE/"trained_semantic/features_validation.npy",mmap_mode="r") @ archive["coef"].T + archive["intercept"]
    mobile = softmax(np.log(np.clip(np.load(BASE/"trained_semantic/mobilenet_validation.npy"),1e-8,1))/old["mobile_temperature"],axis=1)
    previous = .5*mobile + .5*softmax(linear/old["linear_temperature"],axis=1)
    candidates.append(("previous",previous))
    trials = []
    for name,p in candidates:
        trials.append(dict(name=name,**metrics(p,truth["validation"])))
        np.save(OUT/(name.replace(":","_")+"_validation.npy"),p)
    ranked = sorted(candidates,key=lambda item:metrics(item[1],truth["validation"])["accuracy"],reverse=True)
    blend_pool = ranked[:3]
    nearest = next(item for item in ranked if item[0].startswith("knn:"))
    if nearest[0] not in {item[0] for item in blend_pool}:
        blend_pool.append(nearest)
    for name,p in blend_pool:
        for other,q in blend_pool:
            if name >= other:
                continue
            for weight in (.25,.5,.75):
                ensemble = weight*p + (1-weight)*q
                trials.append(dict(name=f"ensemble:{name}+{other}",first=name,second=other,weight=weight,**metrics(ensemble,truth["validation"])))
    selected = max(trials,key=lambda row:(row["accuracy"],row["macro_f1"]))
    by_name = dict(candidates)
    selected_val = (selected["weight"]*by_name[selected["first"]] + (1-selected["weight"])*by_name[selected["second"]]
                    if "first" in selected else by_name[selected["name"]])
    val_logits = np.log(np.clip(selected_val,1e-8,1))
    def nll(value):
        probabilities = softmax(val_logits/value,axis=1)
        return float(-np.log(np.clip(probabilities[np.arange(len(val_logits)),truth["validation"]],1e-8,1)).mean())
    calibrated_temperature = float(minimize_scalar(nll,bounds=(.25,5),method="bounded").x)
    np.save(OUT/"validation_probabilities.npy",softmax(val_logits/calibrated_temperature,axis=1))
    pd.DataFrame(trials).to_csv(OUT/"validation_trials.csv",index=False)
    settings = dict(classes=classes,manifest_sha256=signature,preprocess_version=old["preprocess_version"],feature_source=args.features,inputs=inputs,selected=selected,temperature=calibrated_temperature,fine_members=fine_names,
                    selection="Validation only; test and local photos never fit or select",split_counts={k:len(f) for k,f in frames.items()})
    (OUT/"selection.json").write_text(json.dumps(settings,indent=2))
    print("VALIDATION SELECTION",json.dumps(settings),flush=True)
    # Test is evaluated only after the complete validation selection is saved.
    def predict_test(name):
        if name == "average_fine":
            return np.mean([predict_test(member) for member in fine_names],axis=0)
        if name.startswith("knn:"):
            import ast
            k,t = ast.literal_eval(name[4:])
            return neighbor_probabilities(features["test"],features["train"],truth["train"],len(classes),[(k,t)],device)[str((k,t))]
        if name == "previous":
            with np.load(BASE / "trained_semantic/classifier.npz") as archive:
                l = np.load(BASE/"trained_semantic/features_test.npy",mmap_mode="r") @ archive["coef"].T + archive["intercept"]
            m = softmax(np.log(np.clip(np.load(BASE/"trained_semantic/mobilenet_test.npy"),1e-8,1))/old["mobile_temperature"],axis=1)
            return .5*m + .5*softmax(l/old["linear_temperature"],axis=1)
        checkpoint = torch.load(OUT/f"{name}.pt",map_location="cpu",weights_only=True)
        model = FeatureHead(checkpoint["outputs"],inputs=checkpoint.get("inputs",512)).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        with torch.inference_mode():
            test_x = torch.tensor((features["test"]-mean)/std,device=device)
            raw = torch.cat([model(batch).softmax(1).cpu() for batch in test_x.split(1024)]).numpy()
        p = np.zeros((len(raw),len(classes)),dtype=np.float32)
        for i,target in enumerate(checkpoint["mapping"]):
            p[:,target] += raw[:,i]
        return p
    if "first" in selected:
        p = selected["weight"]*predict_test(selected["first"]) + (1-selected["weight"])*predict_test(selected["second"])
    else:
        p = predict_test(selected["name"])
    p = softmax(np.log(np.clip(p,1e-8,1))/calibrated_temperature,axis=1)
    assert p.shape == (len(frames["test"]),len(classes)) and np.allclose(p.sum(1),1,atol=1e-5)
    np.save(OUT/"test_probabilities.npy",p)
    test_metrics = metrics(p,truth["test"])
    pd.DataFrame(classification_report(truth["test"],p.argmax(1),target_names=classes,output_dict=True,zero_division=0)).T.to_csv(OUT/"classification_report.csv")
    completion = dict(status="completed",validation_metrics={k:selected[k] for k in ("accuracy","macro_f1")},test_metrics=test_metrics,
                      target_met=selected["accuracy"] >= .95 and test_metrics["accuracy"] >= .95,seconds=time.perf_counter()-started,**settings)
    (OUT/"training_completed.json").write_text(json.dumps(completion,indent=2))
    print("COMPLETED",json.dumps(completion),flush=True)


if __name__ == "__main__":
    main()
