"""Export an allowlisted, privacy-checked public snapshot without editing originals."""
import ast
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import zipfile

BASE = Path(__file__).resolve().parent
OUT = BASE / "public_export_20261023"
ARCHIVE = BASE / "EcoSort_AI_Source_and_Evidence_20261023.zip"
TEXT_SUFFIXES = {".py",".md",".json",".csv",".txt",".ipynb",".command"}
PRIVATE_PATTERNS = [r"/Users/[A-Za-z0-9._-]+",r"/var/folders/[a-z0-9]{2}/[^\s\"']+",
                    r"https://colab\.research\.google\.com/drive/[A-Za-z0-9_-]+[^\s\"']*",
                    r"https://docs\.google\.com/document/d/[A-Za-z0-9_-]+[^\s\"']*"]
SECRETS = re.compile(r"github_pat_[A-Za-z0-9_]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{25,}|AKIA[A-Z0-9]{16}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")


def clean(text):
    text = text.replace(str(BASE),"PROJECT_ROOT").replace(str(Path.home()),"HOME_DIRECTORY")
    for pattern in PRIVATE_PATTERNS:
        text = re.sub(pattern,"PRIVATE_REFERENCE_NOT_INCLUDED",text)
    return text


def export(source,relative=None):
    destination = OUT/(relative or source.relative_to(BASE))
    destination.parent.mkdir(parents=True,exist_ok=True)
    if source.suffix == ".ipynb":
        notebook = json.loads(source.read_text())
        notebook["metadata"] = {"kernelspec":{"display_name":"EcoSort Local","language":"python","name":"python3"},"language_info":{"name":"python"}}
        for cell in notebook["cells"]:
            cell["metadata"] = {}
            cell.pop("attachments",None)
            if cell["cell_type"] == "code":
                cell["outputs"],cell["execution_count"] = [],None
            cell["source"] = clean("".join(cell["source"])).splitlines(keepends=True)
        destination.write_text(json.dumps(notebook,ensure_ascii=False,indent=1))
    elif source.suffix in TEXT_SUFFIXES:
        destination.write_text(clean(source.read_text()))
    else:
        shutil.copy2(source,destination)


def main():
    OUT.mkdir(exist_ok=True)
    assert not (OUT/"EXPORT_INVENTORY.json").exists(), "Preserve the already verified export."
    sources = list(BASE.glob("*.py"))+list(BASE.glob("*.ipynb"))
    sources += [BASE/name for name in ("requirements-lock.txt","Start EcoSort.command","active_model.json",
                "FINAL_REPORT_REVIEW_ZH.md","FINAL_REPORT_REVIEW_EN.md","EcoSort_AI_Technical_Guide_EN.pdf",
                "EcoSort_AI_Real_Photo_Update_EN.pdf","REAL_PHOTO_ACCEPTANCE_REPORT.md")]
    sources += [BASE/"acceptance_web"/name for name in ("test_manifest.json", "results.json", "results.csv", "runtime_parity.json", "selection_review.json")]
    sources += [BASE/"trained_objects/deployment_approved.json"]
    sources += [BASE/"trained_objects/local_development.json"]
    sources += [BASE/"data/expanded"/name for name in ("dataset_verification.json","class_counts.csv","class_names.json","category_mapping.json")]
    sources += [BASE/"saved_results"/name for name in ("split_manifest.csv","weighted_vs_unweighted.csv","per_class_recall_comparison.csv")]
    for directory in ("trained_expanded","trained_semantic","trained_nonlinear","trained_multiview","trained_multiseed"):
        for path in (BASE/directory).iterdir():
            if path.suffix == ".json" and not path.name.startswith("features_"):
                sources.append(path)
            elif path.suffix == ".csv" and "predictions" not in path.name:
                sources.append(path)
    for source in sorted(set(sources)):
        export(source)
    export(BASE/"PUBLIC_README.md","README.md")
    export(BASE/"trained_expanded/ecosort_model.keras")
    export(BASE/"trained_semantic/classifier.npz")
    # Keep split identities while removing private machine paths and unused audit columns.
    source = BASE/"data/expanded/split_manifest.csv"
    with source.open() as handle:
        rows = list(csv.DictReader(handle))
    target = OUT/"data/expanded/split_manifest.csv"
    fields = ["dataset","label","fine_label","path","sha256","group_id","split"]
    with target.open("w",newline="") as handle:
        writer = csv.DictWriter(handle,fieldnames=fields)
        writer.writeheader()
        for row in rows:
            row["path"] = str(Path(row["path"]).relative_to(BASE))
            writer.writerow({key:row[key] for key in fields})
    summary = dict(raw_images=167285,retained_images=147303,
                   split_counts=dict(train=117893,validation=14698,test=14712),
                   deployed=load("trained_objects/deployment_approved.json"),
                   former_deployed=load("trained_semantic/training_completed.json"),
                   candidates={name:dict(completion=load(name+"/training_completed.json"),diagnostics=load(name+"/external_diagnostics.json"))
                               for name in ("trained_nonlinear","trained_multiview","trained_multiseed")},
                   warning="Current application is frozen SigLIP2, not the benchmark candidate. New web joint scores: 19/20, 18/20, 18/20. Local six-photo development: 5/6. No universal accuracy guarantee; old benchmark 95% validation/test target remains unmet.")
    (OUT/"EXPERIMENT_SUMMARY.json").write_text(json.dumps(summary,indent=2))
    (OUT/".gitignore").write_text(".venv/\n__pycache__/\n*.pyc\n.DS_Store\nmodels/\nfeatures_clip_dino/\ndata/garbage265/\ndata/garbage265_download/\n*.log\n*.pid\n")
    inventory = []
    for path in sorted(OUT.rglob("*")):
        if not path.is_file():
            continue
        relative = str(path.relative_to(OUT))
        assert not any(part in relative.split("/") for part in (".venv","__pycache__","samples"))
        assert path.suffix.lower() not in {".jpg",".jpeg",".png",".heic",".env"}
        if path.suffix in TEXT_SUFFIXES:
            text = path.read_text()
            assert not SECRETS.search(text), f"Potential secret in {relative}"
            assert not re.search(r"/Users/[A-Za-z0-9._-]+/|/var/folders/[a-z0-9]{2}/",text), relative
            if path.suffix == ".py":
                ast.parse(text)
        inventory.append(dict(path=relative,bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    record = dict(files=inventory,notebooks_sanitized=True,private_photos_included=False,raw_images_included=False,
                  original_manifest_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  exported_manifest_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                  note="Source cells retained; outputs and metadata cleared; private paths redacted. Manifests retain identities and splits.")
    (OUT/"EXPORT_INVENTORY.json").write_text(json.dumps(record,indent=2))
    with zipfile.ZipFile(ARCHIVE,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for path in sorted(OUT.rglob("*")):
            if path.is_file():
                archive.write(path,str(Path("EcoSort-AI")/path.relative_to(OUT)))
    with zipfile.ZipFile(ARCHIVE) as archive:
        assert archive.testzip() is None
    assert ARCHIVE.stat().st_size < 25*1024*1024, "Use a release asset for larger archives."
    print(json.dumps(dict(files=len(inventory)+1,archive_bytes=ARCHIVE.stat().st_size,archive_sha256=hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(),source_files=len(list(OUT.glob('*.py'))),notebooks=len(list(OUT.glob('*.ipynb'))))))


def load(name):
    return json.loads((BASE/name).read_text())


if __name__ == "__main__":
    main()
