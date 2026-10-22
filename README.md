# EcoSort AI

ENGG2112 research prototype for local waste-image identification and conditional City of Sydney disposal assistance. This repository contains the complete project source and notebooks, compact experiment evidence, an English technical guide, and Chinese/English report-review notes.

## Read These First

- [English technical evidence guide](EcoSort_AI_Technical_Guide_EN.pdf)
- [English report review](FINAL_REPORT_REVIEW_EN.md)
- [Chinese report review](FINAL_REPORT_REVIEW_ZH.md)
- [Machine-readable experiment summary](EXPERIMENT_SUMMARY.json)
- [Complete structured source and evidence archive](EcoSort_AI_Source_and_Evidence.zip)

This is a development snapshot, **not a completed 95% real-world classifier**. The latest experimental candidate reached **95.13% validation and 94.85% test accuracy** on the reused public benchmark. It did **not** improve the six-photo local diagnostic (2/6 under the documented label assumptions). The website remains on the earlier CLIP/MobileNet model, with 86.43% validation and 85.88% test accuracy. Experimental results must not be attributed to the deployed website.

## Dataset and Results

Raw candidates: **167,285**. Retained after auditing: **147,303** across **17 classes**. Fixed grouped partitions: **117,893 training / 14,698 validation / 14,712 test**, approximately 80/10/10. No test or local-photo labels train the candidate heads.

| Expanded experiment | Validation accuracy | Test accuracy | Test macro-F1 |
|---|---:|---:|---:|
| MobileNetV3Large frozen baseline | See experiment records | 73.02% | 0.6902 |
| MobileNetV3Large partial fine-tuning | See experiment records | 78.42% | 0.7483 |
| Deployed OpenCLIP/MobileNet ensemble | 86.43% | 85.88% | 0.8362 |
| Nonlinear fine-label CLIP head and neighbors | 93.83% | 93.50% | 0.9198 |
| CLIP/DINOv2 single-head candidate | 94.85% | 94.65% | 0.9315 |
| CLIP/DINOv2 three-head candidate | 95.13% | 94.85% | 0.9338 |

All expanded rows use the same public test split; repeated evaluation is not untouched final validation. Real-photo diagnostics, uncertainty coverage and benchmark scores are separate quantities. Review flags do not remove incorrect predictions from the accuracy denominator.

## Run the Deployed App

The complete ZIP includes the deployed project-trained MobileNet and compact CLIP classification head, with the correct directory structure. Generic pretrained encoders, raw images and feature caches are intentionally excluded. Download and extract that ZIP for the full runnable layout. Source files in the repository are also directly browseable.

Verified environment: **Apple Silicon Mac, Python 3.12**. Other platforms have not been tested. From the extracted project directory:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python semantic_model.py
python app.py
```

The first encoder download is approximately 605 MB, pinned to a revision and SHA-256 checked. Open **http://127.0.0.1:7861/**. After the weights are downloaded, prediction runs locally without an external inference API. Do not expose this development server publicly. In the Git clone, the deployed binary weights are also available in their normal result directories; if omitted by a partial download, extract the complete ZIP.

The interface supports upload and webcam snapshots, candidate scores, explicit uncertain-result messages and conditional disposal guidance based on user-confirmed item types. It cannot certify emptiness, contamination, hidden hazards or legal compliance. There is no automatic multi-object detector, physical sorter, completed batch-contamination audit or evaluated LLM summarizer.

## Reproduce Experiments

- `prepare_dataset.py`, `download_expanded.py`, `prepare_expanded.py`: recover sources, audit and build grouped partitions. Read source licences before downloading or redistributing datasets.
- `train_full.py`: frozen/partially fine-tuned MobileNet comparison.
- `semantic_model.py`, `train_semantic.py`: frozen CLIP features, regularized head and validation-selected ensemble.
- `nonlinear_model.py`, `train_nonlinear.py`: nonlinear material/fine-label heads, train-only visual neighbors and validation selection.
- `extract_dino_features.py`: pinned DINOv2 features with full-image letterboxing.
- `evaluate_candidate.py`: fixed, previously observed diagnostic panels; never use them for fitting.
- `ECOSORT_Nonlinear_Experiments.ipynb`: English text/code walkthrough.

The public manifests preserve image identity and split assignments but redact local machine paths. Their exported SHA-256 therefore differs from the original private manifest signature recorded by historical experiments; the export inventory records both hashes. Rebuild feature markers and train into a new output directory rather than modifying old evidence records. For example:

```bash
python train_nonlinear.py --features features_clip_dino --output trained_reproduction --seeds 3
```

That command requires regenerated CLIP/DINO training features and restored source images. It does not work from metrics files alone. Raw datasets, diagnostic web-image files, private local images and candidate nearest-neighbor caches are not included. Historical migration scripts may deliberately stop to protect existing files; do not blindly run every script in file-name order.

Notebook source cells are preserved for browsing, but outputs, execution metadata, attachments and private local paths are removed in public copies. Thus the public historical export is sanitized, not byte-for-byte identical to the private original Colab export. The original private project remains unchanged by packaging. `requirements-lock.txt` records the measured Mac environment; document-building helpers additionally use ReportLab.

## Privacy and Rights

The export uses an explicit allowlist. No `.venv`, credentials, browser state, raw dataset images, feature arrays, user photographs, camera captures, machine logs or original team-report PDF is uploaded. Aggregate local-photo findings contain no private image pixels. Private-machine path prefixes are redacted. The complete ZIP and inventory are checked before upload.

Public code visibility does not establish a new licence for source datasets, pretrained encoders or third-party libraries. No new project-wide software licence has been selected; retain upstream notices and confirm rights before redistribution or commercial use. The data sources and model references are documented in the guide and preparation scripts. This project used Codex/ChatGPT assistance for coding, debugging, analysis and drafting; course disclosures must describe the actual use and human verification.
