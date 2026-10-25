# EcoSort AI: Project-Trained Submission Edition

The classifier is **MobileNetV3Large, initialized with ImageNet and then trained on the project's audited waste images**. It predicts 17 waste-material categories and provides conditional City of Sydney disposal guidance.

## Five Python Files

| File | Purpose |
|---|---|
| app.py | English upload/webcam interface, user confirmation and local startup. |
| classifier.py | Load the project-trained checkpoint, preprocess images and apply validation-selected review settings. |
| disposal_rules.py | Conditional City of Sydney / NSW EPA guidance. |
| train.py | Reproduce frozen-backbone training and partial fine-tuning on every training image. |
| evaluate.py | Re-evaluate the fixed validation/test splits, rule checks and deployment inputs. |

## Dataset and Training

167,285 raw candidates were audited; 147,303 were retained across 17 material classes. The grouped split contains 117,893 training, 14,698 validation and 14,712 test images. These numbers are not 180,000 training images. Validation selects checkpoints/calibration; test images do not update weights.

The deployed checkpoint comes from the completed 22 October training run: an ImageNet-initialized frozen MobileNetV3 baseline, followed by six epochs of partial fine-tuning. Each fine-tuning epoch visits the complete training partition, without a sample/step cap. Class weighting, training-only augmentation and validation macro-F1 checkpoint selection are implemented in train.py. Batch normalization remains frozen during fine-tuning.

Validation accuracy: 79.39%. Test accuracy: 78.42%, macro-F1 0.7483, rechecked on 25 October 2026. Read evidence/trained_model_verified.json for the measured re-evaluation. No 95% claim is made. These previously observed public-data splits are not fresh independent evaluation or a real-photo accuracy guarantee.

## Run on Apple Silicon / Python 3.12

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python app.py
```

The trained checkpoint and training records are in trained_expanded/. The packaged verification record permits inference without downloading the training dataset. After restoring the dataset, run python evaluate.py to re-evaluate every validation/test image.

Raw images and the private-path split manifest are not published. On the original Mac, the split manifest and source images remain in the local project. Training and evaluation require data/expanded/split_manifest.csv with the columns path, label, sha256, group_id and split, together with the matching licensed image files. A fresh clone supports inference using the included checkpoint; it cannot reproduce training or full evaluation until that dataset and its fixed grouped split are restored. Do not repartition the images and describe the resulting metrics as the same evaluation.

The local interface uses http://127.0.0.1:7862/. Start EcoSort.command launches it after setup. Do not expose the development server publicly.

## Reproduce Training Safely

```bash
python train.py --output trained_reproduction
```

This refuses to overwrite an existing output directory. Only ImageNet initialization is downloaded; the 17-class head and fine-tuned layers are learned from project data. The new run writes checkpoints, training logs, class weights, calibration and test metrics into that separate directory. It is not automatically deployed; evaluate a new checkpoint before replacing trained_expanded. Existing results are preserved.

EcoSort_Demo.ipynb alternates English explanations and code for dataset inspection, training, evaluation and deployment.

## Limits and Honest Reporting

The network predicts material categories, not bottle/bag/container subtypes. Generic plastic, glass and metal therefore require item confirmation before a specific household recycling route is given. Explicitly confirmed item types can select a route; the image cannot confirm hidden hazards, contamination or service availability.

The interface supports upload and webcam inputs, 17-class scores, uncertainty flags, user confirmation and conditional local guidance. Dataset accuracy, review coverage and real-photo performance are separate quantities. No independent real-photo pass is claimed.

Disclose actual Codex/ChatGPT coding and analysis assistance as required by the course. Public visibility does not grant a new licence for upstream models, datasets or dependencies; retain their applicable notices.
