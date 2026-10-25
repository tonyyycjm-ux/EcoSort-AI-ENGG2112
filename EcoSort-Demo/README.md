# EcoSort AI: Project-Trained Submission Edition

The active classifier is **MobileNetV3Large, initialized with ImageNet and then trained on the project's audited waste images**. There is no SigLIP2, text-prompt matcher, CLIP or zero-shot fallback in the active application.

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

Historical validation accuracy: 79.39%. Historical test accuracy: 78.42%, macro-F1 0.7483. Read evidence/trained_model_verified.json for the latest actual re-evaluation. No 95% claim is made. These previously observed public-data splits are not fresh independent evaluation.

## Run on Apple Silicon / Python 3.12

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python app.py
```

The trained checkpoint and training records are in trained_expanded/. The packaged verification record permits inference without downloading the training dataset. After restoring the dataset, run python evaluate.py to re-evaluate every validation/test image.

The local split manifest currently references the original data files in EcoSort-Local; do not delete that dataset. Raw images and the private-path manifest are not published with this edition. To reproduce on another machine, restore the licensed source images and rebuild the same grouped split with the original preparation scripts in the repository, then update image paths without changing labels, hashes, groups or split membership. The public source archive alone is not sufficient to retrain without those data-recovery steps.

The local interface uses http://127.0.0.1:7862/. Start EcoSort.command launches it after setup. Do not expose the development server publicly.

## Reproduce Training Safely

```bash
python train.py --output trained_reproduction
```

This refuses to overwrite an existing output directory. Only ImageNet initialization is downloaded; the 17-class head and fine-tuned layers are learned from project data. The new run writes checkpoints, training logs, class weights, calibration and test metrics into that separate directory. It is not automatically deployed; evaluate a new checkpoint before replacing trained_expanded. Existing results are preserved.

EcoSort_Demo.ipynb alternates English explanations and code for dataset inspection, training, evaluation and deployment.

## Limits and Honest Reporting

The network predicts material categories, not bottle/bag/container subtypes. Generic plastic, glass and metal therefore require item confirmation before a specific household recycling route is given. Explicitly confirmed item types can select a route; the image cannot confirm hidden hazards, contamination or service availability.

Upload and webcam inputs, 17-class scores, uncertainty flags, user confirmation and conditional local guidance are retained. Current model scores, review coverage and real-photo performance are separate quantities. Earlier SigLIP2 panel results do not apply to this model. No new independent real-photo pass is claimed.

The previous SigLIP2 sources were archived locally before this change, and GitHub history retains that version. Historical project training remains available in EcoSort-Local. Disclose actual Codex/ChatGPT coding and analysis assistance as required by the course.
