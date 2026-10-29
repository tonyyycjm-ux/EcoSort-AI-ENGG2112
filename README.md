# EcoSort AI

Open **EcoSort_Demo.ipynb** in VS Code, select the project .venv kernel, and Run All. Dataset preparation, training and full evaluation are disabled by default; the final cell opens the local demo. No separate launcher is needed.

## Six executable notebook modules

- prepare_data.ipynb: collect missing source images, map categories, decode images, group duplicates, exclude conflicting labels and make an 80/10/10 grouped split.
- train.ipynb: ImageNet initialization, frozen-head training and partial MobileNetV3Large fine-tuning on every training image.
- classifier.ipynb: load our trained model, preprocess images and check uncertainty.
- disposal_rules.ipynb: conditional City of Sydney disposal guidance.
- app.ipynb: English upload/webcam interface and human confirmation.
- evaluate.ipynb: full validation/test evaluation and input/rule checks.

## Two working folders

- data/: original photographs in sources/, source lists, fixed split_manifest.csv, category mapping and cleaning summary.
- model/: model.keras is the learned network, settings.json contains class order and review settings, and results.json contains training/evaluation results and compact training logs.

These model files are required for prediction; they are not separate programs. The model file fingerprint checks that results belong to the same weights. No source-code fingerprint system is required.

## Where the 147,303 images came from

The recovered preparation script combines a fixed original selection of 20,000 images with 147,285 Garbage265 source records: 167,285 candidates. It checks decoding/minimum size, manual rejections, exact/pixel/perceptual similarity and source families. Conflicting-label groups are excluded; exact/pixel duplicates are reduced. Groups are assigned 80/10/10 using seed 42. Near-duplicate matching is heuristic, not a proof of independence.

The original 20,000-image choice is explicitly preserved in data/source_lists/original_20000.csv with labels and SHA256 identities. It is an input selection, not a claim that this script recreates every earlier exploratory notebook or manual decision. Garbage265 class names and source lists are included beside it. Source data comes from TrashNet, RealWaste, three Kaggle datasets and Garbage265; download URLs are recorded in prepare_data.ipynb. Source-category labels are separate from Sydney disposal rules.

To rebuild without changing the deployed split, enable REBUILD_DATA in the notebook. The script refuses an existing output directory. --download collects missing source trees; it does not redownload present image collections. Missing/changed upstream files fail verification rather than silently replacing the selected dataset. A newly prepared split must be checked before replacing data/split_manifest.csv; changed data needs fresh training and evaluation.

## Actual training and results

Retained: 147,303 images, 17 classes. Train: 117,893; validation: 14,698; test: 14,712. Only training images update weights. Validation selects the checkpoint and review settings. The completed run trained the head for 16 epochs and fine-tuned for six epochs. Baseline test accuracy: 73.02%; fine-tuned test accuracy: 78.42%; validation: 79.39%. Read model/results.json for evaluation date and details. These are previously observed dataset splits, not independent real-photo performance.

New training uses a separate model_new/ folder and never overwrites model/. The fixed raw photographs are local and are not bundled for public redistribution. Upstream access/licences still apply; preserved manifests are needed for exact reproduction.

## Limits

The network classifies material/waste categories, not exact object identity, contamination or hidden hazards. Confirmation changes guidance, not weights. Local rules are conditional and need maintenance. The app has no zero-shot model fallback. No 95% overall accuracy or measured environmental benefit is claimed. Disclose actual AI coding assistance according to course requirements.

Notebook modules are loaded from code cells tagged module. Optional execution cells never execute during module loading. Restart the kernel after edits.
