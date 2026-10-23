# EcoSort AI: Demonstration Edition

Four Python files, one final model, and no imports from the historical project.

| File | Responsibility |
|---|---|
| `app.py` | English upload/webcam interface, user confirmation, and local startup. |
| `classifier.py` | Pinned frozen SigLIP2, object prompts, category matching and uncertainty. |
| `disposal_rules.py` | Conditional City of Sydney / NSW EPA guidance with source links. |
| `evaluate.py` | Rule/input checks and regression against the original 60-photo outcomes. |

## Run

Verified target: Apple Silicon Mac, Python 3.12. From this directory:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python classifier.py
python app.py
```

The pinned Google SigLIP2 download is approximately 1.50 GB. After download, inference is local and requires no inference API. Open http://127.0.0.1:7862/. The original application uses port 7861 and is left unchanged. The launcher reuses a running demo and refuses to reuse another application's port. Do not expose this development server publicly.

`EcoSort_Demo.ipynb` provides alternating English explanations and code cells. Select this directory's `.venv` kernel. `Start EcoSort.command` provides a double-click launcher after environment setup.

## Verify

```bash
python evaluate.py --restore
```

This restores missing, source-attributed acceptance photos and requires their exact recorded SHA-256 hashes. Network errors and changed source images stop the test; never replace failed photos. For locally available copies, use `python evaluate.py --images /path/to/acceptance_web`.

The regression checks category, suggested object, uncertainty and conditional disposal route for all sixty photos, including the original failures. It also checks invalid inputs and user-confirmed routing. The application checks the resulting source/evidence fingerprints before starting. A code change needs a new regression, not manual editing of the verification record.

## What the results mean

Original category-and-conditional-route results: 19/20, 18/20 and 18/20. Original category-only results: 19/20, 18/20 and 20/20. These prelabelled stratified convenience panels are not a random population sample. A refactor regression is **not new independent accuracy evidence**. The printed plastic bag still has an incorrect recycling suggestion; four other outcomes require review.

The current model is frozen, pretrained Google SigLIP2. It was not trained on the project's 147,303 retained images. Historical training, alternatives and benchmark scores remain in the complete repository; they must not be attributed to this model. Relative candidate scores are not calibrated probabilities. The original prompts, image preprocessing, weights and thresholds are retained.

Routes are conditional on the suggested object being correct, packaging being eligible and empty, and service availability. Images cannot establish hidden contents, contamination or damaged batteries. Human confirmation remains necessary; this is not safe autonomous hazardous-waste sorting.

## Q&A Map

Explain the flow as: image -> image/text embeddings -> best object match -> material category -> uncertainty check -> conditional local rule -> interface.

Explain the development honestly: historical trained models scored better on public benchmarks than on everyday photos, motivating a frozen semantic model for the final demo. Preserve the historical comparison, the known failures, and actual Codex/ChatGPT assistance in course disclosures. Fewer files do not remove responsibility for understanding the model and evaluation limits.

The complete historical project is not deleted. This separate edition removes runtime dependencies on TensorFlow, OpenCLIP, DINOv2, SciPy and old trained heads. It retains the same main user features, not the full training/research toolchain. Pretrained encoder weights, source test images, private photos and credentials are excluded from GitHub. Retain upstream model/data licences; no new project-wide licence is granted by public visibility.
