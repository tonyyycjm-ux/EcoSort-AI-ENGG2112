"""Generate the English accuracy-improvement workflow, keeping older experiments."""
from pathlib import Path
import nbformat as nbf

BASE=Path(__file__).resolve().parent
sections=[
    ("# EcoSort AI - Accuracy Improvement\n\n## 1. Check the environment\n\nUse the isolated local environment. The existing MobileNet remains archived. The new frozen OpenCLIP encoder and trained linear head operate locally, with all 117,893 training images retained.","from pathlib import Path\nimport json\nimport torch\nPROJECT=Path.cwd()\nassert (PROJECT/'train_semantic.py').exists()\nprint('MPS available:',torch.backends.mps.is_available())"),
    ("## 2. Load verified pretrained weights\n\nDownload only if absent. Check the pinned revision and SHA-256 before use. Full-object letterboxing preserves bottle ends and other shapes that default center cropping can remove. No image is sent to a remote model.","%run semantic_model.py"),
    ("## 3. Cache the unchanged MobileNet predictions\n\nThe original class order, manifest order and model hash are preserved for fair comparison and ensemble inference.","%run cache_mobilenet.py"),
    ("## 4. Train and select on validation\n\nExtract all training, validation and test features with identical full-object preprocessing. Fit the classification head only on the training partition. Use validation to select regularization and model combinations; reject unconverged heads. Test images never update parameters. Features are resumable and tied to the manifest and preprocessing version.","%run train_semantic.py"),
    ("## 5. Inspect the untouched final evaluation\n\nThe original ten photos are regression cases. The 55 TACO crops exposed the center-crop defect and are now development diagnostics, not independent evidence. The separate final whole-photo panel was visually reviewed, duplicate-checked and frozen before prediction. Re-running evaluation does not permit replacing failed photos.","completion=json.loads((PROJECT/'trained_semantic/training_completed.json').read_text())\ndisplay(completion)\nfinal=PROJECT/'final_test/evaluation.json'\nif final.exists():\n    display(json.loads(final.read_text()))\nelse:\n    print('Final external evaluation has not completed.')"),
    ("## 6. Open the active local application\n\nThe inference notebook reads active_model.json and loads the selected model. City of Sydney guidance remains conditional and separate from visual scores. Run only one application instance. An observed test accuracy is not a guarantee for arbitrary future photos.","import webbrowser\nwebbrowser.open('http://127.0.0.1:7861')"),
]
cells=[]
for text,code in sections:
    cells.extend([nbf.v4.new_markdown_cell(text),nbf.v4.new_code_cell(code)])
notebook=nbf.v4.new_notebook(cells=cells,metadata={"kernelspec":{"name":"ecosort-local","language":"python","display_name":"EcoSort Local (Python 3.12 / Metal)"}})
nbf.write(notebook,BASE/"ECOSORT_Accuracy_Improvement.ipynb")
print(BASE/"ECOSORT_Accuracy_Improvement.ipynb")
