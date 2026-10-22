"""Generate the English inference notebook while retaining the legacy version."""
import shutil
from pathlib import Path
import nbformat as nbf

BASE = Path(__file__).resolve().parent
target = BASE / "ECOSORT_Local_Run.ipynb"
backup = BASE / "ECOSORT_Legacy_12class_Run.ipynb"
if not backup.exists():
    shutil.copy2(target, backup)
cells = []
for text, code in [
    ("# EcoSort AI - Expanded Local Run\n\n## 1. Check the active completed model\n\nRead the active model pointer and its actual completion record. This notebook does not retrain. Historical results remain separate.", "from pathlib import Path\nimport json\nPROJECT = Path.cwd()\nassert (PROJECT / 'ecosort_runtime.py').exists(), 'Open this notebook from EcoSort-Local.'\nactive = json.loads((PROJECT / 'active_model.json').read_text())['results'] if (PROJECT / 'active_model.json').exists() else 'trained_expanded'\nRESULTS = PROJECT / active\ncompletion = json.loads((RESULTS / 'training_completed.json').read_text())\nassert completion['status'] == 'completed'\ncompletion"),
    ("## 2. Load the selected model\n\nLoad the active 17-class classifier and validation-only calibration. The semantic model combines MobileNet with a frozen OpenCLIP encoder and a trained head. Letterboxing preserves the entire object. Print available devices; do not assume a GPU is present.", "import tensorflow as tf\nfrom ecosort_runtime import EcoSort\nprint('GPU devices:', tf.config.list_physical_devices('GPU'))\nengine = EcoSort()\nprint('Visual classes:', engine.classes)\nprint('Review threshold:', engine.settings['review_threshold'])"),
    ("## 3. Open the English application\n\nReuse the running local application, or start it if absent. Classification remains separate from user-confirmed item guidance. Official City of Sydney and NSW EPA links accompany conditional disposal routes.", "from ecosort_runtime import application_running, launch\nimport webbrowser\nif not application_running():\n    demo = launch(prevent_thread_lock=True)\nwebbrowser.open('http://127.0.0.1:7861')"),
    ("## 4. Read the final whole-photo evaluation\n\nThe thirty whole photos were frozen and duplicate-checked before their first prediction. The original ten photos and the 55 TACO diagnostic crops are regression/development sets, not new independent evidence. Report every case, including incorrect predictions and manual review. These convenience samples do not replace 50-100 local Sydney photos or guarantee performance in all scenes.", "import pandas as pd\nreport = json.loads((PROJECT / 'final_test/evaluation.json').read_text())\ndisplay(report)\ndisplay(pd.read_csv(PROJECT / 'final_test/predictions.csv')[['expected_label', 'predicted', 'confidence', 'manual_review', 'correct']])\ndisplay(json.loads((PROJECT / 'trained_semantic/taco_diagnostic/evaluation.json').read_text()))"),
]:
    cells.extend([nbf.v4.new_markdown_cell(text), nbf.v4.new_code_cell(code)])
notebook = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec":{"display_name":"EcoSort Local (Python 3.12 / Metal)", "language":"python", "name":"ecosort-local"}})
nbf.write(notebook, target)
print(target)
