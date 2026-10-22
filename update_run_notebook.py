"""Generate the English inference notebook while retaining the legacy version."""
import shutil
from pathlib import Path
import nbformat as nbf

BASE = Path(__file__).resolve().parent
target = BASE / "ECOSORT_Local_Run.ipynb"
backup = BASE / "ECOSORT_Legacy_12class_Run.ipynb"
if not backup.exists():
    shutil.copy2(target, backup)
semantic_backup = BASE / "ECOSORT_Semantic_Run_Archive.ipynb"
if not semantic_backup.exists():
    shutil.copy2(target, semantic_backup)
cells = []
for text, code in [
    ("# EcoSort AI - Object-Level Local Run\n\n## 1. Read the active model and acceptance evidence\n\nThe current application uses a frozen, locally executed SigLIP2 object recognizer. It does not retrain on the sixty acceptance photos. Historical trained benchmark classifiers remain separate and their validation/test scores must not be attributed to this model.", "from pathlib import Path\nimport json\nPROJECT = Path.cwd()\nassert (PROJECT / 'ecosort_runtime.py').exists(), 'Open this notebook from EcoSort-Local.'\nactive = json.loads((PROJECT / 'active_model.json').read_text())\nprint(active)\nRESULTS = PROJECT / active['results']\nrecord = 'deployment_approved.json' if active['results'] == 'trained_objects' else 'training_completed.json'\ncompletion = json.loads((RESULTS / record).read_text())\ncompletion"),
    ("## 2. Load the verified local model\n\nLoad pinned model weights and object descriptions. The strongest object match determines the material category, then a separate official-rule module gives conditional disposal guidance. Relative scores are not calibrated probabilities. Ambiguous classifications require manual review.", "import torch\nfrom ecosort_runtime import EcoSort\nprint('Apple GPU available:', torch.backends.mps.is_available())\nengine = EcoSort()\nprint('Visual classes:', engine.classes)"),
    ("## 3. Open the English application\n\nReuse the running local application, or start it if absent. Classification remains separate from user-confirmed item guidance. Official City of Sydney and NSW EPA links accompany conditional disposal routes.", "from ecosort_runtime import application_running, launch\nimport webbrowser\nif not application_running():\n    demo = launch(prevent_thread_lock=True)\nwebbrowser.open('http://127.0.0.1:7861')"),
    ("## 4. Inspect all sixty acceptance outcomes\n\nThree fixed panels contain twenty whole web photos each. Each panel needs at least seventeen correct category-and-route outputs. Uncertain outputs count as failures. This is a stratified convenience sample, not a guarantee for arbitrary phone photos. The six local development photos are not independent acceptance evidence.", "import pandas as pd\nreport = json.loads((PROJECT / 'acceptance_web/results.json').read_text())\ndisplay(pd.DataFrame(report['groups']))\ndisplay(pd.read_csv(PROJECT / 'acceptance_web/results.csv'))\nprint(report['warning'])"),
]:
    cells.extend([nbf.v4.new_markdown_cell(text), nbf.v4.new_code_cell(code)])
notebook = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec":{"display_name":"EcoSort Local (Python 3.12 / Metal)", "language":"python", "name":"ecosort-local"}})
nbf.write(notebook, target)
print(target)
