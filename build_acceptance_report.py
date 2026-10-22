"""Build an English current-version addendum without replacing historical reports."""
import json
from pathlib import Path
from xml.sax.saxutils import escape

import build_team_guide as g

BASE = Path(__file__).resolve().parent


def main():
    results = json.loads((BASE/"acceptance_web/results.json").read_text())
    manifest = json.loads((BASE/"acceptance_web/test_manifest.json").read_text())
    assert results["passed"] and len(results["predictions"]) == 60
    g.PDF = BASE/"EcoSort_AI_Real_Photo_Update_EN.pdf"
    g.styles["TableGuide"].fontSize = 8
    g.styles["TableGuide"].leading = 10
    g.paragraph("EcoSort AI: Real-Photo Update", "GuideTitle")
    g.paragraph("23 October 2026 | Current application and sixty-photo acceptance evidence", "Heading2")
    g.paragraph("The application now uses a pinned Google SigLIP2 base model locally on the Mac. Object descriptions are matched to the full photograph, then mapped to 17 project categories. A separate City of Sydney / NSW EPA rule module supplies conditional disposal advice. This is a zero-shot object recognizer, not a newly trained 147,303-image classifier.")
    g.table(["Panel", "Category correct", "Category + route correct", "Required"],
            [[r["panel"], f"{r['category_correct']}/20", f"{r['joint_correct']}/20 ({r['joint_correct']/20:.0%})", "17/20"] for r in results["groups"]], [20,43,68,34])
    g.paragraph("<b>All three panels passed on their first recorded evaluation.</b> Total category accuracy is 57/60 (95%); total category-and-conditional-route accuracy is 55/60 (91.7%). No failed image was replaced after predictions. Uncertain outputs count as joint failures, even when the suggested category is correct.")
    g.paragraph("<b>Limits:</b> these are stratified convenience samples from Wikimedia Commons, not randomly sampled Sydney household photos. Sixty examples do not establish universal 80% or 95% real-world accuracy. Foundation-model pretraining membership is unknown. The mixed-items class is not covered. Routes do not verify hidden contents, contamination, damaged batteries, or access to a service.")
    g.paragraph("The six previously observed phone photos improved from the former deployed model's 2/6 to 5/6 category matches. They are development examples, not fresh validation. The blue-green plastic puzzle remains an uncertain incorrect cardboard suggestion. No private phone photo is included in the public repository.")
    g.section("What changed and what did not")
    g.table(["Component", "Current implementation"], [
        ["Object model", "Google SigLIP2 base patch16 224; frozen weights; local MPS execution; individual object prompts instead of a benchmark-trained material head."],
        ["Inputs", "Whole RGB photo with EXIF orientation corrected, then official model preprocessing; no manual crop or ground-truth item confirmation supplied during acceptance."],
        ["Category", "Maximum object-text similarity per class. Relative softmax scores are displayed as uncalibrated candidate scores, not accuracy probabilities."],
        ["Uncertainty", "Review if top two class logits differ by less than 0.5, or highest logit is below -10. Frozen before acceptance prediction."],
        ["Disposal", "Inferred item guidance is conditional and distinct from a user-confirmed item. Batteries, lamps and paint need special collection; soft plastic and paper towels are not yellow-bin recycling."],
        ["Dataset history", "167,285 indexed images; 147,303 retained; 117,893 train / 14,698 validation / 14,712 test. These counts describe archived project experiments, not new SigLIP2 training."],
        ["Historical benchmark", "95.13% validation and 94.85% public test belong to an archived CLIP+DINO candidate. The public test has been reused; do not attribute these numbers to the current application."],
    ], [35,130])
    g.paragraph("Key files: object_model.py, sydney_rules.py, ecosort_runtime.py, active_model.json, trained_objects/deployment_approved.json, acceptance_web/test_manifest.json, acceptance_web/results.csv, and test_runtime_acceptance.py. The English local-run notebook retains explanation/code cell pairs.")
    g.paragraph("Setup: install requirements-lock.txt in a Python 3.12 virtual environment; run python object_model.py once to download the pinned ~1.50 GB model; then run python app.py. Inference remains local. To restore the frozen web panel, run python restore_acceptance.py, then python test_runtime_acceptance.py. Download failures or changed source files must not silently replace an evaluated image.")
    g.section("Every failure is retained")
    failed = [r for r in results["predictions"] if not r["joint_correct"]]
    g.table(["Panel / ID", "Ground truth", "Actual output", "Why joint failure"],
            [[f"{r['panel']} / {r['id']}", r["expected_object"], r["predicted_object"],
              "Manual review" if r["uncertain"] else "Wrong category and route"] for r in failed], [23,48,50,44])
    g.paragraph("The printed soft-plastic bag was confidently mistaken for paper and received an incorrect recycling route. This is a remaining safety-relevant failure, not an item to hide from the report. The application must not be used for unattended hazardous-waste decisions.")
    g.paragraph("Protocol: twenty predefined object slots, with three different photographs per slot. Visual review excluded drawings, unrelated search results, ambiguous material, and repeated views before model prediction. SHA-256, resized-pixel hashes and perceptual hashes were checked against the entire project pool and prior panels; no detected duplicates were found. These are heuristic checks, not proof of complete independence.")
    g.paragraph('Official rule source: <link href="https://www.cityofsydney.nsw.gov.au/waste-recycling-services/recycle-household-items">City of Sydney household recycling</link>. Additional food, garden, doorstep and NSW EPA links are embedded in the runtime rule module. Google model source: <link href="https://huggingface.co/google/siglip2-base-patch16-224">SigLIP2 official model card</link>.')
    by_id = {r["id"]: r for r in manifest["images"]}
    for group in results["groups"]:
        g.section(f"Panel {group['panel']}: all twenty outcomes")
        g.paragraph("Source-page links are attached to each photo ID. Complete route outputs, hashes, author and licence metadata are in the accompanying JSON/CSV files.")
        g.table(["Photo ID", "Reference object", "Predicted object", "Joint result"],
                [[r["id"], r["expected_object"], r["predicted_object"], "Correct" if r["joint_correct"] else "FAILED"]
                 for r in results["predictions"] if r["panel"] == group["panel"]], [18,54,68,25])
        sources = [r for r in results["predictions"] if r["panel"] == group["panel"]]
        g.paragraph("Photo sources: " + " | ".join(f'<link href="{escape(by_id[r["id"]]["source_url"], {chr(34): "&quot;"})}">{r["id"]}</link>' for r in sources))
    def footer(canvas, document):
        canvas.setFont("Helvetica", 8)
        canvas.drawString(50, 28, "EcoSort AI | Real-photo update | 23 October 2026")
        canvas.drawRightString(545, 28, str(document.page))
    g.SimpleDocTemplate(str(g.PDF), pagesize=g.A4, leftMargin=18*g.mm, rightMargin=18*g.mm,
                        topMargin=16*g.mm, bottomMargin=18*g.mm).build(g.story, onFirstPage=footer, onLaterPages=footer)
    lines = ["# Real-Photo Acceptance Update", "", "23 October 2026. Current application: local frozen SigLIP2 object recognition.", "",
             "| Panel | Category | Category + conditional route |", "|---|---|---|"]
    lines += [f"| {r['panel']} | {r['category_correct']}/20 | {r['joint_correct']}/20 ({r['joint_correct']/20:.0%}) |" for r in results["groups"]]
    lines += ["", "All three first evaluations passed the >=17/20 joint criterion. All 60 UI callback decisions reproduce the recorded predictions, including failures.", "",
              "This is a prelabelled stratified convenience sample, not a random population estimate or a guarantee. Uncertainty counts as failure. Foundation pretraining membership is unknown. No ground-truth item confirmation was supplied to inference.", "",
              "The six previously observed local photos are development-only: 5/6 category matches; the plastic puzzle remains unresolved. Private photos are excluded from GitHub.", "",
              "Archived dataset: 167,285 indexed / 147,303 retained / 117,893 train / 14,698 validation / 14,712 test. Current frozen SigLIP2 was NOT trained on those images. Historical benchmark scores must not be attributed to it.", "",
              "## Failures", ""]
    lines += [f"- Panel {r['panel']}, {r['id']}: {r['expected_object']} -> {r['predicted_object']}; {r['predicted_route']}." for r in failed]
    lines += ["", "The printed plastic bag has an unsafe wrong yellow-bin suggestion. Human confirmation remains important.", "",
              "See EcoSort_AI_Real_Photo_Update_EN.pdf, acceptance_web/results.csv and acceptance_web/test_manifest.json for full evidence and source attribution."]
    (BASE/"REAL_PHOTO_ACCEPTANCE_REPORT.md").write_text("\n".join(lines)+"\n")
    print(g.PDF)


if __name__ == "__main__":
    main()
