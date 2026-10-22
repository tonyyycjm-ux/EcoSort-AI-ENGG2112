"""Build the English evidence guide from completed local measurements."""
import json
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle, PageBreak

BASE = Path(__file__).resolve().parent
PDF = BASE / "EcoSort_AI_Technical_Guide_EN.pdf"
REPO = "https://github.com/tonyyycjm-ux/EcoSort-AI-ENGG2112"
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="BodyTextGuide",fontName="Helvetica",fontSize=10.5,leading=14,spaceAfter=8))
styles.add(ParagraphStyle(name="TableGuide",fontName="Helvetica",fontSize=9,leading=12))
styles.add(ParagraphStyle(name="GuideTitle",fontName="Helvetica-Bold",fontSize=24,leading=28,spaceAfter=12,textColor=colors.black))
for name in ("Heading1","Heading2"):
    styles[name].textColor = colors.black
    styles[name].spaceAfter = 9
    styles[name].spaceBefore = 10
story = []


def paragraph(text,style="BodyTextGuide"):
    story.append(Paragraph(text,styles[style]))


def table(headers,rows,widths):
    values = [[Paragraph(escape(str(value)),styles["TableGuide"]) for value in row] for row in [headers]+rows]
    item = Table(values,colWidths=[value*mm for value in widths],repeatRows=1,hAlign="LEFT")
    item.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#e5edf2")),
        ("GRID",(0,0),(-1,-1),.45,colors.HexColor("#d9d9d9")),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("LEFTPADDING",(0,0),(-1,-1),7),("RIGHTPADDING",(0,0),(-1,-1),7),
        ("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6)]))
    story.extend([item,Spacer(1,9)])


def section(title):
    if story:
        story.append(PageBreak())
    paragraph(title,"Heading1")


def load(name):
    return json.loads((BASE/name).read_text())


def footer(canvas,document):
    canvas.setFont("Helvetica",8)
    canvas.setFillColor(colors.HexColor("#555555"))
    canvas.drawString(18*mm,13*mm,"EcoSort AI  |  ENGG2112 team evidence guide  |  22 October 2026")
    canvas.drawRightString(192*mm,13*mm,str(document.page))


def main():
    verification = load("data/expanded/dataset_verification.json")
    current = load("trained_semantic/training_completed.json")
    first = load("trained_nonlinear/training_completed.json")
    second = load("trained_multiview/training_completed.json")
    latest = load("trained_multiseed/training_completed.json")
    external = load("trained_multiseed/external_diagnostics.json")
    active = load("active_model.json")
    assert active["results"] == "trained_semantic"
    assert latest["target_met"] is False
    paragraph("EcoSort AI Technical Evidence Guide","GuideTitle")
    paragraph("Implementation details and verified results for the ENGG2112 report team","Heading2")
    paragraph("This guide explains what the program implements, how its experiments were run, which numbers can be cited, and what remains incomplete. It is an evidence pack for report writing, not a claim that the system is ready for unattended waste sorting.")
    paragraph("<b>Main conclusion:</b> the latest candidate achieved 95.13% validation accuracy and 94.85% test accuracy on the reused 17-class public benchmark. The joint 95% target has not been met. The six local-photo diagnostic remains 2/6 under the stated label assumptions. The deployed website therefore remains the earlier semantic ensemble.")
    paragraph(f'Public code repository: <link href="{REPO}" color="#184b70">{REPO}</link>')
    table(["Model or experiment","Validation","Test","Test macro F1"],[
        ["MobileNetV3Large frozen baseline","Not shown here","73.02%","0.6902"],
        ["MobileNetV3Large partial fine tuning","Not shown here","78.42%","0.7483"],
        ["Deployed CLIP and MobileNet ensemble","86.43%",f"{current['test_metrics']['accuracy']:.2%}","0.8362"],
        ["Nonlinear CLIP head and neighbors",f"{first['validation_metrics']['accuracy']:.2%}",f"{first['test_metrics']['accuracy']:.2%}","0.9198"],
        ["CLIP and DINOv2 single head and neighbors",f"{second['validation_metrics']['accuracy']:.2%}",f"{second['test_metrics']['accuracy']:.2%}","0.9315"],
        ["CLIP and DINOv2 three heads and neighbors",f"{latest['validation_metrics']['accuracy']:.2%}",f"{latest['test_metrics']['accuracy']:.2%}",f"{latest['test_metrics']['macro_f1']:.4f}"]],[77,31,31,35])
    paragraph("All six expanded-model rows use the same 14,712-image test partition. Historical 12-class results use different data and must not be presented as controlled comparisons. A model score for one image is not overall accuracy.")

    section("Data preparation and exact image counts")
    paragraph("The raw pool contained 167,285 candidates: 147,285 indexed Garbage265 images and the previously selected 20,000 images. After decoding, duplicate/group checks, conflicting-label exclusions and targeted manual review, 147,303 images remained. The raw pool is not the number used to train a single epoch.")
    table(["Partition","Images","Purpose"],[["Training","117,893","Fits classifier parameters"],["Validation","14,698","Selects checkpoints, combinations and calibration"],["Test","14,712","Evaluates the selected candidate; never fits weights"]],[36,30,108])
    paragraph("The approximate 80/10/10 split uses seed 42 and per-class duplicate/source-family groups. Detected groups cannot cross partitions. SHA-256, resized pixel hashes and rotation/mirror-aware perceptual hashes reduce leakage; they do not prove complete independence. Source labels are preserved as fine_label. Chinese source disposal categories are not treated as Sydney bin rules.")
    table(["Retained source","Images"],[["Garbage265","128,212"],["RealWaste","4,597"],["TrashNet","2,103"],["Household dataset","7,979"],["E waste dataset","2,630"],["Phenom dataset","1,782"]],[119,55])
    paragraph("These are retained project counts, not publisher totals. The project used automated selection/auditing and selected human checks, not documented manual inspection of every image. Raw datasets and private user photos are not redistributed in the public package.")

    section("The seventeen output categories")
    import pandas as pd
    counts = pd.read_csv(BASE/"data/expanded/class_counts.csv",index_col=0)
    table(["Category","Train","Validation","Test"],[[label.replace("_"," "),f"{int(row['train']):,}",f"{int(row['validation']):,}",f"{int(row['test']):,}"] for label,row in counts.iterrows()],[87,29,29,29])
    paragraph("The outputs mix material categories, object types and residual-waste categories. They are not seventeen mutually exclusive physical materials. Ceramics has 754 training images, not 38. Plain water-cup and bowl source labels do not identify material and were mapped to mixed items, creating a real limitation for complete ceramic mugs.")
    paragraph("Paper material and residual categories also overlap for tissues. Exact disposal depends on the item and its condition, not material alone. The report should discuss this ontology limitation rather than implying that class balancing has removed it.")

    section("How the models and selection work")
    paragraph("<b>MobileNet comparison.</b> An ImageNet-pretrained MobileNetV3Large provides a frozen baseline. The later experiment fine-tunes late layers with BatchNorm frozen, training-only augmentation and validation macro-F1 checkpoint selection. This is distinct from the earlier MobileNetV3Small prototype.")
    paragraph("<b>Deployed semantic model.</b> A frozen local OpenCLIP ViT-B-32 encoder produces normalized 512-dimensional features. A regularized classification head is fitted to all training rows. Validation selects a 50/50 combination with the fine-tuned MobileNet. The selected zero-shot weight is zero. Temperature calibration changes score interpretation, not the underlying top class by itself.")
    paragraph("<b>New nonlinear candidate.</b> Training-only feature means and standard deviations normalize inputs. Each head uses a linear hidden layer with 768 units, LayerNorm, GELU, dropout 0.3 and an output layer. Training uses AdamW, learning rate 0.001, weight decay 0.01, batch size 1,024, label smoothing 0.03 and gradient clipping. At most 100 epochs are allowed; twelve non-improving epochs stop a run. This candidate does not use the historical class_weight setting.")
    paragraph("<b>Fine object labels.</b> A head learns the original fine_label combined with its mapped coarse label. Probabilities for fine labels belonging to a coarse class are summed into the same seventeen outputs. This preserves useful object distinctions but does not fix incorrect or ambiguous material annotations.")
    paragraph("<b>Complementary DINOv2 features.</b> The frozen Small model uses experimental 224-pixel whole-image letterboxing, adapted positional embeddings, a 384-dimensional class token and a 384-dimensional average patch vector. Together with CLIP, this produces 1,280 dimensions. Encoder weights are pinned and SHA-256 checked; no remote inference service receives photos.")
    paragraph("<b>Selected latest candidate.</b> Average three fine-label heads, then combine that average with fifteen cosine-similar training references at a 75/25 probability weight. Neighbor temperature is 0.03. The combination and final probability calibration are selected on validation only. The test labels and six local photos do not fit or select the heads.")
    paragraph("The public test has now been reused across several experiments. It remains useful for benchmark comparison, but is not untouched final evidence; repeated comparisons can introduce indirect adaptation. A fresh local confirmation set is still required.")

    section("Program functions and source map")
    table(["Source file","What it implements"],[
        ["app.py and ecosort_runtime.py","Local Gradio entry point, upload/webcam inference, input checks, candidate scores and uncertainty"],
        ["sydney_rules.py","Conditional item-confirmed City of Sydney and NSW EPA disposal guidance with official links"],
        ["prepare_dataset.py and prepare_expanded.py","Source recovery, audits, label mapping, duplicate groups and fixed partitions"],
        ["train_full.py","MobileNetV3Large baseline and partial fine tuning"],
        ["semantic_model.py and train_semantic.py","Pinned CLIP encoder, full-object preprocessing, trained linear head and ensemble selection"],
        ["nonlinear_model.py and train_nonlinear.py","Nonlinear/fine-label heads, train-only nearest neighbors, selection and calibration"],
        ["extract_dino_features.py","Pinned frozen DINOv2 extraction with resumable feature files"],
        ["evaluate_candidate.py and diagnose_local_photos.py","Fixed diagnostic panels, component comparisons and whole/cropped-photo checks"],
        ["ECOSORT notebooks","English text/code sections for environment, training, evaluation and local running"],
        ["Completion records and class reports","Machine-readable metrics, sample counts, per-class results and provenance"]],[65,109])
    paragraph("The interface accepts an uploaded image or a webcam snapshot, predicts one visual category, returns candidate scores and flags uncertain results. The disposal function additionally uses a user-confirmed item type and service availability. It is a transparent rule layer, not an LLM, retrieval-augmented generation or legal-compliance checker.")
    paragraph("No automatic object detector currently selects the user's intended item among several objects. The image cannot prove emptiness, contamination, hidden batteries or hazardous contents. The system does not operate physical sorting machinery or measure municipal batch-contamination rates.")

    section("Real photo performance and honest evaluation")
    table(["Panel","Deployed semantic model","Latest candidate","Interpretation"],[
        ["Thirty whole web photos","28/30",f"{external['web_thirty']['correct']}/30","Ten classes; previously observed convenience sample"],
        ["Fifty five TACO object crops","36/55",f"{external['taco_diagnostic']['correct']}/55","Development diagnostic, not whole-camera scenes"],
        ["Original ten web photos","8/10",f"{external['old_ten']['correct']}/10","Regression evidence only"],
        ["Six user photos","2/6",f"{external['local_six']['correct']}/6","Mug/towel assumptions plus owner-confirmed board and plastic piece"]],[43,39,31,61])
    paragraph("The higher benchmark candidate has not improved the six-photo local diagnostic, and its thirty-photo result remains below the deployed model. It has not replaced the website. Publishing the candidate's benchmark metrics does not make it the deployed or recommended real-world model.")
    paragraph("The gray board was confirmed as cardboard and the blue/green puzzle piece as plastic by the owner. The other two failure labels assume the visible mug is ceramic and the white sheet is kitchen paper towel. The candidate recognizes the water bottle and predicts residual trash for the towel, but still fails the mug, board, box and plastic piece. This panel is too small and partly assumption-labelled to estimate population accuracy.")
    paragraph("Review flags must remain in the top-1 accuracy denominator. Report coverage and accepted-only accuracy separately if studied; do not rename accepted-only accuracy as overall accuracy. Confidence is not a guarantee of correctness. Normal background should be allowed; manual cropping has sometimes made correct predictions wrong.")
    paragraph("Webcam capture and prediction were functionally verified on the deployed model, with 1,760 by 1,328 frames. This was an unlabelled smoke test, not measured camera accuracy. The camera was closed afterward. Foundation-model overlap with public web photos is unknown.")

    section("Development issues and report corrections")
    paragraph("<b>Migration and reproducibility.</b> Colab paths and file APIs were adapted to a project virtual environment and local file links while retaining the original export. Apple Metal/MPS was verified. Local predictions do not require a cloud inference connection after weights are present.")
    paragraph("<b>Imbalance and duplicates.</b> Historical class weighting improved mean recall but reduced accuracy; it did not remove bias. Hash/group filtering reduced duplicate leakage and excluded conflicting labels. Do not claim all images received individual manual review or that all duplicates were proven removed.")
    paragraph("<b>Preprocessing.</b> Center cropping could remove narrow objects' ends. The CLIP branch was changed to full-image letterboxing; features and heads were rebuilt and training/inference parity checked. This did not solve real-world domain shift or replace object detection.")
    paragraph("<b>Taxonomy and domain shift.</b> Cups, bowls and tissues expose inconsistent material/object/disposal supervision. Nonlinear and complementary-feature candidates substantially improved the benchmark but not the six local photos. A complete training-label audit and representative local training plus fresh independent testing remain necessary.")
    paragraph("<b>Draft corrections.</b> The uploaded seven-page draft describes an older 20,000-image, twelve-class MobileNetV3Small stage. Its actual Sections 3-7 are empty headings. An explicitly labelled indigestion example starts on page 3 and must not appear as EcoSort results. Its 80/20 split, cross-validation, AUC, binary/tabular preprocessing, investment claims and unrelated references are not this project's evidence.")
    paragraph("<b>Contribution and AI use.</b> The user states that one team member was responsible for the code. Record actual contributions rather than copying equal-contribution or weekly-meeting claims from the template. Codex/ChatGPT assisted coding, debugging, analysis and drafting; disclose this according to course requirements and describe human verification. No names, meetings or unverified contributions are invented here.")

    section("Reproducing the package and using it in the report")
    paragraph("The repository provides browseable source files and a structured archive containing the runnable local layout, compact experiment evidence and the deployed project-trained model. The archive excludes raw datasets, large generic pretrained encoders, feature caches, virtual environments, private photographs, credentials and the team's original report PDF. Public notebook outputs and execution metadata are cleared; source cells are retained with private-path redaction where necessary.")
    paragraph("For the complete layout, download and extract EcoSort_AI_Source_and_Evidence.zip. Use Python 3.12, create a virtual environment and install the supplied requirements. Apple Silicon is the verified platform; do not claim Windows or other hardware has been tested. Run semantic_model.py once to download the pinned CLIP encoder, then start app.py. The app loads the deployed model without a full retraining. Generic encoder downloads require internet access; subsequent local inference does not.")
    paragraph("Training reproduction additionally requires restoring the original selected data and Garbage265 sources, auditing their licences and restoring the fixed manifest. The original image archives and private photos are deliberately absent. Candidate nearest-neighbor inference also requires regenerated training feature references. Read the archive README before running migration scripts; some historical scripts intentionally stop rather than overwrite existing evidence.")
    paragraph("For the report, cite exact counts and the correct model version; include per-class results, not accuracy alone. Separate historical Colab experiments, expanded benchmarks, real-photo diagnostics and webcam functionality. Do not claim 95% test accuracy, improved real-photo accuracy, commercial readiness, measured carbon savings or completed batch/LLM functions.")
    paragraph("Key evidence files are dataset_verification.json, class_counts.csv, each experiment's training_completed.json, classification_report.csv, validation_trials.csv, external_diagnostics.json and active_model.json. The English and Chinese report-review documents list page-specific corrections and evidence references.")
    paragraph("Core sources: <link href='https://github.com/garythung/trashnet'>TrashNet</link>; <link href='https://archive.ics.uci.edu/dataset/908/realwaste'>RealWaste UCI</link>; <link href='https://arxiv.org/abs/1905.02244'>MobileNetV3</link>; <link href='https://github.com/mlfoundations/open_clip'>OpenCLIP</link>; <link href='https://arxiv.org/abs/2103.00020'>CLIP</link>; <link href='https://github.com/facebookresearch/dinov2'>DINOv2</link>; <link href='https://huggingface.co/timm/vit_small_patch14_dinov2.lvd142m'>DINOv2 weights</link>; <link href='https://www.cityofsydney.nsw.gov.au/waste-recycling-services/recycle-household-items'>City of Sydney recycling</link>; <link href='https://www.epa.nsw.gov.au/Your-environment/Recycling-and-reuse/household-recycling-overview/community-recycling-centres'>NSW EPA recycling centres</link>. Dataset source URLs are also recorded in preparation scripts. Access checked on 22 October 2026; no third-party ownership or licence is transferred by this repository.")
    SimpleDocTemplate(str(PDF),pagesize=A4,rightMargin=18*mm,leftMargin=18*mm,topMargin=18*mm,bottomMargin=22*mm,
                      title="EcoSort AI Technical Evidence Guide",author="EcoSort AI project evidence").build(story,onFirstPage=footer,onLaterPages=footer)
    print(PDF)


if __name__ == "__main__":
    main()
