# EcoSort AI Final Report Review and Evidence Pack

Reviewed on 22 October 2026, Australia/Sydney. Document: Downloads/ENGG 2112 Final report.pdf, seven PDF pages. The original PDF has not been edited. Page references below refer to PDF pages, not the example's internal numbering.

## 1. Overall Assessment

Pages 1-2 and the beginning of page 3 contain the EcoSort draft, largely describing the earlier 20,000-image, 12-class version. The actual Related Work, Methodology, Results, Wider Adoption and Conclusion sections on page 3 are headings without project-specific content. An explicitly labelled example begins at the bottom of page 3 and continues through page 7, discussing student indigestion. It is an example, not evidence that the team claims to have conducted those experiments. Remove it from the submission body or keep it separately as a writing reference.

This is not yet a completed final report. Substantial verified methods, results and development issues are available to populate it, but the 95% target, representative local-photo validation and parts of the original proposal remain incomplete.

## 2. Differences Between the Draft and Implementation

| Location | Draft statement or omission | Evidence-based correction |
|---|---|---|
| Page 1; page 2 Objective 2 | MobileNetV3Small is the underlying architecture | Historically valid, not the current complete method. Expanded experiments use MobileNetV3Large; the deployed model adds frozen OpenCLIP ViT-B-32 features and a trained classification head. A newer nonlinear/fine-label/nearest-neighbor candidate has completed one experiment. Describe version history rather than combining all methods into one model. |
| Page 2, Section 2.3 | The most recent experiment has 20,000 images and 12 categories | Outdated. The current raw pool has 167,285 candidates; 147,303 images in 17 categories remain after auditing. Keep 20,000/12 as an earlier stage. |
| Page 2, Section 2.3 | All 20,000 images were manually selected | The evidence supports automated inventory selection, decoding, duplicate/group checks and targeted manual inspection, not individual human selection of every image. Use selected and audited images with automated checks and targeted manual review. |
| Pages 1 and 3 | The system does not determine a bin or assess local compliance | It is still not a compliance-certification system, but it now provides conditional, source-linked City of Sydney / NSW EPA guidance with user-confirmed item types. Do not describe guidance as absent, or claim autonomous guaranteed disposal decisions. |
| Page 2, Section 2.1 | Automation of waste sorting | The prototype assists image-based identification; it does not operate physical sorting equipment. Use image-based waste identification and disposal assistance. |
| Page 2 | Categories described simply as materials | The 17 outputs mix material categories, object types and residual-waste categories. They are not 17 mutually exclusive material types; discuss the taxonomy limitation. |
| Page 1 | Repeated scope-refinement and AI for Good paragraphs | Consolidate the repeated content. Environmental benefit is motivation, not measured recycling improvement or carbon savings. |
| Page 1 links; References | Two bare dataset links and example references | Cite every actual source, relevant model papers/implementations and official guidance. Publicly accessible does not mean public domain; check licences rather than asserting unrestricted rights. |
| Page 3, Sections 3-7 | Only section headings | Populate with actual EcoSort methods, measurements, limitations and conclusions. Do not substitute the attached example's settings or figures. |
| Page 5, example Section 4.4 | 80/20 split, ten-instance cross-validation, averaging three best models, AUC, Colab-only experiments | These are example settings, not this implementation. Current experiments use a fixed approximately 80/10/10 grouped split, validation-only model selection, local Mac execution and accuracy/macro-F1/per-class metrics. Ten-fold validation and AUC have not been established as completed project results. |
| Page 5, example Sections 4.1-4.3 | Mean imputation, tabular one-hot encoding, feature1/feature2, binary classification | These belong to the example. EcoSort processes RGB images and performs multiclass classification; a fine-label head can aggregate specific object predictions into 17 classes. |
| Pages 6-7, example | Investment, Deloitte evidence, competitiveness, equal contributions, weekly meetings, 85%/95%, Einstein/Knuth/LaTeX references | Not EcoSort evidence. Remove these example statements; do not invent funding, team meetings, commercial readiness or equal contributions. 95% is an outstanding target, not an achieved result. |

## 3. Verified Data and Results

### Data Preparation and Partitions

- Raw candidates: 167,285, comprising 147,285 indexed Garbage265 images and the previously selected 20,000 images.
- Retained images: 147,303, not the entire raw pool.
- Training: 117,893; validation: 14,698; test: 14,712, approximately 80/10/10.
- Seed 42 is used to assign duplicate/source-family groups within each class. Image counts are not exactly the nominal proportions. Detected groups do not cross partitions; heuristic near-duplicate checks do not prove complete independence.
- Retained source counts: Garbage265 128,212; RealWaste 4,597; TrashNet 2,103; household 7,979; ewaste 2,630; phenom 1,782. These are retained project counts, not publisher totals.
- Outputs: battery, cardboard, ceramics, ewaste, food_organics, glass, household_problem_waste, light_bulbs, medicines, metal, mixed_items, paper, plastic, textile_trash, trash, vegetation and wood.
- Ceramics currently has 943 retained images: 754 training, 97 validation and 92 test, not 38 training examples.

### Keep Experiment Versions Separate

| Experiment | Validation accuracy | Test accuracy | Test macro-F1 | Interpretation |
|---|---:|---:|---:|---|
| Historical 20,000-image, 12-class unweighted head | Not rechecked here | 85.48% | 0.8071 | Earlier Colab experiment; different data/splits prevent direct controlled comparison with the expanded models |
| Historical 20,000-image, 12-class weighted head | Not rechecked here | 84.17% | 0.7968 | Mean recall increased from 79.62% to 81.61%, while accuracy decreased |
| Expanded MobileNetV3Large frozen baseline | Not separately reported here | 73.02% | 0.6902 | Same expanded test partition |
| Expanded MobileNetV3Large partial fine-tuning | Not separately reported here | 78.42% | 0.7483 | Same expanded test partition |
| Full-object letterbox OpenCLIP head + MobileNet, 50/50 | 86.43% | 85.88% | 0.8362 | Deployed model until active_model.json identifies another completed version |
| New nonlinear fine-label head + training-reference neighbors, 50/50 | 93.83% | 93.50% | 0.9198 | Completed candidate experiment; not 95% and not proof of improved local-photo performance or deployment |
| CLIP + DINOv2 features with one fine-label head and neighbors | 94.85% | 94.65% | 0.9315 | Candidate only; local photos remain 2/6; not deployed |
| CLIP + DINOv2 features with three averaged fine-label heads and neighbors | 95.13% | 94.85% | 0.9338 | Best completed public-benchmark candidate in this run; validation alone exceeds 95%, test does not; local photos remain 2/6; not deployed |

The new candidate uses all 117,893 training rows through cached frozen features. Material-level and fine-label nonlinear heads were compared; validation selected a combination of the fine-label classifier and training-only neighbor probabilities. Its 28.06-second measured run reused existing feature extraction and excludes data download, extraction and earlier feature computation. It is not the full end-to-end training time.

Subsequent diagnostics for that candidate were 2/6 local photos, 23/30 web whole photos, 29/55 TACO crops and 9/10 old photos. The local panel did not improve and two external panels regressed, so the candidate did not replace the deployed site. This negative result is important: do not selectively report only 93.50% or 9/10. Evidence: trained_nonlinear/external_diagnostics.json.

The subsequent frozen DINOv2 Small encoder contributes complementary shape/texture features using 224-pixel letterboxed inputs, a class token and mean patch features, concatenated with CLIP into 1,280 dimensions. A single-head candidate scored 94.65% test accuracy. Averaging three fine-label heads and blending them 75/25 with training-reference neighbors yielded 95.13% validation accuracy and 94.85% test accuracy after validation-only selection and calibration. Diagnostics were 25/30 web photos, 36/55 TACO crops, 9/10 old photos and 2/6 local photos. Do not round the test result into an achieved 95% target or claim a real-photo improvement. Evidence: trained_multiview and trained_multiseed. The 76.56-second three-head experiment excludes DINO feature extraction, pretrained-weight download and data preparation.

The public test partition has been evaluated across several model versions. No test labels update model weights, but repeated benchmark comparisons can still create indirect adaptation. Describe it as a reused benchmark and add a fresh local confirmation set rather than calling it untouched final evidence.

### External and Local Checks Must Be Reported Separately

- The previously deployed OpenCLIP/MobileNet model scored 28/30 (93.33%) on thirty fixed whole-object web photos covering ten classes, three photos per class. This is a convenience sample, not representative household accuracy; foundation-model pretraining overlap is unknown.
- The original ten web photos scored 8/10 after previous use: regression evidence only.
- The 55 TACO object crops scored 36/55 (65.45%). This panel informed preprocessing diagnosis and is development evidence, not an independent final test.
- Six new user photos exposed severe local errors. The box and bottle were correct; owner-confirmed cardboard was predicted plastic, and an owner-confirmed plastic puzzle piece was predicted paper. Assuming the other objects are a ceramic mug and kitchen paper towel, top-1 accuracy is 2/6. The mug/towel labels still include visual inference; this is not a fully independently annotated benchmark.
- Five of six images triggered review. Review flags do not turn incorrect top-1 predictions into correct predictions and must not be removed from the accuracy denominator.
- Webcam frame capture, photography and prediction were tested, followed by closing the camera. There is no labelled webcam accuracy experiment.

## 4. Implemented Functions and How They Work

| Function | Actual method and scope | Report section |
|---|---|---|
| Multi-source data recovery/expansion | Public archives, preserved source labels, SHA-256 verification of restored selected images | 4.1 Data Preparation |
| Auditing and grouping | Image decoding and size checks; SHA-256/pixel hashes; rotation/mirror-aware perceptual hashes; source-family grouping; conflicting-group exclusion; targeted manual review | 4.1 |
| MobileNet comparison | ImageNet-pretrained MobileNetV3Large; frozen baseline; partial late-layer fine-tuning with BatchNorm frozen; training-only augmentation; validation macro-F1 checkpoint selection | 4.2-4.3 |
| General frozen visual features | Local OpenCLIP ViT-B-32, full-image letterboxing, regularized classification head and validation-selected MobileNet combination | 4.2-4.3 |
| New candidate experiment | Nonlinear fine-label classification aggregated into 17 classes, combined with references from the training partition only | 4.3, 5.1 |
| Evaluation and provenance | Accuracy, macro-F1, per-class precision/recall/F1, confusion matrices, prediction files, fixed manifests and completion records | 5.1 |
| Local user interface | English Gradio interface, image upload, webcam snapshot, candidate scores and explicit uncertainty; localhost rather than public internet hosting | 4.4, 5.1 |
| Conditional disposal assistance | Predicted category plus user-confirmed object and food-service availability enter a rule function with City of Sydney / NSW EPA source links | 4.3, 6 |
| Reproducibility | Python 3.12 virtual environment, dependency lock, English text/code notebook sections, unchanged original Colab export and model/feature integrity checks | 4.4, Appendix |

Not implemented as validated capabilities: automatic object detection/target boxes, verified emptiness or contamination recognition, hidden-hazard detection, physical sorting, municipal batch-contamination auditing, or evaluated LLM-generated summaries. The disposal layer is rule-based, not an LLM or autonomous legal-compliance system.

## 5. Evidence-Based Issues Faced

| Issue | Diagnosis | Implemented response / measured outcome | Remaining limitation |
|---|---|---|---|
| Slow cloud workflow and local migration | Colab-specific /content paths, downloads and environment differed from the Mac | Preserved original export, built .venv, adapted paths/file links and verified local Metal/MPS inference/training | Do not claim every step runs on the GPU or present cached-head timing as full training |
| Class imbalance | Common classes outnumbered ceramics/vegetation and other rare categories | Historical weighted/unweighted comparison documented the recall/accuracy trade-off | No guarantee of zero bias; the current nonlinear experiment uses label smoothing, not class_weight |
| Duplicate images and conflicting labels | Multiple archives and near-identical source families risk leakage | Group-aware filtering, conflict exclusion and fixed partitions | Heuristic duplicate detection does not establish complete independence |
| Preprocessing removed object shape | Center crops could discard the ends of narrow bottles | Changed the CLIP branch to whole-object letterboxing; rebuilt features/heads and checked training/inference parity | Not automatic object localization; background interference remains possible |
| Optimistic public benchmark performance | Product photos and real household scenes differ | Separated whole-photo, TACO diagnostic and local-photo evaluations; preserved failures | Real-world performance remains inadequate |
| Cup/bowl/tissue ontology conflicts | Water cups and bowls mapped to mixed_items; paper material and residual-waste labels overlap | Preserved fine labels, diagnosed the conflict and added a fine-label learning experiment | No complete image-level material relabelling has been performed |
| Unstable manual cropping | A correctly classified bottle became glass after cropping | Logged whole/crop/component comparisons rather than assuming background removal fixes recognition | Manual crops are not a validated detector |
| Restricted classification head | A linear boundary may not capture complex class structure | Nonlinear fine-label head + neighbors reached 93.83% validation and 93.50% test | The explanation is an engineering hypothesis; the real-photo improvement and 95% target remain unproven |
| Misleading display of uncertain predictions | A prominent top label could appear definitive | Prioritized Unable to identify reliably and renamed scores as candidates | Display changes do not improve accuracy; confident errors remain possible |
| Material does not establish disposal eligibility | Image labels do not reveal contents or all service conditions | Separate conditional rules, item confirmation and official-source links | Not a guarantee of safe or legally compliant disposal |
| Repeated local service startup | Notebook/app launches could compete for the same port | Detect and reuse the running EcoSort service | Other historical ports may still show older versions; identify the current URL and active model |

Do not invent exception traces, crash counts, energy savings, user surveys or end-to-end training times not recorded in the evidence.

## 6. Writing Guidance for Missing Sections

### Methodology

Explain the fixed grouped manifest, audited multi-source images, transfer learning and model-specific preprocessing. Distinguish frozen encoders, trained heads, partial MobileNet fine-tuning, validation selection and benchmark evaluation. Keep the deployed version separate from experimental candidates.

### Key Findings

On the same expanded test partition, compare 73.02% frozen MobileNet, 78.42% partially fine-tuned MobileNet, 85.88% CLIP/MobileNet and 93.50% nonlinear/neighbor candidate, with macro-F1. Do not call historical 12-class results controlled comparisons. The local-photo failures are a central finding about domain shift and label semantics, not a result to hide.

### Wider Adoption and Conclusion

The evidence supports a local single-item assistance prototype with conditional source-linked guidance. Further work should address representative local training/evaluation, label refinement, target selection/localization and error risk. A target of 95% does not establish commercial readiness, environmental impact or suitability to replace human inspection.

### Individual Contributions and AI Use

The user states that one team member was responsible for the code. Record software implementation, pipeline integration, local deployment and model experiments accordingly; other members should confirm their own literature, writing and coordination contributions. Do not copy the example's equal-contribution or weekly-meeting statements without records.

Codex/ChatGPT assistance was used for coding, debugging, analysis and drafting. Do not claim unaided authorship. The final AI Use Appendix should identify tools, purposes, verification and human responsibility in accordance with the course requirements. This evidence pack is not a declaration already approved by every team member.

## 7. Evidence to Share with the Report Team

- data/expanded/dataset_verification.json, class_counts.csv and split_manifest.csv: exact retained counts and partitions.
- trained_expanded/baseline_vs_finetuned.csv: controlled MobileNet comparison.
- trained_semantic/training_completed.json and classification_report.csv: deployed semantic ensemble metrics.
- trained_nonlinear/training_completed.json, validation_trials.csv and classification_report.csv: the completed 93.50% candidate experiment.
- final_test/evaluation.json, predictions.csv and test_manifest.json: web whole-photo results, attribution and limits.
- trained_semantic/taco_diagnostic/evaluation.json: complex-scene diagnostic failures.
- local_photo_diagnostic/predictions.json and FINDINGS.md: six local-photo predictions and owner confirmation.
- accuracy_ui_verification.json: upload/webcam functionality, not camera accuracy.
- active_model.json: the actual deployed model; candidate scores must not be attributed to the live site prematurely.

## 8. References to Use Instead of the Example Bibliography

Cite [TrashNet](https://github.com/garythung/trashnet), [RealWaste / UCI](https://archive.ics.uci.edu/dataset/908/realwaste), all other actual source datasets and original publishers, [MobileNetV3](https://arxiv.org/abs/1905.02244), [OpenCLIP](https://github.com/mlfoundations/open_clip), [CLIP](https://arxiv.org/abs/2103.00020), [City of Sydney recycling](https://www.cityofsydney.nsw.gov.au/waste-recycling-services/recycle-household-items) and [NSW EPA community recycling centres](https://www.epa.nsw.gov.au/Your-environment/Recycling-and-reuse/household-recycling-overview/community-recycling-centres). Add the DINOv2 paper and exact weights only if that experiment is ultimately included. Einstein and LaTeX references do not support EcoSort methodology or findings.

## 9. Outstanding Work

The joint 95% validation/test target remains incomplete: validation is 95.13%, test is 94.85%. Representative independent local-photo evaluation, comprehensive image-level label cleanup, automatic multi-object localization, original-proposal batch-contamination metrics, LLM-summary comparison, final report and verified team contribution statements also remain separate incomplete items. The team should explicitly explain which original-proposal functions were removed by the revised software scope rather than treating omitted functions as completed.
