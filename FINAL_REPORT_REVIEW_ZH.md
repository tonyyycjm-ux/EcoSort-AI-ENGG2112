# EcoSort AI Final Report 核对与写作材料：中文版

核对日期：2026-10-22，Australia/Sydney。对象：Downloads/ENGG 2112 Final report.pdf，7 页。原 PDF 未修改。页码均指 PDF 页码，不是模板内部页码。

## 1. 总体判断

前两页和第三页开头是 EcoSort 草稿，但基本停留在旧的 20,000 张、12 类阶段。第三页的 Related Work、Methodology、Results、Wider Adoption 和 Conclusion 只有标题，没有实际内容。第三页下方明确出现 example)，随后到第七页是学生消化不良的示例报告。这是模板示例，不应被误认为同学声称完成的 EcoSort 实验；正式稿应删去或移出提交正文。

目前不能把这个文件当作已完成的 final report。我们已有足够的真实方法、实验与问题记录来填补大部分技术章节，但准确率 95%、大规模本地实拍验证和原 proposal 的部分功能尚未完成。

## 2. 报告与实际实现的差异

| 位置 | 草稿内容或缺口 | 核对结果及应如何修改 |
|---|---|---|
| 第 1 页、第 2 页 Objective 2 | 核心模型是 MobileNetV3Small | 这是旧版。现有扩展实验使用 MobileNetV3Large；已部署模型加入冻结 OpenCLIP ViT-B-32 特征与分类头。新非线性、细分类和近邻候选已经完成一轮，见下表；后续候选不能提前写成最终部署版本。应按版本写演进过程。 |
| 第 2 页 2.3 | most recent experiment 是 20,000 张、12 类 | 已过时。当前原始候选 167,285 张，审核保留 147,303 张、17 类。20,000 张、12 类可保留为历史阶段，不应称为最新版本。 |
| 第 2 页 2.3 | 20,000 manually selected images | 有自动清单选择、解码、去重、分组和部分人工检查的证据，没有逐张人工挑选全部 20,000 张的证据。改为 selected and audited images with automated checks and targeted manual review。 |
| 第 1 页、第 3 页 | 不判断具体垃圾桶；不做地方规则合规判定 | 后一句“不是法规合规审查系统”仍然正确；但现在已有 City of Sydney / NSW EPA 来源链接的条件式投放建议和物品确认选项。应写成提供 guidance，而不是完全不提供投放信息，也不是自动保证正确投放。 |
| 第 2 页 2.1 | automation of sorting waste | 实际为辅助识别软件，不驱动分拣机械。改为 image-based waste identification and disposal assistance，避免暗示实物自动分拣。 |
| 第 2 页 | 全部称 waste materials | 17 类混合了材质、物品类型和残余垃圾类别，并非 17 种互斥材质。这是需要讨论的 taxonomy limitation。 |
| 第 1 页 | 缩小范围和 AI for Good 的两组段落 | 有明显重复，保留一组完整表述即可。AI for Good 是动机，不是已测量的减排或回收率提升。 |
| 第 1 页开头、References | 两个裸 dataset 链接；参考文献为模板 | 正式补充全部实际数据源、模型和实现、相关论文、地方投放规则与访问日期。公开可下载不等于 public domain，不能不核对许可就声称无版权限制。 |
| 第 3 页 Sections 3-7 | 方法、结果、问题、结论只列标题 | 需要填真实内容。不能用后面 example 的实验设置、曲线和数字代替。 |
| 第 5 页示例 4.4 | 80/20、十次交叉验证、平均三个最佳模型、AUC、全在 Colab | 均不是当前 EcoSort 实验流程。当前约 80/10/10、固定分组清单、验证集选方案、一次选定方案后评估测试；本地 Mac 运行。AUC 和十次交叉验证未作为本项目现有主要结果完成。 |
| 第 5 页示例 4.1-4.3 | 缺失值均值填补、表格 one-hot、feature1/feature2、binary classification | 属于示例，不适用于我们的 RGB 图像多类别分类流程。不能写成实际实现。细分类头使用具体物品标签学习后汇总为 17 类，也不是二分类。 |
| 第 6-7 页示例 | 商业竞争力、Deloitte、投资、每周会议、贡献平均、85%和95%、爱因斯坦/Knuth/LaTeX 文献 | 不是 EcoSort 的实验证据，应删除。不能依据模板声称实际会议、贡献、融资或商业可行性；95% 是当前目标，尚未达成。 |

## 3. 可核对的数据与结果

### 数据规模和划分

- 原始候选：167,285 张，即新增 Garbage265 元数据索引 147,285 张，加原有选定 20,000 张。
- 实际保留：147,303 张，不是 167,285 张全部进入训练。
- Training：117,893；Validation：14,698；Test：14,712。约 80% / 10% / 10%。
- 采用 seed 42，在每个类别内按重复/来源家族分组划分；因此按图片数不恰好等于 80/10/10。已检测分组不跨集合，但近重复检测是启发式，不能保证绝对无泄漏。
- 保留数据源数量：Garbage265 128,212；RealWaste 4,597；TrashNet 2,103；household 7,979；ewaste 2,630；phenom 1,782。这里是审核保留数，不是各源网站标注的原始总数。
- 17 类：battery、cardboard、ceramics、ewaste、food_organics、glass、household_problem_waste、light_bulbs、medicines、metal、mixed_items、paper、plastic、textile_trash、trash、vegetation、wood。
- 目前陶瓷不是只有 38 张：943 张保留图片，其中 754 training、97 validation、92 test。

### 结果必须按实验版本区分

| 版本 / 数据范围 | Validation accuracy | Test accuracy | Test macro-F1 | 状态 |
|---|---:|---:|---:|---|
| 旧 20,000 张、12 类、不加权头 | 此处未另核对 | 85.48% | 0.8071 | 历史 Colab 实验；不可与新测试集直接声称公平优劣 |
| 旧 20,000 张、12 类、加权头 | 此处未另核对 | 84.17% | 0.7968 | mean recall 从 79.62% 到 81.61%，但 accuracy 下降，说明权重有取舍 |
| 扩展 17 类 MobileNetV3Large frozen baseline | 此处不另报 | 73.02% | 0.6902 | 在同一扩展测试集上 |
| 扩展 17 类 MobileNetV3Large partial fine-tuning | 此处不另报 | 78.42% | 0.7483 | 在同一扩展测试集上 |
| 全物体 letterbox OpenCLIP 分类头 + MobileNet，50/50 | 86.43% | 85.88% | 0.8362 | 当前已部署版本，直到 active_model.json 指向其他完成版本 |
| 新非线性细分类头 + 训练图片近邻，50/50 | 93.83% | 93.50% | 0.9198 | 本次已完成的候选实验；尚未达到 95%，不可称为已部署或证明实拍更好 |
| CLIP + DINOv2 特征，单细分类头与近邻组合 | 94.85% | 94.65% | 0.9315 | 候选实验，未部署；实拍仍 2/6 |
| CLIP + DINOv2，三个细分类头平均后与近邻组合 | 95.13% | 94.85% | 0.9338 | 当前本轮最好公开 benchmark 候选；只有 validation 达 95%，test 未达，实拍仍 2/6；未部署 |

本轮新候选使用现有冻结特征、所有 117,893 个训练样本，比较 17 类头和 fine_label 细分类头，并按验证集选择与训练图片近邻匹配的组合。新头训练和评估耗时 28.06 秒，复用了已经提取的特征；不能把这个时间写成全部图片下载、特征提取和完整系统训练总时间。

该候选后续诊断：本地六张仍为 2/6；网上 30 张为 23/30；TACO 为 29/55；旧十张为 9/10。它没有改善这组六张实拍，而且两个外部面板退步，所以没有替换网站上的已部署版本。这是报告中值得讨论的负面结果，不应只挑 93.50% 和 9/10 来写。证据：trained_nonlinear/external_diagnostics.json。

后续冻结 DINOv2 Small 提供形状/纹理互补特征，使用 224 像素补边输入、CLS token 与平均 patch 特征，与 CLIP 合并为 1,280 维。单头候选 94.65% test；三个细分类头的平均预测与训练近邻按 75/25 组合，经验证集选择与校准后，validation 95.13%、test 94.85%。该版本网上 30 张为 25/30、TACO 为 36/55、旧十张为 9/10、本地仍 2/6。不能四舍五入成 test 达到 95%，也不能说实拍已改善。证据目录：trained_multiview、trained_multiseed。三头实验的 76.56 秒不包含 DINO 特征提取、预训练模型下载和数据准备。

现在的公开测试集已经被多个版本反复用于结果比较，因此是重复使用的 benchmark，不是从未见过的新最终确认集。没有用其标签更新参数，但长期迭代仍可能对基准产生间接适应，最终应增加独立本地测试集。

### 实拍及外部检查要单独报告

- 对此前部署的 OpenCLIP + MobileNet 版本，固定的 30 张网上完整物品图是 28/30（93.33%），仅覆盖 10 类、每类 3 张，是便利样本，不能代表真实家庭使用准确率。基础模型是否在预训练时见过这些公开图片未知。
- 原先 10 张已经使用过的照片：8/10，只能作为 regression check，不是新独立测试。
- TACO 55 张物体裁剪图：36/55（65.45%）。它已参与定位预处理问题，因此是 development diagnostic，不再称独立测试。
- 你新增的 6 张实拍：纸箱和水瓶正确；纸板被猜 Plastic、塑料拼接块被猜 Paper。按另外两张分别是陶瓷杯与厨房纸判断，为 2/6。两种材质由物主确认，杯子和厨房纸仍含视觉判断，不应包装成完善的人工标注集。
- 六张中五张触发 review，仍应计入原始 top-1 错误。不能删掉“不确定”图片后只报准确率。
- 摄像头已实际取帧、拍照、完成预测，之后关闭；没有已标注的摄像头准确率实验。

## 4. 代码实际实现的功能与方法

| 功能 | 实际方法与边界 | 对应报告章节 |
|---|---|---|
| 多源数据恢复与扩展 | 下载公开档案，保留原文件与来源标签；恢复原清单时校验 SHA-256 | 4.1 Data Preparation |
| 数据审核及分组划分 | 图像解码、尺寸检查、SHA-256、像素哈希、旋转/镜像感知哈希、来源家族分组、标签冲突排除；只有部分样例人工审查 | 4.1 |
| MobileNet 对照 | ImageNet 预训练 MobileNetV3Large；冻结基线；部分末层微调，BatchNorm 保持冻结，训练集增强，验证 macro-F1 选 checkpoint | 4.2-4.3 |
| 冻结通用视觉特征 | 本地 OpenCLIP ViT-B-32；完整物体补边；训练正则化分类头，验证选择与 MobileNet 的组合 | 4.2-4.3 |
| 本轮新候选 | 非线性网络学习细分类再汇总；与仅来自训练分区的近邻概率组合；所有选择在 validation 上做 | 4.3、5.1 |
| 评估与留痕 | Accuracy、macro-F1、每类 precision/recall/F1、混淆矩阵、清单、预测文件和完成状态 | 5.1 |
| 本地界面 | Gradio 英文界面、上传、摄像头拍照、候选分数、明确不确定提示；localhost，不是互联网公开部署 | 4.4、5.1 |
| 条件式投放建议 | 分类结果 + 用户确认具体物品 + 食物服务条件，进入规则函数；附 City of Sydney / NSW EPA 官方链接 | 4.3、6 |
| 复现与版本保留 | Python 3.12 虚拟环境、依赖锁、本地 Notebook 英文文字/代码分段、原 Colab 导出不修改、模型与特征校验 | 4.4、Appendix |

没有自动目标框检测、没有自动判断物品是否空/干净/危险、没有真实分拣机械、没有 municipal batch contamination 验证，也没有当前已验证的 LLM 生成总结。规则建议不是 LLM 输出，不应写成大模型检索或自动法规合规审查。

## 5. 可写入 Issues Faced 的真实开发问题

| 问题 | 诊断或原因 | 实际修正 / 结果 | 仍然存在的限制 |
|---|---|---|---|
| 云端运行慢，本地迁移涉及路径/API差异 | Colab 的 /content、文件下载 API、环境与本机不同 | 保留原导出，建立 .venv，改本地路径/文件链接，验证 Metal/MPS 本地推理与训练 | 本地时长按具体流程测量，不能泛称所有步骤都在 GPU |
| 类别数量不均衡 | 常见类多，陶瓷、植被等少 | 历史比较加权与不加权，记录 recall/accuracy 取舍；后续仍按具体模型的真实损失配置描述 | 没有“完全无偏”；本轮非线性头没有使用 class_weight，而是 label smoothing |
| 重复图片与冲突标签 | 多源档案重复、相同/相近图片可能有不同标签 | 分组检查、冲突组排除、固定分组分区 | 近重复检测不证明全部训练/测试独立 |
| 图片中心裁剪丢失形状 | 瓶子的顶部、底部可能被裁掉 | CLIP 输入改为完整图片 letterbox；训练与推理一致，重新提取特征并训练 | 不是自动目标定位，保留背景也会带来干扰 |
| 单靠公开 benchmark 过于乐观 | 网上物品图与本地随手拍存在场景差异 | 加网上整图、复杂 TACO 裁剪和本地照片分开诊断；未把失败测试隐藏 | 真实场景准确率远不足，不能以公开集成绩替代 |
| 杯子、碗、纸巾标签语义冲突 | 1,066 张水杯和 462 张碗归 Mixed items；纸巾材质与残余垃圾定义混合 | 保留来源标签，定位问题；本轮增加细分类学习实验 | 尚未完成逐张材质重标注，不能声称 taxonomy 已修好 |
| 裁背景不稳定 | 水瓶整图正确，手工裁后变成 Glass | 不把“一律去背景”作为默认补救；记录整图/裁图/子模型诊断 | 手工裁图不能当成已实现、已验证的自动检测 |
| 分类器限制表达能力 | 线性头对数据的复杂边界拟合不足 | 本轮细分类非线性头 + 近邻：validation 93.83%，test 93.50% | 因果解释是工程假设；实拍提升须另测，95%未达到 |
| 不确定分数容易误导 | 第一名显示醒目，用户可能当成确定结论 | 改为优先显示 Unable to identify reliably，并将分数称候选 | 显示方式改变不等于准确率提高；高分也可能错 |
| 图像类别不等于投放资格 | 塑料不一定都能进黄桶，图片无法证明隐藏内容物 | 独立规则层、用户确认、官方来源和条件提示 | 不是保证安全或全面法规判定 |
| 本地重复启动服务 | Notebook / app 可能重复占用同一端口 | 检查已运行的 EcoSort 服务并复用 | 历史其他端口可能仍是旧版本，演示需明确当前地址和 active model |

不要补写没有证据的错误堆栈、崩溃次数、准确率、节能量、用户调查或完整训练时长。

## 6. 可据此撰写的段落要点

### Methodology

描述公开多源图片审核后形成固定分组清单，再用预训练视觉编码器进行迁移学习。明确区分冻结编码器、训练分类头、MobileNet 部分微调、验证集选择以及最终 benchmark 评估。当前部署版本与实验候选分开，不应混写成一个模型。

### Key Findings

写同一测试集上 MobileNet 冻结 73.02%、微调 78.42%、CLIP 组合 85.88%、本轮非线性组合 93.50% 的版本对照，同时展示 macro-F1。强调不同数据规模/划分的旧 12 类结果不是严格同条件比较。最重要的负面结果是实拍差距，说明数据语义与使用场景的重要性，而不是承诺系统已经可靠。

### Wider Adoption / Conclusion

可写本地、单物体辅助识别和有来源的条件式提示具有原型可用性；应先扩充代表性实拍训练/验证、修正标签、加入目标选择或检测并评估错误风险。不能因为准确率目标是 95% 就称可商业化、可替代人工或已经提高市政回收效率。

### 个人贡献与 AI 使用

你说明由你负责代码，因此可以据实写：软件实现、数据流程整合、本地部署与模型实验主要由一位组员负责；其他成员的文献、报告、协调贡献分别由本人确认。不要沿用模板的“每人相同贡献”和“每周两小时会议”，除非确有记录。

本项目有 Codex/ChatGPT 辅助代码生成、调试、分析和文字整理的真实使用，不能声称完全独立、无 AI 编程。最终 AI Use Appendix 应按课程规则列工具、用途、核验方法和人的责任。不要把这份辅助材料本身当作已经由全体组员审核过的最终声明。

## 7. 建议交给组员的证据文件

- data/expanded/dataset_verification.json、class_counts.csv、split_manifest.csv：人数以外的所有样本规模、类别与分区依据。
- trained_expanded/baseline_vs_finetuned.csv：同分区 MobileNet 基线与微调比较。
- trained_semantic/training_completed.json、classification_report.csv：已部署语义组合结果。
- trained_nonlinear/training_completed.json、validation_trials.csv、classification_report.csv：本轮 93.50% 候选的实际结果。
- final_test/evaluation.json、predictions.csv、test_manifest.json：30 张网上整图的结果、来源和样本限制。
- trained_semantic/taco_diagnostic/evaluation.json：复杂场景失败案例。
- local_photo_diagnostic/predictions.json、FINDINGS.md：六张本地照片的问题和用户确认。
- accuracy_ui_verification.json：上传和摄像头功能验证，不是摄像头准确率验证。
- active_model.json：演示实际在用哪个版本；不要拿候选成绩冒充网站当前成绩。

## 8. 参考文献应替换成什么

使用 [TrashNet 原项目](https://github.com/garythung/trashnet)、[RealWaste / UCI](https://archive.ics.uci.edu/dataset/908/realwaste)、其他实际数据源及原发布者、[MobileNetV3 论文](https://arxiv.org/abs/1905.02244)、[OpenCLIP 实现](https://github.com/mlfoundations/open_clip)、[CLIP 论文](https://arxiv.org/abs/2103.00020)，以及 [City of Sydney 官方回收页面](https://www.cityofsydney.nsw.gov.au/waste-recycling-services/recycle-household-items) 和 [NSW EPA CRC 页面](https://www.epa.nsw.gov.au/Your-environment/Recycling-and-reuse/household-recycling-overview/community-recycling-centres)。如最终纳入 DINOv2，再加入其论文和具体权重版本；在实验完成前不能提前当成最终方法。模板的爱因斯坦与 LaTeX 文献不能充当这些内容的依据。

## 9. 尚未完成 / 不应声称完成

同时达到 95% 验证与测试的目标仍未完成（validation 已 95.13%，test 94.85%）；代表性独立本地实拍验证、完整逐图标签清理、自动多物体定位、原 proposal 的批量污染比例及其精确率/召回率、LLM 总结对比、最终报告及全组贡献确认仍需分别补齐。是否继续原 proposal 的附加功能应由团队按目前已缩小的软件范围决定，并在报告解释变更，而不是悄悄把未做内容写成完成。
