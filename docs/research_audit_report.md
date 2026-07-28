# Báo cáo audit nghiên cứu: Pneumonia XAI Pipeline

Ngày audit: 2026-07-27  
Phạm vi: `docs/reference_paper.pdf`, `paper/main.tex`, `paper/main.pdf`, mã nguồn trong `pipeline/`, và artifacts trong `pipeline/outputs/`.

## Kết luận ngắn

Đánh giá theo chuẩn bài Q1 hoặc hội nghị A/A*: **Needs major revision before submission**.

Điểm sẵn sàng công bố hiện tại: **3.0 / 10**.

Lý do chính không phải vì ý tưởng kém. Ý tưởng leakage-aware split, group-disjoint sealed test, mask-guided attention, calibration, và định lượng XAI là hướng tốt. Nhưng bản hiện tại có nhiều lỗi nghiêm trọng ở provenance thực nghiệm, mâu thuẫn số liệu, claim quá mạnh, và một số bug code làm cho các kết luận về XAI validity không đáng tin.

Nếu sửa triệt để, rerun đúng protocol và viết lại paper trung thực hơn, hướng nghiên cứu có thể nâng lên khoảng **6.5-7.0 / 10** cho workshop/venue vừa; muốn tới Q1/A/A* cần thêm external validation, baseline hiện đại, statistical testing sạch, và release artifact tái lập đầy đủ.

## Bằng chứng đã kiểm tra

- Paper hiện tại: `paper/main.tex`, `paper/main.pdf`.
- Bài gốc: `docs/reference_paper.pdf`.
- Pipeline chính: `pipeline/src/*.py`, `pipeline/build_notebook.py`, `pipeline/run_local_xai.py`.
- Kết quả classification: `pipeline/outputs/json/*_test_metrics.json`, `pipeline/outputs/csv/all_results_test.csv`, `pipeline/outputs/csv/run_ledger.csv`.
- Kết quả XAI: `pipeline/outputs/csv/*_xai_lrr.csv`, `pipeline/outputs/csv/XAI_ALL_SAMPLES_COMBINED.csv`, `pipeline/outputs/csv/XAI_AGGREGATED_SUMMARY.csv`.
- Dataset manifests: `pipeline/outputs/gate46_development/manifests/*.csv`, `pipeline/input/SEALED_test_label_key.csv`.
- Protocol locks: `pipeline/outputs/gate46_development/configs/r0_source_faithful_v1.json`, `pipeline/outputs/gate46_development/locks/*.json`.

Nguồn ngoài dùng để đối chiếu chuẩn:

- CLAIM 2024 Update, Radiology: Artificial Intelligence: https://pubs.rsna.org/doi/10.1148/ryai.240300
- CLAIM checklist page, RSNA: https://pubs.rsna.org/page/ai/claim
- Adebayo et al., "Sanity Checks for Saliency Maps", NeurIPS 2018: https://papers.neurips.cc/paper_files/paper/2018/hash/294a8ed24b1ad22ec2e7efea049b8737-Abstract.html
- Tomsett et al., "Sanity Checks for Saliency Metrics", AAAI 2020: https://ojs.aaai.org/index.php/AAAI/article/view/6064
- Recent Kermany leakage-aware example, PMC article noting patient-ID overlap handling: https://pmc.ncbi.nlm.nih.gov/articles/PMC12942337/

## 1. Audit bảng kết quả thực nghiệm

### 1.1 Classification table

Paper claim chính:

- Abstract nói D-P sealed-test mean AUROC = **0.974**.
- Results table nói DenseNet121 P = **0.974 +- 0.003** và ResNet50 P = **0.972 +- 0.002**.
- Discussion/Conclusion lại nói DenseNet121 P = **0.958 +- 0.006** và ResNet50 P = **0.971 +- 0.007**.
- Ablation table ghi D-P tuning/test = **0.9995 / 0.538**.

Artifacts hiện có cho P-only:

| Model | Seeds | AUROC mean | AUROC std | Accuracy mean | Sensitivity mean | Specificity mean | Brier mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| DenseNet121 P | 3407, 42, 2024 | 0.9746 | 0.0031 | 0.8248 | 0.9957 | 0.5399 | 0.1340 |
| ResNet50 P | 3407, 42, 2024 | 0.9716 | 0.0024 | 0.8798 | 0.9906 | 0.6952 | 0.1105 |

Đánh giá:

1. **High severity: mâu thuẫn số liệu trực tiếp trong paper.** Không thể để đồng thời 0.974, 0.958 và 0.538 cho cùng D-P sealed test. Artifacts JSON ủng hộ 0.9746, không ủng hộ 0.958 hoặc 0.538.

2. **High severity: baseline C0/R0 không có artifact đủ để support bảng.** Trong `pipeline/outputs/` không thấy JSON/CSV/checkpoint rõ ràng cho D-C0, R-C0, D-C1-D-C4, R0-D, R0-R. Paper lại claim D-C0 AUROC 0.932, C0 subset AUROC 0.9987, ablation AUROC 0.9986-0.9995. Đây hiện là unsupported.

3. **High severity: claim "D-P outperforms baseline" chưa được chứng minh.** Nếu baseline là D-C0 sealed-test 0.932 thì D-P tốt hơn, nhưng artifact không có. Nếu nhìn ngay table, D-C0 0.9987 còn cao hơn D-P 0.974, nhưng lại footnote là subset/tuning/single seed. Paper đang trộn cohort, split, và seed khác nhau trong một bảng so sánh.

4. **Medium severity: accuracy/specificity trade-off chưa được trình bày đúng.** DenseNet121 P có AUROC cao nhưng specificity trung bình chỉ khoảng 0.54, với rất nhiều false positives. Nếu paper hướng medical imaging, cần báo cáo operating point rõ hơn: threshold, sensitivity-specificity target, PPV/NPV, balanced accuracy, confidence intervals, và clinical trade-off.

5. **Medium severity: `all_results_test.csv` bị append trùng.** File có 33 dòng, nhưng chỉ 6 run thật theo ledger. DenseNet121 seed 3407 xuất hiện 10 lần; ResNet50 seed 3407 xuất hiện 7 lần. Nguyên nhân nằm ở `pipeline/src/12_export.py`, nơi export append vào CSV nếu file đã tồn tại. JSON per-run đáng tin hơn CSV tổng hợp.

Kết luận cho bảng classification: **chưa đạt chuẩn publishable**. Có 6 P-run sealed-test đáng dùng, nhưng bảng cần viết lại từ artifact sạch và phải bổ sung/rerun C0/R0/ablation.

### 1.2 Ablation table

Paper claim:

- D-C0, D-C1, D-C2, D-C3, D-C4 đều có AUROC tuning khoảng 0.9984-0.9995.
- D-P tuning/test ghi 0.9995 / 0.538.
- Text lại nói D-P sealed test = 0.958 +- 0.006.

Code/artifact:

- `pipeline/src/02_config.py` chỉ có một global config mặc định: CLAHE=True, focal loss, CBAM=True, mask loss=True, TTA=True.
- Không thấy config files hoặc scripts riêng cho C0-C4 trong outputs.
- Không thấy metrics JSON cho D-C0-D-C4.

Đánh giá:

1. **High severity: ablation không tái lập được từ repo hiện tại.** Một bài Q1/A* cần mỗi arm có frozen config, checkpoint, seed, split, metric file, và script invocation.

2. **High severity: ablation design không khớp code hiện tại.** Paper mô tả C0 raw/BCE/no mask/no CBAM, C1 CLAHE, C2 sampler, C3 focal, C4 fine-tune. Nhưng config mặc định luôn bật nhiều thành phần cùng lúc. Nếu trước đó bạn từng chạy C0-C4 bằng sửa tay config, artifact đó chưa được lưu đủ.

3. **High severity: số 0.538 rất đáng nghi.** Nó không khớp AUROC P-run hiện có, không khớp accuracy mean, và cũng không được giải thích. Đây là lỗi cần sửa ngay trước mọi submission.

Kết luận cho ablation: **không thể chấp nhận ở dạng hiện tại**.

### 1.3 XAI table

Artifacts hiện có:

| Model | Rows | Seeds | Grad-CAM LRR+ mean | Guided Grad-CAM LRR+ mean | Chance mean |
|---|---:|---|---:|---:|---:|
| DenseNet121 P | 384 | 42, 2024, 3407 | 0.7743 | 0.8022 | 0.3424 |
| ResNet50 P | 384 | 42, 2024, 3407 | 0.7789 | 0.8004 | 0.3424 |

Đánh giá:

1. **High severity: paper mâu thuẫn XAI sample size.** Abstract nói 180-case tuning sample; Results nói 100 cases; table note nói 128 valid sealed test cases; Discussion nói 178-case tuning sample. Artifact thực tế là 128 ảnh mỗi seed mỗi architecture, 384 rows per architecture.

2. **High severity: paper mâu thuẫn kết quả LRR.** Results nói Guided Grad-CAM LRR+ khoảng 0.802, Grad-CAM 0.774. Discussion/Conclusion lại nói median 0.245 và 0.107, rồi kết luận "poor lung-field concentration". Hai narrative này không thể cùng đúng.

3. **High severity: validity metrics không đủ cho 3 seed.** Chỉ `*_seed3407_xai_lrr.csv` có deletion/insertion/stability/randomization columns. Seed 42 và 2024 chỉ có LRR cơ bản. Vì vậy claim về deletion AUC, insertion AUC, parameter randomization, input stability không được support trên pooled 3-seed analysis.

4. **High severity: parameter randomization implementation có bug nghiêm trọng trong kết quả cũ.** Trước cleanup, `pipeline/src/10_xai.py` tìm `model.fc`, `model.classifier`, hoặc `model.head`, trong khi `ClassificationModel` chứa head ở `model.backbone.fc` hoặc `model.backbone.classifier`. Do đó `head_layers` nhiều khả năng rỗng, không layer nào bị randomize. Điều này giải thích vì sao `gcam_rand` và `guided_rand` gần như 1.0. Code active đã được sửa ngày 2026-07-27, nhưng mọi kết quả randomization cũ vẫn phải rerun.

5. **High severity: deletion/insertion trong kết quả cũ có thể sai target class.** Trước cleanup, `compute_deletion_insertion_auc()` luôn lấy sigmoid của output logit dương, trong khi XAI đang giải thích predicted class, có thể là normal bằng cách negate logit. Với normal cases, deletion/insertion cũ không đo xác suất class được giải thích. Code active đã được sửa ngày 2026-07-27, nhưng các metric cũ vẫn phải rerun.

6. **Medium severity: method text nói "pneumonia logit", code dùng predicted class.** Paper dòng method mô tả scalar target là pneumonia logit, nhưng code ở `generate_explanations()` chọn predicted label và dùng negative logit cho class 0. Cần chọn một định nghĩa và viết nhất quán.

7. **Medium severity: paper gọi LRP nhưng code thực tế là Guided Grad-CAM.** Trước cleanup, `explainability_methods` ghi `["Grad-CAM", "LRP"]`, comment/docstring nói LRP, nhưng implementation dùng GuidedBackpropReLUModel * Grad-CAM. Code active đã được đổi sang `["Grad-CAM", "Guided Grad-CAM"]` và bỏ `zennit`, nhưng paper vẫn cần loại bỏ toàn bộ claim LRP nếu Guided Grad-CAM là phương pháp cuối.

Kết luận cho XAI: LRR cơ bản có thể dùng sau khi làm sạch sample definition, nhưng các claim về faithfulness/sanity/stability hiện **không đáng tin**.

### 1.4 Calibration and operating points

Paper claim:

- Temperature scaling T = 1.50-1.51.
- Brier 0.0072-0.0098 và ECE 0.008-0.012 trên tuning.
- Caption nói 95% CI và cluster bootstrap.

Artifacts:

- Test JSON có Brier khoảng 0.103-0.148, không phải 0.007-0.009.
- Không thấy saved temperature file, ECE table, bootstrap CI file, calibration raw outputs, hoặc confidence interval artifacts.
- Calibration figures tồn tại trong `paper/figures`, nhưng không thấy script/data regenerate trong repo.

Đánh giá:

1. **High severity: calibration claims thiếu provenance.**
2. **Medium severity: test Brier khá cao so với narrative "robust confidence".**
3. **Medium severity: caption hứa 95% cluster-bootstrap CI nhưng bảng không có CI.**

Kết luận: cần rerun/export calibration đầy đủ hoặc hạ claim.

## 2. Reproduction của bài gốc

Bài gốc `docs/reference_paper.pdf` claim cho pneumonia:

- ResNet50: accuracy 84.4%, AUC 0.95, F1/AP có mâu thuẫn nhẹ giữa table và text.
- DenseNet121: accuracy 89.1%, AUC 0.98.
- Split mô tả là 70/15/15.
- Grad-CAM chỉ qualitative.

Repo hiện tại:

- Có `r0_source_faithful_v1.json` và source reference scripts trong `pipeline/outputs/gate46_development/reference/`.
- Nhưng config R0 ghi test access disabled.
- Paper vừa nói "completed R0 execution", vừa nói limitation là "R0 TensorFlow source reconstruction was omitted".
- Không thấy R0 metrics JSON/CSV/checkpoints/logs để chứng minh reproduce +-2%.

Đánh giá:

1. **High severity: mục tiêu reproduce +-2% bài gốc chưa được chứng minh.**
2. **High severity: paper không nên claim "completed R0" nếu không có R0 test metrics và logs.**
3. **Medium severity: nếu dùng leakage-aware group-disjoint split mới, kết quả sẽ không còn trực tiếp so với bài gốc 70/15/15.** Cần tách rõ "source reproduction" và "clean protocol extension".

Khuyến nghị: chạy lại R0 đúng nhất có thể theo source paper/code, trên split source-defined, report metric gốc riêng; sau đó mới so sánh clean leakage-aware protocol như một evaluation mới, không gọi là reproduce trực tiếp.

## 3. Audit dataset và split

Điểm mạnh:

- Clean development manifest có 5190 rows: train 4411, tuning 779.
- Sealed test có 624 rows: pneumonia 390, normal 234.
- Development và sealed test không overlap `operational_group_id`.
- Không overlap `content_sha256`.
- Không overlap `source_relative_path`.

Điểm cần sửa/clarify:

1. **Medium severity: sealed test có 428 operational groups cho 624 images.** Có nhiều ảnh cùng group trong test. Không phải leakage, nhưng bootstrap/CI phải cluster theo group, không image-level.

2. **Medium severity: audit script hiện chỉ kiểm tra sự tồn tại test labels và count.** Nó không tự động report overlap dev-test, duplicate clusters, or mask availability. Những check này nên được script hóa và export.

3. **Medium severity: paper nói "0 near-duplicate" nhưng repo chưa có near-duplicate detection evidence.** Có content hash overlap = 0, nhưng near-duplicate cần perceptual hash/embedding or documented audit.

## 4. Audit method

Phương pháp P hiện tại theo code:

- DenseNet121/ResNet50 pretrained ImageNet.
- CBAM attention trước global pooling.
- Mask loss BCE giữa spatial attention và lung mask.
- CLAHE enabled.
- Focal loss enabled.
- Weighted sampler enabled trong `build_notebook.py`.
- Two-stage training, 5 head epochs + 25 fine-tuning epochs.
- Temperature scaling + Youden threshold from tuning.
- TTA horizontal flip on test.

Vấn đề:

1. **High severity: paper method chưa tách rõ contribution.** P là một bundle nhiều kỹ thuật. Nếu ablation không đầy đủ, không thể nói kỹ thuật nào tạo hiệu quả.

2. **High severity: claim "mitigating shortcut learning" chưa được chứng minh.** LRR cao trong lung mask không đủ chứng minh giảm shortcut. Cần C0-vs-P comparison trên background sensitivity, mask occlusion, site/source subgroup, external data, or failure analysis.

3. **Medium severity: mask loss dùng "ground-truth binary lung label" là cách diễn đạt sai.** Lung mask không phải label bệnh, nên nên gọi là automatically generated lung segmentation mask/reference anatomical mask.

4. **Medium severity: TTA horizontal flip trong CXR cần justification.** Horizontal flip có thể đổi laterality và anatomy. Dù pneumonia binary có thể vẫn chấp nhận như sensitivity, Q1 reviewer sẽ hỏi.

5. **Medium severity: model selection dùng validation loss, threshold dùng Youden on tuning, calibration dùng tuning.** Hợp lý nếu tuning cố định, nhưng phải export threshold/temperature per run và tránh mọi chỉnh sửa sau khi nhìn test.

## 5. Audit code/reproducibility

Vấn đề nghiêm trọng:

1. **Invalid import/package structure.** Files như `src/11_test_evaluation.py` có `from src.08_validation import calculate_metrics`, không hợp lệ trong Python package chuẩn vì module bắt đầu bằng số. Local runner bypass bằng cách stitch code và remove imports. Với reviewer, repo hiện không phải package reproducible sạch.

2. **Notebook chưa phải execution artifact.** `pipeline/Pneumonia_XAI_Pipeline.ipynb` có 27 cells nhưng execution_count = 0 và outputs = 0.

3. **CSV append không idempotent.** `export_metrics()` append vào `all_results_test.csv`, gây trùng rows.

4. **Training skip mặc định.** `skip_training_if_checkpoint_exists=True` khiến rerun mặc định load checkpoints, không retrain. Điều này tiện cho audit nhưng phải ghi rõ. Muốn reproduce training cần flag riêng.

5. **No complete environment lock for PyTorch P pipeline.** R0 có requirements reference, nhưng P pipeline thiếu frozen requirements/environment, package versions, CUDA/cuDNN, commit hash, and command log.

6. **`fix_fn.py` rất nguy hiểm cho paper.** Script hardcode một case và in "Saved fake FN". Nếu figure này đi vào paper hoặc qualitative figure, cần loại bỏ hoặc đổi tên/ghi rõ là manually generated diagnostic visualization. Từ "fake" trong repo là red flag lớn cho reviewer hoặc audit nội bộ.

## 6. Audit từng phần paper

### Abstract

Trạng thái: **Needs rewrite**.

Lỗi chính:

- Claim D-P AUROC 0.974 nhưng Discussion/Conclusion nói 0.958.
- Claim "effectively mitigating severe domain-shift shortcut learning" quá mạnh.
- Claim XAI "exceptional concentration" mâu thuẫn với Discussion nói poor concentration.
- "Within frozen D-P checkpoint ... 180-case tuning sample" không khớp artifacts.

Sửa đề xuất:

- Chỉ nêu số được artifact support: P sealed-test AUROC 0.9746 DenseNet121 và 0.9716 ResNet50, nếu chấp nhận P-only.
- Hạ shortcut claim thành "designed to test/reduce reliance on non-lung regions".
- Nêu rõ XAI is organ-concentration, not lesion localization.

### Introduction

Trạng thái: **Directionally good, but novelty framing needs tightening**.

Điểm tốt:

- Nêu đúng rủi ro leakage, shortcut, Grad-CAM overinterpretation.
- Tách clinical utility khỏi dataset-label classification.

Vấn đề:

- Contribution quá rộng so với evidence: reproduction, ablation, calibration, XAI validity, mask loss, external-readiness.
- Cần nói rõ đây là internal methodological study, không clinical diagnostic validation.

### Related Work

Trạng thái: **Needs stronger modern positioning**.

Cần bổ sung:

- CLAIM 2024 / TRIPOD+AI style reporting expectations.
- Saliency sanity literature beyond Adebayo: saliency metric reliability, faithfulness caveats.
- Recent Kermany dataset leakage/patient overlap papers.
- Stronger modern CXR baselines or foundation-model baselines nếu muốn A/A*.

### Methods

Trạng thái: **Potentially strong, but not reproducible enough**.

Cần sửa:

- Đưa exact configs per arm C0-C4/P/R0 vào supplement.
- Tách source reproduction protocol khỏi clean protocol.
- Viết đúng target class trong XAI.
- Viết đúng layer target thực tế: CBAM output node vs backbone final conv.
- Đưa mask generation/QC script and QC metrics.

### Experiments

Trạng thái: **Major revision**.

Cần có:

- One command per experiment arm.
- Frozen split manifest hashes.
- Per-seed checkpoint hashes.
- Per-seed threshold/temperature.
- Full prediction CSV per run.
- Bootstrap CI script.
- Separate tables for tuning vs sealed test.
- External validation plan or explicit limitation.

### Results

Trạng thái: **Not publishable in current form**.

Cần sửa ngay:

- Remove all unsupported C0/R0/ablation numbers or rerun and export.
- Resolve 0.974 vs 0.958 vs 0.538.
- Resolve LRR 0.802 vs 0.245.
- Remove "95% CI" caption until CI exists.
- Do not compare subset/tuning C0 against full sealed P in same row.

### Discussion

Trạng thái: **Needs rewrite after results cleanup**.

Điểm tốt:

- Có caveat external/prospective/lesion-localization.

Vấn đề:

- Principal findings đang dùng số cũ/khác với Results.
- "will be presented separately", "pending", "anticipated limitations" còn sót như draft protocol, không phải final paper.
- "LRP's marginal superiority" sai thuật ngữ nếu không có LRP.

### Conclusion

Trạng thái: **Needs rewrite**.

Không nên kết luận clean leakage-aware performance và XAI findings khi tables chưa sạch. Kết luận hiện nên hạ xuống: internal P-runs show high AUROC but operating specificity and reproducibility/provenance gaps remain.

## 7. Các claim quá mạnh hoặc sai

1. "D-P outperformed baseline" - chưa có artifact baseline sealed-test đầy đủ.
2. "Mitigating shortcut learning" - chưa chứng minh causal/mechanistic reduction.
3. "Exceptional concentration" - có thể đúng theo LRR artifact, nhưng mâu thuẫn với phần khác và không phải lesion localization.
4. "Parameter randomization sanity checks" - implementation bug làm kết quả hiện không hợp lệ.
5. "Reproduced reference paper" - chưa chứng minh +-2%.
6. "95% cluster-bootstrap confidence intervals" - chưa thấy CI artifacts.
7. "Full ablation" - chưa có configs/artifacts đầy đủ.

## 8. Những gì có thể giữ

1. Group-disjoint sealed test design: có evidence tốt, overlap dev-test = 0 theo group/hash/path.
2. P-run classification JSON cho 6 model-seed combinations: có thể dùng sau khi làm sạch provenance.
3. LRR basic metric for 3 seeds x 2 architectures x 128 samples: có artifact.
4. Ý tưởng mask-guided CBAM: hợp lý để viết thành methodological extension.
5. Cách giới hạn clinical claim: paper đã có tinh thần đúng, cần nhất quán hơn.

## 9. Việc cần làm trước khi sửa paper

Ưu tiên bắt buộc:

1. Tạo `experiments/` hoặc `configs/` với YAML/JSON cho từng arm: R0-D, R0-R, D-C0..D-C4, D-P, R-C0, R-P.
2. Rerun hoặc locate artifacts cho R0 reproduction. Nếu không có, remove claim reproduce +-2%.
3. Rerun hoặc locate artifacts cho C0/C1-C4/R-C0. Nếu không có, remove ablation and baseline claims.
4. Fix `compute_parameter_randomization()` để randomize `model.backbone.classifier` hoặc `model.backbone.fc`.
5. Fix deletion/insertion to use the same explained class score.
6. Export prediction CSV per run, threshold, temperature, checkpoint hash, dataset manifest hash.
7. Make `all_results_test.csv` idempotent or regenerate from JSON only.
8. Regenerate XAI combined CSV with common columns for all seeds or separate LRR-only from validity-only.
9. Decide one sample definition: tuning or sealed test; 100, 128, 178, or 180.
10. Rewrite all tables from generated artifacts, not manually typed numbers.

Ưu tiên để nâng chuẩn Q1/A/A*:

1. External validation on another pediatric or mixed CXR dataset.
2. Compare against stronger baselines: EfficientNet, ConvNeXt, ViT/Swin, DenseNet121 without mask loss but same training budget, and possibly self-supervised/foundation-model embeddings.
3. Add confidence intervals and paired statistical tests for P-vs-C0 on the same test cases.
4. Add decision-curve or clinically meaningful operating thresholds.
5. Add robust XAI validation: model randomization, data randomization, insertion/deletion with consistent baselines, mask perturbation, counterfactual/background occlusion.
6. Include CLAIM 2024 checklist as supplement.

## 10. Suggested revised score after fixes

Current:

- Novelty: 4/10
- Methodological rigor: 3/10
- Experimental validity: 2/10
- Reproducibility: 2/10
- Writing/positioning: 4/10
- Overall: 3/10

After required cleanup and reruns, without external validation:

- Likely overall: 5.5-6.5/10.

After cleanup + external validation + strong baselines + full XAI validity:

- Possible overall: 7.0-8.0/10.

## 11. Final recommendation

Không nên submit bản `main.pdf` hiện tại. Bản này có quá nhiều mâu thuẫn số liệu và unsupported claims để qua review nghiêm túc.

Hướng tốt nhất là biến paper thành một nghiên cứu methodological audit trung thực:

- Stage 1: reproduce paper gốc, nói rõ có/không đạt +-2%.
- Stage 2: leakage-aware clean split làm giảm hoặc thay đổi performance như thế nào.
- Stage 3: P method có cải thiện gì so với C0 dưới cùng split, cùng seed, cùng budget.
- Stage 4: XAI metric chỉ được claim sau khi sanity checks chạy đúng.

Nếu phải ưu tiên trước mắt, hãy sửa bảng kết quả trước. Bảng hiện tại là phần reviewer sẽ bắt lỗi nhanh nhất.

## 12. Addendum: Kaggle Gate-4 R0 kernels pulled on 2026-07-27

Kernels checked:

- `hintrngia/gate-4-r0-densenet121-development-only`
- `hintrngia/gate-4-r0-resnet50-development-only`

Local source files pulled:

- `gate-4-r0-densenet121-development-only.py`
- `gate-4-r0-resnet50-development-only.py`

Kaggle status:

- Both kernels report `KernelWorkerStatus.COMPLETE`.

Important finding:

- These are explicitly **Gate-4 development-only** R0 runs.
- The source code verifies that `GATE46_DEVELOPMENT.g46blob` contains no test images/manifests.
- The exported result schema sets:
  - `status = COMPLETE_DEVELOPMENT_ONLY`
  - `test_images_opened = 0`
  - `test_inference_performed = false`
- Therefore these runs **cannot establish reproduction of the reference paper's test performance**.

Extracted run results from Kaggle logs:

| R0 kernel | Train samples | Validation samples | Epochs run | Best val accuracy | Best val accuracy epoch | Best val loss | Best val loss epoch | Runtime |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| DenseNet121 | 4278 | 938 | 33 | 0.9828 | 28 | 0.0520 | 23 | 2193.97 s |
| ResNet50 | 4278 | 938 | 37 | 0.9537 | 26 | 0.1281 | 27 | 2469.27 s |

Runtime details:

- GPU: Tesla P100-PCIE-16GB.
- Python: 3.12.13.
- TensorFlow: 2.20.0.
- Keras: 3.13.2.
- Both runs used seed 42.
- DenseNet121 froze 320 / 427 base layers.
- ResNet50 froze 140 / 175 base layers.

Protocol deviations recorded by the kernels:

- TensorFlow 2.20/Keras 3 replaces source-pinned TensorFlow/Keras 2.11 because Kaggle Python 3.12 cannot host the original pin.
- TensorFlow seed 42 was added; the source script reportedly sets NumPy seed only.
- Validation remains augmented through `ImageDataGenerator(validation_split=0.18)`, matching the source behavior but weakening the interpretability of validation metrics.
- `steps_per_epoch` and `validation_steps` use floor division, so some samples are skipped per epoch/evaluation.

Suitability for paper:

1. **Can be reported only as R0 development-validation reconstruction evidence.**
   These results support the statement that the source-like R0 training pipeline ran successfully on the development-only bundle.

2. **Cannot be used as reference-paper reproduction within +-2%.**
   The reference paper reports pneumonia test metrics: DenseNet121 accuracy 89.1%, AUC 0.98; ResNet50 accuracy 84.4%, AUC 0.95. The Gate-4 kernels report development-validation accuracy/loss, not test accuracy/AUC/F1/AP on the reference split.

3. **Should not appear in the main classification table beside sealed-test P/C0 results.**
   Mixing development-validation R0 values with sealed-test P values would be methodologically invalid.

4. **If included, use a separate R0 development-only table.**
   Label clearly: "development-validation only; no test inference; not a confirmatory reproduction of published test performance."

Required next step:

- Run a separate **Gate-7 R0 test evaluation** using the frozen R0 checkpoints and a locked test protocol, or rerun the original source-defined train/validation/test protocol if the goal is to verify the reference paper within +-2%.
- Required outputs for that next step:
  - test prediction CSV for DenseNet121 and ResNet50;
  - accuracy, AUROC, F1, average precision;
  - confusion matrix;
  - checkpoint hash used for inference;
  - exact test split/cohort definition;
  - explicit statement whether source-defined test or clean sealed test was used.

## 13. Addendum: Kaggle Gate-5 G5R1 kernels pulled on 2026-07-27

Kernels checked:

- `hintrngia/gate5-g5r1-densenet-selection-v2`
- `hintrngia/gate5-g5r1-densenet-c0-repeats-v2`
- `hintrngia/gate5-g5r1-resnet-c0-v2`
- `hintrngia/gate5-g5r1-densenet-p-v1`
- `hintrngia/gate5-g5r1-resnet-p-v1`

Local source files pulled:

- `gate5-g5r1-densenet-selection-v2.py`
- `gate5-g5r1-densenet-c0-repeats-v2.py`
- `gate5-g5r1-resnet-c0-v2.py`
- `gate5-g5r1-densenet-p-v1.py`
- `gate5-g5r1-resnet-p-v1.py`

Kaggle status:

- All five kernels report `KernelWorkerStatus.COMPLETE`.

Boundary and protocol finding:

- All five source files begin with the same locked scope: `Gate-5 clean classification runner; development/tuning only, never test`.
- The code extracts only `GATE46_DEVELOPMENT.g46blob`, verifies the clean development manifest hash, and enforces exactly 5190 rows with splits `{train, tuning}`.
- Split counts enforced by code:
  - train NORMAL: 1139
  - train PNEUMONIA: 3272
  - tuning NORMAL: 201
  - tuning PNEUMONIA: 578
- Every `RUN_RESULT.json` and `KERNEL_SUMMARY.json` sets:
  - `status = COMPLETE_DEVELOPMENT_TUNING_ONLY`
  - `test_images_opened = 0`
  - `test_inference_performed = false`

Therefore Gate-5 results are valid only as **development/tuning model-selection or ablation evidence**. They are not sealed-test/internal-test performance.

Extracted Gate-5 tuning results from Kaggle logs:

| Run | Selected epoch | Epochs run | AUROC | Log loss | ECE-15 | Brier | Youden accuracy | Sensitivity | Specificity | Confusion matrix TN/FP/FN/TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| D-C0 seed 42 | 12 | 20 | 0.998631 | 0.056511 | 0.013574 | 0.013106 | 0.985879 | 0.986159 | 0.985075 | 198/3/8/570 |
| D-C1 seed 42 | 10 | 18 | 0.998993 | 0.052358 | 0.015761 | 0.014547 | 0.988447 | 0.991349 | 0.980100 | 197/4/5/573 |
| D-C2 seed 42 | 10 | 18 | 0.998752 | 0.054623 | 0.017068 | 0.014570 | 0.987163 | 0.987889 | 0.985075 | 198/3/7/571 |
| D-C3 seed 42 | 12 | 20 | 0.998433 | 0.054661 | 0.022079 | 0.013699 | 0.988447 | 0.989619 | 0.985075 | 198/3/6/572 |
| D-C4 seed 42 | 16 | 24 | 0.999475 | 0.043978 | 0.008752 | 0.008486 | 0.989730 | 0.987889 | 0.995025 | 200/1/7/571 |
| D-C0 seed 2025 | 12 | 20 | 0.998700 | 0.052810 | 0.011767 | 0.011023 | 0.988447 | 0.989619 | 0.985075 | 198/3/6/572 |
| D-C0 seed 3407 | 12 | 20 | 0.998520 | 0.068502 | 0.018398 | 0.016131 | 0.979461 | 0.974048 | 0.995025 | 200/1/15/563 |
| R-C0 seed 42 | 13 | 21 | 0.998821 | 0.056592 | 0.012037 | 0.012032 | 0.988447 | 0.987889 | 0.990050 | 199/2/7/571 |
| D-P seed 42 | 21 | 29 | 0.999509 | 0.043192 | 0.012198 | 0.010617 | 0.985879 | 0.980969 | 1.000000 | 201/0/11/567 |
| D-P seed 2025 | 17 | 25 | 0.999484 | 0.054653 | 0.012623 | 0.011106 | 0.984596 | 0.979239 | 1.000000 | 201/0/12/566 |
| D-P seed 3407 | 25 | 33 | 0.999544 | 0.026804 | 0.009193 | 0.007642 | 0.992298 | 0.989619 | 1.000000 | 201/0/6/572 |
| R-P seed 42 | 19 | 27 | 0.999604 | 0.034195 | 0.010214 | 0.008769 | 0.993582 | 0.994810 | 0.990050 | 199/2/3/575 |

Aggregate tuning summaries:

| Group | n | Mean AUROC | SD AUROC | Mean Youden accuracy | Mean sensitivity | Mean specificity |
|---|---:|---:|---:|---:|---:|---:|
| DenseNet121 C0 | 3 | 0.998617 | 0.000091 | 0.984596 | 0.983276 | 0.988391 |
| DenseNet121 P | 3 | 0.999512 | 0.000030 | 0.987591 | 0.983276 | 1.000000 |
| ResNet50 C0 | 1 | 0.998821 | 0.000000 | 0.988447 | 0.987889 | 0.990050 |
| ResNet50 P | 1 | 0.999604 | 0.000000 | 0.993582 | 0.994810 | 0.990050 |

Configuration interpretation:

- `D-C0`: raw input, no imbalance sampler, BCE loss, base two-stage fine-tuning.
- `D-C1`: CLAHE only versus C0.
- `D-C2`: inverse-frequency weighted sampler only versus C0.
- `D-C3`: focal loss only versus C0.
- `D-C4`: wider DenseNet fine-tuning only versus C0.
- `D-P`: locked selected recipe = CLAHE + inverse-frequency weighted sampler + fine-tune two backbone stages.
- `R-P`: same selected recipe applied to ResNet50.

Important implementation note:

- Gate-5 P kernels do **not** implement the CBAM + Mask Loss architecture described in the paper text. In the pulled code, P means the selected clean recipe:
  - `clahe_clip2_tile8`
  - `inverse_frequency_weighted_sampler`
  - `fine_tune_two_backbone_stages`
- There is no CBAM module and no BCE mask loss against lung masks in these Gate-5 source files.

Suitability for the current paper:

1. **Usable for Table `DenseNet121-only controlled ablation` only if labeled as tuning/development.**
   The paper's values for D-C0 through D-C4 match the Gate-5 tuning AUROCs after rounding:
   D-C0 0.9986, D-C1 0.9990, D-C2 0.9988, D-C3 0.9984, D-C4 0.9995.

2. **D-P tuning value should be reported as 0.9995 only as tuning, not test.**
   Gate-5 D-P seed 42 AUROC is 0.999509, and the 3-seed mean is 0.999512 +/- 0.000030 on tuning.

3. **R-C0 and R-P Gate-5 values are tuning-only.**
   R-C0 seed 42 AUROC is 0.998821; R-P seed 42 AUROC is 0.999604.

4. **Do not use Gate-5 results in the main sealed-test classification table.**
   The current paper table labels C0 rows with daggers as tuning, but places them inside "Clean internal-test classification performance." This is confusing and would likely be criticized. Split this into:
   - a development/tuning ablation table; and
   - a separate sealed-test classification table using only Gate-7/test-evaluation outputs.

5. **The paper currently contains a severe D-P sealed-test contradiction.**
   It states D-P sealed-test AUROC as 0.974 +/- 0.003 in the Abstract/Results, 0.958 +/- 0.006 in Discussion/Conclusion, and `0.538` in the ablation table footnote row. Gate-5 cannot resolve this because it never opens test images.

6. **The paper's method claim does not match these Gate-5 P kernels.**
   If the paper claims CBAM + Mask Loss as the proposed method, the Gate-5 P results cannot support that claim. Either:
   - replace the method description with the actual selected clean recipe used by Gate-5/Gate-7; or
   - provide separate kernels/artifacts that actually train/evaluate CBAM + Mask Loss.

7. **Specificity of 1.000 for all D-P tuning seeds is plausible but suspicious enough to discuss.**
   D-P tuning produced zero false positives on 201 tuning NORMAL images for all three seeds. This may be real, but under Q1/A/A* standards it requires sealed-test confirmation, confidence intervals, and subgroup/error analysis. It cannot be marketed as robust clinical specificity from tuning alone.

Required next step:

- Pull/check the corresponding Gate-7 or sealed-test evaluation kernels for D-C0, D-P, R-C0, and R-P.
- Required artifacts:
  - sealed-test prediction CSV per model/seed;
  - checkpoint hash linking Gate-5 training to Gate-7 inference;
  - AUROC, AUPRC, Brier, log loss, accuracy, sensitivity, specificity, F1, confusion matrix;
  - confidence intervals and paired P-vs-C0 comparison on the same 624 test cases;
  - clear threshold provenance: fixed 0.5 versus Youden threshold selected on tuning and then applied unchanged to test.

Current conclusion for Gate-5:

- Gate-5 numbers are internally consistent and suitable for a **tuning-only ablation/model-selection section**.
- Gate-5 numbers are **not suitable** as evidence for sealed-test classification performance.
- Gate-5 P code, as pulled, does **not support** the paper's stated CBAM + Mask Loss proposed-method claim.

## 14. Addendum: Kaggle Gate-6 XAI notebooks pulled on 2026-07-27

Kernels/notebooks checked:

- `hintrngia/gate6-g6r1-xai-smoke-v1`
- `hintrngia/gate6-g6r1e-full-battery-v1`
- `hintrngia/g6-r1e-ptb-mask-sensitivity`
- `hintrngia/notebook94ebf5c236`

Local source files pulled:

- `gate6-g6r1-xai-smoke-v1.py`
- `gate6-g6r1e-full-battery-v1.py`
- `g6-r1e-ptb-mask-sensitivity.py`
- `notebook94ebf5c236.ipynb`

Kaggle status:

| Kernel | Status | Paper suitability |
|---|---|---|
| `gate6-g6r1-xai-smoke-v1` | `KernelWorkerStatus.COMPLETE` | Smoke/sensitivity sanity only; not main result |
| `gate6-g6r1e-full-battery-v1` | `KernelWorkerStatus.COMPLETE` | Main Gate-6 tuning-only XAI battery |
| `g6-r1e-ptb-mask-sensitivity` | `KernelWorkerStatus.COMPLETE` | Mask sensitivity artifact only; tuning-only |
| `notebook94ebf5c236` | `KernelWorkerStatus.CANCEL_ACKNOWLEDGED` | Not usable as completed evidence |

Boundary and protocol finding:

- The first three Gate-6 source files explicitly lock scope to tuning-only, never test.
- `gate6-g6r1e-full-battery-v1.py` verifies exactly 180 `pilot_phase = frozen_validation` rows, all from `final_split = tuning`.
- The full-battery summary reports:
  - `status = PRIMARY_FULL_BATTERY_COMPLETE`
  - `cohort.rows = 180`
  - `cohort.split = tuning`
  - `test_images_opened = 0`
  - `test_inference_performed = false`
- Therefore these XAI results cannot be described as sealed-test XAI results.

### 14.1 Gate-6 smoke result

`gate6-g6r1-xai-smoke-v1` is a 6-case smoke/sensitivity run:

- status: `SENSITIVITY_COMPLETE`
- rows: 6
- records: 24
- strata: normal 2, bacteria 2, virus 2
- all maps finite: true
- test images opened: 0
- test inference performed: false

Important smoke finding:

- DenseNet121 relevance conservation errors are small:
  - epsilon-gamma-box median relative error: 0.001729
  - epsilon-plus-flat median relative error: 0.001744
- ResNet50 relevance conservation is poor:
  - epsilon-gamma-box median relative error: 1.040582
  - epsilon-plus-flat median relative error: 0.863796
- This supports the decision not to run/report ResNet50 LRP as a confirmatory method.

### 14.2 Gate-6 full battery result

Downloaded artifacts:

- `LOCALIZATION.csv`
- `FAITHFULNESS_AUC.csv`
- `FAITHFULNESS_CURVES.csv`
- `ROBUSTNESS.csv`
- `RANDOMIZATION.csv`
- `MASK_QC.csv`
- `GATE6_FULL_BATTERY.json` equivalent summary from log

Summary from Kaggle log:

- status: `PRIMARY_FULL_BATTERY_COMPLETE`
- rows: 180 tuning cases
- strata: normal 46, bacteria 88, virus 46
- Danilov mask hard failures: 0
- all original maps finite: true
- all randomized maps finite: true
- elapsed runtime: 807.97 s on Tesla P100
- record counts:
  - localization: 5040
  - faithfulness AUC: 1260
  - faithfulness curves: 166320
  - robustness: 5040
  - randomization: 1440
  - mask QC: 180

Original-mask localization results recalculated from `LOCALIZATION.csv`:

| Backbone | Method | n | Mean LRR+ | Median LRR+ | Median LRR abs | Median mask area | Median enrichment | Median relative conservation error |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| DenseNet121 | Grad-CAM | 180 | 0.232622 | 0.257274 | 0.257274 | 0.218312 | 1.239264 | 0 |
| DenseNet121 | epsilon-gamma-box LRP | 180 | 0.276575 | 0.245070 | 0.239215 | 0.218312 | 1.090806 | 0.001379 |
| DenseNet121 | epsilon-plus-flat LRP | 180 | 0.261294 | 0.243434 | 0.260000 | 0.218312 | 1.125987 | 0.001416 |
| ResNet50 | Grad-CAM | 180 | 0.081678 | 0.071261 | 0.071261 | 0.218312 | 0.330445 | 0 |

Primary paired DenseNet121 endpoint from log and recomputed artifacts:

- Pair: DenseNet121 epsilon-gamma-box LRP minus DenseNet121 Grad-CAM LRR+
- Common defined cases: 178
- Mean difference: +0.044016
- Median difference: -0.000223
- Bootstrap 95% CI for mean: [0.017796, 0.071113]
- Bootstrap unit: image; exploratory Gate-6 tuning analysis

Interpretation:

- The mean paired difference favors DenseNet121 LRP over Grad-CAM.
- The median paired difference is approximately zero and slightly negative.
- Absolute lung concentration is weak-to-moderate, not "exceptional": median DenseNet121 LRP LRR+ is about 0.245 and median DenseNet121 Grad-CAM LRR+ is about 0.257, with median lung mask area about 0.218.
- ResNet50 Grad-CAM is very poor by LRR+: median 0.071, below mask-area chance.

Faithfulness results recalculated from `FAITHFULNESS_AUC.csv`:

| Backbone | Method | Baseline | n | Median deletion AUC | Median insertion AUC | Median deletion advantage vs random | Median insertion advantage vs random |
|---|---|---|---:|---:|---:|---:|---:|
| DenseNet121 | Grad-CAM | zero | 180 | 0.994815 | 0.999889 | 0.005162 | -0.000084 |
| DenseNet121 | Grad-CAM | blur sigma 5 | 180 | 0.999992 | 1.000000 | 0.000006 | 0.000000 |
| DenseNet121 | epsilon-gamma-box LRP | zero | 180 | 0.999976 | 0.999972 | 0.000001 | -0.000006 |
| DenseNet121 | epsilon-plus-flat LRP | zero | 180 | 0.998885 | 0.999797 | 0.001030 | -0.000181 |
| ResNet50 | Grad-CAM | zero | 180 | 0.999316 | 0.999995 | 0.000679 | 0.000001 |

Interpretation:

- Faithfulness evidence is weak in the current form because deletion/insertion AUCs are near 1.0 for both method-ranked and random-ranked perturbations.
- The method-vs-random advantages are tiny. This should be reported as a limitation, not as strong faithfulness validation.

Parameter-randomization sanity checks recalculated from `RANDOMIZATION.csv`:

| Backbone | Method | Randomization level | n | Median Pearson | Median cosine | Median top-10 overlap |
|---|---|---|---:|---:|---:|---:|
| DenseNet121 | Grad-CAM | head | 180 | 0.139325 | 0.369663 | 0.071642 |
| DenseNet121 | Grad-CAM | last stage + head | 180 | 0.086321 | 0.459881 | 0.087585 |
| DenseNet121 | Grad-CAM | full | 180 | 0.258529 | 0.744464 | 0.056596 |
| DenseNet121 | epsilon-gamma-box LRP | head | 180 | 0.024085 | 0.023264 | 0.581108 |
| DenseNet121 | epsilon-gamma-box LRP | last stage + head | 180 | -0.182441 | -0.192506 | 0.460941 |
| DenseNet121 | epsilon-gamma-box LRP | full | 180 | 0.004232 | 0.018648 | 0.143185 |
| ResNet50 | Grad-CAM | head | 180 | -0.223453 | 0.214012 | 0.006477 |
| ResNet50 | Grad-CAM | full | 180 | -0.222217 | 0.531438 | 0.002790 |

Interpretation:

- Unlike the older local pipeline finding where randomization was effectively broken, this Gate-6 full-battery code actually resets modules and produces nontrivial changes.
- It is still not uniformly clean: DenseNet121 Grad-CAM full-randomized cosine median remains high at 0.744464, which requires cautious discussion.

Robustness results recalculated from `ROBUSTNESS.csv`:

| Backbone | Method | n | Median Pearson | Median cosine | Median top-10 overlap | Median absolute LRR+ change |
|---|---|---:|---:|---:|---:|---:|
| DenseNet121 | Grad-CAM | 1260 | 0.994697 | 0.998380 | 0.953559 | 0.002845 |
| DenseNet121 | epsilon-gamma-box LRP | 1260 | 0.003126 | 0.005459 | 0.165449 | 0.125810 |
| DenseNet121 | epsilon-plus-flat LRP | 1260 | 0.166127 | 0.279116 | 0.400664 | 0.054084 |
| ResNet50 | Grad-CAM | 1260 | 0.988784 | 0.993786 | 0.926805 | 0.003261 |

Interpretation:

- Grad-CAM is highly stable under the tested perturbations.
- DenseNet121 LRP, especially epsilon-gamma-box, is unstable under small input perturbations. This undermines any claim that LRP is robust in this setup.

### 14.3 pTB mask sensitivity result

`g6-r1e-ptb-mask-sensitivity` result:

- status: `PTB_MASK_SENSITIVITY_INFERENCE_COMPLETE`
- cohort rows: 180 tuning cases
- predicted masks: 176
- technical failures: 4
- tuning images opened: 180
- test images opened: 0
- test inference performed: false
- wall time: 1102.53 s

Failure breakdown from `G6_R1E_PTB_MASK_RECORDS.csv`:

| Status | Failure reason | Count |
|---|---|---:|
| OK | - | 176 |
| TECHNICAL_FAILURE | no_detection_at_confidence_0.5 | 3 |
| TECHNICAL_FAILURE | bbox_out_of_bounds_or_nonpositive | 1 |

Valid pTB mask area fraction:

- valid masks: 176
- median area fraction at 224 grid: 0.307438
- min area fraction: 0.195851
- max area fraction: 0.548151

Interpretation:

- This notebook generates an alternative pTB mask set for the same 180 tuning cases.
- It does not by itself recompute LRR sensitivity against pTB masks in the full XAI table unless a downstream comparison consumes `G6_R1E_PTB_MASKS.npz`.
- It should be cited as a mask-generation/sensitivity-input artifact, not as completed XAI sensitivity analysis unless the downstream comparison exists.

### 14.4 `notebook94ebf5c236`

Status:

- Kaggle reports `KernelWorkerStatus.CANCEL_ACKNOWLEDGED`.

Source-level finding:

- The notebook is not part of the locked Gate-5/Gate-6 protocol style.
- It attempts to locate `SEALED_test_label_key.csv`, extract `opaque_test_images.zip`, build a test dataset, train models, calibrate on validation, and evaluate the test set.
- It has no completed Kaggle result and no reliable final metric artifact.

Suitability:

- Do not cite this notebook as experimental evidence.
- Do not use it to support final sealed-test metrics.
- If it is intended as a sealed-test runner, it needs to be replaced by a locked, auditable Gate-7-style notebook with:
  - checkpoint hashes;
  - test image/label manifest hashes;
  - single-pass inference;
  - no training or model selection after test labels are mounted;
  - exported prediction CSV and metrics JSON.

### 14.5 Direct implications for `main.tex`

The current paper still has XAI contradictions:

- Abstract/Results say Guided Grad-CAM or Guided/LRP-like LRR around 0.802 and Grad-CAM around 0.774.
- Discussion/Conclusion say median LRR+ around 0.245 for Guided/LRP and 0.107 for Grad-CAM.
- Gate-6 full-battery artifacts show:
  - DenseNet121 epsilon-gamma-box LRP median LRR+ = 0.245070.
  - DenseNet121 Grad-CAM median LRR+ = 0.257274.
  - ResNet50 Grad-CAM median LRR+ = 0.071261.
- Therefore the 0.802/0.774 values likely come from older/local CSVs using a different method, mask, sample, or metric definition and should not be mixed with Gate-6 results.

Terminology correction:

- Gate-6 full-battery does not compute "Guided Grad-CAM" as the primary method.
- It computes:
  - Grad-CAM;
  - DenseNet121 epsilon-gamma-box LRP through Zennit;
  - DenseNet121 epsilon-plus-flat LRP sensitivity.
- Calling epsilon-gamma-box LRP "Guided Grad-CAM" would be incorrect.

Current conclusion for Gate-6:

- Gate-6 full battery is a usable, auditable **tuning-only exploratory XAI validity battery**.
- It supports a cautious claim: DenseNet121 LRP has higher mean LRR+ than DenseNet121 Grad-CAM on 178 common tuning cases, but the median paired difference is approximately zero and robustness/faithfulness evidence is mixed.
- It does not support claims of exceptional lung-field concentration.
- It does not support sealed-test XAI claims.
- pTB mask sensitivity is partially complete as mask generation, but downstream LRR recomputation against pTB masks is not proven by these checked artifacts.

## 15. Addendum: Gate-7 / Drive outputs / Guided Grad-CAM replacement checked on 2026-07-27

Items checked:

- Kaggle kernels:
  - `hintrngia/bsmsrs-c0`
  - `hintrngia/bsmsc0`
  - `hintrngia/bsmsp`
- Local notebooks:
  - `GATE7_CLEAN_PIPELINE.ipynb`
  - `GATE7_CLEAN_PIPELINE (1).ipynb`
- Local artifacts:
  - `pipeline/outputs/models/*.pt`
  - `pipeline/outputs/json/*_test_metrics.json`
  - `pipeline/outputs/csv/*.csv`

Kaggle status:

| Kernel | Status | Finding |
|---|---|---|
| `bsmsrs-c0` | `KernelWorkerStatus.CANCEL_ACKNOWLEDGED` | Not usable as completed evidence |
| `bsmsc0` | `KernelWorkerStatus.COMPLETE` | Completed, but metrics are very poor and still log XAI as Grad-CAM & LRP |
| `bsmsp` | `KernelWorkerStatus.COMPLETE` | Completed, but metrics are very poor and still log XAI as Grad-CAM & LRP |

### 15.1 Local checkpoints found

Six model checkpoints exist locally:

| Checkpoint | Size |
|---|---:|
| `pipeline/outputs/models/DenseNet121_seed3407_best.pt` | 28,964,235 bytes |
| `pipeline/outputs/models/DenseNet121_seed42_best.pt` | 28,962,763 bytes |
| `pipeline/outputs/models/DenseNet121_seed2024_best.pt` | 28,964,235 bytes |
| `pipeline/outputs/models/ResNet50_seed3407_best.pt` | 96,462,080 bytes |
| `pipeline/outputs/models/ResNet50_seed42_best.pt` | 96,461,358 bytes |
| `pipeline/outputs/models/ResNet50_seed2024_best.pt` | 96,462,080 bytes |

These checkpoints are consistent with Drive/local outputs existing, but the current audit did not find a formal manifest that records:

- checkpoint SHA-256;
- source notebook/run id;
- exact Git/source snapshot;
- training dataset hash;
- threshold/temperature artifact hash;
- test prediction CSV hash.

This should be added before submission.

### 15.2 Local sealed-test metrics from `pipeline/outputs/json`

The local JSON metrics support the paper's high AUROC claim:

| Model | Seed | AUROC | AUPRC | Accuracy | Sensitivity | Specificity | Brier | Confusion matrix TN/FP/FN/TP |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| DenseNet121 P | 3407 | 0.974447 | 0.981270 | 0.801282 | 0.997436 | 0.474359 | 0.140704 | 111/123/1/389 |
| DenseNet121 P | 42 | 0.971247 | 0.980277 | 0.804487 | 0.997436 | 0.482906 | 0.148422 | 113/121/1/389 |
| DenseNet121 P | 2024 | 0.977712 | 0.985905 | 0.868590 | 0.992308 | 0.662393 | 0.112766 | 155/79/3/387 |
| ResNet50 P | 3407 | 0.973515 | 0.981048 | 0.871795 | 0.989744 | 0.675214 | 0.106041 | 158/76/4/386 |
| ResNet50 P | 42 | 0.972650 | 0.979809 | 0.879808 | 0.992308 | 0.692308 | 0.103667 | 162/72/3/387 |
| ResNet50 P | 2024 | 0.968584 | 0.975031 | 0.887821 | 0.989744 | 0.717949 | 0.121867 | 168/66/4/386 |

Aggregates:

| Model | n | Mean AUROC | SD AUROC | Mean accuracy | Mean sensitivity | Mean specificity | Mean Brier |
|---|---:|---:|---:|---:|---:|---:|---:|
| DenseNet121 P | 3 | 0.974469 | 0.003233 | 0.824786 | 0.995726 | 0.539886 | 0.133964 |
| ResNet50 P | 3 | 0.971583 | 0.002633 | 0.879808 | 0.990598 | 0.695157 | 0.110525 |

Interpretation:

- These local JSON metrics are the only checked source that supports the abstract-level result around D-P AUROC `0.974 +/- 0.003`.
- They do **not** support the conflicting `0.958 +/- 0.006` claim elsewhere in the manuscript.
- They also do **not** support `0.538` as sealed-test D-P AUROC in the ablation table.

Major limitation:

- No sealed-test prediction CSV was found for these local JSON metrics. Without per-case predictions, the paper cannot compute:
  - cluster/bootstrap confidence intervals;
  - paired D-P vs D-C0 tests;
  - DeLong/permutation tests;
  - subgroup analyses;
  - calibration plots from raw predictions;
  - error-case tables with image ids;
  - reproducible verification of the JSON metrics.

### 15.3 Kaggle `bsmsc0` and `bsmsp` outputs conflict with local metrics

Downloaded Kaggle output CSVs from both complete kernels.

`bsmsc0` run ledger:

| Run | AUROC | Accuracy |
|---|---:|---:|
| DenseNet121 seed 3407 | 0.369702 | 0.625000 |
| ResNet50 seed 3407 | 0.365138 | 0.407051 |
| DenseNet121 seed 42 | 0.411993 | 0.589744 |
| ResNet50 seed 42 | 0.453479 | 0.604167 |
| DenseNet121 seed 2024 | 0.449540 | 0.628205 |
| ResNet50 seed 2024 | 0.449781 | 0.625000 |

`bsmsp` run ledger:

| Run | AUROC | Accuracy |
|---|---:|---:|
| DenseNet121 seed 3407 | 0.547661 | 0.625000 |
| ResNet50 seed 3407 | 0.275362 | 0.625000 |
| DenseNet121 seed 42 | 0.423855 | 0.625000 |
| ResNet50 seed 42 | 0.418343 | 0.625000 |
| DenseNet121 seed 2024 | 0.641973 | 0.626603 |
| ResNet50 seed 2024 | 0.496976 | 0.625000 |

Interpretation:

- These Kaggle complete kernels cannot be used as final reported performance.
- They contradict the local `pipeline/outputs/json` metrics by a very large margin.
- Accuracy near 0.625 with sensitivity 1.0 and specificity 0.0 indicates a near-all-positive operating point for many runs.
- The Kaggle logs still state `[XAI] Running Explainability Analysis (Grad-CAM & LRP)`, not Guided Grad-CAM.

Required action:

- Do not mix `bsmsc0` / `bsmsp` metrics with local Drive metrics.
- Either explain and discard these kernels as failed/obsolete runs, or rerun a locked Gate-7 pipeline and make that the single source of truth.

### 15.4 Guided Grad-CAM replacement status

Local notebooks are not equivalent:

| Notebook | XAI method in source | Status |
|---|---|---|
| `GATE7_CLEAN_PIPELINE.ipynb` | Grad-CAM + Guided Grad-CAM | Relevant to the user's replacement direction |
| `GATE7_CLEAN_PIPELINE (1).ipynb` | Grad-CAM + LRP | Older/inconsistent with replacement |
| `bsmsc0.ipynb` / `bsmsp.ipynb` | Grad-CAM + LRP in logs/artifacts | Not updated to Guided Grad-CAM |

Local Guided Grad-CAM CSVs:

| File | Rows | Grad-CAM mean | Grad-CAM median | Guided Grad-CAM mean | Guided Grad-CAM median | Has faithfulness/randomization/stability columns |
|---|---:|---:|---:|---:|---:|---|
| `DenseNet121_seed3407_xai_lrr.csv` | 128 | 0.775463 | 0.776439 | 0.800273 | 0.805381 | Yes |
| `DenseNet121_seed42_xai_lrr.csv` | 128 | 0.753628 | 0.759359 | 0.774345 | 0.780241 | No |
| `DenseNet121_seed2024_xai_lrr.csv` | 128 | 0.793720 | 0.796658 | 0.832047 | 0.836481 | No |
| `ResNet50_seed3407_xai_lrr.csv` | 128 | 0.817280 | 0.818805 | 0.864734 | 0.869085 | Yes |
| `ResNet50_seed42_xai_lrr.csv` | 128 | 0.755985 | 0.759297 | 0.770700 | 0.780381 | No |
| `ResNet50_seed2024_xai_lrr.csv` | 128 | 0.763507 | 0.766307 | 0.765807 | 0.765322 | No |

Aggregate local XAI summary:

- DenseNet121 mean Grad-CAM LRR+: 0.774270
- DenseNet121 mean Guided Grad-CAM LRR+: 0.802222
- ResNet50 mean Grad-CAM LRR+: 0.778924
- ResNet50 mean Guided Grad-CAM LRR+: 0.800414
- Mean chance LRR: 0.342431

Important caveats:

- Each XAI CSV has only 128 rows, not the full 624 sealed-test cases.
- Only seed 3407 files contain deletion/insertion/stability/randomization columns.
- Other seeds contain LRR only.
- The local randomization columns for seed 3407 remain suspicious: `gcam_rand` and `guided_rand` are approximately 1.0 in earlier local inspection, suggesting the randomization test was not meaningful or not actually randomizing the right module.
- Therefore the Guided Grad-CAM replacement can support a descriptive LRR table, but **not yet a full Q1-level XAI validity claim**.

### 15.5 What is still missing before the paper is complete and defensible

Highest priority reruns/fixes:

1. **Single locked Gate-7 source of truth.**
   Rerun one final locked pipeline for D-C0, D-P, R-C0, and R-P. It must export prediction CSVs and metrics JSON. Current sources conflict: local JSON says AUROC ~0.97, Kaggle `bsmsc0/bsmsp` says AUROC ~0.28-0.64.

2. **D-C0 and R-C0 sealed-test controls.**
   The current reliable local JSON only covers P-style checkpoints. The C0 rows in the manuscript are tuning-only or failed/obsolete Kaggle runs. For the primary claim, D-P must be compared against D-C0 on the same sealed test set.

3. **Per-case sealed-test prediction CSVs.**
   Required columns:
   - sample_id / image_id
   - operational_group_id
   - label
   - model
   - seed
   - checkpoint_sha256
   - raw_logit
   - probability
   - calibrated_probability
   - threshold_used
   - predicted_label
   - split = sealed_test

4. **Confidence intervals and paired tests.**
   Required for AUROC, AUPRC, Brier/log loss, sensitivity, specificity, F1, and P-vs-C0 paired contrasts. Use cluster bootstrap by operational/acquisition group where possible.

5. **Checkpoint and artifact manifest.**
   Add a manifest mapping every result row to checkpoint hash, notebook hash, dataset hash, threshold/temperature, and created artifact hashes.

6. **Reproduction of the reference paper.**
   Gate-4 R0 was development-only. A real reference-paper reproduction still needs R0 test evaluation or source-defined test evaluation for DenseNet121 and ResNet50, with accuracy, AUROC, F1, and AP, to check the claimed +/-2%.

7. **Guided Grad-CAM full validity battery.**
   If LRP is replaced by Guided Grad-CAM, rerun XAI validity for Guided Grad-CAM, not just LRR:
   - 624-case sealed-test LRR or a pre-locked 180/128 case sample with justification;
   - deletion/insertion for all seeds or a single pre-specified primary checkpoint;
   - parameter randomization that actually changes the target layers;
   - input perturbation stability;
   - mask sensitivity with Danilov and pTB masks;
   - method-vs-random attribution ranking baselines.

8. **Resolve XAI cohort inconsistency.**
   The paper currently mentions 100, 128, 178, and 180 cases in different places. Pick one primary XAI cohort and label it exactly as tuning or sealed test.

9. **Resolve method inconsistency.**
   Remove all LRP text/tables if the final method is Guided Grad-CAM. Also remove Gate-6 LRP full-battery claims unless moved to supplementary obsolete/sensitivity material.

10. **Resolve proposed-method inconsistency.**
    Gate-5 P was not CBAM + Mask Loss, but local Gate-7 notebooks include CBAM + Mask Loss. The paper must clearly separate:
    - Gate-5 clean recipe P; and
    - Gate-7 CBAM + Mask Loss P.
    If the final proposed method is CBAM + Mask Loss, rerun clean C0/P comparisons under one consistent codebase and stop using Gate-5 P as if it were the same method.

11. **Calibration artifacts.**
    Save tuning logits/probabilities, temperature, Youden threshold, and show that thresholds were selected only on tuning then applied once to test.

12. **Error and subgroup analysis.**
    Report false positives/false negatives, normal/bacteria/virus subgroup performance if labels exist, and duplicate-group-aware summaries.

13. **Ablation completion.**
    DenseNet C1-C4 are tuning-only. If they remain in the paper, keep them as tuning ablation only. If claiming sealed-test effect, rerun sealed-test evaluation for each ablation arm.

14. **External validation.**
    For Q1/A/A* level, an independent external pediatric CXR cohort is strongly recommended. Without it, the paper should position itself as internal/leakage-aware methodological evaluation only.

15. **Clean paper tables.**
    Regenerate all tables from machine-readable artifacts. Do not manually mix:
    - Gate-5 tuning values;
    - local Gate-7 sealed-test values;
    - failed/obsolete Kaggle runs;
    - Gate-6 LRP tuning values;
    - local Guided Grad-CAM sealed/sampled values.

Current Gate-7 conclusion:

- The local Drive/pipeline outputs are promising and support D-P/R-P AUROC around 0.97 on the sealed test, but they are not yet sufficiently auditable for Q1/A/A* standards because prediction CSVs, paired C0 controls, confidence intervals, and a locked artifact manifest are missing.
- The complete Kaggle kernels `bsmsc0` and `bsmsp` should be treated as failed/obsolete because their metrics are incompatible with the local results.
- Guided Grad-CAM replacement is partially present locally, but not consistently propagated to the Kaggle kernels or full validity battery.

## 16. Cleanup performed on 2026-07-27

Repository cleanup goal:

- keep only the artifacts that can currently support manuscript tables;
- separate tuning-only/supporting evidence from final paper-ready evidence;
- archive failed, canceled, obsolete, or deprecated runs;
- remove generated cache and credential files from the active tree.

Current clean structure:

```text
docs/
  reference_paper.pdf
  research_audit_report.md
  locked_final_rerun_plan.md
  q1_astar_readiness_plan.md
notebooks/
  kaggle/
paper/
  main.tex
  main.pdf
pipeline/
results/
  locked_final/
    reference_reproduction/
    classification_runs/
    classification_statistics/
    xai_manifest/
    xai_runs/
    paper_evidence_audit.md
    retained_artifact_manifest.csv
tools/
```

Clean-repo artifacts retained:

- Locked sealed-test prediction CSVs, metric JSONs, thresholds, calibration
  metadata, run configs, environment snapshots, and audit summaries for all
  12 B01-B12 classification runs.
- Reference reproduction predictions and metrics from A01.
- C00 XAI manifest and E01-E05 XAI metric/statistical evidence.
- Paper figures and the final LNCS source/PDF.
- `results/locked_final/retained_artifact_manifest.csv` with SHA-256 hashes for
  all retained evidence files.

Checkpoint weights are excluded from the clean GitHub package. They can be
regenerated by rerunning the locked Kaggle notebooks or attached externally as
Kaggle datasets when rerunning the XAI notebooks.

Supporting artifacts retained:

- Gate-5 Kaggle source/logs for development/tuning-only ablation.
- Gate-4 R0 source pulls for development-only reproduction evidence.

Archived as obsolete/deprecated:

- `bsmsrs-c0`, `bsmsc0`, and `bsmsp` notebooks/logs/outputs because they are canceled or metric-inconsistent with the local paper-ready results.
- `notebook94ebf5c236.ipynb` because Kaggle status was canceled.
- Gate-6 LRP/pTB tuning artifacts because the user decided to replace LRP with Guided Grad-CAM and these outputs are not final paper evidence.
- Old local Gate-7 LRP notebook.
- Previous `pipeline/outputs` tree was moved to archive after paper-ready artifacts were copied.

Removed from the active tree:

- `pipeline/input/kaggle.json` credential file.
- Python `__pycache__`.
- Generated/extracted duplicate `gate46_development` output trees and redundant Kaggle extracted outputs.
- LaTeX build byproducts from `paper/` (`.aux`, `.bbl`, `.blg`, `.log`, `.out`).

Important caveat after cleanup:

- Cleanup does not make the paper complete. It only prevents accidental citation of wrong or obsolete artifacts.
- The final paper still needs a locked rerun with D-C0/D-P/R-C0/R-P, prediction CSVs, paired CIs/tests, and a complete Guided Grad-CAM validity battery before Q1/A/A* submission.
- Active XAI source was corrected after the audit: `pipeline/src/02_config.py` now lists Guided Grad-CAM instead of LRP; `pipeline/src/10_xai.py` now uses target-class probabilities for deletion/insertion and randomizes `model.backbone.fc` / `model.backbone.classifier` for sanity checks.
- Therefore old faithfulness/randomization artifacts remain invalid and must not be carried forward without rerun. `notebooks/active/GATE7_CLEAN_PIPELINE_guided_gradcam.ipynb` was regenerated from the cleaned source.
