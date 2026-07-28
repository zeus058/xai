# Final Paper Evidence Audit

Ngày audit: 2026-07-27

## Phạm vi

File paper chính: `paper/main.tex`

PDF đã build: `paper/main.pdf`

Trạng thái build: PDF sinh thành công 16 trang sau khi khôi phục và tinh chỉnh các sơ đồ/biểu đồ chính. `pdflatex` trả exit code khác 0 do MiKTeX không ghi được log người dùng ở `C:\Users\trang\AppData\Local\MiKTeX\miktex\log\pdflatex.log`; log paper không có undefined reference, fatal error, overfull box, hoặc yêu cầu rerun cross-reference. Cảnh báo còn lại đáng kể là `amsmath Warning: Unable to redefine math accent \vec`, không ảnh hưởng số liệu.

## Nguồn Bằng Chứng Chính

| Thành phần trong paper | File bằng chứng |
|---|---|
| Bảng kết quả classification chính | `results/locked_final/classification_statistics/classification_summary_table.csv` |
| Paired bootstrap CI P-minus-C0 | `results/locked_final/classification_statistics/paired_delta_bootstrap_ci.csv` |
| Tóm tắt thống kê classification | `results/locked_final/classification_statistics/classification_statistics_summary.md` |
| Bảng XAI chính | `results/locked_final/xai_runs/aggregate/xai_summary_table.csv` |
| Paired bootstrap CI XAI | `results/locked_final/xai_runs/aggregate/xai_p_minus_c0_paired_delta_ci.csv` |
| XAI all rows by image/seed/method | `results/locked_final/xai_runs/aggregate/xai_all_by_image.csv` |
| Tóm tắt XAI aggregate | `results/locked_final/xai_runs/aggregate/xai_statistics_summary.md` |
| Hình qualitative DenseNet trong paper | `paper/figures/xai_qualitative_densenet_guided_cases.png` |
| Sơ đồ locked study flow trong paper | TikZ trong `paper/main.tex` |
| Biểu đồ classification delta CI trong paper | `paper/figures/classification_p_minus_c0_delta_ci.png` |
| Biểu đồ XAI localization delta CI trong paper | `paper/figures/xai_p_minus_c0_localization_delta_ci.png` |
| Audit chọn case qualitative | `results/locked_final/xai_runs/aggregate/selected_qualitative/` |
| Script sinh panel qualitative | `tools/generate_paper_xai_panels.py` |

## Kết Quả Classification Được Phép Báo Cáo

Tất cả B01-B12 là 12 run locked final battery: D-C0, D-P, R-C0, R-P, mỗi điều kiện 3 seed. Mỗi run có prediction CSV trên 624 ảnh test sealed. Tóm tắt chính:

| Điều kiện | AUROC | Balanced accuracy | Specificity | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| D-C0 | 0.9677 +/- 0.0029 | 0.7833 +/- 0.0122 | 0.5726 +/- 0.0280 | 0.1386 +/- 0.0122 | 0.1779 +/- 0.0136 |
| D-P | 0.9746 +/- 0.0055 | 0.7963 +/- 0.0208 | 0.5969 +/- 0.0428 | 0.1258 +/- 0.0115 | 0.1627 +/- 0.0177 |
| R-C0 | 0.9699 +/- 0.0018 | 0.7828 +/- 0.0562 | 0.5698 +/- 0.1138 | 0.1399 +/- 0.0049 | 0.1847 +/- 0.0052 |
| R-P | 0.9731 +/- 0.0030 | 0.7993 +/- 0.0531 | 0.6011 +/- 0.1062 | 0.1273 +/- 0.0081 | 0.1700 +/- 0.0068 |

Paired deltas cần diễn giải thận trọng:

| Backbone | Metric | Delta | 95% CI | Kết luận |
|---|---|---:|---:|---|
| DenseNet121 | AUROC | +0.0069 | [0.0034, 0.0108] | Supported |
| DenseNet121 | Balanced accuracy | +0.0130 | [0.0014, 0.0246] | Supported |
| DenseNet121 | Specificity | +0.0242 | [0.0014, 0.0470] | Supported |
| DenseNet121 | Brier | -0.0128 | [-0.0166, -0.0089] | Supported |
| DenseNet121 | ECE | -0.0153 | [-0.0190, -0.0113] | Supported |
| ResNet50 | AUROC | +0.0032 | [-0.0003, 0.0067] | Directional only |
| ResNet50 | Balanced accuracy | +0.0165 | [0.0034, 0.0296] | Supported |
| ResNet50 | Specificity | +0.0313 | [0.0057, 0.0570] | Supported |
| ResNet50 | Brier | -0.0127 | [-0.0164, -0.0088] | Supported |
| ResNet50 | ECE | -0.0147 | [-0.0184, -0.0107] | Supported |

Claim đúng mức: proposed configuration cho improvement nhỏ nhưng nhất quán hơn ở specificity, balanced accuracy, F1, Brier, và ECE. DenseNet AUROC supported; ResNet AUROC chỉ là directional vì CI cắt 0.

## Kết Quả XAI Được Phép Báo Cáo

XAI locked battery gồm E01-E04, 128 ảnh XAI cân bằng, 3 seed mỗi setting, tổng 1536 rows.

| Điều kiện | Grad-CAM LRR+ | Guided Grad-CAM LRR+ | Grad-CAM pointing | Guided Grad-CAM pointing |
|---|---:|---:|---:|---:|
| D-C0 | 0.3798 +/- 0.0111 | 0.3616 +/- 0.0259 | 0.4036 +/- 0.0401 | 0.2135 +/- 0.0352 |
| D-P | 0.7752 +/- 0.0215 | 0.7979 +/- 0.0359 | 0.9948 +/- 0.0045 | 0.9688 +/- 0.0271 |
| R-C0 | 0.3606 +/- 0.0312 | 0.3827 +/- 0.0628 | 0.2057 +/- 0.1144 | 0.2943 +/- 0.0813 |
| R-P | 0.7616 +/- 0.0008 | 0.8218 +/- 0.0140 | 0.9870 +/- 0.0045 | 0.9271 +/- 0.0352 |

Paired XAI deltas đều strongly supported cho lung-region concentration:

| Backbone | Metric | Delta | 95% CI |
|---|---|---:|---:|
| DenseNet121 | Grad-CAM LRR+ | +0.3955 | [0.3867, 0.4044] |
| DenseNet121 | Guided Grad-CAM LRR+ | +0.4363 | [0.4220, 0.4502] |
| DenseNet121 | Grad-CAM pointing | +0.5911 | [0.5417, 0.6406] |
| DenseNet121 | Guided Grad-CAM pointing | +0.7552 | [0.7109, 0.7995] |
| ResNet50 | Grad-CAM LRR+ | +0.4010 | [0.3924, 0.4093] |
| ResNet50 | Guided Grad-CAM LRR+ | +0.4391 | [0.4232, 0.4550] |
| ResNet50 | Grad-CAM pointing | +0.7812 | [0.7422, 0.8203] |
| ResNet50 | Guided Grad-CAM pointing | +0.6328 | [0.5807, 0.6823] |

Claim đúng mức: proposed CBAM + mask-guided auxiliary loss làm tăng mạnh concentration của Grad-CAM và Guided Grad-CAM bên trong lung fields. Không được diễn giải thành lesion localization hay clinical reasoning.

## Hình Qualitative XAI

Figure trong paper: `paper/figures/xai_qualitative_densenet_guided_cases.png`

Panel phụ chưa đưa main paper: `paper/figures/xai_qualitative_resnet_guided_cases.png`

Rule chọn case được lưu ở `selected_qualitative_summary.json`:

> For each available P-model outcome stratum in the 128-case XAI sample, select the case with both proposed Grad-CAM and proposed Guided Grad-CAM pointing inside lung, then largest minimum proposed LRR+ across the two methods, then largest total proposed-minus-control LRR+ delta.

Ghi chú quan trọng: không có proposed false-negative stratum trong XAI sample seed 3407, nên paper không nên hứa có đủ TN/TP/FP/FN qualitative cases. Caption hiện tại đã ghi rõ điểm này.

## Reproduction Bài Gốc

R0 reproduction không được dùng làm superiority baseline. Lý do đúng để trình bày: bài gốc/source không công bố đầy đủ image-level split, seed, model-selection details và môi trường TensorFlow/Keras cũ không khớp runtime hiện tại. Vì vậy R0 là reproducibility limitation, còn improvement claim phải dựa trên C0 vs P locked protocol.

Kết quả R0 cần giữ như limitation:

| Backbone | Reproduced accuracy | Reference accuracy | Reproduced AUROC | Reference AUROC | Pass +/-2% |
|---|---:|---:|---:|---:|---|
| DenseNet121 | 0.812500 | 0.891 | 0.954562 | 0.980 | No |
| ResNet50 | 0.815171 | 0.844 | 0.954854 | 0.950 | No |

## Claim Không Được Nói Quá

- Không claim lesion localization vì không có expert lesion annotation.
- Không claim clinical utility, deployment readiness, hoặc diagnostic benefit ngoài internal sealed test.
- Không claim ResNet AUROC significant improvement vì CI cắt 0.
- Không claim proposed model có randomization sanity tốt hơn; head-randomization correlations cao hơn ở proposed models.
- Không claim deletion/insertion uniformly better; kết quả mixed.
- Không claim CBAM và mask loss là hai causal ablation riêng biệt; locked P configuration gộp cả hai.
- Không claim exact reproduction bài gốc; chỉ best-effort reproduction vì thiếu seed/split/selection details.

## Kiểm Tra Rác/Mâu Thuẫn

Đã scan `paper/main.tex` cho các pattern rủi ro: `LRP`, `D05`, `P-only`, `TBD`, `PENDING`, `TO LOCK`, `ground-truth lung`, số liệu cũ `0.958`/`0.971`, và tên figure cũ. Không còn match rác sau chỉnh sửa.

Các match còn lại là caveat đúng chủ đích: `not lesion localization`, `clinical deployment readiness`, `clinical diagnostic benefit`, và câu availability `will be released`.

## Việc Còn Thiếu Nếu Nhắm Q1/A*

- External cohort validation.
- Expert lesion-level annotation nếu muốn claim localization tổn thương.
- Ablation tách riêng CBAM-only và mask-loss-only nếu muốn chứng minh từng thành phần.
- Reader study hoặc clinical workflow evaluation nếu muốn claim clinical utility.
- Public release package hoàn chỉnh sau khi kiểm tra license dataset/model weights.

Kết luận audit: paper hiện tại đủ nhất quán với locked final evidence để báo cáo một contribution về leakage-aware evaluation và lung-focused explanation improvement. Mức claim hiện tại phù hợp hơn cho một bài methodology/audit nghiêm túc; chưa đủ để claim clinical deployment hoặc lesion-level explanation.

## Final Pass Sau Khi Rút Gọn

Sau khi rút paper xuống 14 trang, đã kiểm tra lại abstract, Methods, Results, Discussion, figure/caption XAI, và các bảng chính. Inconsistency cũ trong abstract về `5216 images` đã được sửa thành locked clean protocol: 4411 training images, 779 tuning images, và 624 sealed-test images.

Sau yêu cầu khôi phục trực quan hóa, paper được cập nhật lên 16 trang với bốn thành phần trực quan chính: locked study-flow schematic, classification P-minus-C0 delta CI plot, XAI localization P-minus-C0 delta CI plot, và qualitative XAI panel. XAI aggregate bar plot được thay bằng delta-CI plot vì dạng này thể hiện trực tiếp effect size và 95% CI của claim P-versus-C0, phù hợp hơn với reviewer so với so sánh cột trung bình. Hình qualitative XAI vẫn giữ layout không có lung-mask panel: original radiograph, C0 Grad-CAM, P Grad-CAM, và P Guided Grad-CAM. Tất cả hình/sơ đồ trong main paper hiện có caption và câu dẫn hoặc đoạn diễn giải liền kề: study-flow được giải thích trong Methods, hai delta-CI plots có câu dẫn trước hình, và qualitative XAI panel có câu dẫn trước hình cùng caption nêu rõ rule chọn case. LaTeX build cuối sinh `paper/main.pdf` 16 trang; log không có undefined reference, fatal error, overfull box, hoặc yêu cầu rerun cross-reference. Sơ đồ study-flow đã được chỉnh thành bố cục nhiều tầng, giãn khoảng cách box, khóa vị trí để caption/prose không bị tách trang, và dùng arrow routing thẳng/gấp khúc với đầu mũi tên chạm đúng box đích. Paper cũng thêm một đoạn Methods giải thích cách đọc sơ đồ như các evidence streams tách biệt: dev/tuning cho training-thresholding, sealed test cho locked prediction/calibration/XAI, và C0/P là paired contrasts.

## Final Reviewer Audit Sau Chỉnh Sơ Đồ Và Evidence Trace

Đã thực hiện final audit theo hướng reviewer Q1/A*: kiểm tra source `paper/main.tex`, rendered PDF 16 trang, LaTeX log, figure files, evidence CSV/MD, và các claim nhạy cảm.

Các chỉnh sửa cuối đã đưa vào paper:

- Làm rõ cohort arithmetic: source có 5840 readable images, nhưng 5814 ảnh có canonical lung masks và đi vào locked clean C0/P protocol. Điều này giải thích vì sao 4411 training + 779 tuning + 624 sealed test = 5814, đồng thời bảo đảm C0 và P dùng cùng mask-eligible population.
- Sửa wording `4411 development images` thành `4411 training images` ở abstract, Methods, Results và TikZ diagram để tránh nhầm với development set tổng.
- Thêm F1 vào Table `classification-deltas` và regenerate Figure `classification_p_minus_c0_delta_ci.png` để paper nhiều lần claim F1 improved đều có visible evidence trong cả bảng lẫn hình: DenseNet121 `+0.0066 [0.0011,0.0124]`, ResNet50 `+0.0082 [0.0020,0.0145]`.
- Kiểm tra lại study-flow diagram sau các chỉnh sửa mũi tên: bus line được nâng lên giữa khoảng trống hai hàng box, mũi tên dọc thẳng hàng với thân, caption/prose đi cùng figure.

Validation checks đã chạy:

- LaTeX build sinh `paper/main.pdf` 16 trang. `pdflatex` vẫn trả exit code khác 0 do MiKTeX không ghi được user log, nhưng paper log ghi `Output written on main.pdf (16 pages, 9002938 bytes)`.
- Log scan không có `Overfull`, `Undefined`, `Fatal`, `Citation undefined`, `Label(s) may have changed`, hoặc yêu cầu rerun thật.
- Figure files trong paper tồn tại: `classification_p_minus_c0_delta_ci.png`, `xai_p_minus_c0_localization_delta_ci.png`, `xai_qualitative_densenet_guided_cases.png`; study-flow là TikZ trong source.
- Spot-check tự động đối chiếu 60 số chính từ `classification_summary_table.csv`, `paired_delta_bootstrap_ci.csv`, `xai_summary_table.csv`, và `xai_p_minus_c0_paired_delta_ci.csv` với `.tex`. Không có discrepancy thực; một check ResNet AUROC CI high khác do rounding script (`0.006649...`) trong khi paper-ready summary dùng `0.0067`.
- Pattern scan cho các cụm rủi ro (`LRP`, `D05`, `P-only`, `TBD`, `PENDING`, `TO LOCK`, `ground-truth lung`, số liệu cũ `0.958`/`0.971`, `5216`) không phát hiện số liệu rác trong paper. Các match còn lại là caveat chủ đích như `not lesion localization`, `not clinical utility`, `not deployment readiness`, và `not globally superior faithfulness`.

Overall reviewer-readiness assessment: paper hiện ở mức **Ready to share with required caveats**. Bài đã nhất quán với locked final evidence cho claim chính: leakage-aware matched C0/P evaluation và improvement rõ nhất ở lung-region attribution concentration, specificity/balanced accuracy/F1/calibration. Các caveat bắt buộc vẫn phải giữ nguyên trong submission: chưa có external cohort, chưa có expert lesion annotation, chưa có ablation tách CBAM-only/mask-loss-only, và chưa có reader/clinical workflow study.

## CPU-Only Paper Asset Regeneration Layer

Đã bổ sung `tools/regenerate_paper_assets.py` để tái sinh các paper-ready
figures và SHA-256 manifest trực tiếp từ locked final CSV/MD artifacts. Script
này không dùng raw CXR images, lung masks, model checkpoints, Kaggle credentials,
GPU, hoặc notebook execution. Đây là lớp tái lập cuối cùng từ locked evidence
sang manuscript assets.

Command:

```bash
python tools/regenerate_paper_assets.py
```

Outputs:

- `paper/figures/classification_p_minus_c0_delta_ci.png`
- `paper/figures/xai_p_minus_c0_localization_delta_ci.png`
- `results/locked_final/paper_asset_manifest.csv`
- `results/locked_final/paper_asset_manifest.json`
- `results/locked_final/paper_asset_regeneration_report.md`

Validation sau khi chạy script: LaTeX build đầy đủ (`pdflatex`, `bibtex`,
`pdflatex`, `pdflatex`) sinh `paper/main.pdf` 16 trang. Scan `paper/main.log`
không có `LaTeX Error`, `Emergency stop`, `Fatal error`, undefined citation,
undefined reference, overfull box, hoặc yêu cầu rerun cross-reference. MiKTeX
vẫn trả non-zero ở cuối do không ghi được user-level log tại
`C:\Users\trang\AppData\Local\MiKTeX\miktex\log`, không phải lỗi manuscript.
