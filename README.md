# Báo Cáo Bài Làm Lab Day 2 — Deep Learning Advance

**Học viên:** Phan Danh Đạt  
**MSSV:** 02627  
**Lớp / Track:** K4 — Track 4: Deep Learning Advance  
**Môi trường thực nghiệm:** NVIDIA GeForce RTX 3060 12GB GDDR6 (CUDA 12.1), Windows 11  
**Phiên bản môi trường:** Python 3.11.9, PyTorch 2.5.1+cu121, timm 1.0.15, pandas 2.2.3, scikit-learn 1.5.2, openpyxl 3.1.5  

---

## 🏆 Tóm Tắt Kết Quả & Điểm Tự Chấm RUBRIC (Phần I: 20 / 20 Điểm)

Kết quả đánh giá chính thức qua công cụ `eval.py grade` độc lập (trung bình 3 seed `[0, 1, 2]` trên toàn bộ tập Test Fold 0 gồm 3.507 ảnh):

| Mã | Tiêu chí đánh giá | Kết quả đạt được | Mốc đối chiếu / Ngưỡng | Điểm đạt | Điểm tối đa |
|:---:|:---|:---:|:---:|:---:|:---:|
| **I1** | **Top-1 Accuracy trên Test** | **97.53% ± 0.09%** | ≥ 95.7% (vượt mốc 100 epoch bài báo: 95.7%) | **7** | 7 |
| **I2** | **Cải thiện Macro-F1 so với mốc** | **0.9686 ± 0.0013** | Mốc T00: 0.8069 ± 0.0149 (Δ = +0.1617 > s = 0.0149) | **5** | 5 |
| **I3** | **Recall 2 lớp khó nhất** | **Chinee Apple: 94.1% ± 1.4%**<br>**Snake Weed: 95.3% ± 0.7%** | Mốc bài báo: 88.5%<br>Mốc bài báo: 88.8% | **4** | 4 |
| **I4a**| **Hiệu chuẩn Temperature Scaling** | **ECE: 0.0821 → 0.0071 ± 0.0009** | ECE sau TS < ECE trước TS | **1** | 1 |
| **I4b**| **Độ ổn định Val / Test** | Val: 0.9658 vs Test: 0.9686 | \|Val - Test\| = 0.0028 ≤ 0.02 | **1** | 1 |
| **I5** | **Thời gian thực (Real-time Budget)**| **p95 = 5.78 ms** (batch 1, RTX 3060) | Ngân sách ≤ 100 ms (đo chuẩn sync + warmup) | **2** | 2 |
| **Σ** | **Tổng điểm Phần I** | | | **20** | **20** |

---

## 📁 Cấu Trúc Sản Phẩm Nộp Bài

```text
K4-Track4-Day2-PhanDanhDat-02627-Deeplearning-Advance/
├── README.md                      # Báo cáo tổng quan, môi trường, hướng dẫn tái lập & đề bài
├── report.md                      # Báo cáo khoa học phân tích chuyên sâu đầy đủ các bước (tiếng Việt)
├── results.xlsx                   # Bảng số liệu hoàn chỉnh 7 sheets (Backbones, Training, Inference, Final, PerClass, Latency, Summary)
├── requirements.txt               # Danh sách thư viện và phiên bản chính xác đã chạy trên RTX 3060
├── eval.py                        # Công cụ chấm điểm và đánh giá chuẩn mực của BTC (giữ nguyên gốc)
├── code/                          # Toàn bộ mã nguồn hoàn chỉnh
│   ├── dataset.py                 # Tải dữ liệu, kiểm tra tính toàn vẹn Fold 0 (S1-S6), transforms
│   ├── model.py                   # Khởi tạo 7 kiến trúc, 3 parameter groups, phân rã GMACs/Params
│   ├── losses.py                  # Label Smoothing, Focal Loss, Class-Weighted CE, Mixup/CutMix
│   ├── train.py                   # Vòng lặp huấn luyện chuẩn, AMP, Cosine Warmup, EMA, checkpointing
│   ├── inference.py              # TTA (lật/multi-scale), Temperature Scaling, Fused Conv-BN
│   ├── benchmark.py              # Đo độ trễ chuẩn mực (warmup >= 20, cuda.synchronize, p50/p95/p99)
│   ├── run_experiments.py        # Runner tự động chạy Bước 1, Bước 2, Bước 4
│   ├── run_inference.py          # Runner đánh giá suy luận và đo độ trễ chuẩn mực Bước 3
│   ├── build_results.py          # Tổng hợp 100% số liệu thực nghiệm vào results.xlsx
│   ├── generate_report_charts.py # Tạo biểu đồ tổng hợp cho báo cáo (F1 vs Latency, Confusion Matrix, ECE)
│   ├── eda_plot.py               # Biểu đồ phân bố 9 lớp dữ liệu Fold 0
│   └── lab_day2.ipynb            # Notebook tương tác từng bước từ 0 đến 5
├── curves/                        # 30 biểu đồ PNG chất lượng cao (26 tiến trình train/val + 4 biểu đồ phân tích)
│   ├── B02_resnext50_32x4d.png ... B07_mobilenetv3_large_100.png
│   ├── T00_resnet50.png ... T17_resnet50.png
│   ├── F01_convnext_tiny.png
│   ├── calibration_curve.png, confusion_matrix_test.png, eda_class_distribution.png
│   └── f1_vs_latency_backbones.png, f1_vs_latency_inference.png
└── predictions/                   # Toàn bộ 12 file CSV dự đoán trên Test và Val (F01, F01_uncal, T00 qua 3 seed)
    ├── F01_seed{0,1,2}_test.csv, F01_uncal_seed{0,1,2}_test.csv, F01_seed{0,1,2}_val.csv
    └── T00_seed{0,1,2}_test.csv, T00_seed{0,1,2}_val.csv
```

---

## 🚀 Hướng Dẫn Tái Lập Toàn Bộ Kết Quả (Reproduction Guide)

Mọi thí nghiệm được thiết kế để có thể tái lập 100% với các lệnh dưới đây (PowerShell trên Windows hoặc Bash trên Linux):

### 1. Kích hoạt môi trường và thiết lập mã hóa UTF-8
```powershell
.\.venv\Scripts\Activate.ps1
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
```

### 2. Kiểm tra bộ dữ liệu và pipeline (Bước 0)
```powershell
python -c "from code.dataset import check_split; check_split('data/images', 'data/labels', fold=0)"
python code/eda_plot.py
```

### 3. Chạy kiểm thử tự động toàn bộ Unit Tests (Không cần GPU)
```powershell
python -m unittest discover -s tests
```

### 4. Chạy Bước 1: So sánh 7 Backbone (B01 - B07)
```powershell
python code/run_experiments.py --step backbone
```

### 5. Chạy Bước 2: Khảo sát 16 cấu hình công thức huấn luyện (T01 - T17)
```powershell
python code/run_experiments.py --step training
```

### 6. Chạy Bước 3: Đánh giá phương pháp suy luận và đo độ trễ chuẩn mực (I00 - I08)
```powershell
python code/run_inference.py --exp-id T00 --backbone resnet50 --seed 0
```

### 7. Chạy Bước 4: Vòng chung kết 3 seed cho baseline T00 và model chung kết F01
```powershell
python code/run_experiments.py --step final
```

### 8. Xuất file kết quả tổng hợp `results.xlsx` và các biểu đồ phân tích
```powershell
python code/build_results.py
python code/generate_report_charts.py
```

### 9. Chạy công cụ đánh giá chính thức `eval.py`
```powershell
# Chấm điểm chi tiết model F01
python eval.py score --pred "predictions/F01_seed*_test.csv" --test-csv data/labels/test_subset0.csv --labels data/labels/labels.csv --tag F01 --out eval_out

# Tự chấm điểm Phần I RUBRIC
python eval.py grade --final "predictions/F01_seed*_test.csv" --baseline "predictions/T00_seed*_test.csv" --uncal "predictions/F01_uncal_seed*_test.csv" --final-val "predictions/F01_seed*_val.csv" --latency-p95-ms 5.78 --latency-method proper --test-csv data/labels/test_subset0.csv --labels data/labels/labels.csv
```

---

# Đề Bài Gốc: Lab Day 2 — Backbone, công thức huấn luyện và suy luận trên DeepWeeds

> Track 4 · Ngày 2 · *Tích chập, chuỗi, attention · backbone · huấn luyện · suy luận*
> Bài lab này mở rộng **Lab #2** trong slide Day 2. Slide chỉ yêu cầu 1 backbone, 3 cách khởi tạo, có/không CutMix và TTA. Ở đây bạn làm đầy đủ: **≥ 5 backbone**, **nhiều công thức huấn luyện**, **nhiều cách suy luận**, rồi chọn cấu hình tốt nhất và báo cáo.

Repo gồm **hướng dẫn, tiêu chí chấm, bộ khung code (pseudo-code) và công cụ đánh giá**. Bộ khung `starter/` chỉ có chữ ký hàm, docstring và các bước `TODO`: **bạn tự viết phần ruột** (model, loss, augmentation, vòng huấn luyện, TTA, đo độ trễ). Riêng `eval.py` đã hoàn chỉnh, bạn không sửa. Làm vậy để bạn hiểu từng thành phần trong slide, nhưng vẫn đo bằng cùng một thước.

| File | Dùng để làm gì |
|---|---|
| `README.md` (file này) | Tổng quan, dataset, quy tắc chia dữ liệu, cách đánh giá, sản phẩm phải nộp, cách nộp bài |
| [`GUIDE.md`](GUIDE.md) | Quy trình từng bước, danh sách thí nghiệm, cấu trúc file xlsx và báo cáo, bẫy thường gặp |
| [`RUBRIC.md`](RUBRIC.md) | Thang điểm 100, tiêu chí đạt, lỗi bị trừ điểm |
| [`eval.py`](eval.py) | **Đã hoàn chỉnh.** Tính chỉ số đúng định nghĩa ở mục 2.2 và tự chấm phần I của RUBRIC (mục 2.4) |
| [`starter/`](starter) | **Pseudo-code** để bạn hoàn thiện: `dataset.py`, `model.py`, `losses.py`, `train.py`, `inference.py`, `benchmark.py`, `lab_day2.ipynb` |
| [`tests/`](tests) | Test của `eval.py` và của bộ khung (chạy được không cần GPU) |

---


## 1. Mục tiêu học tập

Sau bài lab, bạn có thể:

1. So sánh công bằng nhiều backbone (CNN và transformer) trên cùng một bài toán, cùng một công thức huấn luyện.
2. Đo riêng đóng góp của từng yếu tố trong **công thức huấn luyện** (khởi tạo, augmentation, loss, optimizer/LR, EMA). Slide chương 5 nhấn mạnh công thức quan trọng ngang kiến trúc.
3. So sánh các **kỹ thuật suy luận** (TTA, độ phân giải kiểm tra, ensemble, hiệu chuẩn, gộp BatchNorm/FP16) bằng cả độ chính xác lẫn **độ trễ đo đúng cách**.
4. Phân biệt chênh lệch thật với **nhiễu** do hạt giống (seed) bằng mean ± std.
5. Rút ra kết luận có bằng chứng, nêu rõ cấu hình nào tốt nhất và vì sao.

## 2. Dataset: DeepWeeds

| Thuộc tính | Giá trị |
|---|---|
| Bài toán | Phân loại ảnh cỏ dại ngoài đồng (robot nông nghiệp, Queensland, Úc) |
| Số ảnh | 17.509 ảnh RGB 256×256 |
| Số lớp | 9: 8 loài cỏ dại + `Negative` (thực vật không phải loài mục tiêu) |
| Đặc điểm | **Mất cân bằng lớp**: `Negative` có 9.106 ảnh (khoảng 52%), mỗi loài cỏ có 1.009–1.125 ảnh (xem bảng dưới). Số liệu theo Table 1 của bài báo; bạn phải tự đếm lại ở bước EDA và đối chiếu |
| Dung lượng | Khoảng 490 MB |
| Giấy phép | CC BY 4.0 |
| Bài báo | Olsen et al., *DeepWeeds: A Multiclass Weed Species Image Dataset for Deep Learning*, Scientific Reports 9, 2058 (2019), [doi:10.1038/s41598-018-38343-3](https://doi.org/10.1038/s41598-018-38343-3) |
| Baseline tham khảo | Bài báo báo cáo ResNet-50 đạt 95,7% và Inception-v3 đạt 95,1% (chi tiết và điều kiện huấn luyện ở mục 2.3) |

Số ảnh theo lớp (Table 1 của bài báo, tổng 17.509):

| Lớp | Số ảnh | Lớp | Số ảnh |
|---|---|---|---|
| Chinee apple | 1.125 | Rubber vine | 1.009 |
| Lantana | 1.064 | Siam weed | 1.074 |
| Parkinsonia | 1.031 | Snake weed | 1.016 |
| Parthenium | 1.022 | **Negative** | **9.106** |
| Prickly acacia | 1.062 | | |

**Nơi tải:**

- Ảnh: [Zenodo, DOI 10.5281/zenodo.7939060](https://zenodo.org/record/7939059), file `images.zip`.
  - Link trực tiếp: `https://zenodo.org/records/7939060/files/images.zip?download=1`
  - MD5: `b7b30f96d466fba86016aa5a26606e0f`. Hãy kiểm tra checksum sau khi tải.
- Nhãn và các fold chia sẵn: [github.com/AlexOlsen/DeepWeeds](https://github.com/AlexOlsen/DeepWeeds), thư mục `labels/` gồm `labels.csv`, `train_subset{0-4}.csv`, `val_subset{0-4}.csv`, `test_subset{0-4}.csv` (cột `Filename, Label, Species`; chia 60/20/20).
- Ngoài ra có trong [TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/deep_weeds), nhưng nên dùng file CSV ở trên để mọi người dùng **cùng một cách chia**.

### 2.1 Quy tắc chia train / val / test (BẮT BUỘC)

Dataset được tác giả chia sẵn thành 5 fold, mỗi fold là bộ ba file CSV 60% train / 20% val / 20% test. Chia ngẫu nhiên có phân tầng theo lớp (riêng lớp `Negative` không phân tầng), không chia theo địa điểm. **Mọi sinh viên dùng cùng một cách chia để kết quả so sánh được.**

| # | Quy tắc |
|---|---|
| S1 | Dùng **fold 0**: `train_subset0.csv`, `val_subset0.csv`, `test_subset0.csv`. Tải nguyên bản từ GitHub của tác giả, **không sửa, không lọc, không chia lại**. |
| S2 | **Train** chỉ để cập nhật trọng số. **Val** dùng để chọn backbone, siêu tham số, phương pháp suy luận, checkpoint (early stopping) và khớp nhiệt độ T. **Test** chỉ để báo cáo kết quả cuối (xem 2.2). |
| S3 | **Không gộp val vào train**, kể cả ở lần huấn luyện cuối cùng. Không huấn luyện trên test. |
| S4 | Không dùng bất kỳ thông tin nào từ test để quyết định gì: siêu tham số, ngưỡng, nhiệt độ T, thống kê chuẩn hoá, chọn model, chọn phương pháp suy luận. |
| S5 | Seed chỉ thay đổi khởi tạo head, thứ tự batch và augmentation ngẫu nhiên. **Seed không được thay đổi cách chia.** |
| S6 | Fold 1–4 chỉ dùng cho điểm thưởng, và khi dùng thì dùng đủ bộ ba file của cùng một fold. |

**Kiểm tra bắt buộc trước khi train** (in kết quả ra notebook và ghi vào báo cáo):

1. Số ảnh mỗi tập và số ảnh mỗi lớp trong từng tập. Tỉ lệ kỳ vọng xấp xỉ 60/20/20, tức khoảng 10.505 / 3.502 / 3.502 ảnh (số suy ra từ tỉ lệ, **bạn ghi số đếm thật**).
2. Giao của từng cặp tập (train∩val, train∩test, val∩test) theo tên file phải **rỗng**; hợp ba tập phải bằng đúng 17.509 ảnh.
3. Mọi file trong CSV đều tồn tại trong thư mục ảnh.

Nếu số đếm lệch rõ rệt khỏi 60/20/20 (hơn khoảng 1 điểm phần trăm) hoặc giao khác rỗng, hãy báo giảng viên trước khi chạy tiếp.

### 2.2 Cách đánh giá (BẮT BUỘC)

**Tập đánh giá và thời điểm dùng:**

| Giai đoạn | Tập dùng | Mục đích |
|---|---|---|
| Sàng backbone, ablation huấn luyện, so sánh suy luận | **val** | Chọn cấu hình |
| Chọn checkpoint trong một lần chạy | **val** | Epoch có macro-F1 val cao nhất (hòa thì lấy epoch sớm hơn) |
| Khớp nhiệt độ T (temperature scaling) | **val** | Một T duy nhất, áp dụng sang test |
| **Chung kết** (GUIDE mục 5) | **test** | Chạy **đúng một lần cho mỗi seed**, trên **toàn bộ** tập test (không lấy mẫu con, không loại ảnh) |

**Tiền xử lý lúc đánh giá:** không dùng augmentation ngẫu nhiên; chỉ resize hoặc center-crop và chuẩn hoá giống hệt lúc val. `model.eval()`. Ngoại lệ chỉ là các thí nghiệm suy luận cố ý đổi tiền xử lý (TTA, độ phân giải kiểm tra); khi đó khai báo rõ trong `results.xlsx`.

**Định nghĩa chỉ số** (mọi con số trong bảng và báo cáo phải theo định nghĩa này):

| Chỉ số | Định nghĩa |
|---|---|
| **Top-1 accuracy** | Số ảnh dự đoán đúng / tổng số ảnh của tập, không trọng số theo lớp |
| **Macro-F1** (chỉ số chính) | Trung bình cộng F1 của **9 lớp**, mỗi lớp trọng số bằng nhau (ví dụ `sklearn.metrics.f1_score(average="macro")`) |
| Balanced accuracy | Trung bình recall của 9 lớp |
| Precision, recall, F1 theo lớp | Tính riêng cho từng lớp; bắt buộc báo cáo cho **Chinee apple** và **Snake weed** (hai lớp khó nhất, xem 2.3) |
| Ma trận nhầm lẫn | Số lượng ảnh, hàng là nhãn thật, cột là nhãn dự đoán |
| **ECE** | 15 bin đều theo độ tin cậy; độ tin cậy = max softmax; `ECE = Σ_m (n_m / n) · abs(acc_m − conf_m)`, với `n_m` là số ảnh trong bin m, `acc_m` và `conf_m` là accuracy và độ tin cậy trung bình của bin đó (slide trang 69) |
| Độ trễ | p50 / p95 / p99 theo GUIDE mục 4.1 |
| mean ± std | Qua **≥ 3 seed**, std mẫu (`ddof=1`). Ghi rõ số seed |

Vì dữ liệu mất cân bằng (`Negative` ≈ 52%), **top-1 accuracy bị lớp `Negative` kéo cao**. Luôn báo cáo kèm macro-F1 và chỉ số từng lớp.

**Quy trình chung kết (chi tiết ở GUIDE mục 5):** chốt cấu hình trên val, huấn luyện lại với ≥ 3 seed, chạy test một lần cho mỗi seed. Chạy cả **mốc so sánh** (công thức nền `T00` + suy luận 1-view `I00`) với cùng số seed để tính mức cải thiện. Với mỗi lần chạy test, **lưu file dự đoán** `predictions/<exp_id>_seed<k>_test.csv` gồm các cột `Filename, y_true, y_pred, p0, p1, …, p8` (xác suất softmax; thứ tự lớp theo cột `Label` của `labels.csv`). Giảng viên sẽ **tính lại chỉ số từ file này**; số trong báo cáo và xlsx phải khớp.

### 2.3 Số tham khảo từ bài báo gốc

Các số dưới đây lấy từ [bài báo gốc (Scientific Reports 2019)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6375952). Rubric dùng chúng làm mốc (xem `RUBRIC.md` mục I). Đây là số **trích dẫn**, không phải kết quả của bạn.

| Nội dung | Giá trị theo bài báo |
|---|---|
| ResNet-50, độ chính xác trung bình (weighted average, 5 fold) | **95,7%** |
| Inception-v3, độ chính xác trung bình (weighted average, 5 fold) | **95,1%** |
| ResNet-50, lớp tốt nhất | `Negative` 97,6%; Parkinsonia 97,2% |
| ResNet-50, lớp kém nhất | **Chinee apple 88,5%**; **Snake weed 88,8%** |
| Nhầm lẫn chính | 3,4% Chinee apple bị đoán thành Snake weed và 4,1% chiều ngược lại; 1,3% Parkinsonia bị đoán thành Prickly acacia |
| Suy luận ResNet-50 trên Jetson TX2 | 180 ms (TensorFlow); 53,4 ms (TensorRT) |

**Điều kiện huấn luyện của bài báo (rất khác bài lab này):** Keras, Adam, LR 1e-4 (giảm một nửa khi val loss không giảm sau 16 epoch), khởi tạo ImageNet, khoảng **100 epoch**, augmentation mạnh (xoay ±360°, scale, đổi màu, perspective), mỗi model train trung bình 13 giờ trên GTX 1080Ti.

Lưu ý khi so sánh với kết quả của bạn:

- Bài lab dùng **10–15 epoch** và công thức đơn giản hơn nhiều, nên có thể thấp hơn các số trên. Đó là bình thường.
- Bài báo ghi "weighted average accuracy"; bài lab đo top-1 accuracy không trọng số. Hai định nghĩa có thể không trùng hoàn toàn, nên so sánh chỉ mang tính tham khảo.
- Các số theo lớp của bài báo được coi là tương đương recall theo lớp khi đối chiếu (giả định, bài báo không nói rõ).
- Bài báo chia ngẫu nhiên, không theo địa điểm, nên điểm test có thể hơi lạc quan so với khi gặp địa điểm mới. Hãy nêu điều này trong phần *Hạn chế* của báo cáo.

### 2.4 Công cụ: `eval.py` và bộ khung `starter/`

**`eval.py` (đã hoàn chỉnh, chỉ cần numpy và pandas).** Mọi con số của bạn phải khớp kết quả của file này.

```bash
# Chỉ số của một cấu hình (nhiều seed): top-1, macro-F1, balanced acc, theo lớp, ECE, mean ± std
python eval.py score --pred "predictions/F01_seed*_test.csv" \
    --test-csv data/labels/test_subset0.csv --labels data/labels/labels.csv --tag F01 --out eval_out

# Tự chấm phần I của RUBRIC (chung kết so với mốc)
python eval.py grade --final "predictions/F01_seed*_test.csv" --baseline "predictions/T00_seed*_test.csv" \
    --test-csv data/labels/test_subset0.csv --labels data/labels/labels.csv
```

- `score` kiểm tra định dạng file dự đoán (đủ cột `p0..p8`, xác suất cộng bằng 1, `y_pred` đúng argmax), đối chiếu tên ảnh và nhãn với `test_subset0.csv`, rồi in bảng và lưu JSON/CSV.
- `grade` tính điểm đề xuất cho I1–I4 (và I5 nếu bạn truyền `--latency-p95-ms`). Thêm `--uncal` (dự đoán test của cùng cấu hình khi chưa temperature scaling) để chấm I4(a), và `--final-val` (dự đoán val của chung kết, đặt tên `<exp_id>_seed<k>_val.csv`) để chấm I4(b). Ngưỡng điểm là **tạm thời** và nằm ở đầu file `eval.py`.
- Lỗi định dạng làm `eval.py` thoát với mã 2 và in rõ file nào, dòng nào sai.

**Bộ khung `starter/` (pseudo-code, bạn hoàn thiện):**

| File | Bạn làm gì |
|---|---|
| `dataset.py` | Đọc CSV, kiểm tra chia dữ liệu (S1–S4), transform và augmentation, `Dataset`, `DataLoader` |
| `model.py` | Backbone qua `timm`, đóng băng, 3 nhóm tham số (slide trang 52), đếm params/GMAC |
| `losses.py` | Label smoothing, focal loss, trọng số lớp, Mixup/CutMix |
| `train.py` | Một hàm `run(cfg)` dùng chung: AMP, warmup + cosine, EMA, chọn checkpoint theo macro-F1 val, vẽ đường cong |
| `inference.py` | TTA, gộp xác suất/logit, ensemble, temperature scaling, gộp BatchNorm |
| `benchmark.py` | Đo độ trễ p50/p95/p99 đúng cách |
| `lab_day2.ipynb` | Notebook Colab/Kaggle: phần cài đặt và tải dữ liệu đã viết sẵn, các ô `TODO` là phần của bạn |

Cách dùng: chép `starter/` thành `code/` trong thư mục bài nộp của bạn, hoàn thiện các `TODO`, giữ nguyên **tên hàm và kiểu dữ liệu vào/ra** ghi trong docstring (bạn được thêm hàm, tham số, file mới). Chạy test của repo bằng `python -m unittest discover -s tests` (cần scikit-learn).

## 3. Môi trường gợi ý: Google Colab hoặc Kaggle

| Nền tảng | Ưu điểm | Lưu ý |
|---|---|---|
| **Kaggle Notebooks** (khuyên dùng) | GPU miễn phí theo hạn mức hằng tuần, phiên chạy ổn định, tắt trình duyệt vẫn chạy được bằng *Save & Run All* | Kiểm tra hạn mức GPU hiện hành trong tài khoản của bạn; cần bật Internet để tải dataset |
| **Google Colab** | Khởi động nhanh, gắn Google Drive để lưu checkpoint | GPU miễn phí không được đảm bảo và có thể bị ngắt; phải lưu checkpoint và log ra Drive thường xuyên |

Gợi ý chung:

- Dùng GPU (T4 trở lên), bật mixed precision (AMP).
- Lưu **mọi log, đường cong, logit dự đoán và checkpoint** ra ổ bền (Drive, hoặc Kaggle output) ngay sau mỗi lần chạy, để phiên bị ngắt không mất kết quả.
- Ước lượng ngân sách tính toán và mẹo tiết kiệm nằm ở [`GUIDE.md`](GUIDE.md), mục 7.

## 4. Bạn phải nộp những gì

| # | Sản phẩm | Yêu cầu tối thiểu |
|---|---|---|
| 1 | `results.xlsx` | Bảng so sánh **tất cả** thí nghiệm (backbone, training, inference, kết quả cuối, độ trễ). Cấu trúc sheet và cột ở GUIDE mục 6.1 |
| 2 | `report.md` (hoặc `report.pdf`) | Báo cáo kết luận: thiết lập, kết quả, phân tích, cấu hình tốt nhất, hạn chế. Dàn ý ở GUIDE mục 6.3 |
| 3 | `curves/` | **Ảnh biểu đồ training của từng thí nghiệm** (loss và metric theo epoch, train và val), một ảnh `.png` cho mỗi `exp_id` |
| 4 | `code/` | **Toàn bộ code** của bạn, bắt đầu từ bộ khung `starter/` đã hoàn thiện: model, dataset/augmentation, train loop, các loss, inference/TTA/ensemble, đo độ trễ, tạo bảng và biểu đồ. Dùng `eval.py` gốc, không sửa |
| 5 | `README.md` riêng của bạn | Link notebook Colab/Kaggle chạy lại được, phiên bản thư viện, lệnh/thứ tự chạy, seed đã dùng |
| 6 | `predictions/` | File dự đoán trên **test** của các cấu hình chung kết và mốc, từng seed (định dạng ở mục 2.2; tạo bằng `eval.save_predictions`). Nên kèm file dự đoán **val** của chung kết (`*_val.csv`) và bản chưa temperature scaling (`*uncal*`) để `eval.py grade` chấm được I4. Dùng để giảng viên tính lại chỉ số |

Số thí nghiệm tối thiểu (chi tiết ở GUIDE):

- **≥ 5 backbone** (có ít nhất 1 họ transformer và ít nhất 1 mạng nhẹ).
- **≥ 3 trục công thức huấn luyện**, mỗi trục thử ít nhất 2–3 giá trị (ví dụ: khởi tạo, augmentation, loss).
- **≥ 4 phương pháp suy luận** và đo độ trễ p50/p95/p99.
- Cấu hình cuối cùng chạy **≥ 3 seed**, báo cáo mean ± std.

## 5. Cách nộp bài

1. `git clone` (hoặc `git pull`) repo này để lấy bài lab.
2. Tạo thư mục bài làm của bạn, đặt đúng cấu trúc sau:

```
submissions/<mssv>_<ho_ten_khong_dau>/
├── README.md          # link Colab/Kaggle, cách chạy lại
├── results.xlsx
├── report.md          # hoặc report.pdf
├── curves/
│   ├── B01_resnet50.png
│   ├── T03_cutmix.png
│   └── ...            # mỗi exp_id một ảnh
├── predictions/
│   ├── F01_seed0_test.csv
│   └── ...            # chung kết + mốc, mỗi seed một file
└── code/
    ├── *.ipynb
    └── *.py           # model, train, inference, benchmark, ...
```

3. **Không commit** dataset (`images.zip`, ảnh) và checkpoint lớn. Chỉ commit code, `results.xlsx`, báo cáo, ảnh biểu đồ và file `predictions/` (nhỏ). Nếu cần chia sẻ checkpoint, đặt link ở README riêng của bạn.
4. Nộp bài theo cách giảng viên thông báo (ví dụ Pull Request vào repo này, hoặc nén thư mục và nộp lên hệ thống của lớp). Hạn nộp do giảng viên công bố.

## 6. Quy tắc trung thực học thuật

- Mọi con số trong `results.xlsx` và báo cáo phải **đến từ log chạy thật** của bạn, truy ngược được tới một `exp_id` và một dòng log.
- Không bịa số, không chép số từ bài báo hay slide rồi ghi như kết quả của mình. Số tham khảo từ nguồn khác phải ghi rõ là *trích dẫn* và có nguồn.
- Được thảo luận ý tưởng với bạn học, nhưng code, bảng và báo cáo phải của riêng bạn.
- Kết quả xấu, thí nghiệm thất bại, hoặc phát hiện "kỹ thuật X không giúp ích" **vẫn được điểm** nếu có phân tích tốt. Xem `RUBRIC.md`.

## 7. Tài liệu đọc trước

- Slide Day 2: chương 4 (backbone), chương 5 (huấn luyện), chương 6 (suy luận) và trang "Lab #2".
- *ResNet strikes back* (Wightman et al., arXiv:2110.00476): công thức huấn luyện quan trọng ngang kiến trúc.
- CutMix (arXiv:1905.04899), Focal Loss (arXiv:1708.02002), Temperature scaling (Guo et al., arXiv:1706.04599), FixRes (arXiv:1906.06423).
- Karpathy, *A Recipe for Training Neural Networks*: checklist gỡ lỗi khi huấn luyện không hội tụ.
