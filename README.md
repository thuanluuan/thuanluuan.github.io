# HỆ THỐNG HỖ TRỢ CHẨN ĐOÁN BỆNH LÝ LỒNG NGỰC QUA ẢNH X-QUANG
## Ứng dụng Transfer Learning (DenseNet121 + Channel Attention) cho Phân loại Đa nhãn 14 Bệnh lý

<div align="center">

![Platform](https://img.shields.io/badge/Nền_tảng-Kaggle_Kernels_(GPU_T4_×2)-blue?style=flat-square&logo=kaggle)
![Framework](https://img.shields.io/badge/Framework-TensorFlow_%2F_Keras-orange?style=flat-square&logo=tensorflow)
![Dataset](https://img.shields.io/badge/Dữ_liệu-NIH_ChestX--ray14_(112K_ảnh)-green?style=flat-square)
![Model](https://img.shields.io/badge/Mô_hình-DenseNet121_%2B_SE_Block-red?style=flat-square)
![Best AUC](https://img.shields.io/badge/Mean_AUC_(Test)-0.8267-brightgreen?style=flat-square)

**Sinh viên thực hiện:** Lưu An Thuận &nbsp;|&nbsp; **Giáo viên hướng dẫn:** Trần Văn Thiện

</div>

---

## 📑 MỤC LỤC

| # | Nội dung |
|:---:|:---|
| 1 | [Tóm tắt Dự án](#1-tóm-tắt-dự-án) |
| 2 | [Giới thiệu & Bối cảnh](#2-giới-thiệu--bối-cảnh) |
| 3 | [Dữ liệu & Phương pháp](#3-dữ-liệu--phương-pháp) |
| 4 | [Kiến trúc Mô hình](#4-kiến-trúc-mô-hình) |
| 5 | [Quy trình Thực nghiệm](#5-quy-trình-thực-nghiệm) |
| 6 | [Kết quả & Đánh giá](#6-kết-quả--đánh-giá) |
| 7 | [Phân tích & Thảo luận](#7-phân-tích--thảo-luận) |
| 8 | [Kết luận & Hướng phát triển](#8-kết-luận--hướng-phát-triển) |
| 9 | [Tài liệu tham khảo](#9-tài-liệu-tham-khảo) |

---

## 1. TÓM TẮT DỰ ÁN

Dự án xây dựng hệ thống trí tuệ nhân tạo hỗ trợ chẩn đoán tự động 14 bệnh lý lồng ngực từ ảnh X-quang, sử dụng phương pháp **Transfer Learning thuần** dựa trên kiến trúc **DenseNet121** kết hợp **Channel Attention (SE Block)**. Hệ thống được huấn luyện và đánh giá trên bộ dữ liệu chuẩn quốc tế **NIH Chest X-ray 14** (112,120 ảnh).

### Điểm nổi bật chính

| Tiêu chí | Kết quả |
|:---|:---:|
| Mean AUC trên tập Test (tốt nhất) | **0.8267** |
| Số bệnh vượt Baseline Wang et al. (2017) | **14 / 14** ✅ |
| Số bệnh vượt CheXNet SOTA (Stanford, 2017) | **5 / 14** 🏆 |
| Khoảng cách với CheXNet (Mean AUC) | **−0.015** |
| Bệnh đạt AUC cao nhất | **Hernia — 0.946** |
| Dữ liệu xác nhận (Validation Leakage) | **Không** — tách theo Patient ID |
| Khả năng giải thích (Explainability) | ✅ Gradient Saliency Map |

---

## 2. GIỚI THIỆU & BỐI CẢNH

### 2.1. Vấn đề đặt ra

X-quang lồng ngực là xét nghiệm hình ảnh phổ biến nhất trong y tế, được chỉ định hàng ngày tại các cơ sở khám chữa bệnh. Tuy nhiên, việc đọc và phân tích phim X-quang đang đối mặt với ba thách thức lớn:

- **Thiếu hụt chuyên gia:** Số lượng bác sĩ X-quang không đáp ứng được khối lượng phim ngày càng tăng, đặc biệt tại các vùng xa, nước đang phát triển.
- **Nguy cơ sai sót:** Chẩn đoán thủ công chịu ảnh hưởng của yếu tố chủ quan, mệt mỏi và quá tải — dễ bỏ sót tổn thương vi mô hoặc chẩn đoán nhầm.
- **Độ phức tạp đa bệnh:** Trên thực tế, bệnh nhân thường mắc nhiều bệnh lý đồng thời (multi-label), khiến bài toán phân loại khó hơn so với đơn nhãn truyền thống.

### 2.2. Mục tiêu dự án

Xây dựng một hệ thống AI có khả năng:

1. **Phân loại đa nhãn đồng thời** 14 bệnh lý lồng ngực phổ biến chỉ từ một ảnh X-quang duy nhất.
2. **Đạt hiệu suất cạnh tranh** với các công bố quốc tế trên cùng bộ dữ liệu chuẩn NIH, đặc biệt vượt Baseline Wang et al. (2017) và tiệm cận CheXNet (Stanford, 2017).
3. **Hỗ trợ lâm sàng thực tế** với 3 chế độ ngưỡng phân loại khác nhau, phù hợp từng bối cảnh sử dụng: chẩn đoán chính xác, tối ưu F1, và sàng lọc diện rộng.
4. **Đảm bảo tính giải thích (Explainability)** thông qua Gradient Saliency Map — trực quan hóa vùng tổn thương mà mô hình tập trung, tăng tin cậy cho bác sĩ.

---

## 3. DỮ LIỆU & PHƯƠNG PHÁP

### 3.1. Bộ dữ liệu NIH Chest X-ray 14

| Thông số | Chi tiết |
|:---|:---|
| Nguồn | National Institutes of Health (NIH), Mỹ |
| Quy mô | 112,120 ảnh X-quang từ 30,805 bệnh nhân |
| Định dạng | PNG, Grayscale (chuyển sang RGB 3 kênh khi đưa vào mô hình) |
| Kích thước gốc | 1,024 × 1,024 pixels |
| Số nhãn | 14 bệnh lý + 1 nhãn "No Finding" (Bình thường) |
| Loại nhãn | **Multi-label** — một ảnh có thể chứa nhiều bệnh đồng thời |
| Nhãn phân phối | Mất cân bằng nghiêm trọng (No Finding: 60,361 vs Hernia: 227) |

**14 bệnh lý mục tiêu:**

| STT | Tiếng Anh | Tiếng Việt | Số ca |
|:---:|:---|:---|---:|
| 1 | Atelectasis | Xẹp phổi | 11,559 |
| 2 | Cardiomegaly | Tim to | 2,776 |
| 3 | Effusion | Tràn dịch màng phổi | 13,317 |
| 4 | Infiltration | Thâm nhiễm phổi | 19,894 |
| 5 | Mass | Khối u | 5,782 |
| 6 | Nodule | Nốt phổi | 6,331 |
| 7 | Pneumonia | Viêm phổi | 1,431 |
| 8 | Pneumothorax | Tràn khí màng phổi | 5,302 |
| 9 | Consolidation | Đông đặc phổi | 4,667 |
| 10 | Edema | Phù phổi | 2,303 |
| 11 | Emphysema | Khí phế thũng | 2,516 |
| 12 | Fibrosis | Xơ phổi | 1,686 |
| 13 | Pleural_Thickening | Dày màng phổi | 3,385 |
| 14 | Hernia | Thoát vị hoành | 227 |

### 3.2. Tiền xử lý & Tăng cường dữ liệu

**Tiền xử lý (Preprocessing):**
- Resize ảnh: **224 × 224** cho Phase 1 & 2, nâng lên **384 × 384** ở Phase 3 (Deep Fine-tuning).
- Chuyển đổi Grayscale → **RGB 3 kênh** để tương thích với trọng số ImageNet pre-trained.
- Chuẩn hóa pixel về `[0, 1]` bằng Rescaling `1./255`.

**Tăng cường dữ liệu (Data Augmentation) — chỉ áp dụng trên tập Train:**

| Kỹ thuật | Tham số | Lý do lâm sàng |
|:---|:---:|:---|
| Rotation | ±10° | Mô phỏng góc chụp nhẹ lệch |
| Width / Height Shift | ±10% | Mô phỏng bệnh nhân không đứng chính giữa |
| Zoom | ±15% | Mô phỏng khoảng cách chụp khác nhau |
| Brightness | [0.85 – 1.15] | Mô phỏng máy chụp cũ/mới (thiếu sáng / dư sáng) |
| Fill mode | `constant`, `cval=0` | Viền đen tự nhiên, không tạo artifact |
| Horizontal Flip | **Tắt** | X-quang có tính đối xứng tim-phổi → lật sai vị trí tim |
| Vertical Flip | **Tắt** | Tuyệt đối không lật dọc ảnh y tế |

### 3.3. Chia tập dữ liệu — Chống Data Leakage

> ⚠️ **Nguyên tắc vàng:** Tách biệt tuyệt đối theo **Patient ID**, không theo ảnh. Một bệnh nhân có thể có nhiều ảnh — nếu chia ngẫu nhiên theo ảnh, thông tin bệnh nhân sẽ rò rỉ từ Train sang Test, làm phồng kết quả một cách giả tạo.

| Tập | Tỷ lệ | Mục đích |
|:---:|:---:|:---|
| Train | ~80% | Huấn luyện mô hình |
| Validation | ~10% | Theo dõi quá trình train, tối ưu ngưỡng |
| Test | ~10% | Đánh giá hiệu năng cuối, **không tham gia huấn luyện** |

---

## 4. KIẾN TRÚC MÔ HÌNH

Dự án áp dụng phương pháp **Transfer Learning thuần** — toàn bộ sức mạnh đặc trưng đến từ DenseNet121 đã được pre-trained, không kết hợp hay thay thế bởi CNN tùy chỉnh từ đầu.

### 4.1. Tại sao chọn Transfer Learning với DenseNet121?

**Về lý thuyết:**
- **Dense Connections:** DenseNet121 kết nối trực tiếp mỗi lớp với tất cả các lớp phía sau trong cùng Dense Block. Điều này cho phép **tái sử dụng đặc trưng (feature reuse)** tối đa — đặc biệt quan trọng với ảnh y tế khi các đặc trưng vi mô (nốt nhỏ, mờ mờ) cần được bảo toàn qua nhiều tầng.
- **Gradient Flow tốt:** Dense Connections giảm thiểu vấn đề vanishing gradient, giúp mô hình sâu hội tụ ổn định hơn.

**Về thực nghiệm:**
- DenseNet121 là kiến trúc nền tảng của **CheXNet (Stanford, 2017)** — mô hình đầu tiên vượt qua mức chẩn đoán của bác sĩ X-quang trên bộ NIH, xác nhận tính phù hợp của kiến trúc này với bài toán.
- **Trọng số ImageNet** cung cấp nền tảng đặc trưng cấp thấp (cạnh, kết cấu, hình dạng) — khi fine-tune trên X-quang, mô hình không cần học lại từ đầu mà chỉ cần **điều chỉnh** đặc trưng cấp cao theo miền y tế.

### 4.2. Kiến trúc chi tiết

```
Input (Batch, 224×224×3)  ←── Phase 1 & 2
Input (Batch, 384×384×3)  ←── Phase 3 (Deep Fine-tuning)
        │
        ▼
┌──────────────────────────────────────────────┐
│              BACKBONE                        │
│   DenseNet121 (pre-trained ImageNet)         │
│   • 4 Dense Blocks + 3 Transition Layers     │
│   • training=False → BN dùng ImageNet stats  │
│   • Phase 1: frozen toàn bộ                  │
│   • Phase 2: mở 100 layers cuối              │
│   • Phase 3: mở toàn bộ                      │
│   Output: Feature Map (H, W, 1024)           │
└──────────────────┬───────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────┐
│        CHANNEL ATTENTION — SE Block          │
│                                              │
│   Feature Map (H, W, 1024)                   │
│         │                                    │
│   GlobalAvgPool2D → (1024,)                  │
│         │                                    │
│   Reshape → (1, 1, 1024)                     │
│         │                                    │
│   Dense(64, ReLU, no bias)                   │
│         │                                    │
│   Dense(1024, Sigmoid, no bias)              │
│         │                                    │
│   Multiply(Feature Map × Attention Weight)   │
│         │                                    │
│   Output: Re-weighted Map (H, W, 1024)       │
└──────────────────┬───────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────┐
│           CLASSIFICATION HEAD                │
│                                              │
│   GlobalAveragePooling2D  → (1024,)          │
│   BatchNormalization                         │
│                                              │
│   Dense(512, L2=1e-4)                        │
│   LeakyReLU(α=0.1)                           │
│   Dropout(0.5)                               │
│                                              │
│   Dense(256, L2=1e-4)                        │
│   LeakyReLU(α=0.1)                           │
│   Dropout(0.4)                               │
│                                              │
│   Dense(15)                                  │
│   Sigmoid [float32]  → Output (15,)          │
└──────────────────────────────────────────────┘
        │
        ▼
  Xác suất 15 nhãn ∈ [0, 1]
  (14 bệnh lý + No Finding)
```

**Giải thích từng thành phần:**

| Thành phần | Tham số | Vai trò | Lý do thiết kế |
|:---|:---|:---|:---|
| DenseNet121 (Backbone) | `training=False` | Trích xuất đặc trưng không gian | Tái sử dụng tri thức ImageNet; BN giữ nguyên running stats để ổn định |
| SE Block (Attention) | ratio=16, no bias | Cân trọng số kênh đặc trưng | Tập trung kênh liên quan tổn thương, triệt tiêu nhiễu |
| GlobalAvgPooling2D | — | Tổng hợp đặc trưng toàn cục | Ít tham số hơn Flatten, tránh overfit |
| BatchNormalization | — | Chuẩn hóa phân phối | Ổn định gradient khi fine-tune |
| Dense(512) + LeakyReLU(α=0.1) | L2=1e-4 | Học kết hợp đặc trưng cấp cao | LeakyReLU tránh dying ReLU; L2 chống overfit |
| Dropout(0.5) | — | Chính quy hóa lớp 1 | Giảm co-adaptation giữa các neuron |
| Dense(256) + LeakyReLU(α=0.1) | L2=1e-4 | Tinh chỉnh biểu diễn | Thu hẹp dần chiều → tổng quát hóa tốt hơn |
| Dropout(0.4) | — | Chính quy hóa lớp 2 | Nhẹ hơn lớp trước vì đã thu hẹp chiều |
| Dense(15, **Sigmoid**) | dtype=float32 | Đầu ra đa nhãn | Sigmoid vì 15 nhãn **độc lập**; float32 bắt buộc khi Mixed Precision FP16 |

### 4.3. Chiến lược Fine-Tuning 3 Giai đoạn

Huấn luyện theo lộ trình từ đơn giản đến phức tạp để tránh phá vỡ trọng số pre-trained:

| Giai đoạn | Cell | Lớp được train | Input Size | Learning Rate | Số Epoch | Điểm đặc biệt |
|:---:|:---:|:---|:---:|:---:|:---:|:---|
| **Phase 1** — Feature Extraction | Cell 29 | Chỉ Classification Head (DenseNet121 **frozen** hoàn toàn) | 224×224 | `1e-4` | 20 | BN backbone dùng ImageNet stats (`training=False`) |
| **Phase 2** — Fine-tuning | Cell 32 | Head + **100 layers cuối** DenseNet121 | 224×224 | `1e-4` | 20 | Mở băng dần từ cuối lên, BN layers giữ frozen |
| **Phase 3** — Deep Fine-tuning | Cell 32+ | Toàn bộ mô hình | **384×384** | `5e-6` | 15 | Mixed Precision FP16, Dual GPU (MirroredStrategy) |

---

## 5. QUY TRÌNH THỰC NGHIỆM

### 5.1. Môi trường thực nghiệm

| Thành phần | Cấu hình |
|:---|:---|
| Nền tảng | Kaggle Notebooks |
| GPU | NVIDIA Tesla T4 × 2 |
| Framework | TensorFlow 2.x / Keras |
| Input size | 384 × 384 × 3 |
| Batch size | 32 |
| Loss function | **Asymmetric Focal Loss** (custom) — γ_pos=0.5, γ_neg=2.5, α=0.80, label smoothing=0.05 + No Finding penalty |
| Metric chính | AUC (Area Under ROC Curve) — Multi-label |
| Optimizer | Adam |

### 5.2. Cơ chế Auto-Resume (Xử lý Timeout 12 giờ Kaggle)

Kaggle giới hạn mỗi phiên GPU 12 giờ — không đủ cho toàn bộ quá trình fine-tuning. Giải pháp **Checkpoint + Auto-Resume** được tích hợp:

- **Lưu tự động** sau mỗi epoch: `best_deep_model.keras` (theo Val AUC) và `last_deep_model.keras` (mới nhất).
- **Phát hiện checkpoint:** Khi khởi động lại, notebook tự kiểm tra file checkpoint và resume đúng epoch.
- **Callbacks:** `ModelCheckpoint` + `EarlyStopping (patience=5)` + `ReduceLROnPlateau (factor=0.5, patience=3)`.

### 5.3. Tối ưu Ngưỡng Phân loại (Threshold Optimization)

Sau khi huấn luyện, xác suất đầu ra cần được chuyển thành nhãn nhị phân (0/1) dựa trên một ngưỡng. Thay vì dùng ngưỡng cố định 0.5, dự án **tối ưu ngưỡng riêng biệt cho từng bệnh** trên tập Validation theo 3 chiến lược:

| Chế độ | Chiến lược tối ưu | Đặc điểm | Phù hợp với |
|:---:|:---|:---|:---|
| **Dạng 1 — Precise** | Youden Index (max Sensitivity + Specificity − 1) | Cân bằng tốt giữa phát hiện và độ chính xác | Chẩn đoán hỗ trợ tổng quát |
| **Dạng 2 — Advanced** | Tối ưu F1-score + Rare disease adjust | Cân bằng Precision–Recall, F1 cao nhất | Môi trường muốn giảm False Positive |
| **Dạng 3 — Screening** | Tối ưu Recall ≥ 0.75 | Ưu tiên không bỏ sót bệnh (Recall cao) | Sàng lọc cộng đồng, tầm soát |

> **Xử lý đặc biệt cho bệnh hiếm (Rare Disease Adjustment):** Hernia, Pneumonia, Emphysema, Fibrosis có số mẫu rất ít → ngưỡng được hạ thêm 15% để tăng độ nhạy phát hiện.

### 5.4. Suy diễn & Giải thích (Inference & Explainability)

- **Test-Time Augmentation (TTA):** Dự đoán nhiều lần với các biến đổi nhỏ của ảnh, lấy trung bình xác suất — giảm phương sai, tăng độ ổn định.
- **Ensemble (Dạng 2 — Advanced):** Kết hợp có trọng số 2 checkpoint tốt nhất — `best_deep_model` (w=0.6) + `best_recall_deep_model` (w=0.4) — ưu tiên mô hình AUC cao nhưng vẫn giữ tỷ trọng cho mô hình recall tốt.
- **Gradient Saliency Map:** Tính đạo hàm của đầu ra so với pixel đầu vào (Percentile Normalization + Gaussian Blur) để trực quan hóa vùng tổn thương mô hình tập trung — giúp bác sĩ kiểm chứng quyết định của AI.

---

## 6. KẾT QUẢ & ĐÁNH GIÁ

### 6.1. Kết quả tổng quan 3 chế độ (Tập Test)

| Chế độ | **Mean AUC** ↑ | Mean F1 | Mean Recall | Mean Specificity |
|:---|:---:|:---:|:---:|:---:|
| Dạng 1 — Precise | 0.8264 | 0.2475 | 0.7727 | 0.7318 |
| **Dạng 2 — Advanced** | **0.8267** | **0.3306** | 0.5210 | **0.8597** |
| Dạng 3 — Screening | 0.8260 | 0.1679 | **0.9210** | 0.3287 |

> 🏆 **Kết quả tốt nhất:** Dạng 2 — Advanced đạt **Mean AUC = 0.8267** trên tập Test, đây là con số được sử dụng để so sánh với các nghiên cứu quốc tế.

### 6.2. So sánh AUC từng bệnh với nghiên cứu quốc tế

**Đối chiếu với 2 mốc quan trọng:**
- **Wang et al. (2017)** — Nhóm tác giả NIH, công bố bộ dữ liệu, kết quả Baseline gốc.
- **CheXNet (2017, Stanford)** — Mô hình SOTA, được train bởi đội ngũ của Andrew Ng, sử dụng cùng kiến trúc DenseNet121 trên quy mô lớn hơn.

| STT | Bệnh lý | Tên Việt | Wang et al. | CheXNet | **Dự án này** | vs. Wang | vs. CheXNet |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|
| 1 | Atelectasis | Xẹp phổi | 0.7003 | 0.8094 | **0.807** | `+0.107` ✅ | `−0.002` ≈ |
| 2 | Cardiomegaly | Tim to | 0.8100 | 0.9248 | **0.899** | `+0.089` ✅ | `−0.026` |
| 3 | Effusion | Tràn dịch màng phổi | 0.7593 | 0.8638 | **0.875** | `+0.116` ✅ | `+0.011` 🏆 |
| 4 | Infiltration | Thâm nhiễm phổi | 0.6614 | 0.7345 | **0.714** | `+0.053` ✅ | `−0.021` |
| 5 | Mass | Khối u | 0.6933 | 0.8676 | **0.818** | `+0.125` ✅ | `−0.050` |
| 6 | Nodule | Nốt phổi | 0.6687 | 0.7802 | **0.752** | `+0.083` ✅ | `−0.028` |
| 7 | Pneumonia | Viêm phổi | 0.6580 | 0.7680 | **0.749** | `+0.091` ✅ | `−0.019` |
| 8 | Pneumothorax | Tràn khí màng phổi | 0.7993 | 0.8887 | **0.870** | `+0.071` ✅ | `−0.019` |
| 9 | Consolidation | Đông đặc phổi | 0.7032 | 0.8027 | **0.805** | `+0.102` ✅ | `+0.002` 🏆 |
| 10 | Edema | Phù phổi | 0.8052 | 0.8878 | **0.898** | `+0.093` ✅ | `+0.010` 🏆 |
| 11 | Emphysema | Khí phế thũng | 0.8330 | 0.9371 | **0.917** | `+0.084` ✅ | `−0.020` |
| 12 | Fibrosis | Xơ phổi | 0.7859 | 0.8047 | **0.809** | `+0.023` ✅ | `+0.004` 🏆 |
| 13 | Pleural_Thickening | Dày màng phổi | 0.6835 | 0.8062 | **0.777** | `+0.094` ✅ | `−0.029` |
| 14 | Hernia | Thoát vị hoành | 0.8717 | 0.9164 | **0.946** | `+0.074` ✅ | `+0.030` 🏆 |
| | **MEAN AUC** | | **0.7381** | **0.8412** | **0.8267** | **`+0.089` ✅** | **`−0.015`** |

### 6.3. Kết quả chi tiết theo từng bệnh (Tập Test — Dạng 2 Advanced)

| Bệnh lý | AUC | F1 | Recall |
|:---|:---:|:---:|:---:|
| Xẹp phổi | 0.802 | 0.396 | 0.529 |
| Tim to | 0.897 | 0.329 | 0.410 |
| Tràn dịch màng phổi | 0.875 | 0.503 | **0.785** 🏆 |
| Thâm nhiễm phổi | 0.714 | 0.422 | 0.519 |
| Khối u | 0.818 | 0.344 | 0.439 |
| Nốt phổi | 0.751 | 0.299 | 0.257 |
| Viêm phổi | 0.747 | 0.044 | **0.785** 🏆 |
| Tràn khí màng phổi | 0.870 | 0.304 | **0.769** 🏆 |
| Đông đặc phổi | 0.804 | 0.247 | 0.382 |
| Phù phổi | 0.884 | 0.256 | 0.431 |
| Khí phế thũng | 0.917 | 0.396 | 0.584 |
| Xơ phổi | 0.809 | 0.142 | 0.354 |
| Dày màng phổi | 0.776 | 0.190 | 0.351 |
| Thoát vị hoành | **0.945** | 0.339 | 0.312 |

> 🏆 Các bệnh đạt Recall ≥ 0.75 được đánh dấu — quan trọng trong ngữ cảnh lâm sàng (bệnh nguy hiểm không được bỏ sót).
>
> *Specificity chỉ có ở mức trung bình toàn bộ — xem bảng tổng quan mục 6.1.*

---

## 7. PHÂN TÍCH & THẢO LUẬN

### 7.1. Điểm mạnh của mô hình

**① Vượt Baseline trên toàn bộ 14/14 bệnh (+8.9% AUC trung bình)**
Đây là minh chứng rõ ràng nhất cho hiệu quả của Transfer Learning so với phương pháp xây dựng CNN truyền thống từ đầu mà Wang et al. (2017) áp dụng. Khoảng cách trung bình **+0.089 AUC** cho thấy tri thức từ ImageNet có giá trị chuyển giao rất cao sang miền ảnh y tế.

**② Cạnh tranh với CheXNet mặc dù giới hạn tài nguyên**
CheXNet được phát triển bởi đội ngũ Stanford ML Group với cơ sở hạ tầng tính toán quy mô lớn, trong khi dự án này thực hiện hoàn toàn trên **GPU miễn phí của Kaggle (T4)**. Khoảng cách chỉ **−0.015 AUC** là kết quả rất đáng kể.

**③ Vượt CheXNet trên 5 bệnh lý quan trọng**
- **Hernia (+0.030):** Dạng bệnh hiếm, thường khó phân loại — kết quả AUC 0.946 cho thấy kết hợp SE Block + Rare Adjustment ngưỡng phát huy tốt.
- **Effusion (+0.011) & Edema (+0.010):** Hai bệnh lý liên quan dịch — SE Block có thể đã học cách chú ý đặc biệt đến vùng góc sườn hoành và cung tim.
- **Consolidation (+0.002) & Fibrosis (+0.004):** Hai bệnh có đặc trưng hình thái tương đồng — Attention giúp phân biệt tốt hơn.

**④ Khả năng giải thích bằng Gradient Saliency Map**
Mô hình không chỉ cho ra con số AUC mà còn trực quan hóa được **vùng tổn thương** mà nó tập trung — đây là yếu tố thiết yếu để bác sĩ có thể tin tưởng và kiểm chứng quyết định của AI trong môi trường lâm sàng.

### 7.2. Hạn chế và hướng cải thiện

| Hạn chế | Phân tích | Hướng cải thiện đề xuất |
|:---|:---|:---|
| **Infiltration (0.714)** thấp nhất | Đặc trưng hình ảnh mờ, chồng lấp nhiều bệnh | Augmentation chuyên biệt, tăng trọng số class |
| **Mass (−0.050 vs CheXNet)** | Khối u đòi hỏi đặc trưng cục bộ cao | Object detection head, ROI Pooling |
| **F1 thấp nhìn chung** | Mất cân bằng nhãn nghiêm trọng | Class-Balanced Sampling, tăng RARE_THR_SCALE |
| **Viêm phổi F1 = 0.044** | Dữ liệu Pneumonia cực ít (1,431 ca), dù Recall cao | Few-shot learning, bổ sung external data |
| Tài nguyên GPU giới hạn | Không thể train kiến trúc lớn hơn | Gradient checkpointing, TPU Kaggle |

---

## 8. KẾT LUẬN & HƯỚNG PHÁT TRIỂN

### 8.1. Kết luận

Dự án đã thành công xây dựng hệ thống phân loại đa nhãn 14 bệnh lý lồng ngực từ ảnh X-quang bằng phương pháp **Transfer Learning thuần** (DenseNet121 + Channel Attention SE Block), đạt **Mean AUC = 0.8267** trên tập Test độc lập.

Kết quả đạt được **vượt toàn bộ 14/14 bệnh** so với Baseline Wang et al. (2017) — chứng minh tính hiệu quả rõ ràng của Transfer Learning. Đặc biệt, dự án **vượt CheXNet trên 5 bệnh** và chỉ kém **0.015 AUC trung bình** so với mô hình SOTA của Stanford — một kết quả rất cạnh tranh trong điều kiện sử dụng tài nguyên tính toán miễn phí.

Hệ thống đã hoàn thiện đầy đủ pipeline từ tiền xử lý, huấn luyện, đánh giá đến triển khai với 3 chế độ phân loại lâm sàng khác nhau và khả năng giải thích qua Gradient Saliency Map.

### 8.2. Hướng phát triển tương lai

- **Backbone nâng cao:** Thử nghiệm EfficientNetV2, ConvNeXt, hoặc Vision Transformer (ViT) — có thể cải thiện thêm 2–4% AUC.
- **Self-supervised pre-training:** Sử dụng CheXpert hoặc MIMIC-CXR để pre-train trực tiếp trên ảnh X-quang trước khi fine-tune — gần hơn với miền dữ liệu đích.
- **Xử lý mất cân bằng nâng cao:** Áp dụng Class-Balanced Sampling hoặc MixUp Augmentation để cải thiện thêm các bệnh hiếm (Hernia, Pneumonia).
- **Localization:** Kết hợp CAM (Class Activation Mapping) để dự đoán không chỉ nhãn bệnh mà còn **khoanh vùng tổn thương** cụ thể trên ảnh.
- **Web API / Mobile App:** Đóng gói mô hình thành REST API hoặc ứng dụng di động để bác sĩ có thể sử dụng ngay tại điểm chăm sóc.

---

## 9. TÀI LIỆU THAM KHẢO

1. **Wang, X. et al.** (2017). *ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks*. IEEE CVPR. — Công bố bộ dữ liệu NIH và Baseline.

2. **Rajpurkar, P. et al.** (2017). *CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning*. Stanford ML Group. — Mô hình SOTA DenseNet121.

3. **Huang, G. et al.** (2017). *Densely Connected Convolutional Networks (DenseNet)*. IEEE CVPR. — Kiến trúc nền tảng của mô hình.

4. **Hu, J. et al.** (2018). *Squeeze-and-Excitation Networks*. IEEE CVPR. — Cơ sở lý thuyết Channel Attention (SE Block).

5. **Simonyan, K. & Zisserman, A.** (2014). *Very Deep Convolutional Networks for Large-Scale Image Recognition*. ICLR. — Nền tảng Transfer Learning từ ImageNet.

---

<div align="center">

**© 2025 — Lưu An Thuận | Giáo viên hướng dẫn: Trần Văn Thiện**

</div>
