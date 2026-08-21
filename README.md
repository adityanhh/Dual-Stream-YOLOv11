# Dual-Stream YOLOv11 (DE-YOLOv11): Arsitektur Deteksi Objek Multi-Modalitas (RGB + IR)

Repositori ini berisi implementasi lengkap arsitektur **Dual-Stream YOLOv11** dengan masukan ganda (*dual input*: citra tampak/Visible RGB dan citra termal/Infrared IR) yang mengadaptasi dan mengembangkan konsep **DE-YOLO** (*Dual-Feature-Enhancement YOLO*, ICPR 2024) ke dalam basis arsitektur state-of-the-art **YOLOv11** (Ultralytics).

---

## 🌟 Fitur Utama & Inovasi Arsitektur

1. **Dual-Stream Backbone Paralel:**
   - **Stream 1 (Visible RGB):** Mengekstraksi informasi detail spasial, tekstur, dan warna.
   - **Stream 2 (Infrared IR):** Mengekstraksi radiasi termal objek, kontras siluet, dan ketahanan dalam kondisi minim cahaya, silau, atau asap.
2. **`C3k2_BiFocus` (Bi-directional Decoupled Focus untuk YOLOv11):**
   - Mengintegrasikan modul `FocusH` (horizontal) dan `FocusV` (vertikal) dengan *DepthWise Separable Convolution* pada blok `C3k2` level $P_2$ untuk memperluas *effective receptive field*.
3. **`C2PSA` (Pointwise Spatial Attention):**
   - Mempertahankan modul atensi spasial global bawaan YOLOv11 pada stage terdalam ($P_5$) pada kedua stream.
4. **`DEA` (Dual-context Collaborative Enhancement Attention):**
   - **`DECA` (*Dual Semantic Enhancing Channel Weight Assignment*):** Mempelajari korelasi antar-kanal lintas modalitas menggunakan MLP dan piramida konvolusi multi-skala.
   - **`DEPA` (*Dual Spatial Enhancing Pixel Weight Assignment*):** Mempelajari representasi dependensi spasial/posisi piksel lintas modalitas dengan kernel $3\times 3$ dan $7\times 7$.
5. **Modern PANet Neck & Decoupled Detect Head:**
   - Agregasi fitur multi-skala berbasis `C3k2` dan deteksi decoupled head yang efisien.
6. **Pretrained Weight Transfer:**
   - Mendukung transfer bobot dari checkpoint resmi `yolo11n.pt`, `yolo11s.pt`, dsb. ke kedua cabang backbone sekaligus.

---

## 📁 Struktur Direktori Proyek

```text
DualStreamYOLO11/
├── configs/                          # Konfigurasi Model & Dataset
│   ├── yolo11-dualstream.yaml        # Master model config
│   ├── yolo11n-dualstream.yaml       # Varian Nano (6.19M parameter)
│   ├── yolo11s-dualstream.yaml       # Varian Small
│   ├── yolo11m-dualstream.yaml       # Varian Medium
│   ├── yolo11l-dualstream.yaml       # Varian Large
│   ├── M3FD.yaml                     # Template dataset M3FD (6 kelas)
│   └── LLVIP.yaml                    # Template dataset LLVIP (1 kelas: person)
├── modules/                          # Modul Neural Network Kustom
│   ├── bifocus.py                    # FocusH, FocusV, DepthWiseConv, BiFocus
│   ├── c3k2_bifocus.py               # C3k2_BiFocus (integrasi C3k2 + BiFocus)
│   └── dea.py                        # DECA, DEPA, dan DEA Cross-Modality Fusion
├── models/                           # Core Model Definition
│   └── dualstream_model.py           # DualStreamDetectionModel & DualStreamYOLO
├── data/                             # Dataloader & Sinkronisasi Augmentasi
│   ├── augment.py                    # Synchronized Letterbox, Flip, Affine
│   └── dataset.py                    # DualStreamDataset & collate function
├── engine/                           # Training & Inference Loop
│   ├── trainer.py                    # DualStreamTrainer (AMP, EMA, Cosine LR, Best Save)
│   └── predictor.py                  # DualStreamPredictor (Inference, NMS, Dual Viz)
├── train.py                          # CLI script untuk pelatihan model
├── predict.py                        # CLI script untuk prediksi & visualisasi
├── test_dualstream.py                # Uji forward/backward/strides model
├── test_end_to_end.py                # Uji integrasi pipeline end-to-end
└── README.md                         # Dokumentasi teknis & panduan riset
```

---

## 📊 Detail Spesifikasi Model (YOLO11n-DualStream)

| Properti | Nilai |
| :--- | :--- |
| **Total Layers** | 38 layers (388 sequential modules) |
| **Total Parameter** | 6.191.628 (6.19 M) |
| **Trainable Parameters** | 6.191.612 |
| **Multi-Scale Output Strides** | $P_3$ (stride 8), $P_4$ (stride 16), $P_5$ (stride 32) |
| **Dimensi Output Deteksi (640x640)** | `[Batch, 4 + nc, 8400]` anchors |

---

## 🚀 Panduan Penggunaan

### 1. Struktur Dataset
Siapkan dataset berpasangan (misal M3FD atau LLVIP) dengan struktur direktori berikut:

```text
datasets/M3FD/
├── images/
│   ├── vis_train/ (1.jpg, 2.jpg, ...)
│   ├── vis_val/   (100.jpg, 101.jpg, ...)
│   ├── Ir_train/  (1.jpg, 2.jpg, ...)
│   └── Ir_val/    (100.jpg, 101.jpg, ...)
└── labels/
    ├── vis_train/ (1.txt, 2.txt, ...)
    └── vis_val/   (100.txt, 101.txt, ...)
```

Sesuaikan path di `DualStreamYOLO11/configs/M3FD.yaml` atau `LLVIP.yaml`.

---

### 2. Pelatihan Model (Training)

Jalankan perintah berikut pada terminal:

```bash
# Training dari awal (from scratch)
python DualStreamYOLO11/train.py \
    --model DualStreamYOLO11/configs/yolo11n-dualstream.yaml \
    --data DualStreamYOLO11/configs/M3FD.yaml \
    --epochs 100 \
    --batch 8 \
    --imgsz 640 \
    --device 0

# Training dengan transfer bobot pretrained YOLO11
python DualStreamYOLO11/train.py \
    --model DualStreamYOLO11/configs/yolo11n-dualstream.yaml \
    --data DualStreamYOLO11/configs/M3FD.yaml \
    --pretrained yolo11n.pt \
    --epochs 100 \
    --batch 8 \
    --imgsz 640 \
    --device 0
```

Hasil bobot terbaik akan otomatis tersimpan di `runs/train/exp/best.pt`.

---

### 3. Prediksi & Visualisasi (Inference)

Jalankan inferensi pada pasangan citra RGB dan IR:

```bash
python DualStreamYOLO11/predict.py \
    --weights runs/train/exp/best.pt \
    --model DualStreamYOLO11/configs/yolo11n-dualstream.yaml \
    --vis-img path/to/visible_image.jpg \
    --ir-img path/to/infrared_image.jpg \
    --conf 0.25 \
    --save-dir runs/predict
```

Hasil visualisasi akan disimpan dalam 3 format:
- `*_vis_pred.jpg`: Deteksi pada citra Visible RGB
- `*_ir_pred.jpg`: Deteksi pada citra Infrared IR
- `*_dual_pred.jpg`: Tampilan perbandingan berdampingan (*side-by-side*)

---

## 🧪 Verifikasi & Unit Testing

Anda dapat menjalankan pengujian otomatis kapan saja dengan menjalankan:

```bash
# 1. Uji arsitektur model, forward pass, dimensi output, dan backward gradients:
python DualStreamYOLO11/test_dualstream.py

# 2. Uji integrasi end-to-end (dataset sintetis -> training 1 epoch -> checkpoint save -> inferensi):
python DualStreamYOLO11/test_end_to_end.py
```

---

## 📖 Landasan Teori untuk Laporan Tugas Akhir (Skripsi)

### Formulasi Matematis DEA (Dual-context Collaborative Enhancement):
1. **DECA (Channel Attention):**
   $$w_{vi} = \sigma\left(W_2 \cdot \text{ReLU}(W_1 \cdot \text{GAP}(x_{vi}))\right)$$
   $$w_{ir} = \sigma\left(W_2 \cdot \text{ReLU}(W_1 \cdot \text{GAP}(x_{ir}))\right)$$
   $$G = \text{ConvPyramid}(\text{Conv}_{3\times 3}([x_{vi}, x_{ir}]))$$
   $$\hat{x}_{vi} = x_{vi} \odot \sigma(w_{ir} \odot G), \quad \hat{x}_{ir} = x_{ir} \odot \sigma(w_{vi} \odot G)$$

2. **DEPA (Spatial Pixel Attention):**
   $$S_{vi} = \text{Conv}_{5\times 5}([\text{Conv}_{3\times 3}(x_{vi}), \text{Conv}_{7\times 7}(x_{vi})])$$
   $$S_{ir} = \text{Conv}_{5\times 5}([\text{Conv}_{3\times 3}(x_{ir}), \text{Conv}_{7\times 7}(x_{ir})])$$
   $$G_{sp} = \sigma(\text{Conv}_{3\times 3}(x_{vi}) + \text{Conv}_{3\times 3}(x_{ir}))$$
   $$\tilde{x}_{vi} = x_{vi} \odot \sigma(G_{sp} + S_{ir}), \quad \tilde{x}_{ir} = x_{ir} \odot \sigma(G_{sp} + S_{vi})$$

3. **Fusi Akhir (DEA Output):**
   $$x_{fused} = \sigma(\tilde{x}_{vi} + \tilde{x}_{ir})$$
