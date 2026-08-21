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

## 📖 Landasan Teori & Formulasi Matematis (Untuk Bab 3 / Bab 4 Skripsi)

Bagian ini menyajikan penurunan rumus matematis lengkap dari setiap modul kustom yang dibangun pada arsitektur **Dual-Stream YOLOv11**, mencakup **BiFocus**, **DECA**, **DEPA**, dan fusi **DEA**.

---

### 1. Tabel Notasi Simbol Matematis

| Simbol | Keterangan & Dimensi Tensor |
| :--- | :--- |
| $\mathbf{x}_{vi} \in \mathbb{R}^{B \times C \times H \times W}$ | *Feature map* masukan dari stream citra Visible (RGB) |
| $\mathbf{x}_{ir} \in \mathbb{R}^{B \times C \times H \times W}$ | *Feature map* masukan dari stream citra Infrared (IR) |
| $B, C, H, W$ | Berturut-turut: *Batch size*, Jumlah *Channel*, Tinggi (*Height*), Lebar (*Width*) |
| $[\mathbf{a}; \mathbf{b}]$ | Operasi konkatenasi tensor (*channel concatenation*) pada dimensi kanal ($C$) |
| $\odot$ | Perkalian Hadamard (*element-wise multiplication*) |
| $\sigma(\cdot)$ | Fungsi aktivasi Sigmoid: $\sigma(z) = \frac{1}{1 + e^{-z}} \in [0, 1]$ |
| $\text{GAP}(\cdot)$ | *Global Average Pooling*: $\text{GAP}(\mathbf{x})_c = \frac{1}{H \times W} \sum_{i=1}^H \sum_{j=1}^W \mathbf{x}_{c, i, j}$ |
| $\text{Conv}_{k \times k}(\cdot)$ | Operasi konvolusi 2D standar dengan ukuran kernel $k \times k$ |
| $\text{DWConv}_{k \times k}(\cdot)$ | *Depthwise Separable Convolution* dengan kernel $k \times k$ |

---

### 2. Modul `C3k2_BiFocus` & `BiFocus` (Bi-directional Decoupled Focus)

Modul **BiFocus** membagi masukan spasial menjadi dua komponen ortogonal (horizontal dan vertikal) untuk menangkap dependensi piksel bertetangga (*local*) maupun jarak jauh (*remote*) sebelum digabungkan melalui *depthwise convolution*:

1. **Horizontal Decoupled Focus ($\text{FocusH}$):**
   Memisahkan piksel spasial genap dan ganjil pada sumbu horizontal:
   $$\mathbf{x}_1^H[:, :, 2i, :] = \mathbf{x}[:, :, 2i, 2j], \quad \mathbf{x}_1^H[:, :, 2i+1, :] = \mathbf{x}[:, :, 2i+1, 2j+1]$$
   $$\mathbf{x}_2^H[:, :, 2i, :] = \mathbf{x}[:, :, 2i, 2j+1], \quad \mathbf{x}_2^H[:, :, 2i+1, :] = \mathbf{x}[:, :, 2i+1, 2j]$$
   Fitur yang terpisah diproses dengan konvolusi dan direkonstruksi kembali ke dimensi $(B, C, H, W)$ menghasilkan $F_H(\mathbf{x})$.

2. **Vertical Decoupled Focus ($\text{FocusV}$):**
   Memisahkan piksel spasial genap dan ganjil pada sumbu vertikal dengan prinsip yang sama menghasilkan $F_V(\mathbf{x}) \in \mathbb{R}^{B \times C \times H \times W}$.

3. **Fusi BiFocus:**
   Menggabungkan fitur awal dengan respon horizontal dan vertikal:
   $$\mathbf{x}_{cat} = [\mathbf{x}; F_H(\mathbf{x}); F_V(\mathbf{x})] \in \mathbb{R}^{B \times 3C \times H \times W}$$
   $$\text{BiFocus}(\mathbf{x}) = \text{Conv}_{1 \times 1}\left(\text{DWConv}_{3 \times 3}(\mathbf{x}_{cat})\right) \in \mathbb{R}^{B \times C \times H \times W}$$

---

### 3. Modul `DECA` (*Dual Semantic Enhancing Channel Weight Assignment*)

**Tujuan:** Mengekstrak dependensi semantik kanal antar-modalitas sehingga informasi kanal yang menonjol pada IR dapat memperkuat representasi RGB, dan sebaliknya.

```text
[x_vi, x_ir] ──> GAP ──> MLP ──> w_vi (1x1xC), w_ir (1x1xC)
      │
      └──> Concat ──> Conv3x3 ──> ConvPyramid ──> Glob (1x1xC)
                                                    │
x_vi' = x_vi * Sigmoid(w_ir * Glob) <───────────────┤
x_ir' = x_ir * Sigmoid(w_vi * Glob) <───────────────┘
```

#### Langkah-langkah Matematis:

1. **Vektor Konteks Kanal Tiap Modalitas:**
   Menggunakan *Global Average Pooling* diikuti Multi-Layer Perceptron (MLP) dengan rasio reduksi $r = 16$:
   $$\mathbf{w}_{vi} = \sigma\left(\mathbf{W}_2 \cdot \text{ReLU}(\mathbf{W}_1 \cdot \text{GAP}(\mathbf{x}_{vi}))\right) \in \mathbb{R}^{B \times C \times 1 \times 1}$$
   $$\mathbf{w}_{ir} = \sigma\left(\mathbf{W}_2 \cdot \text{ReLU}(\mathbf{W}_1 \cdot \text{GAP}(\mathbf{x}_{ir}))\right) \in \mathbb{R}^{B \times C \times 1 \times 1}$$
   di mana $\mathbf{W}_1 \in \mathbb{R}^{\frac{C}{r} \times C}$ dan $\mathbf{W}_2 \in \mathbb{R}^{C \times \frac{C}{r}}$.

2. **Konteks Global Bersama (*Joint Semantic Context*):**
   Fitur RGB dan IR digabungkan pada dimensi kanal, lalu dikompresi dan diproses melalui piramida konvolusi bertingkat (*Convolution Pyramid*):
   $$\mathbf{x}_{comp} = \text{SiLU}\left(\text{Conv}_{3\times 3}([\mathbf{x}_{vi}; \mathbf{x}_{ir}])\right) \in \mathbb{R}^{B \times C \times H \times W}$$
   $$\mathbf{G} = \text{Conv}_{k_3}\left(\text{Conv}_{k_2}\left(\text{Conv}_{k_1}(\mathbf{x}_{comp})\right)\right) \in \mathbb{R}^{B \times C \times 1 \times 1}$$
   *(Jika ukuran resolusi $H, W < 80$, digunakan *spatial mean* $\mathbf{G} = \frac{1}{HW}\sum \mathbf{x}_{comp}$)*.

3. **Pemberian Bobot Silang (*Cross-Modal Modulation*):**
   Bobot kanal IR ($\mathbf{w}_{ir}$) digunakan untuk memperkaya fitur RGB ($\mathbf{x}_{vi}$), dan sebaliknya:
   $$\mathbf{x}'_{vi} = \mathbf{x}_{vi} \odot \sigma(\mathbf{w}_{ir} \odot \mathbf{G}) \in \mathbb{R}^{B \times C \times H \times W}$$
   $$\mathbf{x}'_{ir} = \mathbf{x}_{ir} \odot \sigma(\mathbf{w}_{vi} \odot \mathbf{G}) \in \mathbb{R}^{B \times C \times H \times W}$$

---

### 4. Modul `DEPA` (*Dual Spatial Enhancing Pixel Weight Assignment*)

**Tujuan:** Mempelajari korelasi posisi/spasial piksel objek lintas modalitas dengan *multi-scale spatial receptive fields* (kernel $3\times 3$ dan $7\times 7$).

#### Langkah-langkah Matematis:

1. **Ekstraksi Atensi Spasial Multi-Skala:**
   Untuk modalitas Visible ($\mathbf{x}'_{vi}$):
   $$\mathbf{S}_{vi} = \text{Conv}_{5\times 5}\left(\left[\text{Conv}_{3\times 3}(\mathbf{x}'_{vi}); \text{Conv}_{7\times 7}(\mathbf{x}'_{vi})\right]\right) \in \mathbb{R}^{B \times 1 \times H \times W}$$
   Untuk modalitas Infrared ($\mathbf{x}'_{ir}$):
   $$\mathbf{S}_{ir} = \text{Conv}_{5\times 5}\left(\left[\text{Conv}_{3\times 3}(\mathbf{x}'_{ir}); \text{Conv}_{7\times 7}(\mathbf{x}'_{ir})\right]\right) \in \mathbb{R}^{B \times 1 \times H \times W}$$

2. **Peta Spasial Global Bersama:**
   Menggabungkan proyeksi spasial kedua modalitas:
   $$\mathbf{G}_{spatial} = \sigma\left(\text{Conv}_{3\times 3}(\mathbf{x}'_{vi}) + \text{Conv}_{3\times 3}(\mathbf{x}'_{ir})\right) \in \mathbb{R}^{B \times 1 \times H \times W}$$

3. **Pemberian Bobot Spasial Silang:**
   $$\mathbf{W}_{spatial}^{vi} = \sigma(\mathbf{G}_{spatial} + \mathbf{S}_{vi}) \in \mathbb{R}^{B \times 1 \times H \times W}$$
   $$\mathbf{W}_{spatial}^{ir} = \sigma(\mathbf{G}_{spatial} + \mathbf{S}_{ir}) \in \mathbb{R}^{B \times 1 \times H \times W}$$
   $$\tilde{\mathbf{x}}_{vi} = \mathbf{x}'_{vi} \odot \mathbf{W}_{spatial}^{ir} \in \mathbb{R}^{B \times C \times H \times W}$$
   $$\tilde{\mathbf{x}}_{ir} = \mathbf{x}'_{ir} \odot \mathbf{W}_{spatial}^{vi} \in \mathbb{R}^{B \times C \times H \times W}$$

---

### 5. Fusi Akhir `DEA` (Dual-context Collaborative Enhancement)

Hasil peningkatan semantik kanal (DECA) dan peningkatan spasial piksel (DEPA) dari kedua modalitas digabungkan menjadi satu representasi multi-modalitas tunggal yang diperkuat:

$$\mathbf{x}_{fused} = \sigma\left(\tilde{\mathbf{x}}_{vi} + \tilde{\mathbf{x}}_{ir}\right) \in \mathbb{R}^{B \times C \times H \times W}$$

Tensor $\mathbf{x}_{fused}$ pada level $P_3, P_4, P_5$ selanjutnya diteruskan ke blok **PANet Neck** berbasis `C3k2` dan dideteksi oleh **Decoupled Detect Head** YOLOv11.
