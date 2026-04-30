# ♻️ Waste Classification — MobileNetV2 Transfer Learning

Klasifikasi jenis sampah dari gambar menggunakan Transfer Learning MobileNetV2.

---

## Kelas

| Kelas | Jumlah |
|-------|--------|
| cardboard | 393 |
| glass | 491 |
| metal | 400 |
| paper | 584 |
| plastic | 472 |
| trash | 127 |
| **Total** | **~2.467** |

---

## Cara Menjalankan

### 1. Install library
```bash
pip install -r requirements.txt
```

### 2. Siapkan Kaggle API token
1. Login ke https://www.kaggle.com
2. Foto profil → Settings → API → **Create New Token**
3. Taruh file `kaggle.json` di folder ini

### 3. Download dataset
```bash
python download_dataset.py
```

### 4. Training model
```bash
python train_model.py
```
Durasi: 15–45 menit (CPU) / 5–10 menit (GPU)

### 5. Jalankan web app
```bash
streamlit run app.py
```
Buka browser: http://localhost:8501

---

## Struktur Folder

```
waste_classification/
├── kaggle.json              ← taruh di sini
├── download_dataset.py
├── train_model.py
├── app.py
├── requirements.txt
├── dataset/
│   ├── cardboard/
│   ├── glass/
│   ├── metal/
│   ├── paper/
│   ├── plastic/
│   └── trash/
├── model/
│   └── waste_mobilenetv2.h5
└── plots/
    ├── class_distribution.png
    ├── training_history.png
    └── confusion_matrix.png
```

---

## Arsitektur Model

```
Input (224×224×3)
    ↓
MobileNetV2 [pretrained ImageNet]
    ↓
GlobalAveragePooling2D
BatchNormalization
Dense(512, relu) → Dropout(0.4)
Dense(256, relu) → Dropout(0.3)
    ↓
Dense(6, softmax)
```

Training 2 fase:
- **Fase 1** (15 epoch): base frozen, LR = 1e-3
- **Fase 2** (15 epoch): 40 layer terakhir dibuka, LR = 1e-5
