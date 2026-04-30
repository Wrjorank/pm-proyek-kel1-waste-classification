"""
================================================================
train_model.py
================================================================
Transfer Learning MobileNetV2 untuk klasifikasi jenis sampah.

Dataset : Garbage Classification (Kaggle)
Kelas   : cardboard, glass, metal, paper, plastic, trash
Model   : MobileNetV2 pretrained ImageNet → custom head

JALANKAN:
  python train_model.py
================================================================
"""

# ================================================================
# 1. IMPORT
# ================================================================
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (
    Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

print("=" * 60)
print("  Waste Classification — MobileNetV2 Transfer Learning")
print("=" * 60)
print(f"  TensorFlow : {tf.__version__}")
print(f"  GPU        : {bool(tf.config.list_physical_devices('GPU'))}")
print("=" * 60)


# ================================================================
# 2. KONFIGURASI — ubah di sini jika perlu
# ================================================================
DATASET_DIR    = "dataset"
MODEL_PATH     = "model/waste_mobilenetv2.h5"
PLOTS_DIR      = "plots"

IMG_SIZE       = (224, 224)   # ukuran input MobileNetV2
BATCH_SIZE     = 32
VAL_SPLIT      = 0.20         # 80% train, 20% validasi
SEED           = 42

# Fase 1: base model frozen
LR_1           = 1e-3
EPOCHS_1       = 50

# Fase 2: fine-tuning
LR_2           = 1e-5         # LR kecil agar bobot pretrained tidak rusak
EPOCHS_2       = 50
UNFREEZE_LAST  = 40           # berapa layer terakhir yang dibuka

os.makedirs("model", exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


# ================================================================
# 3. VALIDASI DATASET
# ================================================================
def check_dataset():
    if not os.path.isdir(DATASET_DIR):
        raise FileNotFoundError(
            f"Folder '{DATASET_DIR}' tidak ada.\n"
            "Jalankan dulu: python download_dataset.py"
        )
    classes = sorted([
        d for d in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, d))
    ])
    if not classes:
        raise ValueError(f"Folder '{DATASET_DIR}' kosong!")

    print(f"\n📂 Dataset: {DATASET_DIR}/")
    total = 0
    for cls in classes:
        n = len([
            f for f in os.listdir(os.path.join(DATASET_DIR, cls))
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])
        total += n
        bar = "█" * (n // 20)
        print(f"   {cls:12s}: {n:4d}  {bar}")
    print(f"   {'─'*38}")
    print(f"   {'TOTAL':12s}: {total:4d}")
    return classes

classes = check_dataset()
NUM_CLASSES = len(classes)


# ================================================================
# 4. DATA GENERATOR + AUGMENTASI
# ================================================================
"""
Augmentasi hanya untuk training — memperbanyak variasi gambar
secara artifisial sehingga model lebih tahan terhadap kondisi
foto yang berbeda-beda (sudut, cahaya, dll).
Validasi TIDAK diaugmentasi — hanya dinormalisasi.
"""
print(f"\n🔧 Menyiapkan data generator ...")

train_gen = ImageDataGenerator(
    rescale            = 1.0 / 255,  # normalisasi pixel ke [0, 1]
    rotation_range     = 20,
    width_shift_range  = 0.1,
    height_shift_range = 0.1,
    shear_range        = 0.1,
    zoom_range         = 0.1,
    brightness_range   = [0.85, 1.15],
    horizontal_flip    = True,
    fill_mode          = "nearest",
    validation_split   = VAL_SPLIT,
).flow_from_directory(
    DATASET_DIR,
    target_size  = IMG_SIZE,
    batch_size   = BATCH_SIZE,
    class_mode   = "categorical",
    subset       = "training",
    shuffle      = True,
    seed         = SEED,
)

val_gen = ImageDataGenerator(
    rescale          = 1.0 / 255,
    validation_split = VAL_SPLIT,
).flow_from_directory(
    DATASET_DIR,
    target_size  = IMG_SIZE,
    batch_size   = BATCH_SIZE,
    class_mode   = "categorical",
    subset       = "validation",
    shuffle      = False,   # jangan shuffle agar evaluasi konsisten
    seed         = SEED,
)

CLASS_LABELS = list(train_gen.class_indices.keys())
print(f"\n   Kelas ({NUM_CLASSES}): {CLASS_LABELS}")
print(f"   Training   : {train_gen.samples} gambar")
print(f"   Validasi   : {val_gen.samples} gambar")


# ================================================================
# 5. CLASS WEIGHTS — menangani data yang tidak seimbang
# ================================================================
"""
Dataset ini tidak seimbang: 'trash' hanya ~127 gambar,
sementara 'paper' ~584. Tanpa class_weight, model cenderung
mengabaikan kelas minoritas.
"""
cw_array = compute_class_weight(
    class_weight = "balanced",
    classes      = np.arange(NUM_CLASSES),
    y            = train_gen.classes,
)
class_weight = dict(enumerate(cw_array))

print(f"\n⚖️  Class weights (mengatasi imbalanced data):")
for i, (cls, w) in enumerate(zip(CLASS_LABELS, cw_array)):
    print(f"   [{i}] {cls:12s}: {w:.3f}")


# ================================================================
# 6. BANGUN MODEL
# ================================================================
"""
Arsitektur Transfer Learning:

  Input (224×224×3)
       ↓
  MobileNetV2 [frozen saat fase 1]
  → fitur pretrained dari 1.2 juta gambar ImageNet
       ↓
  GlobalAveragePooling2D   → (1280,)
  BatchNormalization
  Dense(512, relu)
  Dropout(0.4)
  Dense(256, relu)
  Dropout(0.3)
       ↓
  Dense(N_classes, softmax) → probabilitas tiap kelas
"""
print(f"\n🧠 Membangun model ...")

base = MobileNetV2(
    input_shape = (*IMG_SIZE, 3),
    include_top = False,        # hapus classification head asli
    weights     = "imagenet",   # gunakan bobot pretrained
)
base.trainable = False          # freeze: bobot base tidak berubah

x = GlobalAveragePooling2D()(base.output)
x = BatchNormalization()(x)
x = Dense(512, activation="relu")(x)
x = Dropout(0.4)(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.3)(x)
out = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs=base.input, outputs=out)

trainable_count = sum(
    tf.size(w).numpy() for w in model.trainable_weights
)
print(f"   Total params    : {model.count_params():,}")
print(f"   Trainable params: {trainable_count:,} (custom head saja)")


# ================================================================
# 7. CALLBACKS
# ================================================================
def make_callbacks():
    return [
        # Hentikan training jika val_accuracy stagnan 6 epoch
        EarlyStopping(
            monitor             = "val_accuracy",
            patience            = 6,
            restore_best_weights= True,
            verbose             = 1,
        ),
        # Simpan otomatis model dengan val_accuracy terbaik
        ModelCheckpoint(
            filepath       = MODEL_PATH,
            monitor        = "val_accuracy",
            save_best_only = True,
            verbose        = 1,
        ),
        # Kurangi learning rate jika val_loss tidak turun 3 epoch
        ReduceLROnPlateau(
            monitor  = "val_loss",
            factor   = 0.5,
            patience = 3,
            min_lr   = 1e-8,
            verbose  = 1,
        ),
    ]


# ================================================================
# 8. TRAINING FASE 1 — base frozen
# ================================================================
print("\n" + "=" * 60)
print("  FASE 1: Training custom head (base model frozen)")
print(f"  Epochs: {EPOCHS_1}  |  LR: {LR_1}")
print("=" * 60)

model.compile(
    optimizer = Adam(learning_rate=LR_1),
    loss      = "categorical_crossentropy",
    metrics   = ["accuracy"],
)

hist1 = model.fit(
    train_gen,
    validation_data = val_gen,
    epochs          = EPOCHS_1,
    class_weight    = class_weight,
    callbacks       = make_callbacks(),
)


# ================================================================
# 9. TRAINING FASE 2 — fine-tuning
# ================================================================
"""
Buka freeze pada N layer terakhir MobileNetV2.
Gunakan learning rate sangat kecil (1e-5) agar bobot pretrained
hanya disesuaikan sedikit — tidak dirusak.
"""
print("\n" + "=" * 60)
print(f"  FASE 2: Fine-tuning ({UNFREEZE_LAST} layer terakhir)")
print(f"  Epochs: {EPOCHS_2}  |  LR: {LR_2}")
print("=" * 60)

base.trainable = True
for layer in base.layers[:-UNFREEZE_LAST]:
    layer.trainable = False

model.compile(
    optimizer = Adam(learning_rate=LR_2),
    loss      = "categorical_crossentropy",
    metrics   = ["accuracy"],
)

hist2 = model.fit(
    train_gen,
    validation_data = val_gen,
    epochs          = EPOCHS_2,
    class_weight    = class_weight,
    callbacks       = make_callbacks(),
)


# ================================================================
# 10. EVALUASI
# ================================================================
print("\n" + "=" * 60)
print("  EVALUASI MODEL")
print("=" * 60)

loss, acc = model.evaluate(val_gen, verbose=0)
print(f"\n  Validation loss    : {loss:.4f}")
print(f"  Validation accuracy: {acc:.4f}  ({acc*100:.2f}%)")

# Prediksi semua data validasi
val_gen.reset()
y_prob = model.predict(val_gen, verbose=1)
y_pred = np.argmax(y_prob, axis=1)
y_true = val_gen.classes

print("\n📋 Classification Report:")
print("─" * 60)
print(classification_report(y_true, y_pred, target_names=CLASS_LABELS))


# ================================================================
# 11. SIMPAN HISTORY
# ================================================================
history = {
    "accuracy"    : hist1.history["accuracy"]     + hist2.history["accuracy"],
    "val_accuracy": hist1.history["val_accuracy"] + hist2.history["val_accuracy"],
    "loss"        : hist1.history["loss"]         + hist2.history["loss"],
    "val_loss"    : hist1.history["val_loss"]     + hist2.history["val_loss"],
}
fase1_len = len(hist1.history["accuracy"])
np.save("model/training_history.npy", history)


# ================================================================
# 12. VISUALISASI
# ================================================================

# ── 12a. Distribusi kelas ──────────────────────────────────────
counts = {}
for cls in sorted(os.listdir(DATASET_DIR)):
    p = os.path.join(DATASET_DIR, cls)
    if os.path.isdir(p):
        counts[cls] = len([
            f for f in os.listdir(p)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

fig, ax = plt.subplots(figsize=(10, 4))
colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD"]
bars = ax.bar(counts.keys(), counts.values(), color=colors,
              edgecolor="white", linewidth=1.5)
for bar, v in zip(bars, counts.values()):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 4, str(v),
            ha="center", fontweight="bold", fontsize=11)
ax.set_title("Distribusi Dataset per Kelas", fontsize=13, fontweight="bold")
ax.set_xlabel("Kelas"); ax.set_ylabel("Jumlah Gambar")
ax.set_ylim(0, max(counts.values()) * 1.18)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/class_distribution.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"✅ Grafik distribusi → {PLOTS_DIR}/class_distribution.png")

# ── 12b. Training history ──────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Training History — MobileNetV2 Waste Classification",
             fontsize=13, fontweight="bold")

epochs = range(1, len(history["accuracy"]) + 1)
for ax, m, vm, title in [
    (ax1, "accuracy",  "val_accuracy", "Accuracy"),
    (ax2, "loss",      "val_loss",     "Loss"),
]:
    ax.plot(epochs, history[m],  "#2196F3", lw=2, marker="o", ms=3, label="Train")
    ax.plot(epochs, history[vm], "#F44336", lw=2, marker="s", ms=3, label="Validasi")
    ax.axvline(fase1_len + 0.5, color="#4CAF50",
               linestyle="--", lw=1.5, label="Fine-tune mulai")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel(title)
    ax.legend(); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/training_history.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"✅ Grafik history    → {PLOTS_DIR}/training_history.png")

# ── 12c. Confusion matrix ──────────────────────────────────────
cm   = confusion_matrix(y_true, y_pred)
cm_n = cm.astype("float") / cm.sum(axis=1, keepdims=True)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Confusion Matrix — MobileNetV2 Waste Classification",
             fontsize=13, fontweight="bold")

for ax, data, fmt, cmap, title in [
    (axes[0], cm,   "d",   "Blues",  "Jumlah"),
    (axes[1], cm_n, ".1%", "Greens", "Normalized (%)"),
]:
    sns.heatmap(
        data, annot=True, fmt=fmt, cmap=cmap,
        xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS,
        ax=ax, linewidths=0.5, annot_kws={"size": 10},
    )
    ax.set_title(f"Confusion Matrix ({title})", fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.tick_params(axis="x", rotation=30)

plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"✅ Confusion matrix  → {PLOTS_DIR}/confusion_matrix.png")


# ================================================================
# 13. SIMPAN MODEL FINAL
# ================================================================
model.save(MODEL_PATH)

print("\n" + "=" * 60)
print("  SELESAI!")
print(f"  Validation Accuracy : {acc * 100:.2f}%")
print(f"  Model tersimpan     : {MODEL_PATH}")
print(f"  Grafik tersimpan    : {PLOTS_DIR}/")
print()
print("  Langkah selanjutnya:")
print("  → streamlit run app.py")
print("=" * 60)
