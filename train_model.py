"""
Train a waste-classification model with MobileNetV2 transfer learning.

Run:
  py -3.13 train_model.py
"""

import os
import warnings

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print("=" * 60)
print("  Waste Classification - MobileNetV2 Transfer Learning")
print("=" * 60)
print(f"  TensorFlow : {tf.__version__}")
print(f"  GPU        : {bool(tf.config.list_physical_devices('GPU'))}")
print("=" * 60)


# ================================================================
# Config
# ================================================================
# Semua parameter utama untuk dataset, model, dan proses training.
DATASET_DIR = "dataset"
MODEL_PATH = "model/waste_mobilenetv2.keras"
PLOTS_DIR = "plots"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
VAL_SPLIT = 0.20
SEED = 42
IMAGE_EXTS = (".jpg", ".jpeg", ".png")

LR_1 = 1e-3
EPOCHS_1 = 50

LR_2 = 1e-6
EPOCHS_2 = 20
UNFREEZE_LAST = 20

os.makedirs("model", exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


# ================================================================
# Dataset
# ================================================================
def scan_dataset():
    """Scan folder dataset dan buat tabel path gambar + label."""
    if not os.path.isdir(DATASET_DIR):
        raise FileNotFoundError(
            f"Folder '{DATASET_DIR}' tidak ada.\n"
            "Jalankan dulu: python download_dataset.py"
        )

    classes = sorted(
        d for d in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, d))
    )
    if not classes:
        raise ValueError(f"Folder '{DATASET_DIR}' kosong!")

    rows = []
    counts = {}

    print(f"\nDataset: {DATASET_DIR}/")
    total = 0
    for cls in classes:
        class_dir = os.path.join(DATASET_DIR, cls)
        files = sorted(
            f for f in os.listdir(class_dir)
            if f.lower().endswith(IMAGE_EXTS)
        )
        counts[cls] = len(files)
        total += len(files)
        bar = "#" * (len(files) // 20)
        print(f"   {cls:12s}: {len(files):4d}  {bar}")
        for filename in files:
            rows.append(
                {
                    "filepath": os.path.abspath(os.path.join(class_dir, filename)),
                    "label": cls,
                }
            )

    print(f"   {'-' * 38}")
    print(f"   {'TOTAL':12s}: {total:4d}")

    dataset_df = pd.DataFrame(rows)
    if dataset_df.empty:
        raise ValueError(f"Tidak ada file gambar di '{DATASET_DIR}'.")

    return classes, counts, dataset_df


def split_dataset(dataset_df):
    """Bagi dataset menjadi training dan validasi secara stratified."""
    train_df, val_df = train_test_split(
        dataset_df,
        test_size=VAL_SPLIT,
        stratify=dataset_df["label"],
        random_state=SEED,
        shuffle=True,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def print_split_summary(classes, train_df, val_df):
    print(f"\nSplit data (stratified, random_state={SEED}):")
    print(f"   Training   : {len(train_df)} gambar")
    print(f"   Validasi   : {len(val_df)} gambar")

    train_counts = train_df["label"].value_counts().reindex(classes, fill_value=0)
    val_counts = val_df["label"].value_counts().reindex(classes, fill_value=0)

    print("\nDistribusi split:")
    for cls in classes:
        print(
            f"   {cls:12s}: train={int(train_counts[cls]):4d} | "
            f"val={int(val_counts[cls]):3d}"
        )


def build_generators(train_df, val_df, classes):
    print("\nMenyiapkan data generator ...")

    train_gen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        brightness_range=[0.85, 1.15],
        horizontal_flip=True,
        fill_mode="nearest",
    ).flow_from_dataframe(
        train_df,
        x_col="filepath",
        y_col="label",
        classes=classes,
        directory=None,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=True,
        seed=SEED,
    )

    val_gen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
    ).flow_from_dataframe(
        val_df,
        x_col="filepath",
        y_col="label",
        classes=classes,
        directory=None,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
        seed=SEED,
    )

    print(f"\n   Kelas ({len(classes)}): {classes}")
    print(f"   Training   : {train_gen.samples} gambar")
    print(f"   Validasi   : {val_gen.samples} gambar")
    return train_gen, val_gen


def build_class_weights(train_df, class_indices):
    y_train = train_df["label"].map(class_indices).to_numpy()
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(class_indices)),
        y=y_train,
    )
    class_weight = dict(enumerate(weights))

    print("\nClass weights:")
    for cls, idx in class_indices.items():
        print(f"   [{idx}] {cls:12s}: {class_weight[idx]:.3f}")
    return class_weight


# ================================================================
# Model
# ================================================================
def build_model(num_classes):
    """Bangun model transfer learning dengan MobileNetV2 dan head kustom."""
    print("\nMembangun model ...")

    base = MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.35)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.25)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)

    trainable_count = sum(tf.size(w).numpy() for w in model.trainable_weights)
    print(f"   Total params    : {model.count_params():,}")
    print(f"   Trainable params: {trainable_count:,}")
    return model, base


def compile_model(model, learning_rate):
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=CategoricalCrossentropy(label_smoothing=0.05),
        metrics=["accuracy"],
    )


def make_callbacks():
    """Sediakan callback untuk early stopping, checkpoint, dan penurunan learning rate."""
    return [
        EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=6,
            min_delta=0.002,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=MODEL_PATH,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
    ]


def merge_histories(histories):
    merged = {}
    for history in histories:
        for key, values in history.history.items():
            merged.setdefault(key, []).extend(values)
    return merged


# ================================================================
# Visualization
# ================================================================
def save_class_distribution(counts):
    """Simpan plot distribusi dataset per kelas ke folder plots/."""
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD"]
    bars = ax.bar(counts.keys(), counts.values(), color=colors, edgecolor="white", linewidth=1.5)
    for bar, value in zip(bars, counts.values()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 4,
            str(value),
            ha="center",
            fontweight="bold",
            fontsize=11,
        )
    ax.set_title("Distribusi Dataset per Kelas", fontsize=13, fontweight="bold")
    ax.set_xlabel("Kelas")
    ax.set_ylabel("Jumlah Gambar")
    ax.set_ylim(0, max(counts.values()) * 1.18)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "class_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Grafik distribusi  : {path}")


def save_training_history(history, phase1_len):
    """Simpan grafik akurasi dan loss training/validasi."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training History - MobileNetV2 Waste Classification", fontsize=13, fontweight="bold")

    epochs = range(1, len(history["accuracy"]) + 1)
    for ax, metric, val_metric, title in [
        (ax1, "accuracy", "val_accuracy", "Accuracy"),
        (ax2, "loss", "val_loss", "Loss"),
    ]:
        ax.plot(epochs, history[metric], "#2196F3", lw=2, marker="o", ms=3, label="Train")
        ax.plot(epochs, history[val_metric], "#F44336", lw=2, marker="s", ms=3, label="Validasi")
        ax.axvline(phase1_len + 0.5, color="#4CAF50", linestyle="--", lw=1.5, label="Fine-tune mulai")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.grid(alpha=0.3)
        ax.legend()

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "training_history.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Grafik history     : {path}")


def save_confusion_matrix(cm, class_labels):
    """Simpan confusion matrix dalam bentuk absolut dan normalisasi."""
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Confusion Matrix - MobileNetV2 Waste Classification", fontsize=13, fontweight="bold")

    for ax, data, fmt, cmap, title in [
        (axes[0], cm, "d", "Blues", "Jumlah"),
        (axes[1], cm_norm, ".1%", "Greens", "Normalized (%)"),
    ]:
        sns.heatmap(
            data,
            annot=True,
            fmt=fmt,
            cmap=cmap,
            xticklabels=class_labels,
            yticklabels=class_labels,
            ax=ax,
            linewidths=0.5,
            annot_kws={"size": 10},
        )
        ax.set_title(f"Confusion Matrix ({title})", fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Confusion matrix   : {path}")


# ================================================================
# Main
# ================================================================
# ================================================================
# Main
# ================================================================
# Jalankan semua langkah: scan dataset, buat generator, training, dan evaluasi.
classes, class_counts, dataset_df = scan_dataset()
train_df, val_df = split_dataset(dataset_df)
print_split_summary(classes, train_df, val_df)

train_gen, val_gen = build_generators(train_df, val_df, classes)
class_weight = build_class_weights(train_df, train_gen.class_indices)

model, base = build_model(len(classes))

print("\n" + "=" * 60)
print("  FASE 1: Training custom head")
print(f"  Epochs: {EPOCHS_1}  |  LR: {LR_1}")
print("=" * 60)

compile_model(model, LR_1)
hist1 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_1,
    class_weight=class_weight,
    callbacks=make_callbacks(),
)

histories = [hist1]
phase1_len = len(hist1.history["accuracy"])

if EPOCHS_2 > 0 and UNFREEZE_LAST > 0:
    print("\n" + "=" * 60)
    print(f"  FASE 2: Fine-tuning ({UNFREEZE_LAST} layer terakhir)")
    print(f"  Epochs: {EPOCHS_2}  |  LR: {LR_2}")
    print("=" * 60)

    base.trainable = True
    for layer in base.layers[:-UNFREEZE_LAST]:
        layer.trainable = False
    for layer in base.layers[-UNFREEZE_LAST:]:
        if isinstance(layer, BatchNormalization):
            layer.trainable = False

    compile_model(model, LR_2)
    hist2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS_2,
        class_weight=class_weight,
        callbacks=make_callbacks(),
    )
    histories.append(hist2)

history = merge_histories(histories)
np.save(os.path.join("model", "training_history.npy"), history)

best_epoch = int(np.argmax(history["val_accuracy"])) + 1
best_val_acc = float(np.max(history["val_accuracy"]))
print(f"\nBest validation accuracy: {best_val_acc:.4f} pada epoch {best_epoch}")

print("\nMemuat checkpoint terbaik dari training ...")
best_model = tf.keras.models.load_model(MODEL_PATH)

print("\n" + "=" * 60)
print("  EVALUASI MODEL")
print("=" * 60)

loss, acc = best_model.evaluate(val_gen, verbose=0)
print(f"\n  Validation loss    : {loss:.4f}")
print(f"  Validation accuracy: {acc:.4f} ({acc * 100:.2f}%)")

val_gen.reset()
y_prob = best_model.predict(val_gen, verbose=1)
y_pred = np.argmax(y_prob, axis=1)
y_true = val_gen.classes

print("\nClassification Report:")
print("-" * 60)
print(
    classification_report(
        y_true,
        y_pred,
        target_names=classes,
        digits=4,
        zero_division=0,
    )
)

cm = confusion_matrix(y_true, y_pred)

save_class_distribution(class_counts)
save_training_history(history, phase1_len)
save_confusion_matrix(cm, classes)

print("\n" + "=" * 60)
print("  SELESAI")
print(f"  Best validation accuracy : {best_val_acc * 100:.2f}%")
print(f"  Model terbaik            : {MODEL_PATH}")
print(f"  Plot tersimpan           : {PLOTS_DIR}/")
print("  Jalankan app             : py -3.13 -m streamlit run app.py")
print("=" * 60)
