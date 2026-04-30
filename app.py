"""
================================================================
app.py
================================================================
Web app klasifikasi sampah menggunakan Streamlit.

JALANKAN:
  streamlit run app.py

Pastikan model sudah ada di model/waste_mobilenetv2.h5
Jika belum: python train_model.py
================================================================
"""

import os
import time
import numpy as np
from PIL import Image

import streamlit as st
import plotly.graph_objects as go
import tensorflow as tf


# ================================================================
# KONFIGURASI HALAMAN
# ================================================================
st.set_page_config(
    page_title = "Waste Classifier",
    page_icon  = "♻️",
    layout     = "wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.hero {
    background: linear-gradient(135deg, #0f2027, #2c5364);
    padding: 2.2rem 2rem;
    border-radius: 16px;
    text-align: center;
    margin-bottom: 1.8rem;
}
.hero h1 { color: #e0f7fa; font-size: 2.2rem; font-weight: 600; margin: 0; }
.hero p  { color: #80cbc4; font-size: .95rem; margin: .5rem 0 0; }

.result-box {
    border-radius: 14px; padding: 1.4rem;
    text-align: center; font-size: 1.5rem;
    font-weight: 600; margin: .8rem 0;
}
.high   { background:#e8f5e9; color:#1b5e20; border:2px solid #4caf50; }
.medium { background:#fff8e1; color:#e65100; border:2px solid #ff9800; }
.low    { background:#fce4ec; color:#880e4f; border:2px solid #e91e63; }

.tip {
    background: #e3f2fd;
    border-left: 4px solid #1976d2;
    padding: .85rem 1rem;
    border-radius: 0 10px 10px 0;
    font-size: .88rem;
    color: #0d47a1;
    margin: .5rem 0;
}

.empty-state {
    background: #f5f5f5;
    border: 2px dashed #bdbdbd;
    border-radius: 14px;
    padding: 3.5rem 2rem;
    text-align: center;
    color: #9e9e9e;
}
</style>
""", unsafe_allow_html=True)


# ================================================================
# KONSTANTA
# ================================================================
MODEL_PATH  = "model/waste_mobilenetv2.h5"
IMG_SIZE    = (224, 224)

# 6 kelas sesuai dataset Garbage Classification Kaggle
CLASS_INFO = {
    "cardboard": {
        "emoji": "📦",
        "label": "Kardus",
        "color": "#A5895A",
        "tips" : "Ratakan dan jaga kering. Kardus bersih mudah didaur ulang menjadi produk kertas baru.",
    },
    "glass": {
        "emoji": "🍶",
        "label": "Kaca",
        "color": "#64B5F6",
        "tips" : "Cuci bersih sebelum dibuang. Kaca bisa didaur ulang berkali-kali tanpa kehilangan kualitas.",
    },
    "metal": {
        "emoji": "🥫",
        "label": "Logam",
        "color": "#90A4AE",
        "tips" : "Bilas kaleng dari sisa makanan. Logam 100% bisa didaur ulang — hemat 95% energi vs produksi baru.",
    },
    "paper": {
        "emoji": "📄",
        "label": "Kertas",
        "color": "#FFD54F",
        "tips" : "Pisahkan kertas basah atau berminyak. Kertas bersih bisa didaur ulang hingga 7 kali.",
    },
    "plastic": {
        "emoji": "♻️",
        "label": "Plastik",
        "color": "#EF9A9A",
        "tips" : "Cek nomor resin (1–7) di bawah kemasan. Kurangi penggunaan plastik sekali pakai.",
    },
    "trash": {
        "emoji": "🗑️",
        "label": "Sampah Umum",
        "color": "#CE93D8",
        "tips" : "Sampah yang tidak bisa didaur ulang. Buang ke tempat sampah umum, jangan sembarangan.",
    },
}

CLASS_NAMES = list(CLASS_INFO.keys())   # urutan harus sama dengan training!


# ================================================================
# LOAD MODEL (di-cache agar tidak reload tiap interaksi)
# ================================================================
@st.cache_resource(show_spinner="Memuat model ...")
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH)


# ================================================================
# FUNGSI PREDIKSI
# ================================================================
def preprocess(image: Image.Image) -> np.ndarray:
    """Konversi gambar ke tensor siap prediksi."""
    image = image.convert("RGB").resize(IMG_SIZE)
    arr   = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)          # (1, 224, 224, 3)


def predict(model, image: Image.Image):
    arr   = preprocess(image)
    probs = model.predict(arr, verbose=0)[0]    # shape (6,)
    idx   = int(np.argmax(probs))
    return CLASS_NAMES[idx], float(probs[idx]), probs


def confidence_label(conf: float):
    if conf >= 0.80:
        return "Tinggi ✅", "high"
    elif conf >= 0.55:
        return "Sedang ⚠️", "medium"
    else:
        return "Rendah ❌", "low"


# ================================================================
# SIDEBAR
# ================================================================
with st.sidebar:
    st.markdown("## ♻️ Waste Classifier")
    st.caption("MobileNetV2 · Transfer Learning · TensorFlow")
    st.divider()

    st.markdown("### Kelas yang dikenali")
    for cls, info in CLASS_INFO.items():
        st.markdown(f"{info['emoji']} **{info['label']}** &nbsp; `{cls}`")

    st.divider()
    st.markdown("### Spesifikasi model")
    st.markdown("""
| | |
|---|---|
| Arsitektur | MobileNetV2 |
| Pretrained | ImageNet |
| Input | 224 × 224 px |
| Kelas | 6 jenis sampah |
| Framework | TensorFlow/Keras |
""")
    st.divider()
    st.caption("Tugas Machine Learning — Transfer Learning")


# ================================================================
# HEADER
# ================================================================
st.markdown("""
<div class="hero">
    <h1>♻️ Klasifikasi Jenis Sampah</h1>
    <p>Transfer Learning MobileNetV2 · Garbage Classification Dataset · 6 Kelas</p>
</div>
""", unsafe_allow_html=True)


# ================================================================
# LOAD MODEL — tampilkan status
# ================================================================
model = load_model()

if model is None:
    st.error(
        f"**Model tidak ditemukan di `{MODEL_PATH}`**\n\n"
        "Jalankan training terlebih dahulu:\n"
        "```\npython train_model.py\n```"
    )
    st.stop()

st.success(f"✅ Model siap — `{MODEL_PATH}`")
st.divider()


# ================================================================
# LAYOUT UTAMA: Upload | Hasil
# ================================================================
col_left, col_right = st.columns(2, gap="large")

# ── Kolom kiri: Upload ──────────────────────────────────────────
with col_left:
    st.subheader("📷 Upload Gambar")
    st.caption("Format: JPG, JPEG, PNG · Pastikan objek terlihat jelas")

    uploaded = st.file_uploader(
        "Pilih gambar",
        type            = ["jpg", "jpeg", "png"],
        label_visibility= "collapsed",
    )

    if uploaded:
        img = Image.open(uploaded)
        st.image(img, caption=f"{uploaded.name}", use_column_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Lebar",  f"{img.width} px")
        c2.metric("Tinggi", f"{img.height} px")
        c3.metric("Format", img.format or uploaded.type.split("/")[-1].upper())


# ── Kolom kanan: Hasil ─────────────────────────────────────────
with col_right:
    st.subheader("🤖 Hasil Prediksi")

    if not uploaded:
        st.markdown("""
        <div class="empty-state">
            <div style="font-size:3rem">📤</div>
            <p style="margin-top:.8rem">
                Upload gambar sampah di sebelah kiri,<br>
                lalu klik tombol Prediksi.
            </p>
        </div>
        """, unsafe_allow_html=True)

    else:
        if st.button("🔍  Prediksi Sekarang", type="primary",
                     use_container_width=True):

            with st.spinner("Menganalisis gambar ..."):
                time.sleep(0.3)
                pred_class, confidence, all_probs = predict(model, img)

            info        = CLASS_INFO[pred_class]
            level, css  = confidence_label(confidence)

            # ── Kotak hasil utama ──
            st.markdown(f"""
            <div class="result-box {css}">
                {info['emoji']} {info['label'].upper()}
                <br>
                <span style="font-size:.85rem; font-weight:400;">
                    Confidence: {confidence:.1%} &nbsp;·&nbsp; Level: {level}
                </span>
            </div>
            """, unsafe_allow_html=True)

            # ── Tips penanganan ──
            st.markdown(
                f'<div class="tip">💡 <b>Tips:</b> {info["tips"]}</div>',
                unsafe_allow_html=True,
            )

            st.divider()

            # ── Bar chart probabilitas semua kelas ──
            st.markdown("**Probabilitas semua kelas:**")

            order      = np.argsort(all_probs)[::-1]
            labels     = [
                f"{CLASS_INFO[CLASS_NAMES[i]]['emoji']} {CLASS_INFO[CLASS_NAMES[i]]['label']}"
                for i in order
            ]
            values     = [float(all_probs[i]) for i in order]
            bar_colors = [CLASS_INFO[CLASS_NAMES[i]]["color"] for i in order]

            fig = go.Figure(go.Bar(
                x            = values,
                y            = labels,
                orientation  = "h",
                marker       = dict(color=bar_colors, line=dict(color="white", width=0.8)),
                text         = [f"{v:.1%}" for v in values],
                textposition = "outside",
                hovertemplate= "<b>%{y}</b><br>%{x:.2%}<extra></extra>",
            ))
            fig.update_layout(
                height       = 260,
                margin       = dict(l=0, r=60, t=8, b=8),
                xaxis        = dict(range=[0, 1.18], showticklabels=False, showgrid=False),
                yaxis        = dict(showgrid=False),
                paper_bgcolor= "rgba(0,0,0,0)",
                plot_bgcolor = "rgba(0,0,0,0)",
                font         = dict(size=12),
            )
            st.plotly_chart(fig, use_container_width=True)


# ================================================================
# FOOTER
# ================================================================
st.divider()
st.markdown(
    "<p style='text-align:center; color:#9e9e9e; font-size:.82rem;'>"
    "Implementasi Transfer Learning MobileNetV2 · "
    "Dataset: Garbage Classification (Kaggle) · "
    "Dibuat dengan TensorFlow & Streamlit"
    "</p>",
    unsafe_allow_html=True,
)
