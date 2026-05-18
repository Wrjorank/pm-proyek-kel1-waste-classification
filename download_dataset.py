"""
================================================================
download_dataset.py
================================================================
Download dataset "Garbage Classification" dari Kaggle secara
otomatis, lalu susun ke struktur folder yang dibutuhkan.

CARA PAKAI:
  1. Login ke https://www.kaggle.com
  2. Foto profil → Settings → API → "Create New Token"
  3. Taruh file kaggle.json di folder yang sama dengan script ini
  4. Jalankan: python download_dataset.py
================================================================
"""

import os, sys, json, shutil, zipfile
from pathlib import Path


# ── Konfigurasi ──────────────────────────────────────────────
# Nama dataset, lokasi download sementara, dan direktori target dataset.
KAGGLE_DATASET = "asdasdasasdas/garbage-classification"
DOWNLOAD_DIR   = Path("downloads")
DATASET_DIR    = Path("dataset")
CLASSES        = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]


# ── Langkah 1: Pastikan kaggle.json tersedia ─────────────────
# Periksa apakah token Kaggle ada di folder proyek; jika tidak, salin ke ~/.kaggle/.
def setup_kaggle():
    kaggle_dir  = Path.home() / ".kaggle"
    kaggle_file = kaggle_dir / "kaggle.json"
    local_file  = Path("kaggle.json")

    # Salin dari folder project ke ~/.kaggle/ jika belum ada
    if local_file.exists() and not kaggle_file.exists():
        kaggle_dir.mkdir(exist_ok=True)
        shutil.copy(local_file, kaggle_file)
        kaggle_file.chmod(0o600)
        print("✅ kaggle.json disalin ke ~/.kaggle/")

    if not kaggle_file.exists():
        print("\n❌ File kaggle.json tidak ditemukan!")
        print("   Cara mendapatkannya:")
        print("   1. Login ke https://www.kaggle.com")
        print("   2. Foto profil → Settings → API → Create New Token")
        print("   3. Taruh kaggle.json di folder ini")
        sys.exit(1)

    creds = json.loads(kaggle_file.read_text())
    print(f"✅ Kaggle login sebagai: {creds.get('username', '?')}")


# ── Langkah 2: Install & import kaggle library ───────────────
# Pastikan dependency Kaggle API tersedia sebelum download dataset.
def install_kaggle():
    try:
        import kaggle
        return kaggle
    except ImportError:
        print("📦 Menginstall kaggle library ...")
        os.system(f"{sys.executable} -m pip install kaggle -q")
        import kaggle
        return kaggle


# ── Langkah 3: Download dataset ──────────────────────────────
# Unduh file zip dari Kaggle dan kembalikan path zip yang selesai didownload.
def download(kaggle):
    DOWNLOAD_DIR.mkdir(exist_ok=True)
    print(f"\n📥 Mendownload dataset ({KAGGLE_DATASET}) ...")
    print("   Ukuran sekitar 75 MB, mohon tunggu ...\n")
    kaggle.api.dataset_download_files(
        dataset=KAGGLE_DATASET,
        path=str(DOWNLOAD_DIR),
        unzip=False,
        quiet=False,
    )
    zips = list(DOWNLOAD_DIR.glob("*.zip"))
    if not zips:
        print("❌ File zip tidak ditemukan setelah download!")
        sys.exit(1)
    return zips[0]


# ── Langkah 4: Ekstrak zip ───────────────────────────────────
# Buka file zip hasil download dan ekstrak seluruh isinya ke folder sementara.
def extract(zip_path: Path):
    print(f"\n📦 Mengekstrak {zip_path.name} ...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(DOWNLOAD_DIR)
    print("   ✅ Ekstraksi selesai")


# ── Langkah 5: Susun folder dataset/ ─────────────────────────
# Temukan setiap folder kelas hasil ekstrak dan salin gambar ke struktur dataset/.
def organize():
    print("\n🗂️  Menyusun folder dataset/ ...")

    # Cari folder kelas di dalam hasil ekstrak
    found = []
    for item in DOWNLOAD_DIR.rglob("*"):
        if item.is_dir() and item.name.lower() in CLASSES:
            found.append(item)

    if not found:
        # Fallback: tampilkan semua folder untuk debug
        print("   ⚠️  Folder kelas tidak langsung ditemukan, mencari ...")
        for item in DOWNLOAD_DIR.rglob("*"):
            if item.is_dir():
                print(f"      Ditemukan folder: {item}")
        sys.exit(1)

    total = 0
    for src in found:
        cls_name = src.name.lower()
        dst = DATASET_DIR / cls_name
        dst.mkdir(parents=True, exist_ok=True)

        images = [f for f in src.iterdir()
                  if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
        for img in images:
            shutil.copy2(img, dst / img.name)

        total += len(images)
        bar = "█" * (len(images) // 20)
        print(f"   ✅ {cls_name:12s}: {len(images):4d} gambar  {bar}")

    print(f"\n   Total: {total} gambar di {len(found)} kelas")


# ── Langkah 6: Verifikasi ─────────────────────────────────────
# Tampilkan ringkasan jumlah gambar per kelas untuk memastikan dataset lengkap.
def verify():
    print("\n📊 Verifikasi dataset:")
    print("   " + "─" * 36)
    total = 0
    for cls in sorted(DATASET_DIR.iterdir()):
        if cls.is_dir():
            n = len(list(cls.glob("*.*")))
            total += n
            print(f"   {cls.name:12s}: {n:4d}")
    print("   " + "─" * 36)
    print(f"   {'TOTAL':12s}: {total:4d}")
    print(f"\n✅ Dataset siap di folder: dataset/")


# ── Main ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 56)
    print("  DOWNLOAD DATASET: Garbage Classification (Kaggle)")
    print("=" * 56)

    # Cek jika dataset sudah ada
    if DATASET_DIR.exists() and any(DATASET_DIR.iterdir()):
        ans = input("\n⚠️  Folder dataset/ sudah ada. Download ulang? (y/n): ")
        if ans.strip().lower() != "y":
            print("   Dibatalkan — dataset sudah tersedia.")
            sys.exit(0)

    setup_kaggle()
    kaggle = install_kaggle()
    zip_path = download(kaggle)
    extract(zip_path)
    organize()
    verify()

    # Hapus folder download sementara
    ans = input(f"\n🗑️  Hapus folder downloads/ yang sementara? (y/n): ")
    if ans.strip().lower() == "y":
        shutil.rmtree(DOWNLOAD_DIR, ignore_errors=True)
        print("   ✅ Folder downloads/ dihapus.")

    print("\n" + "=" * 56)
    print("  Sekarang jalankan: python train_model.py")
    print("=" * 56)
