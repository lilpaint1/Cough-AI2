"""
Preprocessing.py  –  Dataset Preparation (NEW STEP, ไม่ทับของเดิม)
====================================================================
ขั้นตอนนี้คือกล่อง  Data Cleaning → Normalization → Cough Segmentation
ในแผนภาพ Data Flow ของโครงการ

หน้าที่
-------
  1. อ่านไฟล์ metadata_compiled  (auto: .csv หรือ .xlsx)
  2. ดู "เฉพาะคอลัมน์ status" เพื่อ map label:
        healthy      -> healthy
        COVID-19     -> covid
        symptomatic  -> symptomatic
        (ค่าว่าง/อื่น ๆ จะถูกข้าม)
  3. หาไฟล์เสียงจาก uuid (.webm/.ogg/.wav/.mp3) ในโฟลเดอร์รวม
  4. Cleaning  : ข้ามไฟล์เสีย/หาไม่เจอ
     Normalize : peak-normalize แอมพลิจูด (ลด bias เพศ/อายุ/อุปกรณ์)
     Segment   : ตัดความเงียบหัว-ท้าย (librosa.effects.trim) แบบเบา
  5. เซฟเป็น .wav ลงโฟลเดอร์แยกตามคลาส (สร้างโฟลเดอร์ใหม่ให้)
  6. พิมพ์ LOG สรุปครบ พร้อมเอาไปใส่โครงการได้เลย

หมายเหตุ
--------
  • ไฟล์ .webm/.ogg ต้องมี ffmpeg ติดตั้งในเครื่อง (librosa เรียกผ่าน audioread)
    Windows: ดาวน์โหลด ffmpeg แล้วเพิ่มลง PATH  หรือ  pip install imageio-ffmpeg
  • ไม่ resample / ไม่ fix-length ที่ขั้นนี้ → เก็บคุณภาพเสียงไว้
    ให้ cnn_extract.py / rf_extract.py ไป resample เองตาม logic เดิม
"""

import os
import json
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── Paths (แก้ได้ถ้าย้ายที่) ──────────────────────────────────────────────────
SRC_AUDIO_DIR = r"C:\Users\Acer\Downloads\archive (3)\public_dataset_v3\coughvid_20211012"
METADATA_PATH = r"C:\Users\Acer\Downloads\archive (3)\public_dataset_v3\coughvid_20211012\metadata_compiled"
DEST_ROOT     = r"C:\Users\Acer\Downloads\Cough Detection\public_dataset"

CLASSES        = ["covid", "healthy", "symptomatic"]
AUDIO_EXTS     = (".webm", ".ogg", ".wav", ".mp3", ".m4a", ".flac")
TRIM_TOP_DB    = 30          # ตัดความเงียบหัว-ท้าย (Cough Segmentation แบบเบา)
SUMMARY_JSON   = os.path.join(DEST_ROOT, "preprocessing_summary.json")


# ── 1. map เฉพาะคอลัมน์ status ────────────────────────────────────────────────
def map_status(value) -> str | None:
    if not isinstance(value, str):
        return None
    s = value.strip().lower()
    if s == "healthy":
        return "healthy"
    if s in ("covid-19", "covid", "covid19"):
        return "covid"
    if s == "symptomatic":
        return "symptomatic"
    return None


# ── 2. หา metadata (csv หรือ xlsx) ────────────────────────────────────────────
def load_metadata(path: str) -> pd.DataFrame:
    candidates = [path, path + ".csv", path + ".xlsx", path + ".xls"]
    for p in candidates:
        if os.path.isfile(p):
            print(f"📄 อ่าน metadata: {p}")
            if p.lower().endswith((".xlsx", ".xls")):
                return pd.read_excel(p)
            return pd.read_csv(p)
    raise FileNotFoundError(
        f"ไม่พบ metadata จาก: {candidates}\n"
        "เช็ค METADATA_PATH ว่าชี้ไปไฟล์ .csv หรือ .xlsx ถูกต้อง"
    )


# ── 3. index ไฟล์เสียงในโฟลเดอร์รวม (stem -> full path) ────────────────────────
def index_audio_files(folder: str) -> dict[str, str]:
    print(f"🔎 สแกนไฟล์เสียงใน: {folder}")
    table: dict[str, str] = {}
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(AUDIO_EXTS):
                stem = os.path.splitext(f)[0]
                table.setdefault(stem, os.path.join(root, f))
    print(f"   → เจอไฟล์เสียงทั้งหมด: {len(table):,} ไฟล์")
    return table


# ── 4. clean + normalize + segment + save ─────────────────────────────────────
def process_one(src_path: str, dst_path: str) -> bool:
    try:
        y, sr = librosa.load(src_path, sr=None, mono=True)   # ไม่ resample (เก็บ native sr)
        if y is None or len(y) == 0:
            return False

        # Cough Segmentation (เบา): ตัดความเงียบหัว-ท้าย
        y_trim, _ = librosa.effects.trim(y, top_db=TRIM_TOP_DB)
        if len(y_trim) > 0:
            y = y_trim

        # Guard: skip clips shorter than 0.5 s after trim (warn + continue)
        if len(y) < int(0.5 * sr):
            tqdm.write(f"⚠️  ข้าม (สั้นกว่า 0.5s หลัง trim): {os.path.basename(src_path)}")
            return False

        # Normalization: peak-normalize (ใช้ native sr เดิม ไม่ resample)
        peak = np.max(np.abs(y))
        if peak > 0:
            y = y / peak

        sf.write(dst_path, y, sr)   # original sample rate
        return True
    except Exception:
        return False


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("🚀 PREPROCESSING : เตรียม Dataset (Cleaning → Normalize → Segment)")
    print("=" * 70)

    df = load_metadata(METADATA_PATH)
    if "status" not in df.columns:
        raise KeyError(
            f"ไม่พบคอลัมน์ 'status' ใน metadata\nคอลัมน์ที่มี: {list(df.columns)[:20]} ..."
        )

    # นับ status ดิบ (เฉพาะคอลัมน์ status ตามที่สั่ง)
    print("\n📊 การกระจายของคอลัมน์ status (ดิบ):")
    print(df["status"].value_counts(dropna=False).to_string())

    # เตรียมโฟลเดอร์ปลายทาง
    for cls in CLASSES:
        os.makedirs(os.path.join(DEST_ROOT, cls), exist_ok=True)

    audio_index = index_audio_files(SRC_AUDIO_DIR)

    # สร้างรายการงาน
    tasks = []          # (uuid, cls, src_path)
    label_counts = {c: 0 for c in CLASSES}
    skipped_label = 0
    missing_audio = 0

    for _, row in df.iterrows():
        cls = map_status(row.get("status"))
        if cls is None:
            skipped_label += 1
            continue
        uuid = str(row.get("uuid", "")).strip()
        if not uuid:
            skipped_label += 1
            continue
        src = audio_index.get(uuid)
        if src is None:
            missing_audio += 1
            continue
        tasks.append((uuid, cls, src))
        label_counts[cls] += 1

    print(f"\n🗂  สรุปก่อนแปลงไฟล์:")
    for c in CLASSES:
        print(f"   {c:<12}: {label_counts[c]:,} (มีไฟล์เสียงพร้อมแปลง)")
    print(f"   ข้าม (status ว่าง/ไม่เข้าเกณฑ์) : {skipped_label:,}")
    print(f"   หาไฟล์เสียงจาก uuid ไม่เจอ      : {missing_audio:,}")
    print(f"   รวมงานที่จะแปลง                : {len(tasks):,}")

    # แปลงไฟล์
    print("\n⚙️  เริ่มแปลง (clean + normalize + segment + save .wav) ...")
    done = {c: 0 for c in CLASSES}
    failed = 0
    for uuid, cls, src in tqdm(tasks, desc="Processing", unit="file", ncols=90):
        dst = os.path.join(DEST_ROOT, cls, f"{uuid}.wav")
        if process_one(src, dst):
            done[cls] += 1
        else:
            failed += 1

    total_done = sum(done.values())

    # ── LOG สรุปสุดท้าย (เอาไปใส่โครงการได้เลย) ───────────────────────────────
    print("\n" + "=" * 70)
    print("✅ PREPROCESSING เสร็จสิ้น — สรุปผล")
    print("=" * 70)
    print(f"📁 ปลายทาง: {DEST_ROOT}")
    for c in CLASSES:
        print(f"   {c:<12}: {done[c]:,} ไฟล์")
    print(f"   {'รวม':<12}: {total_done:,} ไฟล์")
    print(f"   แปลงไม่สำเร็จ (ไฟล์เสีย) : {failed:,}")
    print("=" * 70)

    summary = {
        "created_at":    datetime.now(tz=timezone.utc).isoformat(),
        "src_audio_dir": SRC_AUDIO_DIR,
        "metadata_path": METADATA_PATH,
        "dest_root":     DEST_ROOT,
        "label_source":  "status column only",
        "trim_top_db":   TRIM_TOP_DB,
        "counts": {
            "per_class_saved":   done,
            "total_saved":       total_done,
            "skipped_no_label":  skipped_label,
            "missing_audio":     missing_audio,
            "failed_convert":    failed,
        },
    }
    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"💾 บันทึกสรุป: {SUMMARY_JSON}")
    print("\n👉 ขั้นต่อไป: รัน rf_extract.py และ cnn_extract.py (ชี้มาที่โฟลเดอร์นี้แล้ว)")


if __name__ == "__main__":
    main()