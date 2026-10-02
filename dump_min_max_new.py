"""
dump_min_max.py — สร้าง cough_min_max.json (เวอร์ชันเบา + มี progress bar)
================================================================
หา global min/max แบบไล่ทีละไฟล์ ไม่ยัดทุกอย่างเข้าแรมพร้อมกัน
→ เร็วกว่าและไม่กินแรม เห็น progress ชัดว่ารันถึงไหนแล้ว

ค่าที่ได้ตรงเป๊ะกับ normalization ใน train_cnn.py
(crop/pad คอลัมน์ก่อนคำนวณ เหมือนตอนเทรน)

รันในโฟลเดอร์เดียวกับ cnn_data_manifest.json:
    python dump_min_max.py
================================================================
"""
import json
import numpy as np
from tqdm import tqdm

DATA_MANIFEST_PATH = "cnn_data_manifest_new.json"
OUT_PATH           = "cough_min_max_new.json"
IMAGE_SHAPE        = (128, 128, 1)   # ต้องตรงกับ train_cnn.py

with open(DATA_MANIFEST_PATH) as f:
    manifest = json.load(f)

cols = IMAGE_SHAPE[1]
gmin = np.inf
gmax = -np.inf
seen = 0
missing = 0

for entry in tqdm(manifest, desc="หา min/max", unit="file", ncols=80, colour="cyan"):
    try:
        feat = np.load(entry["filepath"])
    except FileNotFoundError:
        missing += 1
        continue

    # crop/pad คอลัมน์ให้เท่าตอนเทรน (เป๊ะ ๆ)
    if feat.shape[1] > cols:
        feat = feat[:, :cols, :]
    elif feat.shape[1] < cols:
        feat = np.pad(feat, ((0, 0), (0, cols - feat.shape[1]), (0, 0)))

    fmin = float(feat.min())
    fmax = float(feat.max())
    if fmin < gmin: gmin = fmin
    if fmax > gmax: gmax = fmax
    seen += 1

if seen == 0:
    raise SystemExit("ไม่พบไฟล์ฟีเจอร์เลย — เช็ก path ใน cnn_data_manifest.json")

with open(OUT_PATH, "w") as f:
    json.dump({"min": gmin, "max": gmax}, f, indent=2)

print(f"\nบันทึก {OUT_PATH}")
print(f"   min = {gmin:.4f}   max = {gmax:.4f}")
print(f"   samples = {seen:,}   (missing files = {missing})")