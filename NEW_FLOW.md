# Flow ใหม่ (เริ่มจาก preprocessing เป็นตัวตั้งต้น)

สร้างขึ้นเพื่อให้ pipeline **ต่อกันจริง + reproduce ได้** โดยไม่ทับของเดิม
(ของเดิม `verify_sound` เป็น dataset เก่าที่ไม่มีสคริปต์สร้างซ้ำได้ → เลิกใช้)

> ⚠️ ทุกไฟล์ `_new` ก็อป logic มาจากตัวเดิมแบบเป๊ะ ๆ เปลี่ยนแค่ **path เข้า** และ **ชื่อไฟล์ออก**
> โมเดลที่ deploy อยู่ (`cough_rf_model.pkl`, `cough_cnn_model.h5`) **ไม่ถูกแตะ**

## Dataset ที่ใช้
- ต้นทาง: COUGHVID (`metadata_compiled.csv` + เสียงดิบที่ `archive (3)\...\coughvid_20211012`)
- `preprocessing.py` → เขียนลง **`public_dataset\{covid,healthy,symptomatic}` (root)** ← นี่คือ dataset ใหม่
- `*_new.py` ทุกตัวอ่านจากโฟลเดอร์นี้ (ไม่ใช่ `verify_sound`)

## ลำดับการรัน

```
# 1) (แนะนำ) ล้างโฟลเดอร์คลาสเดิมก่อน เพื่อให้ได้ dataset สดล้วน
#    ลบเฉพาะ 3 โฟลเดอร์นี้: covid\  healthy\  symptomatic\

# 2) สร้าง dataset ใหม่จาก COUGHVID  (logic เดิม ไม่แก้)
python preprocessing.py
#    → public_dataset\covid|healthy|symptomatic\*.wav

# 3) สกัดฟีเจอร์ RF
python rf_extract_new.py
#    → features_raw_new.npz, feature_names_new.npy, metadata_new.json

# 4) เทรน RF
python train_rf_new.py
#    → cough_rf_model_new.pkl, rf_confusion_matrix_new.png, rf_roc_curve_new.png

# 5) สกัด mel-spectrogram สำหรับ CNN
python cnn_extract_new.py
#    → cnn_features_new\*.npy, cnn_data_manifest_new.json

# 6) เทรน CNN
python train_cnn_new.py
#    → cough_cnn_model_new.h5, cnn_confusion_matrix_new.png,
#      cnn_roc_curve_new.png, cnn_training_curve_new.png

# 7) สร้าง min/max สำหรับ normalize ตอน inference
python dump_min_max_new.py
#    → cough_min_max_new.json
```

## ไฟล์เดิม ↔ ไฟล์ใหม่ / output

| ขั้น | สคริปต์ใหม่ | input | output ใหม่ |
|------|-----------|-------|------------|
| extract RF  | `rf_extract_new.py`   | `covid/healthy/symptomatic` (root) | `features_raw_new.npz` |
| train RF    | `train_rf_new.py`     | `features_raw_new.npz` | `cough_rf_model_new.pkl` |
| extract CNN | `cnn_extract_new.py`  | `covid/healthy/symptomatic` (root) | `cnn_features_new\`, `cnn_data_manifest_new.json` |
| train CNN   | `train_cnn_new.py`    | `cnn_data_manifest_new.json` | `cough_cnn_model_new.h5` |
| min/max     | `dump_min_max_new.py` | `cnn_data_manifest_new.json` | `cough_min_max_new.json` |

## เอาโมเดลใหม่ขึ้น production (ทำทีหลัง เมื่อพอใจผลแล้ว)
แก้ใน `app.py`:
```python
RF_MODEL_PATH  = "cough_rf_model_new.pkl"
CNN_MODEL_PATH = "cough_cnn_model_new.h5"
MINMAX_PATH    = "cough_min_max_new.json"
```
(ยังไม่ต้องทำตอนนี้ — แอปที่รันอยู่ใช้ของเดิมต่อได้ปกติ)
