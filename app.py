"""
============================================================
CoughAI Backend — Ensemble (CNN + RF)  ·  Cloud Run ready
============================================================
สิ่งที่เพิ่มจากเวอร์ชันเดิม:
  1. ดาวน์โหลดโมเดลจาก Google Drive ตอน container start (gdown)
  2. /predict บันทึกผลลง history อัตโนมัติ (เว็บไอ → ขึ้น Dashboard)
  3. History เก็บถาวรด้วย Firestore (ถ้าตั้งค่าไว้) + in-memory fallback
  4. คืน risk_level + คำแนะนำเบื้องต้น (ภาษาไทย) กลับไปให้หน้าเว็บ

โมเดลที่ใช้:
  - cough_rf_model.pkl   (required, tabular 416-D)
  - cough_cnn_model.h5   (optional → ถ้าโหลดไม่ได้ fallback เป็น RF อย่างเดียว)
============================================================
"""

import os
import io
import json
import numpy as np
import joblib
import soundfile as sf
from collections import deque
from datetime import datetime, timezone
from threading import Lock

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

# ── Feature extractors (จาก pipeline เดิม — ต้องอยู่ใน repo) ──
from rf_extract  import extract_features            # 416-D vector
from cnn_extract import extract_features_cnn         # mel-spectrogram

# ─── TensorFlow: CPU mode (กัน CUDA error บน Cloud Run) ─────
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# ============================================================
# CONFIG
# ============================================================
LABELS         = ["covid", "healthy", "symptomatic"]
IMAGE_SHAPE    = (128, 128, 1)
RF_MODEL_PATH  = "cough_rf_model.pkl"   # tabular 416-D (Random Forest)
CNN_MODEL_PATH = "cough_cnn_model.h5"
MINMAX_PATH    = "cough_min_max.json"
UPLOAD_FOLDER  = "uploads"
ENSEMBLE_ALPHA = 0.5                  # weight ของ CNN (อีกครึ่งเป็น RF)

# Drive file IDs (ตั้งเป็น environment variable ตอน deploy)
RF_FILE_ID  = os.environ.get("RF_MODEL_FILE_ID",  "")
CNN_FILE_ID = os.environ.get("CNN_MODEL_FILE_ID", "")

# risk + คำแนะนำต่อ class (ภาษาไทย) — กรอบ "คัดกรอง" ไม่ใช่ "วินิจฉัย"
RISK_MAP = {"covid": "HIGH", "symptomatic": "MEDIUM", "healthy": "LOW"}
RECO_MAP = {
    "healthy": (
        "เสียงไอของคุณอยู่ในเกณฑ์ปกติ ดูแลสุขภาพให้แข็งแรงต่อไป "
        "พักผ่อนให้เพียงพอ ดื่มน้ำมาก ๆ หากมีอาการผิดปกติภายหลังให้สังเกตอาการต่อ"
    ),
    "symptomatic": (
        "ตรวจพบลักษณะการไอที่อาจบ่งชี้ภาวะทางเดินหายใจ แนะนำให้พักผ่อน ดื่มน้ำอุ่น "
        "หลีกเลี่ยงการแพร่เชื้อให้ผู้อื่น และพบแพทย์หากอาการไม่ดีขึ้นภายใน 2-3 วัน "
        "หรือมีไข้สูง เจ็บหน้าอก หายใจลำบาก"
    ),
    "covid": (
        "ตรวจพบลักษณะการไอที่อาจสัมพันธ์กับโควิด-19 แนะนำให้ตรวจ ATK ยืนยัน "
        "แยกกักตัว สวมหน้ากากอนามัย และติดต่อสายด่วนกรมควบคุมโรค 1422 เพื่อขอคำแนะนำ "
        "หากมีอาการรุนแรง เช่น หายใจหอบเหนื่อย โทร 1669 ทันที"
    ),
}

# ============================================================
# โหลดโมเดลจาก Drive (ถ้ายังไม่มีในเครื่อง)
# ============================================================
def ensure_model(path: str, file_id: str) -> bool:
    """ดาวน์โหลดโมเดลจาก Google Drive ถ้าไฟล์ยังไม่มี (cache ไว้)"""
    if os.path.exists(path):
        print(f"✅ พบไฟล์ {path} ในเครื่องแล้ว")
        return True
    if not file_id:
        print(f"⚠️  ไม่ได้ตั้ง file id สำหรับ {path}")
        return False
    try:
        import gdown
        print(f"⬇️  กำลังโหลด {path} จาก Google Drive ...")
        gdown.download(id=file_id, output=path, quiet=False)
        return os.path.exists(path)
    except Exception as e:
        print(f"❌ โหลด {path} ไม่ได้: {e}")
        return False


# ============================================================
# Flask app
# ============================================================
app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

print("📂 กำลังเตรียมโมเดล...")

# ── RF (required, tabular 416-D) ──
ensure_model(RF_MODEL_PATH, RF_FILE_ID)
try:
    rf_model = joblib.load(RF_MODEL_PATH)
    print(f"✅ RF model: {RF_MODEL_PATH}")
except Exception as e:
    print(f"❌ โหลด RF ไม่ได้: {e}")
    raise

# ── CNN (optional) ──
cnn_model = None
ensure_model(CNN_MODEL_PATH, CNN_FILE_ID)
try:
    import tensorflow as tf
    cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH)
    print(f"✅ CNN model: {CNN_MODEL_PATH}  (ensemble ON, alpha={ENSEMBLE_ALPHA})")
except Exception as e:
    print(f"⚠️  โหลด CNN ไม่ได้ ({e}) → ใช้แค่ RF (ensemble OFF)")

# global min/max จากตอนเทรน → normalize ตอน inference ให้ตรงกับ train
CNN_MIN, CNN_MAX = None, None
if cnn_model is not None and os.path.exists(MINMAX_PATH):
    try:
        with open(MINMAX_PATH) as f:
            mm = json.load(f)
        CNN_MIN, CNN_MAX = float(mm["min"]), float(mm["max"])
        print(f"✅ CNN normalization: global min={CNN_MIN:.3f} max={CNN_MAX:.3f}")
    except Exception as e:
        print(f"⚠️  อ่าน {MINMAX_PATH} ไม่ได้ ({e}) → fallback per-sample normalize")
else:
    if cnn_model is not None:
        print(f"⚠️  ไม่พบ {MINMAX_PATH} → fallback per-sample normalize (ผลอาจเพี้ยนเล็กน้อย)")

# ── Firestore (optional persistence) ──
db = None
try:
    from google.cloud import firestore
    db = firestore.Client()
    # ทดสอบ connection แบบเบา ๆ
    _ = db.collection("screenings").limit(1).get()
    print("✅ Firestore เชื่อมต่อแล้ว (history เก็บถาวร)")
except Exception as e:
    print(f"⚠️  Firestore ไม่พร้อม ({e}) → history เก็บใน RAM ชั่วคราว")

# in-memory fallback (รีเซ็ตเมื่อ container restart)
MEM_HISTORY = deque(maxlen=100)
HISTORY_LOCK = Lock()


# ============================================================
# HISTORY HELPERS
# ============================================================
def save_history(record: dict):
    """บันทึกผลลง Firestore (ถ้ามี) + in-memory เสมอ"""
    with HISTORY_LOCK:
        MEM_HISTORY.appendleft(record)
    if db is not None:
        try:
            db.collection("screenings").add(record)
        except Exception as e:
            print(f"⚠️  เขียน Firestore ไม่สำเร็จ: {e}")


def load_history(limit: int = 100) -> list:
    """อ่านประวัติล่าสุด — Firestore ก่อน ถ้าไม่มีใช้ RAM"""
    if db is not None:
        try:
            docs = (
                db.collection("screenings")
                .order_by("timestamp", direction=firestore.Query.DESCENDING)
                .limit(limit)
                .stream()
            )
            return [d.to_dict() for d in docs]
        except Exception as e:
            print(f"⚠️  อ่าน Firestore ไม่สำเร็จ: {e}")
    with HISTORY_LOCK:
        return list(MEM_HISTORY)


# ============================================================
# PREDICTION
# ============================================================
def prepare_cnn_input(wav_path: str):
    """สกัด mel-spectrogram → (1, 128, 128, 1)"""
    feat = extract_features_cnn(wav_path)
    if feat is None:
        return None
    cols = IMAGE_SHAPE[1]
    if feat.shape[1] > cols:
        feat = feat[:, :cols, :]
    elif feat.shape[1] < cols:
        feat = np.pad(feat, ((0, 0), (0, cols - feat.shape[1]), (0, 0)))

    # normalize ให้ตรงกับตอนเทรน: ใช้ global min/max ถ้ามี
    if CNN_MIN is not None and CNN_MAX is not None:
        feat = (feat - CNN_MIN) / (CNN_MAX - CNN_MIN + 1e-8)
        feat = np.clip(feat, 0.0, 1.0)        # กันค่าหลุดช่วง
    else:
        fmin, fmax = float(feat.min()), float(feat.max())
        feat = (feat - fmin) / (fmax - fmin + 1e-8)

    return feat[np.newaxis, ...].astype(np.float32)


TRIM_TOP_DB = 30   # ตรงกับ preprocessing.py

def preprocess_wav(path: str) -> None:
    """
    trim ความเงียบหัว-ท้าย + peak-normalize ให้ตรงกับ preprocessing.py
    (กัน train/serve mismatch — ตอนเทรนไฟล์ผ่าน trim มาแล้ว)
    เขียนทับไฟล์เดิม ถ้าพังก็ปล่อยไฟล์เดิมไว้ (ไม่ทำให้ /predict ล้ม)
    """
    try:
        import librosa
        y, sr = librosa.load(path, sr=None, mono=True)
        if y is None or len(y) == 0:
            return
        y_trim, _ = librosa.effects.trim(y, top_db=TRIM_TOP_DB)
        if len(y_trim) > 0:
            y = y_trim
        peak = float(np.max(np.abs(y)))
        if peak > 0:
            y = y / peak
        sf.write(path, y, sr)
    except Exception as e:
        print(f"⚠️  preprocess_wav ข้าม ({e}) → ใช้ไฟล์เดิม")


def predict_ensemble(wav_path: str) -> dict:
    """Soft-voting ensemble (CNN + RF) — fallback เป็น RF ถ้า CNN ไม่พร้อม"""
    feat_rf, err = extract_features(wav_path)
    if feat_rf is None:
        raise RuntimeError(f"feature extraction failed: {err}")
    p_rf = rf_model.predict_proba(feat_rf.reshape(1, -1))[0]

    if cnn_model is not None:
        x_cnn = prepare_cnn_input(wav_path)
        if x_cnn is not None:
            p_cnn = cnn_model.predict(x_cnn, verbose=0)[0]
            p_ens = ENSEMBLE_ALPHA * p_cnn + (1 - ENSEMBLE_ALPHA) * p_rf
            mode  = "ensemble"
        else:
            p_ens, mode = p_rf, "rf_only(cnn_feat_failed)"
    else:
        p_ens, mode = p_rf, "rf_only"

    pred_idx = int(np.argmax(p_ens))
    label    = LABELS[pred_idx]
    top_conf = round(float(p_ens[pred_idx]) * 100, 1)
    probs    = [{"label": l, "score": float(p)} for l, p in zip(LABELS, p_ens)]

    return {
        "classification": label,
        "confidence":     top_conf,
        "risk_level":     RISK_MAP.get(label, "LOW"),
        "recommendation": RECO_MAP.get(label, ""),
        "probabilities":  probs,
        "mode":           mode,
    }


# ============================================================
# STATIC ROUTES
# ============================================================
@app.route("/")
def homepage():          return send_from_directory(".", "homepage.html")

@app.route("/app")
def index_page():        return send_from_directory(".", "index.html")

@app.route("/dashboard")
def dashboard_page():     return send_from_directory(".", "dashboard.html")

@app.route("/homepage.css")
def homepage_css():       return send_from_directory(".", "homepage.css")

@app.route("/homepage.js")
def homepage_js():        return send_from_directory(".", "homepage.js")

@app.route("/style.css")
def css():                return send_from_directory(".", "style.css")

@app.route("/script.js")
def js():                 return send_from_directory(".", "script.js")

@app.route("/dashboard.js")
def dashboard_js_route(): return send_from_directory(".", "dashboard.js")


# ============================================================
# STATUS
# ============================================================
@app.route("/status", methods=["GET"])
def status():
    return jsonify({
        "message":   "Smart Cough Detection API is running 🚀",
        "model":     "ensemble" if cnn_model is not None else "rf_only",
        "alpha_cnn": ENSEMBLE_ALPHA if cnn_model is not None else None,
        "history":   "firestore" if db is not None else "in-memory",
    })


# ============================================================
# PREDICT — ensemble + บันทึกประวัติอัตโนมัติ
# ============================================================
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "ไม่พบไฟล์เสียงในคำขอ"}), 400

    audio_file = request.files["file"]
    filename   = secure_filename(audio_file.filename or "cough.wav")
    filepath   = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    try:
        audio_data, sr = sf.read(io.BytesIO(audio_file.read()))
        sf.write(filepath, audio_data, sr, format="WAV")

        preprocess_wav(filepath)          # trim+normalize ให้ตรงกับตอนเทรน
        result = predict_ensemble(filepath)

        # ── บันทึกลงประวัติ (เว็บไอ → Dashboard) ──
        record = {
            "device_id":      request.form.get("device_id", "web"),
            "classification": result["classification"],
            "confidence":     result["confidence"],
            "risk_level":     result["risk_level"],
            "probabilities":  result["probabilities"],
            "timestamp":      datetime.now(timezone.utc).isoformat(),
        }
        save_history(record)

        return jsonify(result), 200

    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(filepath):
            try: os.remove(filepath)
            except: pass


# ============================================================
# HISTORY ENDPOINTS (หน้า Dashboard เรียกใช้)
#   /device/history เก็บไว้เพื่อ compat กับ dashboard.js เดิม
# ============================================================
@app.route("/history", methods=["GET"])
@app.route("/device/history", methods=["GET"])
def history():
    items = load_history(limit=100)
    return jsonify({"count": len(items), "items": items})


@app.route("/device/latest", methods=["GET"])
def device_latest():
    items = load_history(limit=1)
    return jsonify(items[0] if items else {})


# เผื่ออนาคตต่อ edge device — รับผลจากภายนอกได้เหมือนเดิม
@app.route("/device/result", methods=["POST"])
def device_result():
    try:
        data = request.get_json(force=True, silent=True) or {}
        for f in ("device_id", "classification", "confidence"):
            if f not in data:
                return jsonify({"error": f"missing field: {f}"}), 400
        data.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        data.setdefault("risk_level",
                        RISK_MAP.get(str(data["classification"]).lower(), "LOW"))
        save_history(data)
        return jsonify({"ok": True}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# RUN (local dev เท่านั้น — Cloud Run ใช้ gunicorn)
# ============================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"\n🚀 CoughAI running at http://localhost:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=False)
