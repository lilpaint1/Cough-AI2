"""
============================================================
CoughAI Backend — Ensemble (CNN + RF)
Cloud Run / Vercel ready
============================================================

สิ่งที่รองรับ:
  1. ดาวน์โหลดโมเดลจาก Google Drive เมื่อจำเป็น
  2. โหลด RF + CNN แบบ lazy loading
     -> ไม่โหลดโมเดลตอน import app.py
  3. ลด top-level imports เพื่อลด startup failure
  4. /predict บันทึกผลลง history อัตโนมัติ
  5. History ใช้ Firestore ถ้ามี dependency/credentials
     ไม่เช่นนั้นใช้ in-memory
  6. ใช้ /tmp สำหรับไฟล์ชั่วคราวบน Vercel
  7. CNN + RF soft-voting ensemble 50/50
  8. หน้า HTML/CSS/JS เดิมยังใช้งานได้

โมเดล:
  - cough_rf_model.pkl
  - cough_cnn_model.h5

Vercel Environment Variables:
  RF_MODEL_FILE_ID
  CNN_MODEL_FILE_ID
============================================================
"""

import os

# ============================================================
# ENVIRONMENT
# ต้องตั้งก่อน import TensorFlow
# ============================================================
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import io
import json
import uuid

from collections import deque
from datetime import datetime, timezone
from threading import Lock

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename


# ============================================================
# BASE / TEMP PATHS
# ============================================================
BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

# ------------------------------------------------------------
# Vercel เขียนไฟล์ runtime ได้ใน /tmp
# ------------------------------------------------------------
TMP_DIR = "/tmp/coughai"

MODEL_DIR = os.path.join(
    TMP_DIR,
    "models",
)

UPLOAD_FOLDER = os.path.join(
    TMP_DIR,
    "uploads",
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True,
)

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True,
)


# ============================================================
# MODEL PATHS
# ============================================================
RF_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "cough_rf_model.pkl",
)

CNN_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "cough_cnn_model.h5",
)

# ------------------------------------------------------------
# ไฟล์ read-only จาก repository
# ------------------------------------------------------------
MINMAX_PATH = os.path.join(
    BASE_DIR,
    "cough_min_max.json",
)


# ============================================================
# CONFIG
# ============================================================
LABELS = [
    "covid",
    "healthy",
    "symptomatic",
]

IMAGE_SHAPE = (
    128,
    128,
    1,
)

ENSEMBLE_ALPHA = 0.5

TRIM_TOP_DB = 30


# ============================================================
# GOOGLE DRIVE IDS
# ============================================================
RF_FILE_ID = os.environ.get(
    "RF_MODEL_FILE_ID",
    "",
).strip()

CNN_FILE_ID = os.environ.get(
    "CNN_MODEL_FILE_ID",
    "",
).strip()


# ============================================================
# RISK + RECOMMENDATION
# ============================================================
# กรอบ "คัดกรอง" ไม่ใช่การวินิจฉัย
RISK_MAP = {
    "covid": "HIGH",
    "symptomatic": "MEDIUM",
    "healthy": "LOW",
}

RECO_MAP = {
    "healthy": (
        "เสียงไอของคุณอยู่ในเกณฑ์ปกติ ดูแลสุขภาพให้แข็งแรงต่อไป "
        "พักผ่อนให้เพียงพอ ดื่มน้ำมาก ๆ หากมีอาการผิดปกติภายหลังให้สังเกตอาการต่อ"
    ),
    "symptomatic": (
        "ตรวจพบลักษณะการไอที่อาจบ่งชี้ภาวะทางเดินหายใจ แนะนำให้พักผ่อน "
        "ดื่มน้ำอุ่น หลีกเลี่ยงการแพร่เชื้อให้ผู้อื่น และพบแพทย์หากอาการไม่ดีขึ้น "
        "ภายใน 2-3 วัน หรือมีไข้สูง เจ็บหน้าอก หายใจลำบาก"
    ),
    "covid": (
        "ตรวจพบลักษณะการไอที่อาจสัมพันธ์กับโควิด-19 แนะนำให้ตรวจ ATK ยืนยัน "
        "แยกกักตัว สวมหน้ากากอนามัย และติดต่อสายด่วนกรมควบคุมโรค 1422 "
        "เพื่อขอคำแนะนำ หากมีอาการรุนแรง เช่น หายใจหอบเหนื่อย โทร 1669 ทันที"
    ),
}


# ============================================================
# FLASK APP
# ============================================================
app = Flask(
    __name__,
)

CORS(app)

app.config[
    "UPLOAD_FOLDER"
] = UPLOAD_FOLDER


# ============================================================
# GLOBAL MODEL STATE
# ============================================================
rf_model = None
cnn_model = None

RF_READY = False
CNN_READY = False

MODEL_LOAD_ERROR = None

MODEL_LOCK = Lock()


# ============================================================
# CNN NORMALIZATION STATE
# ============================================================
CNN_MIN = None
CNN_MAX = None


# ============================================================
# FIRESTORE STATE
# ============================================================
db = None
FIRESTORE_INIT_ATTEMPTED = False

FIRESTORE_LOCK = Lock()


# ============================================================
# IN-MEMORY HISTORY
# ============================================================
MEM_HISTORY = deque(
    maxlen=100
)

HISTORY_LOCK = Lock()


# ============================================================
# DOWNLOAD MODEL
# ============================================================
def ensure_model(
    path: str,
    file_id: str,
) -> bool:
    """
    ดาวน์โหลดโมเดลจาก Google Drive ถ้ายังไม่มีใน /tmp
    """

    # --------------------------------------------------------
    # มีไฟล์อยู่แล้ว
    # --------------------------------------------------------
    if os.path.exists(path):

        try:

            size_bytes = os.path.getsize(
                path
            )

            if size_bytes > 0:

                size_mb = (
                    size_bytes
                    / (1024 * 1024)
                )

                print(
                    f"✅ พบโมเดล "
                    f"{os.path.basename(path)} "
                    f"({size_mb:.1f} MB)"
                )

                return True

        except Exception:
            pass

    # --------------------------------------------------------
    # ไม่มี File ID
    # --------------------------------------------------------
    if not file_id:

        print(
            f"⚠️ ไม่มี File ID สำหรับ "
            f"{os.path.basename(path)}"
        )

        return False

    # --------------------------------------------------------
    # Download
    # --------------------------------------------------------
    try:

        import gdown

        temp_path = (
            path
            + ".download"
        )

        # ล้างไฟล์ชั่วคราวเดิม
        if os.path.exists(
            temp_path
        ):

            try:
                os.remove(
                    temp_path
                )
            except Exception:
                pass

        print(
            f"⬇️ กำลังดาวน์โหลด "
            f"{os.path.basename(path)} "
            "จาก Google Drive ..."
        )

        downloaded = gdown.download(
            id=file_id,
            output=temp_path,
            quiet=False,
        )

        if not downloaded:

            print(
                f"❌ ดาวน์โหลด "
                f"{os.path.basename(path)} "
                "ไม่สำเร็จ"
            )

            return False

        if not os.path.exists(
            temp_path
        ):

            print(
                f"❌ ไม่พบไฟล์ที่ดาวน์โหลด "
                f"{temp_path}"
            )

            return False

        size_bytes = os.path.getsize(
            temp_path
        )

        if size_bytes <= 0:

            print(
                f"❌ ไฟล์ "
                f"{os.path.basename(path)} "
                "มีขนาด 0 bytes"
            )

            try:
                os.remove(
                    temp_path
                )
            except Exception:
                pass

            return False

        # ย้ายไฟล์หลัง download สำเร็จ
        os.replace(
            temp_path,
            path,
        )

        size_mb = (
            size_bytes
            / (1024 * 1024)
        )

        print(
            f"✅ ดาวน์โหลดสำเร็จ "
            f"{os.path.basename(path)} "
            f"({size_mb:.1f} MB)"
        )

        return True

    except Exception as e:

        print(
            f"❌ ดาวน์โหลด "
            f"{os.path.basename(path)} "
            f"ไม่ได้: {e}"
        )

        return False


# ============================================================
# FIRESTORE
# ============================================================
def get_firestore():
    """
    Firestore แบบ lazy

    ถ้าไม่มี package หรือ credentials
    ระบบจะ fallback เป็น in-memory
    """

    global db
    global FIRESTORE_INIT_ATTEMPTED

    if FIRESTORE_INIT_ATTEMPTED:
        return db

    with FIRESTORE_LOCK:

        if FIRESTORE_INIT_ATTEMPTED:
            return db

        FIRESTORE_INIT_ATTEMPTED = True

        try:

            from google.cloud import firestore

            db = firestore.Client()

            print(
                "✅ Firestore client พร้อมใช้งาน"
            )

        except Exception as e:

            db = None

            print(
                f"⚠️ Firestore ไม่พร้อม ({e}) "
                "→ ใช้ in-memory history"
            )

    return db


# ============================================================
# LOAD MODELS LAZILY
# ============================================================
def load_models():
    """
    โหลด RF + CNN เฉพาะเมื่อจำเป็น เช่น /predict

    RF:
      required

    CNN:
      optional
      ถ้าโหลดไม่ได้ -> RF only
    """

    global rf_model
    global cnn_model

    global RF_READY
    global CNN_READY

    global MODEL_LOAD_ERROR

    global CNN_MIN
    global CNN_MAX

    with MODEL_LOCK:

        # ====================================================
        # RF
        # ====================================================
        if not RF_READY:

            if not ensure_model(
                RF_MODEL_PATH,
                RF_FILE_ID,
            ):

                MODEL_LOAD_ERROR = (
                    "ไม่สามารถดาวน์โหลด "
                    "Random Forest model ได้ "
                    "กรุณาตรวจสอบ RF_MODEL_FILE_ID"
                )

                print(
                    f"❌ {MODEL_LOAD_ERROR}"
                )

            else:

                try:

                    import joblib

                    rf_model = joblib.load(
                        RF_MODEL_PATH
                    )

                    RF_READY = True

                    print(
                        "✅ RF model โหลดสำเร็จ"
                    )

                except Exception as e:

                    rf_model = None

                    MODEL_LOAD_ERROR = (
                        f"โหลด RF model ไม่ได้: {e}"
                    )

                    print(
                        f"❌ {MODEL_LOAD_ERROR}"
                    )

        # ====================================================
        # CNN
        # ====================================================
        if not CNN_READY:

            if not CNN_FILE_ID:

                print(
                    "⚠️ ไม่มี "
                    "CNN_MODEL_FILE_ID "
                    "→ ใช้ RF อย่างเดียว"
                )

            else:

                cnn_available = ensure_model(
                    CNN_MODEL_PATH,
                    CNN_FILE_ID,
                )

                if cnn_available:

                    try:

                        import tensorflow as tf

                        cnn_model = (
                            tf.keras.models.load_model(
                                CNN_MODEL_PATH,
                                compile=False,
                            )
                        )

                        CNN_READY = True

                        print(
                            "✅ CNN model โหลดสำเร็จ "
                            f"(ensemble ON, "
                            f"alpha={ENSEMBLE_ALPHA})"
                        )

                    except Exception as e:

                        cnn_model = None

                        print(
                            f"⚠️ โหลด CNN ไม่ได้: {e} "
                            "→ ใช้ RF อย่างเดียว"
                        )

                else:

                    print(
                        "⚠️ CNN model "
                        "ดาวน์โหลดไม่ได้ "
                        "→ ใช้ RF อย่างเดียว"
                    )

        # ====================================================
        # CNN NORMALIZATION
        # ====================================================
        if (
            CNN_READY
            and CNN_MIN is None
            and CNN_MAX is None
            and os.path.exists(
                MINMAX_PATH
            )
        ):

            try:

                with open(
                    MINMAX_PATH,
                    "r",
                    encoding="utf-8",
                ) as f:

                    mm = json.load(f)

                CNN_MIN = float(
                    mm["min"]
                )

                CNN_MAX = float(
                    mm["max"]
                )

                print(
                    "✅ CNN normalization: "
                    f"min={CNN_MIN:.3f}, "
                    f"max={CNN_MAX:.3f}"
                )

            except Exception as e:

                print(
                    f"⚠️ อ่าน "
                    f"{MINMAX_PATH} ไม่ได้: {e} "
                    "→ fallback per-sample normalize"
                )

        elif (
            CNN_READY
            and CNN_MIN is None
            and CNN_MAX is None
        ):

            print(
                "⚠️ ไม่พบ "
                "cough_min_max.json "
                "→ fallback per-sample normalize"
            )

    return RF_READY, CNN_READY


# ============================================================
# HISTORY HELPERS
# ============================================================
def save_history(
    record: dict,
):
    """
    เก็บ in-memory เสมอ
    และลองเขียน Firestore ถ้ามี
    """

    with HISTORY_LOCK:

        MEM_HISTORY.appendleft(
            record
        )

    firestore_db = get_firestore()

    if firestore_db is not None:

        try:

            firestore_db.collection(
                "screenings"
            ).add(
                record
            )

        except Exception as e:

            print(
                f"⚠️ เขียน Firestore ไม่สำเร็จ: {e}"
            )


def load_history(
    limit: int = 100,
) -> list:
    """
    อ่าน Firestore ก่อน
    ถ้าไม่ได้ -> in-memory
    """

    firestore_db = get_firestore()

    if firestore_db is not None:

        try:

            from google.cloud import firestore

            docs = (
                firestore_db
                .collection(
                    "screenings"
                )
                .order_by(
                    "timestamp",
                    direction=(
                        firestore.Query.DESCENDING
                    ),
                )
                .limit(
                    limit
                )
                .stream()
            )

            return [
                doc.to_dict()
                for doc in docs
            ]

        except Exception as e:

            print(
                f"⚠️ อ่าน Firestore ไม่สำเร็จ: {e}"
            )

    with HISTORY_LOCK:

        return list(
            MEM_HISTORY
        )


# ============================================================
# CNN INPUT
# ============================================================
def prepare_cnn_input(
    wav_path: str,
):
    """
    สกัด mel-spectrogram
    -> (1, 128, 128, 1)
    """

    import numpy as np
    from cnn_extract import (
        extract_features_cnn,
    )

    feat = extract_features_cnn(
        wav_path
    )

    if feat is None:

        return None

    cols = IMAGE_SHAPE[1]

    # --------------------------------------------------------
    # Crop
    # --------------------------------------------------------
    if feat.shape[1] > cols:

        feat = feat[
            :,
            :cols,
            :,
        ]

    # --------------------------------------------------------
    # Pad
    # --------------------------------------------------------
    elif feat.shape[1] < cols:

        feat = np.pad(
            feat,
            (
                (0, 0),
                (
                    0,
                    cols - feat.shape[1],
                ),
                (0, 0),
            ),
        )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------
    if (
        CNN_MIN is not None
        and CNN_MAX is not None
    ):

        feat = (
            feat - CNN_MIN
        ) / (
            CNN_MAX
            - CNN_MIN
            + 1e-8
        )

        feat = np.clip(
            feat,
            0.0,
            1.0,
        )

    else:

        fmin = float(
            feat.min()
        )

        fmax = float(
            feat.max()
        )

        feat = (
            feat - fmin
        ) / (
            fmax
            - fmin
            + 1e-8
        )

    return (
        feat[
            np.newaxis,
            ...,
        ]
        .astype(
            np.float32
        )
    )


# ============================================================
# AUDIO PREPROCESSING
# ============================================================
def preprocess_wav(
    path: str,
) -> None:
    """
    trim ความเงียบหัว-ท้าย
    + peak normalization
    """

    try:

        import librosa
        import numpy as np
        import soundfile as sf

        y, sr = librosa.load(
            path,
            sr=None,
            mono=True,
        )

        if (
            y is None
            or len(y) == 0
        ):

            return

        y_trim, _ = (
            librosa.effects.trim(
                y,
                top_db=TRIM_TOP_DB,
            )
        )

        if len(y_trim) > 0:

            y = y_trim

        peak = float(
            np.max(
                np.abs(y)
            )
        )

        if peak > 0:

            y = y / peak

        sf.write(
            path,
            y,
            sr,
        )

    except Exception as e:

        print(
            f"⚠️ preprocess_wav ข้าม ({e}) "
            "→ ใช้ไฟล์เดิม"
        )


# ============================================================
# ENSEMBLE
# ============================================================
def predict_ensemble(
    wav_path: str,
) -> dict:
    """
    Soft-voting ensemble

    CNN = 50%
    RF  = 50%

    หาก CNN ใช้ไม่ได้:
      -> RF only
    """

    import numpy as np
    from rf_extract import (
        extract_features,
    )

    rf_ready, cnn_ready = (
        load_models()
    )

    # --------------------------------------------------------
    # RF จำเป็น
    # --------------------------------------------------------
    if not rf_ready:

        raise RuntimeError(
            MODEL_LOAD_ERROR
            or
            "Random Forest model "
            "ไม่พร้อมใช้งาน"
        )

    # --------------------------------------------------------
    # RF prediction
    # --------------------------------------------------------
    feat_rf, err = (
        extract_features(
            wav_path
        )
    )

    if feat_rf is None:

        raise RuntimeError(
            f"feature extraction failed: {err}"
        )

    p_rf = (
        rf_model.predict_proba(
            feat_rf.reshape(
                1,
                -1,
            )
        )[0]
    )

    # --------------------------------------------------------
    # CNN
    # --------------------------------------------------------
    if cnn_ready:

        try:

            x_cnn = (
                prepare_cnn_input(
                    wav_path
                )
            )

            if x_cnn is not None:

                p_cnn = (
                    cnn_model.predict(
                        x_cnn,
                        verbose=0,
                    )[0]
                )

                # ============================================
                # SOFT VOTING
                # ============================================
                p_ens = (
                    ENSEMBLE_ALPHA
                    * p_cnn
                    +
                    (
                        1
                        - ENSEMBLE_ALPHA
                    )
                    * p_rf
                )

                mode = "ensemble"

            else:

                p_ens = p_rf

                mode = (
                    "rf_only("
                    "cnn_feat_failed)"
                )

        except Exception as e:

            print(
                f"⚠️ CNN inference failed: {e}"
            )

            p_ens = p_rf

            mode = "rf_only(cnn_failed)"

    else:

        p_ens = p_rf

        mode = "rf_only"

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------
    pred_idx = int(
        np.argmax(
            p_ens
        )
    )

    label = LABELS[
        pred_idx
    ]

    top_conf = round(
        float(
            p_ens[
                pred_idx
            ]
        )
        * 100,
        1,
    )

    probs = [
        {
            "label": label_name,
            "score": float(
                prob
            ),
        }
        for label_name, prob
        in zip(
            LABELS,
            p_ens,
        )
    ]

    return {
        "classification":
            label,

        "confidence":
            top_conf,

        "risk_level":
            RISK_MAP.get(
                label,
                "LOW",
            ),

        "recommendation":
            RECO_MAP.get(
                label,
                "",
            ),

        "probabilities":
            probs,

        "mode":
            mode,
    }


# ============================================================
# STATIC ROUTES
# ============================================================
@app.route("/")
def homepage():

    return send_from_directory(
        BASE_DIR,
        "homepage.html",
    )


@app.route("/app")
def index_page():

    return send_from_directory(
        BASE_DIR,
        "index.html",
    )


@app.route("/dashboard")
def dashboard_page():

    return send_from_directory(
        BASE_DIR,
        "dashboard.html",
    )


@app.route("/homepage.css")
def homepage_css():

    return send_from_directory(
        BASE_DIR,
        "homepage.css",
    )


@app.route("/homepage.js")
def homepage_js():

    return send_from_directory(
        BASE_DIR,
        "homepage.js",
    )


@app.route("/style.css")
def css():

    return send_from_directory(
        BASE_DIR,
        "style.css",
    )


@app.route("/script.js")
def js():

    return send_from_directory(
        BASE_DIR,
        "script.js",
    )


@app.route("/dashboard.js")
def dashboard_js_route():

    return send_from_directory(
        BASE_DIR,
        "dashboard.js",
    )


@app.route("/i18n.js")
def i18n_js():

    return send_from_directory(
        BASE_DIR,
        "i18n.js",
    )


# ============================================================
# STATUS
# ============================================================
@app.route(
    "/status",
    methods=["GET"],
)
def status():

    return jsonify({

        "message":
            "Smart Cough Detection API is running 🚀",

        "model":
            (
                "ensemble"
                if RF_READY
                and CNN_READY
                else
                "rf_only"
                if RF_READY
                else
                "not_loaded"
            ),

        "rf_ready":
            RF_READY,

        "cnn_ready":
            CNN_READY,

        "alpha_cnn":
            (
                ENSEMBLE_ALPHA
                if RF_READY
                and CNN_READY
                else None
            ),

        "history":
            (
                "firestore"
                if db is not None
                else "in-memory"
            ),

        "runtime":
            (
                "vercel"
                if os.environ.get(
                    "VERCEL"
                )
                else "local/cloud"
            ),

    })


# ============================================================
# PREDICT
# ============================================================
@app.route(
    "/predict",
    methods=["POST"],
)
def predict():

    if "file" not in request.files:

        return jsonify({
            "error":
                "ไม่พบไฟล์เสียงในคำขอ"
        }), 400

    audio_file = request.files[
        "file"
    ]

    original_name = secure_filename(
        audio_file.filename
        or "cough.wav"
    )

    extension = (
        os.path.splitext(
            original_name
        )[1]
        or ".wav"
    )

    unique_name = (
        f"{uuid.uuid4().hex}"
        f"{extension}"
    )

    filepath = os.path.join(
        app.config[
            "UPLOAD_FOLDER"
        ],
        unique_name,
    )

    try:

        # ====================================================
        # READ FILE
        # ====================================================
        audio_bytes = (
            audio_file.read()
        )

        if not audio_bytes:

            return jsonify({
                "error":
                    "ไฟล์เสียงว่างเปล่า"
            }), 400

        # ----------------------------------------------------
        # Lazy import
        # ----------------------------------------------------
        import soundfile as sf

        audio_data, sr = (
            sf.read(
                io.BytesIO(
                    audio_bytes
                )
            )
        )

        # ====================================================
        # WRITE TEMP WAV
        # ====================================================
        sf.write(
            filepath,
            audio_data,
            sr,
            format="WAV",
        )

        # ====================================================
        # PREPROCESS
        # ====================================================
        preprocess_wav(
            filepath
        )

        # ====================================================
        # PREDICT
        # ====================================================
        result = predict_ensemble(
            filepath
        )

        # ====================================================
        # SAVE HISTORY
        # ====================================================
        record = {

            "device_id":
                request.form.get(
                    "device_id",
                    "web",
                ),

            "classification":
                result[
                    "classification"
                ],

            "confidence":
                result[
                    "confidence"
                ],

            "risk_level":
                result[
                    "risk_level"
                ],

            "probabilities":
                result[
                    "probabilities"
                ],

            "timestamp":
                datetime.now(
                    timezone.utc
                ).isoformat(),

        }

        save_history(
            record
        )

        return jsonify(
            result
        ), 200

    except Exception as e:
      import traceback
      print("❌ Prediction error:")
      traceback.print_exc()

      return jsonify({
        "error": str(e),
        "error_type": type(e).__name__,
        "stage": "predict"
      }), 500

    finally:

        if os.path.exists(
            filepath
        ):

            try:

                os.remove(
                    filepath
                )

            except Exception:
                pass


# ============================================================
# HISTORY
# ============================================================
@app.route(
    "/history",
    methods=["GET"],
)
@app.route(
    "/device/history",
    methods=["GET"],
)
def history():

    items = load_history(
        limit=100
    )

    return jsonify({

        "count":
            len(items),

        "items":
            items,

    })


# ============================================================
# LATEST
# ============================================================
@app.route(
    "/device/latest",
    methods=["GET"],
)
def device_latest():

    items = load_history(
        limit=1
    )

    return jsonify(
        items[0]
        if items
        else {}
    )


# ============================================================
# DEVICE RESULT
# ============================================================
@app.route(
    "/device/result",
    methods=["POST"],
)
def device_result():

    try:

        data = request.get_json(
            force=True,
            silent=True,
        ) or {}

        for field in (
            "device_id",
            "classification",
            "confidence",
        ):

            if field not in data:

                return jsonify({
                    "error":
                        f"missing field: {field}"
                }), 400

        data.setdefault(
            "timestamp",
            datetime.now(
                timezone.utc
            ).isoformat(),
        )

        data.setdefault(
            "risk_level",
            RISK_MAP.get(
                str(
                    data[
                        "classification"
                    ]
                ).lower(),
                "LOW",
            ),
        )

        save_history(
            data
        )

        return jsonify({
            "ok": True
        }), 200

    except Exception as e:

        return jsonify({
            "error":
                str(e)
        }), 500


# ============================================================
# LOCAL DEVELOPMENT
# ============================================================
if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            8080,
        )
    )

    print(
        "\n🚀 CoughAI running at "
        f"http://localhost:{port}\n"
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
    )
