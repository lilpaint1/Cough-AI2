"""
============================================================
CoughAI Backend — Ensemble (CNN + RF)
Cloud Run / Vercel ready
============================================================

สิ่งที่รองรับ:
  1. ดาวน์โหลดโมเดลจาก Google Drive เมื่อจำเป็น
  2. โหลดโมเดลแบบ Lazy Loading
  3. RF และ CNN โหลด "ทีละตัว" เพื่อลด RAM
  4. RF prediction -> unload RF -> CNN prediction
  5. Ensemble CNN + RF แบบ Soft Voting 50/50
  6. /predict บันทึกผลลง history อัตโนมัติ
  7. Firestore optional + in-memory fallback
  8. ใช้ /tmp สำหรับไฟล์ runtime บน Vercel
  9. ลด top-level imports เพื่อลด startup failure
 10. หน้า HTML/CSS/JS เดิมยังใช้งานได้

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
os.environ.setdefault(
    "CUDA_VISIBLE_DEVICES",
    "-1",
)

os.environ.setdefault(
    "TF_CPP_MIN_LOG_LEVEL",
    "2",
)

# ลด thread/memory overhead
os.environ.setdefault(
    "OMP_NUM_THREADS",
    "1",
)

os.environ.setdefault(
    "OPENBLAS_NUM_THREADS",
    "1",
)

os.environ.setdefault(
    "MKL_NUM_THREADS",
    "1",
)

os.environ.setdefault(
    "NUMEXPR_NUM_THREADS",
    "1",
)


# ============================================================
# STANDARD LIBRARIES
# ============================================================
import io
import json
import uuid
import traceback

from collections import deque
from datetime import datetime, timezone
from threading import Lock


# ============================================================
# FLASK
# ============================================================
from flask import (
    Flask,
    request,
    jsonify,
    send_from_directory,
)

from flask_cors import CORS

from werkzeug.utils import secure_filename


# ============================================================
# BASE / RUNTIME PATHS
# ============================================================
BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

# Vercel /tmp เป็น writable filesystem
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
# LAZY_LOADER STUB FALLBACK (librosa .pyi หายบน Vercel)
# ============================================================
LIBROSA_STUB_ROOT = "/tmp/coughai/librosa_stubs"
LIBROSA_STUB_LOCK = Lock()


def _download_librosa_stubs() -> bool:
    """
    ดึงไฟล์ .pyi ของ librosa จาก wheel บน PyPI
    เก็บไว้ใน /tmp (ทำครั้งเดียวต่อ cold start)
    """

    import urllib.request
    import zipfile

    marker = os.path.join(
        LIBROSA_STUB_ROOT,
        ".ok",
    )

    with LIBROSA_STUB_LOCK:

        if os.path.exists(marker):
            return True

        try:

            try:
                from importlib.metadata import (
                    version,
                )

                ver = version("librosa")

            except Exception:
                ver = "0.10.2.post1"

            meta_url = (
                f"https://pypi.org/pypi/"
                f"librosa/{ver}/json"
            )

            print(
                f"⬇️ ดึง librosa stubs ({ver}) จาก PyPI..."
            )

            with urllib.request.urlopen(
                meta_url,
                timeout=20,
            ) as r:

                meta = json.loads(
                    r.read().decode("utf-8")
                )

            wheel_url = None

            for item in meta.get("urls", []):

                if (
                    item.get("packagetype")
                    == "bdist_wheel"
                ):

                    wheel_url = item["url"]
                    break

            if not wheel_url:

                print(
                    "❌ ไม่พบ wheel ของ librosa บน PyPI"
                )

                return False

            with urllib.request.urlopen(
                wheel_url,
                timeout=60,
            ) as r:

                wheel_bytes = r.read()

            count = 0

            with zipfile.ZipFile(
                io.BytesIO(wheel_bytes)
            ) as z:

                for name in z.namelist():

                    if (
                        name.startswith("librosa/")
                        and name.endswith(".pyi")
                    ):

                        target = os.path.join(
                            LIBROSA_STUB_ROOT,
                            name,
                        )

                        os.makedirs(
                            os.path.dirname(target),
                            exist_ok=True,
                        )

                        with open(
                            target,
                            "wb",
                        ) as out:

                            out.write(
                                z.read(name)
                            )

                        count += 1

            if count == 0:

                print(
                    "❌ wheel ไม่มีไฟล์ .pyi"
                )

                return False

            with open(marker, "w") as f:
                f.write("ok")

            print(
                f"✅ librosa stubs พร้อม ({count} ไฟล์)"
            )

            return True

        except Exception as e:

            print(
                f"❌ ดึง librosa stubs ไม่สำเร็จ: {e}"
            )

            return False


def _install_lazy_loader_fallback() -> None:
    """
    patch lazy_loader.attach_stub
    ถ้า .pyi หายจาก bundle -> ใช้ stub จาก /tmp แทน
    """

    try:
        import lazy_loader
    except Exception as e:
        print(
            f"⚠️ ไม่มี lazy_loader ({e}) ข้าม patch"
        )
        return

    if getattr(
        lazy_loader,
        "_coughai_patched",
        False,
    ):
        return

    original_attach_stub = (
        lazy_loader.attach_stub
    )

    def patched_attach_stub(
        package_name,
        filename,
    ):

        stub_file = (
            filename
            if filename.endswith("i")
            else os.path.splitext(filename)[0]
            + ".pyi"
        )

        # stub ปกติมีอยู่ -> ใช้ตามเดิม
        if os.path.exists(stub_file):

            return original_attach_stub(
                package_name,
                filename,
            )

        parts = package_name.split(".")

        if parts[0] == "librosa":

            fallback = os.path.join(
                LIBROSA_STUB_ROOT,
                *parts,
                "__init__.pyi",
            )

            if not os.path.exists(fallback):
                _download_librosa_stubs()

            if os.path.exists(fallback):

                # ส่งเป็น .py เพื่อให้ lazy_loader
                # แปลงเป็น .pyi เองได้ทุกเวอร์ชัน
                return original_attach_stub(
                    package_name,
                    os.path.join(
                        os.path.dirname(fallback),
                        "__init__.py",
                    ),
                )

        return original_attach_stub(
            package_name,
            filename,
        )

    lazy_loader.attach_stub = (
        patched_attach_stub
    )

    lazy_loader._coughai_patched = True

    print(
        "✅ lazy_loader stub fallback ติดตั้งแล้ว"
    )


_install_lazy_loader_fallback()

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

# ไฟล์ read-only จาก repository
MINMAX_PATH = os.path.join(
    BASE_DIR,
    "cough_min_max.json",
)


# ============================================================
# MODEL CONFIG
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

# CNN = 50%
# RF  = 50%
ENSEMBLE_ALPHA = 0.5

TRIM_TOP_DB = 30


# ============================================================
# GOOGLE DRIVE FILE IDS
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
# RISK MAP
# ============================================================
RISK_MAP = {
    "covid": "HIGH",
    "symptomatic": "MEDIUM",
    "healthy": "LOW",
}


# ============================================================
# RECOMMENDATIONS
# ============================================================
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
# MODEL STATE
# ============================================================
rf_model = None
cnn_model = None

MODEL_LOAD_ERROR = None

MODEL_LOCK = Lock()


# ============================================================
# CNN NORMALIZATION
# ============================================================
CNN_MIN = None
CNN_MAX = None


# ============================================================
# FIRESTORE
# ============================================================
db = None
FIRESTORE_INIT_ATTEMPTED = False

FIRESTORE_LOCK = Lock()


# ============================================================
# HISTORY
# ============================================================
MEM_HISTORY = deque(
    maxlen=100
)

HISTORY_LOCK = Lock()


# ============================================================
# MODEL FILE HELPER
# ============================================================
def ensure_model(
    path: str,
    file_id: str,
) -> bool:
    """
    ตรวจสอบ model file
    ถ้าไม่มี -> ดาวน์โหลดจาก Google Drive

    ไฟล์ทั้งหมดถูกเก็บใน /tmp
    """

    # --------------------------------------------------------
    # Existing file
    # --------------------------------------------------------
    if os.path.exists(path):

        try:

            size_bytes = os.path.getsize(
                path
            )

            if size_bytes > 0:

                size_mb = (
                    size_bytes
                    / (
                        1024 * 1024
                    )
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
    # Missing File ID
    # --------------------------------------------------------
    if not file_id:

        print(
            f"❌ ไม่มี File ID สำหรับ "
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
                "❌ ไม่พบไฟล์ชั่วคราว "
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

        # ย้ายเมื่อดาวน์โหลดสำเร็จเท่านั้น
        os.replace(
            temp_path,
            path,
        )

        size_mb = (
            size_bytes
            / (
                1024 * 1024
            )
        )

        print(
            f"✅ ดาวน์โหลดสำเร็จ: "
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
# LOAD RF MODEL
# ============================================================
def load_rf_model() -> bool:
    """
    ดาวน์โหลด + โหลด RF เท่านั้น
    """

    global rf_model
    global MODEL_LOAD_ERROR

    if rf_model is not None:
        return True

    if not ensure_model(
        RF_MODEL_PATH,
        RF_FILE_ID,
    ):

        MODEL_LOAD_ERROR = (
            "ไม่สามารถดาวน์โหลด "
            "Random Forest model ได้ "
            "กรุณาตรวจสอบ RF_MODEL_FILE_ID"
        )

        return False

    try:

        import joblib

        print(
            "🧠 กำลังโหลด RF model..."
        )

        rf_model = joblib.load(
            RF_MODEL_PATH
        )

        MODEL_LOAD_ERROR = None

        print(
            "✅ RF model พร้อมใช้งาน"
        )

        return True

    except Exception as e:

        rf_model = None

        MODEL_LOAD_ERROR = (
            f"โหลด RF model ไม่ได้: {e}"
        )

        print(
            f"❌ {MODEL_LOAD_ERROR}"
        )

        return False


# ============================================================
# UNLOAD RF MODEL
# ============================================================
def unload_rf_model():
    """
    ปลด RF ออกจาก memory
    และลบ model file จาก /tmp
    """

    global rf_model

    print(
        "🧹 กำลัง unload RF model..."
    )

    rf_model = None

    try:

        import gc

        gc.collect()

    except Exception:
        pass

    if os.path.exists(
        RF_MODEL_PATH
    ):

        try:

            os.remove(
                RF_MODEL_PATH
            )

            print(
                "🗑️ ลบ RF model จาก /tmp แล้ว"
            )

        except Exception as e:

            print(
                f"⚠️ ลบ RF model ไม่สำเร็จ: {e}"
            )


# ============================================================
# LOAD CNN MODEL
# ============================================================
def load_cnn_model() -> bool:
    """
    ดาวน์โหลด + โหลด CNN เท่านั้น
    """

    global cnn_model
    global CNN_MIN
    global CNN_MAX

    if cnn_model is not None:
        return True

    if not CNN_FILE_ID:

        print(
            "⚠️ ไม่มี CNN_MODEL_FILE_ID"
        )

        return False

    if not ensure_model(
        CNN_MODEL_PATH,
        CNN_FILE_ID,
    ):

        print(
            "❌ ดาวน์โหลด CNN model ไม่สำเร็จ"
        )

        return False

    try:

        import tensorflow as tf

        print(
            "🧠 กำลังโหลด CNN model..."
        )

        cnn_model = (
            tf.keras.models.load_model(
                CNN_MODEL_PATH,
                compile=False,
            )
        )

        # ----------------------------------------------------
        # Load normalization values
        # ----------------------------------------------------
        CNN_MIN = None
        CNN_MAX = None

        if os.path.exists(
            MINMAX_PATH
        ):

            try:

                with open(
                    MINMAX_PATH,
                    "r",
                    encoding="utf-8",
                ) as f:

                    mm = json.load(
                        f
                    )

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
                    "⚠️ อ่าน "
                    f"{MINMAX_PATH} ไม่ได้: {e}"
                )

        if (
            CNN_MIN is None
            or CNN_MAX is None
        ):

            print(
                "⚠️ ไม่มี global normalization "
                "→ ใช้ per-sample normalization"
            )

        print(
            "✅ CNN model พร้อมใช้งาน"
        )

        return True

    except Exception as e:

        cnn_model = None

        print(
            f"❌ โหลด CNN model ไม่ได้: {e}"
        )

        return False


# ============================================================
# UNLOAD CNN MODEL
# ============================================================
def unload_cnn_model():
    """
    ปลด TensorFlow model
    และลบ CNN file จาก /tmp
    """

    global cnn_model
    global CNN_MIN
    global CNN_MAX

    print(
        "🧹 กำลัง unload CNN model..."
    )

    try:

        if cnn_model is not None:

            try:

                import tensorflow as tf

                tf.keras.backend.clear_session()

            except Exception:
                pass

    except Exception:
        pass

    cnn_model = None

    CNN_MIN = None
    CNN_MAX = None

    try:

        import gc

        gc.collect()

    except Exception:
        pass

    if os.path.exists(
        CNN_MODEL_PATH
    ):

        try:

            os.remove(
                CNN_MODEL_PATH
            )

            print(
                "🗑️ ลบ CNN model จาก /tmp แล้ว"
            )

        except Exception as e:

            print(
                f"⚠️ ลบ CNN model ไม่สำเร็จ: {e}"
            )


# ============================================================
# FIRESTORE
# ============================================================
def get_firestore():
    """
    Firestore แบบ lazy
    ถ้าไม่มี package / credentials
    จะ fallback เป็น in-memory
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
# HISTORY
# ============================================================
def save_history(
    record: dict,
):
    """
    เก็บ in-memory เสมอ
    และเขียน Firestore ถ้ามี
    """

    with HISTORY_LOCK:

        MEM_HISTORY.appendleft(
            record
        )

    firestore_db = (
        get_firestore()
    )

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
    Firestore ก่อน
    ถ้าไม่ได้ -> in-memory
    """

    firestore_db = (
        get_firestore()
    )

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
    สกัด Mel-spectrogram
    -> (1, 128, 128, 1)

    ใช้ pipeline เดิมของ cnn_extract.py
    """

    import numpy as np

    from cnn_extract import (
        extract_features_cnn,
    )

    feat = (
        extract_features_cnn(
            wav_path
        )
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
    # Global normalization
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

    # --------------------------------------------------------
    # Fallback per-sample normalization
    # --------------------------------------------------------
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
    Peak-normalize + simple silence trim

    ไม่ใช้ librosa ที่นี่
    เพื่อหลีกเลี่ยงปัญหา librosa __init__.pyi
    บน Vercel
    """

    try:

        import numpy as np
        import soundfile as sf

        y, sr = sf.read(
            path,
            always_2d=False,
        )

        # ----------------------------------------------------
        # Stereo -> mono
        # ----------------------------------------------------
        if getattr(
            y,
            "ndim",
            1,
        ) > 1:

            y = np.mean(
                y,
                axis=1,
            )

        y = np.asarray(
            y,
            dtype=np.float32,
        )

        if y.size == 0:
            return

        # ----------------------------------------------------
        # Peak normalize
        # ----------------------------------------------------
        peak = float(
            np.max(
                np.abs(y)
            )
        )

        if peak > 0:

            y = y / peak

        # ----------------------------------------------------
        # Simple RMS-based silence trim
        # roughly corresponds to 30 dB threshold
        # ----------------------------------------------------
        frame_length = 2048
        hop_length = 512

        if len(y) > frame_length:

            rms_values = []

            for start in range(
                0,
                len(y) - frame_length + 1,
                hop_length,
            ):

                frame = y[
                    start:
                    start
                    + frame_length
                ]

                rms = np.sqrt(
                    np.mean(
                        frame * frame
                    )
                    + 1e-12
                )

                rms_values.append(
                    rms
                )

            if rms_values:

                rms_values = np.asarray(
                    rms_values,
                    dtype=np.float32,
                )

                max_rms = float(
                    np.max(
                        rms_values
                    )
                )

                threshold = (
                    max_rms
                    * (
                        10.0
                        ** (
                            -TRIM_TOP_DB
                            / 20.0
                        )
                    )
                )

                active = np.where(
                    rms_values
                    >= threshold
                )[0]

                if active.size > 0:

                    start_frame = int(
                        active[0]
                    )

                    end_frame = int(
                        active[-1]
                    )

                    start_sample = (
                        start_frame
                        * hop_length
                    )

                    end_sample = min(
                        len(y),
                        (
                            end_frame
                            * hop_length
                        )
                        + frame_length,
                    )

                    if (
                        end_sample
                        > start_sample
                    ):

                        y = y[
                            start_sample:
                            end_sample
                        ]

        # ----------------------------------------------------
        # Final normalize
        # ----------------------------------------------------
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
# ENSEMBLE PREDICTION
# ============================================================
def predict_ensemble(
    wav_path: str,
) -> dict:
    """
    CNN + RF soft-voting ensemble

    Memory strategy:

        RF download
        ↓
        RF load
        ↓
        RF inference
        ↓
        RF unload
        ↓
        CNN download
        ↓
        CNN load
        ↓
        CNN inference
        ↓
        CNN unload
        ↓
        50/50 ensemble

    ไม่เก็บ RF + CNN model ไว้พร้อมกัน
    """

    import numpy as np

    p_rf = None
    p_cnn = None

    # ========================================================
    # RF
    # ========================================================
    with MODEL_LOCK:

        if not load_rf_model():

            raise RuntimeError(
                MODEL_LOAD_ERROR
                or
                "Random Forest model "
                "ไม่พร้อมใช้งาน"
            )

        try:

            from rf_extract import (
                extract_features,
            )

            print(
                "🔬 กำลังสกัด RF features..."
            )

            feat_rf, err = (
                extract_features(
                    wav_path
                )
            )

            if feat_rf is None:

                raise RuntimeError(
                    "RF feature extraction failed: "
                    f"{err}"
                )

            print(
                "🤖 กำลังทำนายด้วย RF..."
            )

            p_rf = (
                rf_model.predict_proba(
                    feat_rf.reshape(
                        1,
                        -1,
                    )
                )[0]
            )

            p_rf = np.asarray(
                p_rf,
                dtype=np.float32,
            )

            print(
                f"✅ RF prediction เสร็จ: "
                f"{p_rf.tolist()}"
            )

        finally:

            # ----------------------------------------------
            # สำคัญ: ปลด RF ก่อนโหลด CNN
            # ----------------------------------------------
            unload_rf_model()


    # ========================================================
    # CNN
    # ========================================================
    with MODEL_LOCK:

        if not load_cnn_model():

            print(
                "⚠️ CNN ไม่พร้อม "
                "→ fallback เป็น RF only"
            )

            p_ens = p_rf
            mode = "rf_only"

        else:

            try:

                print(
                    "🔬 กำลังสกัด CNN features..."
                )

                x_cnn = (
                    prepare_cnn_input(
                        wav_path
                    )
                )

                if x_cnn is None:

                    print(
                        "⚠️ CNN feature extraction failed "
                        "→ fallback เป็น RF only"
                    )

                    p_ens = p_rf

                    mode = (
                        "rf_only("
                        "cnn_feat_failed)"
                    )

                else:

                    print(
                        "🧠 กำลังทำนายด้วย CNN..."
                    )

                    p_cnn = (
                        cnn_model.predict(
                            x_cnn,
                            verbose=0,
                        )[0]
                    )

                    p_cnn = np.asarray(
                        p_cnn,
                        dtype=np.float32,
                    )

                    print(
                        f"✅ CNN prediction เสร็จ: "
                        f"{p_cnn.tolist()}"
                    )

                    # ----------------------------------------
                    # EXACT 50/50 SOFT VOTING
                    # ----------------------------------------
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

                    print(
                        "✅ Ensemble 50/50 สำเร็จ"
                    )

            except Exception as e:

                print(
                    f"⚠️ CNN inference failed: {e}"
                )

                traceback.print_exc()

                p_ens = p_rf

                mode = (
                    "rf_only("
                    "cnn_failed)"
                )

            finally:

                # ----------------------------------------------
                # ปลด CNN
                # ----------------------------------------------
                unload_cnn_model()

    # ========================================================
    # Safety check
    # ========================================================
    if p_ens is None:

        raise RuntimeError(
            "ไม่สามารถสร้าง prediction probability ได้"
        )

    # ========================================================
    # RESULT
    # ========================================================
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
                probability
            ),
        }
        for label_name,
        probability
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

    # --------------------------------------------------------
    # ไม่โหลด model
    # ไม่ต่อ Firestore
    # เพื่อให้ health check เบาที่สุด
    # --------------------------------------------------------
    if (
        rf_model is not None
        and cnn_model is not None
    ):

        model_state = "ensemble"

    elif rf_model is not None:

        model_state = "rf_loaded"

    elif cnn_model is not None:

        model_state = "cnn_loaded"

    else:

        model_state = "not_loaded"

    return jsonify({

        "message":
            "Smart Cough Detection API is running 🚀",

        "model":
            model_state,

        "rf_ready":
            rf_model is not None,

        "cnn_ready":
            cnn_model is not None,

        "alpha_cnn":
            (
                ENSEMBLE_ALPHA
                if (
                    rf_model is not None
                    and cnn_model is not None
                )
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

    audio_file = (
        request.files["file"]
    )

    # --------------------------------------------------------
    # Unique temporary filename
    # --------------------------------------------------------
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
        # READ AUDIO
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
        # soundfile
        # ----------------------------------------------------
        import soundfile as sf

        try:

            audio_data, sr = (
                sf.read(
                    io.BytesIO(
                        audio_bytes
                    )
                )
            )

        except Exception as e:

            print(
                f"❌ อ่านไฟล์เสียงไม่ได้: {e}"
            )

            return jsonify({
                "error":
                    "ไม่สามารถอ่านไฟล์เสียงได้",
                "error_type":
                    type(e).__name__,
                "detail":
                    str(e),
            }), 400

        # ====================================================
        # WRITE WAV TO /tmp
        # ====================================================
        sf.write(
            filepath,
            audio_data,
            sr,
            format="WAV",
        )

        print(
            f"✅ รับไฟล์เสียงแล้ว: "
            f"{os.path.basename(filepath)}"
        )

        # ====================================================
        # PREPROCESS
        # ====================================================
        preprocess_wav(
            filepath
        )

        # ====================================================
        # PREDICTION
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

        print(
            "✅ Prediction สำเร็จ"
        )

        return jsonify(
            result
        ), 200

    except Exception as e:

        print(
            "\n========== "
            "PREDICTION ERROR "
            "=========="
        )

        print(
            "ERROR TYPE:",
            type(e).__name__,
        )

        print(
            "ERROR:",
            str(e),
        )

        traceback.print_exc()

        print(
            "================================\n"
        )

        return jsonify({

            "error":
                str(e),

            "error_type":
                type(e).__name__,

            "stage":
                "predict",

        }), 500

    finally:

        # ====================================================
        # DELETE TEMP AUDIO
        # ====================================================
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
# LATEST RESULT
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

        print(
            "❌ device_result error:",
            e,
        )

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
