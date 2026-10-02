"""
cnn_extract.py
============================================================
CNN audio feature extraction for CoughAI

Vercel-compatible version:
  - Keeps the original CNN preprocessing pipeline
  - Avoids top-level librosa import
  - Works around librosa/lazy_loader .pyi packaging issue
  - Keeps output shape: (128, 300, 1)
============================================================
"""

import os
import json
import numpy as np


# ============================================================
# LIBROSA COMPATIBILITY
# ============================================================
def _get_librosa():
    """
    Import librosa safely on environments where Vercel's Python
    bundler may omit librosa/__init__.pyi.

    librosa 0.10.x uses lazy_loader.attach_stub(), which expects
    the adjacent .pyi file. If the stub is missing, we provide
    the same lazy-loading information needed by this project.
    """

    import lazy_loader

    original_attach_stub = lazy_loader.attach_stub

    def safe_attach_stub(
        package_name,
        filename,
    ):
        stub_file = (
            os.path.splitext(filename)[0]
            + ".pyi"
        )

        if (
            package_name == "librosa"
            and not os.path.exists(stub_file)
        ):
            return lazy_loader.attach(
                package_name,
                submodules={
                    "core",
                    "feature",
                    "effects",
                    "util",
                },
                submod_attrs={
                    "core": [
                        "load",
                        "stft",
                        "power_to_db",
                    ],
                    "feature": [
                        "melspectrogram",
                        "mfcc",
                        "delta",
                        "spectral_centroid",
                        "spectral_bandwidth",
                        "spectral_rolloff",
                        "spectral_flatness",
                        "spectral_contrast",
                        "zero_crossing_rate",
                        "rms",
                        "chroma_stft",
                        "tonnetz",
                    ],
                    "effects": [
                        "trim",
                        "harmonic",
                    ],
                    "util": [
                        "fix_length",
                    ],
                },
            )

        return original_attach_stub(
            package_name,
            filename,
        )

    # Patch only for this import cycle.
    lazy_loader.attach_stub = safe_attach_stub

    try:
        import librosa
        return librosa
    finally:
        # Restore original behavior after librosa has initialized.
        lazy_loader.attach_stub = original_attach_stub


# ============================================================
# CNN FEATURE EXTRACTION
# ============================================================
def extract_features_cnn(
    file_path,
    sr=44100,
    duration=10,
    n_mels=128,
    target_cols=300,
):
    """
    Extract Mel-spectrogram for CNN.

    Original pipeline preserved:
      - sr = 44100
      - duration = 10 seconds
      - n_mels = 128
      - target_cols = 300
      - power_to_db(ref=np.max)

    Returns:
        np.ndarray with shape (128, 300, 1)
        or None on failure.
    """

    try:
        librosa = _get_librosa()

        # ----------------------------------------------------
        # Load audio
        # ----------------------------------------------------
        y, _ = librosa.load(
            file_path,
            sr=sr,
            duration=duration,
        )

        # ----------------------------------------------------
        # Force exact duration
        # ----------------------------------------------------
        y = librosa.util.fix_length(
            y,
            size=sr * duration,
        )

        # ----------------------------------------------------
        # Mel-spectrogram
        # ----------------------------------------------------
        mel_spectrogram = (
            librosa.feature.melspectrogram(
                y=y,
                sr=sr,
                n_mels=n_mels,
            )
        )

        # ----------------------------------------------------
        # Convert to dB
        # ----------------------------------------------------
        mel_spectrogram_db = (
            librosa.power_to_db(
                mel_spectrogram,
                ref=np.max,
            )
        )

        # ----------------------------------------------------
        # Force exact time dimension
        # ----------------------------------------------------
        mel_spectrogram_fixed = (
            librosa.util.fix_length(
                mel_spectrogram_db,
                size=target_cols,
                axis=1,
            )
        )

        # ----------------------------------------------------
        # Add CNN channel dimension
        # ----------------------------------------------------
        return (
            mel_spectrogram_fixed[
                ...,
                np.newaxis,
            ]
        )

    except Exception as e:

        print(
            "❌ Error extracting "
            f"Mel-spectrogram from "
            f"{file_path}: {e}"
        )

        return None


# ============================================================
# BATCH FEATURE EXTRACTION
# ใช้สำหรับ training / offline processing เท่านั้น
# ============================================================
def process_and_save_data_cnn(
    base_path,
    classes,
    output_dir="cnn_features",
    output_manifest="cnn_data_manifest.json",
):
    """
    Process WAV files, extract CNN features, save each feature
    as .npy, and create a manifest.

    This function is for offline training/preprocessing.
    """

    from tqdm import tqdm

    if not os.path.exists(
        output_dir
    ):
        os.makedirs(
            output_dir
        )

    manifest = []

    print(
        "🔄 เริ่มสกัดฟีเจอร์สำหรับ CNN..."
    )

    for idx, cls in enumerate(
        classes
    ):

        folder = os.path.join(
            base_path,
            cls,
        )

        if not os.path.isdir(
            folder
        ):
            print(
                f"⚠️ ไม่พบโฟลเดอร์: {folder}"
            )
            continue

        print(
            f"📁 กำลังประมวลผลคลาส: {cls}"
        )

        files = [
            file
            for file in os.listdir(
                folder
            )
            if file.lower().endswith(
                ".wav"
            )
        ]

        for file in tqdm(
            files,
            desc=f"🔍 {cls}",
            unit="file",
        ):

            path = os.path.join(
                folder,
                file,
            )

            features = (
                extract_features_cnn(
                    path
                )
            )

            if features is None:
                continue

            filename = (
                f"{cls}_"
                f"{os.path.splitext(file)[0]}"
                ".npy"
            )

            filepath = os.path.join(
                output_dir,
                filename,
            )

            np.save(
                filepath,
                features,
            )

            manifest.append({
                "filepath": filepath,
                "label": idx,
                "class_name": cls,
            })

    with open(
        output_manifest,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            manifest,
            f,
            indent=4,
            ensure_ascii=False,
        )

    print(
        f"✅ บันทึกไฟล์ manifest ที่ "
        f"{output_manifest}"
    )

    print(
        "✅ สกัดฟีเจอร์ CNN เสร็จสิ้น"
    )


# ============================================================
# LOCAL ENTRY POINT
# ============================================================
if __name__ == "__main__":

    BASE_PATH = (
        r"C:\Users\Acer\Downloads"
        r"\Cough Detection"
        r"\public_dataset"
    )

    CLASSES = [
        "covid",
        "healthy",
        "symptomatic",
    ]

    process_and_save_data_cnn(
        BASE_PATH,
        CLASSES,
    )
