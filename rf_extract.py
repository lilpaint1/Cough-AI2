"""
rf_extract.py
============================================================
Audio feature extraction for CoughAI Random Forest

Produces exactly 416 features per WAV file.

Vercel-compatible version:
  - Keeps the original 416-D feature pipeline
  - Avoids top-level librosa import
  - Works around librosa/lazy_loader .pyi issue
  - Removes top-level tqdm import
  - Keeps training/offline extraction functionality
============================================================
"""

import os
import json
import time
import traceback
import warnings

from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)

from dataclasses import (
    dataclass,
    asdict,
)

from datetime import (
    datetime,
    timezone,
)

from typing import Optional

import numpy as np


warnings.filterwarnings(
    "ignore"
)


# ============================================================
# CONFIG
# ============================================================
@dataclass
class Config:

    sr: int = 16_000

    duration: int = 5

    n_mfcc: int = 40

    n_mels: int = 32

    n_workers: int = max(
        1,
        (os.cpu_count() or 4) - 1,
    )


CFG = Config()


# ============================================================
# EXPECTED FEATURE COUNT
# ============================================================
N_FEATURES = (
    CFG.n_mfcc * 8      # 320
    + 4                  # spectral scalars
    + 7                  # spectral contrast
    + 3                  # time
    + 12                 # chroma
    + CFG.n_mels * 2     # mel mean/std
    + 6                  # tonnetz
)

# Expected:
# 320 + 4 + 7 + 3 + 12 + 64 + 6 = 416


# ============================================================
# LIBROSA COMPATIBILITY
# ============================================================
def _get_librosa():
    """
    Import librosa safely on environments where Vercel's Python
    bundler may omit librosa/__init__.pyi.

    librosa 0.10.x calls lazy_loader.attach_stub(), which expects
    an adjacent .pyi file. When the stub is missing, we provide
    the required lazy submodules and attributes directly.
    """

    import lazy_loader

    original_attach_stub = (
        lazy_loader.attach_stub
    )

    def safe_attach_stub(
        package_name,
        filename,
    ):
        stub_file = (
            os.path.splitext(
                filename
            )[0]
            + ".pyi"
        )

        if (
            package_name == "librosa"
            and not os.path.exists(
                stub_file
            )
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

    lazy_loader.attach_stub = (
        safe_attach_stub
    )

    try:
        import librosa
        return librosa

    finally:
        lazy_loader.attach_stub = (
            original_attach_stub
        )


# ============================================================
# FEATURE EXTRACTION
# ============================================================
def extract_features(
    file_path: str,
) -> tuple[
    Optional[np.ndarray],
    Optional[str],
]:

    """
    Return:

        (feature_vector, None)
            on success

        (None, traceback)
            on failure

    Produces exactly 416 features.
    """

    try:

        librosa = _get_librosa()

        # ====================================================
        # Load + normalize
        # ====================================================
        y, sr = librosa.load(
            file_path,
            duration=CFG.duration,
            sr=CFG.sr,
            mono=True,
        )

        y = librosa.util.fix_length(
            y,
            size=sr * CFG.duration,
        )

        peak = np.max(
            np.abs(y)
        )

        if peak > 0:
            y /= peak


        # ====================================================
        # Shared spectral representations
        # ====================================================
        D = np.abs(
            librosa.stft(y)
        )

        D_sq = D ** 2

        mel = (
            librosa.feature.melspectrogram(
                S=D_sq,
                sr=sr,
                n_mels=CFG.n_mels,
            )
        )

        mel_db = (
            librosa.power_to_db(
                mel,
                ref=np.max,
            )
        )


        # ====================================================
        # MFCC block
        # ====================================================
        mfcc = (
            librosa.feature.mfcc(
                y=y,
                sr=sr,
                n_mfcc=CFG.n_mfcc,
            )
        )

        delta = (
            librosa.feature.delta(
                mfcc
            )
        )

        delta2 = (
            librosa.feature.delta(
                mfcc,
                order=2,
            )
        )

        mfcc_feats = np.hstack([

            np.mean(
                mfcc,
                axis=1,
            ),

            np.std(
                mfcc,
                axis=1,
            ),

            np.percentile(
                mfcc,
                25,
                axis=1,
            ),

            np.percentile(
                mfcc,
                75,
                axis=1,
            ),

            np.mean(
                delta,
                axis=1,
            ),

            np.std(
                delta,
                axis=1,
            ),

            np.mean(
                delta2,
                axis=1,
            ),

            np.std(
                delta2,
                axis=1,
            ),
        ])


        # ====================================================
        # Spectral block
        # ====================================================
        centroid = float(
            np.mean(
                librosa.feature.spectral_centroid(
                    S=D,
                    sr=sr,
                )
            )
        )

        bandwidth = float(
            np.mean(
                librosa.feature.spectral_bandwidth(
                    S=D,
                    sr=sr,
                )
            )
        )

        rolloff = float(
            np.mean(
                librosa.feature.spectral_rolloff(
                    S=D_sq,
                    sr=sr,
                )
            )
        )

        flatness = float(
            np.mean(
                librosa.feature.spectral_flatness(
                    S=D
                )
            )
        )

        contrast = np.mean(
            librosa.feature.spectral_contrast(
                S=D,
                sr=sr,
            ),
            axis=1,
        )

        spectral_feats = np.concatenate([

            np.array(
                [
                    centroid,
                    bandwidth,
                    rolloff,
                    flatness,
                ],
                dtype=np.float32,
            ),

            contrast.astype(
                np.float32
            ),
        ])


        # ====================================================
        # Time-domain block
        # ====================================================
        zcr = float(
            np.mean(
                librosa.feature.zero_crossing_rate(
                    y
                )
            )
        )

        rms = float(
            np.mean(
                librosa.feature.rms(
                    y=y
                )
            )
        )

        frame_len = (
            sr // 10
        )

        zcr_frames = (
            librosa.feature.zero_crossing_rate(
                y,
                frame_length=frame_len,
                hop_length=frame_len // 2,
            )[0]
        )

        zcr_f0_approx = float(
            np.median(
                zcr_frames
            )
            * sr
        )

        time_feats = np.array([
            zcr,
            rms,
            zcr_f0_approx,
        ])


        # ====================================================
        # Chroma
        # ====================================================
        chroma = np.mean(
            librosa.feature.chroma_stft(
                S=D,
                sr=sr,
            ),
            axis=1,
        )


        # ====================================================
        # Mel statistics
        # ====================================================
        mel_feats = np.hstack([

            np.mean(
                mel_db,
                axis=1,
            ),

            np.std(
                mel_db,
                axis=1,
            ),
        ])


        # ====================================================
        # Tonnetz
        # ====================================================
        y_harm = (
            librosa.effects.harmonic(
                y
            )
        )

        tonnetz = np.mean(
            librosa.feature.tonnetz(
                y=y_harm,
                sr=sr,
            ),
            axis=1,
        )


        # ====================================================
        # Assemble 416 features
        # ====================================================
        features = np.hstack([

            mfcc_feats,

            spectral_feats,

            time_feats,

            chroma,

            mel_feats,

            tonnetz,

        ]).astype(
            np.float32
        )


        # ====================================================
        # Validate
        # ====================================================
        if len(features) != N_FEATURES:

            blocks = {

                "mfcc_feats":
                    len(mfcc_feats),

                "spectral_feats":
                    len(spectral_feats),

                "time_feats":
                    len(time_feats),

                "chroma":
                    len(chroma),

                "mel_feats":
                    len(mel_feats),

                "tonnetz":
                    len(tonnetz),
            }

            raise ValueError(
                "Feature length mismatch: "
                f"expected {N_FEATURES}, "
                f"got {len(features)}. "
                f"Block sizes: {blocks}"
            )


        return (
            features,
            None,
        )

    except Exception:

        return (
            None,
            traceback.format_exc(),
        )


# ============================================================
# WORKER
# ============================================================
def _worker(
    args: tuple,
) -> tuple:

    path, label = args

    feat, err = (
        extract_features(
            path
        )
    )

    return (
        feat,
        label,
        err,
        path,
    )


# ============================================================
# FEATURE NAMES
# ============================================================
def build_feature_names() -> list[str]:

    n = CFG.n_mfcc
    m = CFG.n_mels

    return (

        [
            f"mfcc_mean_{i}"
            for i in range(n)
        ]

        + [
            f"mfcc_std_{i}"
            for i in range(n)
        ]

        + [
            f"mfcc_p25_{i}"
            for i in range(n)
        ]

        + [
            f"mfcc_p75_{i}"
            for i in range(n)
        ]

        + [
            f"delta_mean_{i}"
            for i in range(n)
        ]

        + [
            f"delta_std_{i}"
            for i in range(n)
        ]

        + [
            f"delta2_mean_{i}"
            for i in range(n)
        ]

        + [
            f"delta2_std_{i}"
            for i in range(n)
        ]

        + [
            "centroid",
            "bandwidth",
            "rolloff",
            "flatness",
        ]

        + [
            f"contrast_{i}"
            for i in range(7)
        ]

        + [
            "zcr",
            "rms",
            "zcr_f0_approx",
        ]

        + [
            f"chroma_{i}"
            for i in range(12)
        ]

        + [
            f"mel_mean_{i}"
            for i in range(m)
        ]

        + [
            f"mel_std_{i}"
            for i in range(m)
        ]

        + [
            f"tonnetz_{i}"
            for i in range(6)
        ]
    )


# ============================================================
# OFFLINE FEATURE EXTRACTION
# ============================================================
def run_extract(
    base_path: str,
    classes: list[str],
    out_dir: str = ".",
) -> None:

    from tqdm import tqdm

    print(
        "🔄 Feature extraction starting…"
    )

    print(
        f"   SR={CFG.sr} Hz | "
        f"duration={CFG.duration}s | "
        f"workers={CFG.n_workers}\n"
    )


    # ========================================================
    # Build task list
    # ========================================================
    tasks = []

    class_counts = {}

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
                f"⚠️ Folder not found: "
                f"{folder}"
            )

            continue

        wavs = [
            f
            for f in os.listdir(
                folder
            )
            if f.lower().endswith(
                ".wav"
            )
        ]

        class_counts[cls] = len(
            wavs
        )

        print(
            f"📁 {cls}: "
            f"{len(wavs):,} files"
        )

        tasks.extend(
            (
                os.path.join(
                    folder,
                    f,
                ),
                idx,
            )
            for f in wavs
        )


    total = len(tasks)

    print(
        f"\n🗂 Total: "
        f"{total:,} files\n"
    )

    if total == 0:

        print(
            "❌ No WAV files found. "
            "Check base_path and class folders."
        )

        return


    # ========================================================
    # Parallel extraction
    # ========================================================
    X = []
    y = []

    errors = []

    t0 = time.perf_counter()

    with ThreadPoolExecutor(
        max_workers=CFG.n_workers
    ) as pool:

        futures = {
            pool.submit(
                _worker,
                task,
            ): task
            for task in tasks
        }

        for fut in tqdm(
            as_completed(
                futures
            ),
            total=total,
            desc="Extracting",
            unit="file",
            dynamic_ncols=True,
        ):

            feat, label, err, path = (
                fut.result()
            )

            if feat is not None:

                X.append(
                    feat
                )

                y.append(
                    label
                )

            else:

                errors.append(
                    (
                        path,
                        err,
                    )
                )


    elapsed = (
        time.perf_counter()
        - t0
    )


    # ========================================================
    # Assemble arrays
    # ========================================================
    X_arr = np.array(
        X,
        dtype=np.float32,
    )

    y_arr = np.array(
        y,
        dtype=np.int32,
    )


    # ========================================================
    # Stats
    # ========================================================
    rate = (
        total / elapsed
        if elapsed > 0
        else 0
    )

    print(
        f"\n📊 Shape: "
        f"{X_arr.shape} "
        f"({elapsed:.1f}s, "
        f"{rate:.1f} files/s)"
    )

    for idx, cls in enumerate(
        classes
    ):

        n = int(
            np.sum(
                y_arr == idx
            )
        )

        print(
            f"   {cls}: "
            f"{n:,} samples extracted"
        )


    # ========================================================
    # Errors
    # ========================================================
    if errors:

        print(
            f"\n❌ Failed: "
            f"{len(errors)} file(s)"
        )

        for path, tb in errors[:3]:

            print(
                f"   "
                f"{os.path.basename(path)}: "
                f"{tb.splitlines()[-1]}"
            )

        if len(errors) > 3:

            print(
                f"   … and "
                f"{len(errors) - 3} more"
            )


    # ========================================================
    # Save outputs
    # ========================================================
    os.makedirs(
        out_dir,
        exist_ok=True,
    )

    npz_path = os.path.join(
        out_dir,
        "features_raw.npz",
    )

    names_path = os.path.join(
        out_dir,
        "feature_names.npy",
    )

    meta_path = os.path.join(
        out_dir,
        "metadata.json",
    )


    np.savez_compressed(
        npz_path,
        X=X_arr,
        y=y_arr,
    )

    print(
        f"\n💾 Saved {npz_path}"
    )


    feat_names = (
        build_feature_names()
    )

    np.save(
        names_path,
        np.array(
            feat_names
        ),
    )

    print(
        f"💾 Saved {names_path} "
        f"({len(feat_names)} features)"
    )


    meta = {

        "created_at":
            datetime.now(
                tz=timezone.utc
            ).isoformat(),

        "base_path":
            base_path,

        "classes":
            classes,

        "config":
            asdict(CFG),

        "n_features":
            N_FEATURES,

        "n_samples":
            int(
                X_arr.shape[0]
            ),

        "class_counts":
            {
                cls: int(
                    np.sum(
                        y_arr == i
                    )
                )
                for i, cls
                in enumerate(
                    classes
                )
            },

        "failed_files":
            len(errors),

        "elapsed_s":
            round(
                elapsed,
                2,
            ),

        "files_per_sec":
            round(
                rate,
                2,
            ),
    }


    with open(
        meta_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            meta,
            f,
            indent=2,
            ensure_ascii=False,
        )


    print(
        f"💾 Saved {meta_path}"
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

    run_extract(
        BASE_PATH,
        CLASSES,
    )
