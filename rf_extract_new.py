"""
rf_extract.py  –  Audio feature extraction for cough detection
================================================================
Extracts 416 features per WAV file using librosa, then saves
features_raw.npz + feature_names.npy + metadata.json.

Key optimizations vs. original:
  • ThreadPoolExecutor (librosa releases GIL during FFT; avoids Windows spawn overhead)
  • Single STFT / D² computation reused across all features
  • Harmonic computed once, reused by tonnetz
  • Pre-validated feature length; shape mismatches caught early
  • Richer per-class stats and structured metadata output
"""

import os
import json
import time
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional

import librosa
import numpy as np
from tqdm import tqdm

warnings.filterwarnings("ignore")


# ─── Config ───────────────────────────────────────────────────────────────────

@dataclass
class Config:
    sr: int        = 16_000   # 16 kHz is sufficient for cough audio
    duration: int  = 5        # seconds
    n_mfcc: int    = 40
    n_mels: int    = 32
    n_workers: int = max(1, (os.cpu_count() or 4) - 1)  # leave one core free

CFG = Config()

# Expected feature vector length – used as a sanity-check after assembly
N_FEATURES = (
    CFG.n_mfcc * 8   # mfcc ×4 stats + delta ×2 stats + delta2 ×2 stats  = 320
    + 4              # centroid, bandwidth, rolloff, flatness
    + 7              # spectral contrast (6 bands + 1)
    + 3              # zcr, rms, zcr_f0_approx
    + 12             # chroma
    + CFG.n_mels * 2 # mel mean + std
    + 6              # tonnetz
)   # = 416


# ─── Feature extraction ───────────────────────────────────────────────────────

def extract_features(file_path: str) -> tuple[Optional[np.ndarray], Optional[str]]:
    """Return (feature_vector, None) on success, or (None, error_str) on failure."""
    try:
        # ── Load & normalise ──────────────────────────────────────────────────
        y, sr = librosa.load(file_path, duration=CFG.duration, sr=CFG.sr, mono=True)
        y = librosa.util.fix_length(y, size=sr * CFG.duration)
        peak = np.max(np.abs(y))
        if peak > 0:
            y /= peak

        # ── Shared spectral representations (computed ONCE) ───────────────────
        D      = np.abs(librosa.stft(y))          # magnitude spectrogram
        D_sq   = D ** 2                            # power spectrogram (reused)
        mel    = librosa.feature.melspectrogram(S=D_sq, sr=sr, n_mels=CFG.n_mels)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        # ── MFCC block ────────────────────────────────────────────────────────
        mfcc       = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=CFG.n_mfcc)  # y= ensures n_mfcc rows; S= silently uses n_mels rows instead
        delta      = librosa.feature.delta(mfcc)
        delta2     = librosa.feature.delta(mfcc, order=2)

        mfcc_feats = np.hstack([
            np.mean(mfcc,   axis=1), np.std(mfcc,   axis=1),
            np.percentile(mfcc, 25, axis=1),
            np.percentile(mfcc, 75, axis=1),
            np.mean(delta,  axis=1), np.std(delta,  axis=1),
            np.mean(delta2, axis=1), np.std(delta2, axis=1),
        ])  # 320 values

        # ── Spectral block ────────────────────────────────────────────────────
        # Each of centroid/bandwidth/rolloff/flatness returns shape (1, T);
        # flatten to scalar explicitly before stacking with contrast (1D, 7).
        centroid  = float(np.mean(librosa.feature.spectral_centroid(S=D, sr=sr)))
        bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(S=D, sr=sr)))
        rolloff   = float(np.mean(librosa.feature.spectral_rolloff(S=D_sq, sr=sr)))
        flatness  = float(np.mean(librosa.feature.spectral_flatness(S=D)))
        contrast  = np.mean(librosa.feature.spectral_contrast(S=D, sr=sr), axis=1)  # (7,)

        spectral_feats = np.concatenate([
            np.array([centroid, bandwidth, rolloff, flatness], dtype=np.float32),
            contrast.astype(np.float32),
        ])  # 4 + 7 = 11 values

        # ── Time-domain block ─────────────────────────────────────────────────
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
        rms = float(np.mean(librosa.feature.rms(y=y)))

        frame_len    = sr // 10  # 100 ms
        zcr_frames   = librosa.feature.zero_crossing_rate(
            y, frame_length=frame_len, hop_length=frame_len // 2
        )[0]
        zcr_f0_approx = float(np.median(zcr_frames) * sr)

        time_feats = np.array([zcr, rms, zcr_f0_approx])  # 3 values

        # ── Chroma block ──────────────────────────────────────────────────────
        chroma = np.mean(librosa.feature.chroma_stft(S=D, sr=sr), axis=1)  # 12

        # ── Mel stats ─────────────────────────────────────────────────────────
        mel_feats = np.hstack([
            np.mean(mel_db, axis=1),
            np.std(mel_db,  axis=1),
        ])  # n_mels * 2

        # ── Tonnetz (harmonic component, computed once) ───────────────────────
        y_harm  = librosa.effects.harmonic(y)
        tonnetz = np.mean(librosa.feature.tonnetz(y=y_harm, sr=sr), axis=1)  # 6

        # ── Assemble & validate ───────────────────────────────────────────────
        features = np.hstack([
            mfcc_feats,
            spectral_feats,
            time_feats,
            chroma,
            mel_feats,
            tonnetz,
        ]).astype(np.float32)

        if len(features) != N_FEATURES:
            blocks = {
                "mfcc_feats":     len(mfcc_feats),
                "spectral_feats": len(spectral_feats),
                "time_feats":     len(time_feats),
                "chroma":         len(chroma),
                "mel_feats":      len(mel_feats),
                "tonnetz":        len(tonnetz),
            }
            raise ValueError(
                f"Feature length mismatch: expected {N_FEATURES}, got {len(features)}. "
                f"Block sizes: {blocks}"
            )

        return features, None

    except Exception:
        return None, traceback.format_exc()


def _worker(args: tuple) -> tuple:
    path, label = args
    feat, err = extract_features(path)
    return feat, label, err, path


# ─── Feature names (mirrors assembly order above) ─────────────────────────────

def build_feature_names() -> list[str]:
    n, m = CFG.n_mfcc, CFG.n_mels
    return (
        [f"mfcc_mean_{i}"   for i in range(n)] +
        [f"mfcc_std_{i}"    for i in range(n)] +
        [f"mfcc_p25_{i}"    for i in range(n)] +
        [f"mfcc_p75_{i}"    for i in range(n)] +
        [f"delta_mean_{i}"  for i in range(n)] +
        [f"delta_std_{i}"   for i in range(n)] +
        [f"delta2_mean_{i}" for i in range(n)] +
        [f"delta2_std_{i}"  for i in range(n)] +
        ["centroid", "bandwidth", "rolloff", "flatness"] +
        [f"contrast_{i}"    for i in range(7)] +
        ["zcr", "rms", "zcr_f0_approx"] +
        [f"chroma_{i}"      for i in range(12)] +
        [f"mel_mean_{i}"    for i in range(m)] +
        [f"mel_std_{i}"     for i in range(m)] +
        [f"tonnetz_{i}"     for i in range(6)]
    )


# ─── Main extraction runner ───────────────────────────────────────────────────

def run_extract(base_path: str, classes: list[str], out_dir: str = ".") -> None:
    print("🔄 Feature extraction starting…")
    print(f"   SR={CFG.sr} Hz | duration={CFG.duration}s | workers={CFG.n_workers}\n")

    # ── Build task list ───────────────────────────────────────────────────────
    tasks: list[tuple[str, int]] = []
    class_counts: dict[str, int] = {}

    for idx, cls in enumerate(classes):
        folder = os.path.join(base_path, cls)
        if not os.path.isdir(folder):
            print(f"⚠️  Folder not found: {folder}")
            continue
        wavs = [f for f in os.listdir(folder) if f.lower().endswith(".wav")]
        class_counts[cls] = len(wavs)
        print(f"📁 {cls}: {len(wavs):,} files")
        tasks.extend((os.path.join(folder, f), idx) for f in wavs)

    total = len(tasks)
    print(f"\n🗂  Total: {total:,} files\n")

    if total == 0:
        print("❌ No WAV files found. Check base_path and class folders.")
        return

    # ── Parallel extraction ───────────────────────────────────────────────────
    # ThreadPoolExecutor is used deliberately:
    #   - librosa/numpy release the GIL during C-level FFT and array ops,
    #     so threads DO run in parallel for the heavy computation.
    #   - ProcessPoolExecutor on Windows spawns fresh interpreter processes
    #     that each re-import numpy/librosa, consuming huge amounts of virtual
    #     memory (paging file) and causing "DLL load failed" crashes at scale.
    X: list[np.ndarray] = []
    y: list[int]        = []
    errors: list[tuple[str, str]] = []

    t0 = time.perf_counter()

    with ThreadPoolExecutor(max_workers=CFG.n_workers) as pool:
        futures = {pool.submit(_worker, t): t for t in tasks}
        for fut in tqdm(
            as_completed(futures),
            total=total,
            desc="Extracting",
            unit="file",
            dynamic_ncols=True,
        ):
            feat, label, err, path = fut.result()
            if feat is not None:
                X.append(feat)
                y.append(label)
            else:
                errors.append((path, err))

    elapsed = time.perf_counter() - t0

    # ── Assemble arrays ───────────────────────────────────────────────────────
    X_arr = np.array(X, dtype=np.float32)
    y_arr = np.array(y, dtype=np.int32)

    # ── Per-class stats ───────────────────────────────────────────────────────
    print(f"\n📊 Shape: {X_arr.shape}  ({elapsed:.1f}s, "
          f"{total / elapsed:.1f} files/s)")
    for idx, cls in enumerate(classes):
        n = int(np.sum(y_arr == idx))
        print(f"   {cls}: {n:,} samples extracted")

    if errors:
        print(f"\n❌ Failed: {len(errors)} file(s)")
        for path, tb in errors[:3]:
            print(f"   {os.path.basename(path)}: {tb.splitlines()[-1]}")
        if len(errors) > 3:
            print(f"   … and {len(errors) - 3} more")

    # ── Save outputs ──────────────────────────────────────────────────────────
    os.makedirs(out_dir, exist_ok=True)

    npz_path   = os.path.join(out_dir, "features_raw_new.npz")
    names_path = os.path.join(out_dir, "feature_names_new.npy")
    meta_path  = os.path.join(out_dir, "metadata_new.json")

    np.savez_compressed(npz_path, X=X_arr, y=y_arr)
    print(f"\n💾 Saved {npz_path}")

    feat_names = build_feature_names()
    np.save(names_path, np.array(feat_names))
    print(f"💾 Saved {names_path}  ({len(feat_names)} features)")

    meta = {
        "created_at":    datetime.now(tz=timezone.utc).isoformat(),
        "base_path":     base_path,
        "classes":       classes,
        "config":        asdict(CFG),
        "n_features":    N_FEATURES,
        "n_samples":     int(X_arr.shape[0]),
        "class_counts":  {cls: int(np.sum(y_arr == i)) for i, cls in enumerate(classes)},
        "failed_files":  len(errors),
        "elapsed_s":     round(elapsed, 2),
        "files_per_sec": round(total / elapsed, 2),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"💾 Saved {meta_path}")


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # freeze_support() is a no-op outside frozen (PyInstaller) executables,
    # but is harmless and recommended when shipping scripts on Windows.
    import multiprocessing
    multiprocessing.freeze_support()

    # Dataset ใหม่ = output ของ preprocessing.py (โฟลเดอร์คลาสที่ root) — ไม่ใช่ verify_sound (ของเก่า)
    BASE_PATH = r"C:\Users\Acer\Downloads\Cough Detection\public_dataset"
    CLASSES   = ["covid", "healthy", "symptomatic"]

    run_extract(BASE_PATH, CLASSES)
