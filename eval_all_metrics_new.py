"""
eval_all_metrics_new.py — re-export metric ทุกโมเดลที่ 4 ตำแหน่งทศนิยม
ดึงจาก prediction จริงบน test set เดียวกับตอนเทรน (ไม่ใช่เลขปัดจาก report)
ครอบคลุม: RF, CNN, XGB, CNN+RF, CNN+XGB, CNN+RF+XGB  (เวอร์ชัน leaky/เดิม)
"""
import os, json, warnings
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np, joblib, tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)
from imblearn.over_sampling import RandomOverSampler, SMOTE
warnings.filterwarnings("ignore")

CLASSES = ["covid", "healthy", "symptomatic"]
IMG = (128, 128, 1); SEED = 42

# ── CNN test set (เหมือน train_cnn_new.py) ──
def load_cnn():
    with open("cnn_data_manifest_new.json") as f: man = json.load(f)
    X, y = [], []
    for e in man:
        try:
            ft = np.load(e["filepath"]); c = IMG[1]
            if ft.shape[1] > c: ft = ft[:, :c, :]
            elif ft.shape[1] < c: ft = np.pad(ft, ((0,0),(0,c-ft.shape[1]),(0,0)))
            X.append(ft); y.append(e["label"])
        except FileNotFoundError: pass
    X = np.array(X, np.float32); y = np.array(y)
    X = (X - X.min()) / (X.max() - X.min() + 1e-8)
    idx, yr = RandomOverSampler(random_state=SEED).fit_resample(np.arange(len(X)).reshape(-1,1), y)
    idx = idx.ravel()
    tr, te, _, yte = train_test_split(idx, yr, test_size=0.2, stratify=yr, random_state=SEED)
    return X[te], yte

# ── RF/XGB test set (เหมือน train_rf_new.py) ──
def load_rf():
    d = np.load("features_raw_new.npz"); X, y = d["X"], d["y"]
    Xr, yr = SMOTE(random_state=SEED).fit_resample(X, y)
    _, Xte, _, yte = train_test_split(Xr, yr, test_size=0.2, stratify=yr, random_state=SEED)
    return Xte, yte

def metrics(name, y_true, proba):
    y_pred = np.argmax(proba, axis=1)
    return (name,
            accuracy_score(y_true, y_pred),
            roc_auc_score(y_true, proba, multi_class="ovr", average="macro"),
            precision_score(y_true, y_pred, average="macro"),
            recall_score(y_true, y_pred, average="macro"),
            f1_score(y_true, y_pred, average="macro"))

print("โหลดโมเดล...")
cnn = tf.keras.models.load_model("cough_cnn_model_new.h5")
rf  = joblib.load("cough_rf_model_new.pkl")
xgb = joblib.load("cough_xgb_model_new.pkl")

print("โหลด test set...")
Xc, yc = load_cnn()
Xr, yr = load_rf()
assert np.array_equal(yc, yr), "label order ไม่ตรง!"
y = yc

print("ทำนาย...")
p_cnn = cnn.predict(Xc, verbose=0)
p_rf  = rf.predict_proba(Xr)
p_xgb = xgb.predict_proba(Xr)

rows = [
    metrics("RF",            y, p_rf),
    metrics("CNN",           y, p_cnn),
    metrics("XGBoost",       y, p_xgb),
    metrics("CNN+RF",        y, (p_cnn + p_rf) / 2),
    metrics("CNN+XGB",       y, (p_cnn + p_xgb) / 2),
    metrics("CNN+RF+XGB",    y, (p_cnn + p_rf + p_xgb) / 3),
]

print("\n" + "="*78)
print(f"{'Model':<14}{'Accuracy':>11}{'Macro-AUC':>12}{'Macro-Prec':>12}{'Macro-Rec':>12}{'Macro-F1':>11}")
print("="*78)
for n, acc, au, pr, rc, f1 in rows:
    print(f"{n:<14}{acc:>11.4f}{au:>12.4f}{pr:>12.4f}{rc:>12.4f}{f1:>11.4f}")
print("="*78)
