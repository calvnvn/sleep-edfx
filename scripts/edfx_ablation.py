"""Quick GBDT ablations on per-epoch features (proxy evidence, not the DL model)."""
import sys, os, time
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import f1_score, cohen_kappa_score, accuracy_score, confusion_matrix

d = np.load(sys.argv[1])
X, y, g, r = d["X"], d["y"], d["g"], d["r"]
NAMES = ["W", "N1", "N2", "N3", "REM"]
# column blocks: EEG Fpz (9), EEG Pz (9), EOG (4), EMG (4)
FPZ, PZ, EOG, EMG = list(range(0, 9)), list(range(9, 18)), list(range(18, 22)), list(range(22, 26))


def context(Xb, k=2):
    """stack features of neighbouring epochs (within the same recording)"""
    out = [Xb]
    for s in list(range(-k, 0)) + list(range(1, k + 1)):
        sh = np.roll(Xb, -s, axis=0)
        same = np.roll(r, -s) == r
        sh[~same] = 0
        out.append(sh)
    return np.hstack(out)


def run(name, cols, split="subject", ctx=0, weight=None, n_splits=5):
    Xb = X[:, cols]
    if ctx:
        Xb = context(Xb, ctx)
    cv = GroupKFold(n_splits) if split == "subject" else KFold(n_splits, shuffle=True, random_state=0)
    pred = np.zeros_like(y)
    for tr, te in cv.split(Xb, y, g):
        clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=63,
                                             class_weight=weight, random_state=0)
        clf.fit(Xb[tr], y[tr]); pred[te] = clf.predict(Xb[te])
    f1 = f1_score(y, pred, average=None)
    res = (f"{name:42s} ACC={100*accuracy_score(y,pred):5.1f} MF1={100*f1.mean():5.1f} "
           f"k={cohen_kappa_score(y,pred):.3f} | " + " ".join(f"{n}={100*v:4.1f}" for n, v in zip(NAMES, f1)))
    print(res, flush=True)
    return pred


t0 = time.time()
print(f"epochs={len(y)} subjects={len(np.unique(g))} records={len(np.unique(r))}")
run("A1 EEG Fpz-Cz only", FPZ)
run("A2 EEG Fpz-Cz + Pz-Oz", FPZ + PZ)
run("A3 EEG(2) + EOG", FPZ + PZ + EOG)
run("A4 EEG(2) + EOG + EMG", FPZ + PZ + EOG + EMG)
run("A5 EEG Fpz-Cz + EMG (no EOG)", FPZ + EMG)
run("B1 A4 + context ±2 epochs", FPZ + PZ + EOG + EMG, ctx=2)
run("B2 A4 + context ±2 + class_weight=balanced", FPZ + PZ + EOG + EMG, ctx=2, weight="balanced")
p = run("C1 A4 + ctx, EPOCH-WISE KFold (leaky)", FPZ + PZ + EOG + EMG, ctx=2, split="epoch")
pb = run("C2 A4 + ctx, SUBJECT-WISE GroupKFold", FPZ + PZ + EOG + EMG, ctx=2)
cm = confusion_matrix(y, pb)
print("Confusion (rows=true W,N1,N2,N3,REM; row-normalised %):")
for i, row in enumerate(cm):
    print("  ", NAMES[i].ljust(4), " ".join(f"{100*v/row.sum():5.1f}" for v in row))
# per-subject variability of MF1 in subject-wise run
mf = [f1_score(y[g == s], pb[g == s], average="macro", labels=np.unique(y[g == s])) for s in np.unique(g)]
print(f"per-subject MF1 (C2): mean={100*np.mean(mf):.1f} sd={100*np.std(mf):.1f} min={100*np.min(mf):.1f} max={100*np.max(mf):.1f}")
print("time", round(time.time() - t0))
