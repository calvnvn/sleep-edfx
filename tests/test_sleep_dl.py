"""Check that src/sleep_dl.py is the notebook-04 code, plus the new fold and loss helpers.

Run from the repository root:  .venv/Scripts/python.exe tests/test_sleep_dl.py
Everything runs on the CPU, where PyTorch is deterministic, so "identical" means bit-identical.
Takes about 3 to 5 minutes (a short training run is repeated with both implementations).
"""
import ast
import copy
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import sleep_dl as sd  # noqa: E402

CACHE = ROOT / "data" / "processed" / "sc"
MANIFEST = ROOT / "results" / "tables" / "02_cache_manifest.csv"
sd.set_device("cpu")
torch.set_num_threads(4)
t_start = time.time()


def check(name, ok):
    print(f"[{'ok' if ok else 'FAIL'}] {name}", flush=True)
    assert ok, name


# ---- load the notebook-04 definitions (functions and classes only) and its CONFIG -----------------------
nb = json.load(open(ROOT / "notebooks" / "04_cnn_bilstm_sc20.ipynb", encoding="utf-8"))
cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
ns = dict(np=np, pd=pd, torch=torch, nn=nn, F=F, copy=copy, random=random, Dataset=Dataset, DataLoader=DataLoader,
          f1_score=f1_score, accuracy_score=accuracy_score, cohen_kappa_score=cohen_kappa_score,
          GroupShuffleSplit=GroupShuffleSplit, IGNORE=-1, SEED=0, STAGES=sd.STAGES,
          DEVICE=torch.device("cpu"), USE_AMP=False)
config_node = [n for n in ast.parse(cells[0]).body
               if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) == "CONFIG"]
exec(compile(ast.Module(config_node, []), "nb04-config", "exec"), ns)
CONFIG = dict(ns["CONFIG"])
for src in cells:
    defs = [n for n in ast.parse(src).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    exec(compile(ast.Module(defs, []), "nb04", "exec"), ns)
print("notebook-04 CONFIG:", CONFIG)

# ---- 1. memory-mapped recordings hold the same values as np.load ------------------------------------------
recs_mm = sd.load_records(CACHE, MANIFEST)
check("153 recordings, sorted by subject and night",
      len(recs_mm) == 153 and [(r["subject"], r["night"]) for r in recs_mm] == sorted((r["subject"], r["night"]) for r in recs_mm))
same = True
for r in recs_mm:
    with np.load(CACHE / f"{r['rec']}.npz") as z:
        same &= np.array_equal(r["eeg"], z["eeg"]) and np.array_equal(r["eog"][:, 0], z["eog"])
        same &= np.array_equal(r["emg"][:, 0], z["emg"]) and np.array_equal(r["labels"], z["labels"])
check("memory-mapped EEG/EOG/EMG/labels equal np.load for all 153 recordings", bool(same))
labels = np.concatenate([r["labels"] for r in recs_mm])
check("195,767 epochs, 298 ignored, class counts of notebook 02",
      len(labels) == 195_767 and (labels == -1).sum() == 298
      and np.bincount(labels[labels >= 0], minlength=5).tolist() == [65_941, 21_522, 69_132, 13_039, 25_835])

recs = sd.load_records(CACHE, MANIFEST, subjects=[0, 1], mmap=False)     # the notebook-04 way (copied to RAM)

# ---- 2. dataset ----------------------------------------------------------------------------------------------
a, b = ns["SequenceDataset"](recs, 20, 10), sd.SequenceDataset(recs_mm[:4], 20, 10)
check("same windows (start positions) as notebook 04", a.index == b.index)
check("same window contents", all(all(torch.equal(x, y) for x, y in zip(a[k], b[k])) for k in [0, 7, len(a) - 1]))

# ---- 3. model: identical initial weights and outputs ------------------------------------------------------
torch.manual_seed(0)
m_nb = ns["SleepNet"]()
torch.manual_seed(0)
m_sd = sd.SleepNet()
sd_nb, sd_sd = m_nb.state_dict(), m_sd.state_dict()
check("same parameter names", list(sd_nb) == list(sd_sd))
check("identical initial weights", all(torch.equal(sd_nb[k], sd_sd[k]) for k in sd_nb))
check("1,089,477 parameters", sum(p.numel() for p in m_sd.parameters()) == 1_089_477)
x = [t.unsqueeze(0) for t in b[3][:3]]
with torch.no_grad():
    check("identical outputs", torch.equal(m_nb.eval()(*x), m_sd.eval()(*x)))
for kw in [dict(eeg_channels=(0,)), dict(eog=False), dict(emg=False), dict(eog=False, emg=False),
           dict(temporal="none")]:
    with torch.no_grad():
        out = sd.SleepNet(**kw).eval()(*x)
    check(f"ablation model {kw}: output (1, 20, 5)", tuple(out.shape) == (1, 20, 5) and bool(torch.isfinite(out).all()))

# ---- 4. losses --------------------------------------------------------------------------------------------------
torch.manual_seed(1)
logits, target = torch.randn(40, 5), torch.randint(0, 5, (40,))
target[:5] = -1
w = sd.class_weights(recs)
check("class weights equal", torch.equal(w, ns["class_weights"](recs)))
check("focal loss equals notebook 04",
      torch.equal(sd.FocalLoss(2.0, w)(logits, target), ns["FocalLoss"](2.0, w)(logits, target)))
check("'ce' is unweighted cross-entropy (as in notebook 04)",
      torch.isclose(sd.make_loss("ce", w)(logits, target), nn.CrossEntropyLoss(ignore_index=-1)(logits, target)))
keep = target >= 0
manual_ce_w = (w[target[keep]] * F.cross_entropy(logits[keep], target[keep], reduction="none")).mean()
check("'ce_w' = mean of class-weighted cross-entropy terms", torch.isclose(sd.make_loss("ce_w", w)(logits, target), manual_ce_w))
check("'focal_uw' = focal loss without weights",
      torch.equal(sd.make_loss("focal_uw", w)(logits, target), ns["FocalLoss"](2.0)(logits, target)))
check("'focal' = notebook-04 focal loss", torch.equal(sd.make_loss("focal", w)(logits, target), ns["FocalLoss"](2.0, w)(logits, target)))

# ---- 5. a short training run gives bit-identical results ---------------------------------------------------
short = dict(CONFIG, max_epochs=2, patience=1)
ns["CONFIG"] = short                                             # notebook-04 functions read the global CONFIG
train, val = recs[:2], recs[2:3]
t0 = time.time()
model_nb, hist_nb, best_nb, ep_nb = ns["train_fold"](train, val, "focal", seed=7)
model_sd, hist_sd, best_sd, ep_sd = sd.train_fold(recs_mm[:2], recs_mm[2:3], dict(short, loss="focal"), seed=7)
check(f"identical training history ({time.time() - t0:.0f} s): {hist_sd}", hist_nb == hist_sd and ep_nb == ep_sd)
check("identical trained weights", all(torch.equal(v, model_sd.state_dict()[k]) for k, v in model_nb.state_dict().items()))
p_nb, p_sd = ns["predict_probs"](model_nb, recs[3:4]), sd.predict_probs(model_sd, recs_mm[3:4], 20)
check("identical predicted probabilities", np.array_equal(p_nb[0], p_sd[0]))
ns["CONFIG"] = CONFIG

# ---- 6. folds -------------------------------------------------------------------------------------------------------
loso = sd.loso_folds(list(range(20)))
check("SC-20 validation subjects equal notebook 04", all(f["val"] == ns["validation_subjects"](k) for k, f in enumerate(loso)))
folds78 = sd.group_kfold_folds(recs_mm)
tested = sorted(s for f in folds78 for s in f["test"])
check("SC-78: 10 folds, every subject tested exactly once", len(folds78) == 10 and tested == sorted({r["subject"] for r in recs_mm}))
check("SC-78: validation and test subjects never overlap", all(not set(f["test"]) & set(f["val"]) for f in folds78))
base = np.load(ROOT / "data" / "processed" / "baseline_predictions.npz")
key = {(rec, int(e)): i for i, (rec, e) in enumerate(zip(base["rec"], base["epoch_idx"]))}
subj = np.concatenate([[r["subject"]] * len(r["labels"]) for r in recs_mm])
order = [key[(r["rec"], int(e))] for r in recs_mm for e in r["epoch_idx"]]
pred78, y78 = base["pred_sc78"][order], base["y"][order]
check("baseline labels line up with the cache", np.array_equal(y78, labels))
per_fold = [f1_score(y78[np.isin(subj, f["test"]) & (y78 >= 0)], pred78[np.isin(subj, f["test"]) & (y78 >= 0)],
                     labels=range(5), average="macro") for f in folds78]
text = f"{100 * np.mean(per_fold):.1f} ± {100 * np.std(per_fold):.1f}"
stored = pd.read_csv(ROOT / "results" / "tables" / "03_baseline_metrics.csv", index_col=0).loc["SC-78 (10 folds)", "MF1 per fold"]
check(f"SC-78 folds reproduce the baseline per-fold MF1 ({text} vs stored {stored})", text == stored)
sc20 = [r for r in recs_mm if r["subject"] < 20]
leak = sd.epochwise_folds(sc20)
n_scored = sum((r["labels"] >= 0).sum() for r in sc20)
cover = sum(m.sum() for f in leak for m in f["test_mask"])
disjoint = all(not (fm & vm).any() and not (fm & tm).any() and not (vm & tm).any()
               for f in leak for fm, vm, tm in zip(f["fit_mask"], f["val_mask"], f["test_mask"]))
check(f"epoch-wise folds: 20 folds, each labelled epoch tested once ({cover:,} of {n_scored:,})",
      len(leak) == 20 and cover == n_scored and disjoint)

# ---- 7. statistics ----------------------------------------------------------------------------------------------
check("Holm adjustment", np.allclose(sd.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06]))
r = sd.paired(pd.Series([0.8, 0.7, 0.9, 0.6, 0.75]), pd.Series([0.7, 0.7, 0.8, 0.5, 0.7]))
check("paired difference", abs(r["mean diff"] - 7.0) < 1e-9 and r["better"] == 4 and r["CI low"] <= 7 <= r["CI high"])

print(f"all checks passed in {time.time() - t_start:.0f} s")
