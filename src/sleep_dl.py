"""Shared deep-learning code for phases 3-5 (notebooks 04, 05 and 06).

The data, model, loss and training code is extracted from notebooks/04_cnn_bilstm_sc20.ipynb. With default
arguments every function creates the same layers in the same order and uses the random-number generators in the
same order as the notebook, so a notebook-04 run and a run through this module are the same experiment.
tests/test_sleep_dl.py checks this (identical initial weights, outputs, losses and a short CPU training run).

Additions for phases 4 and 5:
- recordings are memory-mapped instead of copied into RAM (SC-78 needs about 3.5 GB),
- SleepNet can drop modalities, the second EEG channel or the Bi-LSTM (ablation arms),
- four losses forming a 2x2 design: focusing (gamma 0 or 2) x class weights (off or on),
- folds identical to notebook 03 for SC-78 (GroupKFold) and for the epoch-wise leakage check,
- run_cv saves after every fold, so a long run can be stopped and resumed,
- paired statistics per subject (bootstrap interval, Wilcoxon, Holm correction).
"""
import copy
import json
import random
import struct
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import wilcoxon
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset

IGNORE = -1
STAGES = ["Wake", "N1", "N2", "N3", "REM"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"


def set_device(name):
    """Switch the device used by every function in this module ('cpu' makes runs bit-reproducible)."""
    global DEVICE, USE_AMP
    DEVICE = torch.device(name)
    USE_AMP = DEVICE.type == "cuda"


# --------------------------------------------------------------------------------------------- data

def _npz_memmap(path, key):
    """Memory-map one array stored uncompressed inside an .npz file (np.load cannot do this for .npz)."""
    with zipfile.ZipFile(path) as zf:
        info = zf.getinfo(f"{key}.npy")
    if info.compress_type != zipfile.ZIP_STORED:
        raise ValueError(f"{path.name}:{key} is compressed and cannot be memory-mapped")
    with open(path, "rb") as fh:
        fh.seek(info.header_offset)
        name_len, extra_len = struct.unpack("<HH", fh.read(30)[26:30])
        fh.seek(info.header_offset + 30 + name_len + extra_len)
        version = np.lib.format.read_magic(fh)
        read_header = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        shape, fortran, dtype = read_header(fh)
        offset = fh.tell()
    return np.memmap(path, dtype=dtype, mode="r", shape=shape, offset=offset, order="F" if fortran else "C")


def load_records(cache, manifest_csv, subjects=None, mmap=True):
    """Phase-1 recordings sorted by subject and night. EEG/EOG stay on disk (memory-mapped) when mmap=True;
    the values are identical to np.load. Each record: rec, subject, night, age, flag_emg, eeg (n,2,3000),
    eog (n,1,3000), emg (n,1,30), labels (n,) int64 with -1 for ignored epochs, epoch_idx."""
    manifest = pd.read_csv(manifest_csv)
    if subjects is not None:
        manifest = manifest[manifest["subject"].isin(subjects)]
    records = []
    for r in manifest.sort_values(["subject", "night"]).itertuples():
        path = Path(cache) / f"{r.rec}.npz"
        with np.load(path) as z:
            rec = dict(rec=r.rec, subject=int(z["subject"]), night=int(z["night"]), age=int(z["age"]),
                       flag_emg=bool(r.flag_emg), emg=z["emg"][:, None, :],
                       labels=z["labels"].astype(np.int64), epoch_idx=z["epoch_idx"])
            if not mmap:
                rec["eeg"], rec["eog"] = z["eeg"], z["eog"][:, None, :]
        if mmap:
            rec["eeg"], rec["eog"] = _npz_memmap(path, "eeg"), _npz_memmap(path, "eog")[:, None, :]
        records.append(rec)
    return records


class SequenceDataset(Dataset):
    """Windows of `seq_len` consecutive epochs, always inside one recording. Label -1 stays in the window."""

    def __init__(self, recs, seq_len, stride):
        self.recs, self.seq_len = recs, seq_len
        self.index = []
        for i, r in enumerate(recs):
            n = len(r["labels"])
            starts = list(range(0, n - seq_len + 1, stride))
            if starts[-1] != n - seq_len:
                starts.append(n - seq_len)                       # make sure the end of the night is covered
            self.index += [(i, s) for s in starts if (r["labels"][s:s + seq_len] >= 0).any()]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, k):
        i, s = self.index[k]
        r, e = self.recs[i], s + self.seq_len
        return (torch.from_numpy(r["eeg"][s:e].astype(np.float32)),
                torch.from_numpy(r["eog"][s:e].astype(np.float32)),
                torch.from_numpy(r["emg"][s:e].astype(np.float32)),
                torch.from_numpy(r["labels"][s:e]))


# --------------------------------------------------------------------------------------------- model

def conv_block(c_in, c_out, kernel, stride=1):
    return nn.Sequential(nn.Conv1d(c_in, c_out, kernel, stride=stride, padding=kernel // 2, bias=False),
                         nn.BatchNorm1d(c_out), nn.GELU())


class DualScaleBranch(nn.Module):
    """Two parallel CNN paths over one 30 s epoch (DeepSleepNet style):
    a small kernel (0.5 s) for fast waves such as spindles, a large kernel (4 s) for slow waves."""

    def __init__(self, c_in, width, out_dim, dropout):
        super().__init__()
        self.small = nn.Sequential(conv_block(c_in, width, 50, stride=6), nn.MaxPool1d(8), nn.Dropout(dropout),
                                   conv_block(width, 2 * width, 8), conv_block(2 * width, 2 * width, 8),
                                   nn.MaxPool1d(4), nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.large = nn.Sequential(conv_block(c_in, width, 400, stride=50), nn.MaxPool1d(4), nn.Dropout(dropout),
                                   conv_block(width, 2 * width, 6), conv_block(2 * width, 2 * width, 6),
                                   nn.MaxPool1d(2), nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.project = nn.Linear(4 * width, out_dim)

    def forward(self, x):
        return self.project(torch.cat([self.small(x), self.large(x)], dim=1))


class SleepNet(nn.Module):
    """Multi-branch 1D-CNN (EEG, EOG, EMG) per epoch, then a 2-layer Bi-LSTM across the window.

    Ablation switches (defaults = notebook-04 model, 1,089,477 parameters):
    eeg_channels: (0, 1) = Fpz-Cz and Pz-Oz, (0,) = Fpz-Cz only; eog / emg: keep the branch or drop it;
    temporal: "lstm" or "none" (each epoch classified from its own CNN features, no context)."""

    def __init__(self, dropout=0.3, hidden=128, eeg_channels=(0, 1), eog=True, emg=True, temporal="lstm"):
        super().__init__()
        self.eeg_channels, self.use_eog, self.use_emg = tuple(eeg_channels), eog, emg
        self.eeg = DualScaleBranch(len(self.eeg_channels), 48, 128, dropout)
        if eog:
            self.eog = DualScaleBranch(1, 24, 64, dropout)
        if emg:
            self.emg = nn.Sequential(conv_block(1, 16, 5), conv_block(16, 32, 3),   # 30 values, no downsampling
                                     nn.AdaptiveAvgPool1d(1), nn.Flatten())
        width = 128 + 64 * eog + 32 * emg
        self.fuse = nn.Sequential(nn.LayerNorm(width), nn.Dropout(dropout))
        if temporal == "lstm":
            self.lstm = nn.LSTM(width, hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * hidden, 5))
        elif temporal == "none":
            self.lstm = None
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(width, 5))
        else:
            raise ValueError(temporal)

    def forward(self, eeg, eog, emg):                             # inputs: (batch, seq_len, channels, samples)
        b, t = eeg.shape[:2]
        flat = lambda x: x.reshape(b * t, *x.shape[2:])
        if self.eeg_channels != (0, 1):
            eeg = eeg[:, :, list(self.eeg_channels)]
        parts = [self.eeg(flat(eeg))]
        if self.use_eog:
            parts.append(self.eog(flat(eog)))
        if self.use_emg:
            parts.append(self.emg(flat(emg)))
        feats = self.fuse(torch.cat(parts, dim=1)).reshape(b, t, -1)
        if self.lstm is not None:
            feats, _ = self.lstm(feats)
        return self.head(feats)                                   # (batch, seq_len, 5) logits, one per epoch


# --------------------------------------------------------------------------------------------- losses

def class_weights(recs):
    """Inverse square-root class frequency of the training labels, normalised to mean 1."""
    y = np.concatenate([r["labels"] for r in recs])
    counts = np.bincount(y[y >= 0], minlength=5).astype(np.float64)
    w = np.sqrt(counts.sum() / counts)
    return torch.tensor(w / w.mean(), dtype=torch.float32)


class FocalLoss(nn.Module):
    """FL = -alpha_t (1 - p_t)^gamma log p_t, with p_t the predicted probability of the true class.
    alpha (class weights) only multiplies the loss; it does not change p_t. The mean is taken over epochs,
    so gamma=0 with weights is a class-weighted cross-entropy on the same scale."""

    def __init__(self, gamma, alpha=None, ignore_index=IGNORE):
        super().__init__()
        self.gamma, self.ignore_index = gamma, ignore_index
        self.register_buffer("alpha", alpha if alpha is not None else torch.ones(5))

    def forward(self, logits, target):                            # logits (N, 5), target (N,)
        keep = target != self.ignore_index
        logp = F.log_softmax(logits[keep].float(), dim=-1)
        logp_t = logp.gather(1, target[keep][:, None]).squeeze(1)
        p_t = logp_t.exp()
        return (-self.alpha[target[keep]] * (1 - p_t) ** self.gamma * logp_t).mean()


LOSSES = {  # name: (gamma, class weights?)  -- the 2x2 design of phase 5
    "ce": (0.0, False),        # notebook 04 control (implemented with nn.CrossEntropyLoss, numerically identical)
    "ce_w": (0.0, True),
    "focal_uw": (2.0, False),
    "focal": (2.0, True),      # notebook 04 "focal"
}


def make_loss(name, weights, gamma=2.0):
    if name == "ce":
        return nn.CrossEntropyLoss(ignore_index=IGNORE)
    g, weighted = LOSSES[name]
    return FocalLoss(gamma if g else 0.0, weights if weighted else None)


# --------------------------------------------------------------------------------------------- training

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict_probs(model, recs, seq_len, windows_per_batch=64):
    """Class probabilities per epoch. Windows overlap by half; each epoch averages the windows covering it."""
    model.eval()
    L = seq_len
    out = []
    for r in recs:
        n = len(r["labels"])
        starts = list(range(0, n - L + 1, L // 2))
        if starts[-1] != n - L:
            starts.append(n - L)
        probs, counts = np.zeros((n, 5)), np.zeros(n)
        for i in range(0, len(starts), windows_per_batch):
            chunk = starts[i:i + windows_per_batch]
            x = [torch.from_numpy(np.stack([r[k][s:s + L] for s in chunk]).astype(np.float32)).to(DEVICE)
                 for k in ["eeg", "eog", "emg"]]
            with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=USE_AMP):
                p = torch.softmax(model(*x).float(), dim=-1).cpu().numpy()
            for s, ps in zip(chunk, p):
                probs[s:s + L] += ps
                counts[s:s + L] += 1
        out.append(probs / counts[:, None])
    return out


def summarise(y_true, y_pred):
    f1 = f1_score(y_true, y_pred, labels=range(5), average=None, zero_division=0)
    return {"ACC": accuracy_score(y_true, y_pred), "MF1": f1.mean(), "kappa": cohen_kappa_score(y_true, y_pred),
            **{f"F1_{s}": v for s, v in zip(STAGES, f1)}}


def macro_f1_of(model, recs, seq_len):
    y = np.concatenate([r["labels"] for r in recs])
    p = np.concatenate(predict_probs(model, recs, seq_len)).argmax(1)
    return f1_score(y[y >= 0], p[y >= 0], labels=range(5), average="macro", zero_division=0)


def train_fold(train_recs, val_recs, cfg, seed):
    """Train on train_recs; keep the weights with the best validation Macro-F1 (early stopping).
    cfg: seq_len, train_stride, batch_size, lr, weight_decay, max_epochs, patience, grad_clip, focal_gamma,
    dropout, loss, and optionally model (keyword arguments of SleepNet)."""
    set_seed(seed)
    loader = DataLoader(SequenceDataset(train_recs, cfg["seq_len"], cfg["train_stride"]),
                        batch_size=cfg["batch_size"], shuffle=True, drop_last=True, pin_memory=USE_AMP)
    model = SleepNet(dropout=cfg["dropout"], **cfg.get("model", {})).to(DEVICE)
    criterion = make_loss(cfg["loss"], class_weights(train_recs), cfg["focal_gamma"]).to(DEVICE)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cfg["max_epochs"])
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
    best_mf1, best_state, best_epoch, wait, history = -1.0, None, 0, 0, []
    for epoch in range(1, cfg["max_epochs"] + 1):
        model.train()
        losses = []
        for eeg, eog, emg, y in loader:
            eeg, eog, emg, y = (t.to(DEVICE, non_blocking=True) for t in (eeg, eog, emg, y))
            optim.zero_grad(set_to_none=True)
            with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=USE_AMP):
                logits = model(eeg, eog, emg)
            loss = criterion(logits.float().reshape(-1, 5), y.reshape(-1))       # loss in float32 for stability
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            scaler.step(optim)
            scaler.update()
            losses.append(loss.item())
        sched.step()
        val_mf1 = macro_f1_of(model, val_recs, cfg["seq_len"])
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_mf1": val_mf1})
        if val_mf1 > best_mf1:
            best_mf1, best_epoch, wait = val_mf1, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= cfg["patience"]:
                break
    model.load_state_dict(best_state)
    return model, history, best_mf1, best_epoch


# --------------------------------------------------------------------------------------------- folds

def loso_folds(subjects, seed=0, val_fraction=0.1):
    """One test subject per fold; ~10% of the remaining subjects for validation (notebook 03/04 rule:
    GroupShuffleSplit with random_state = seed + fold)."""
    folds = []
    for k, test in enumerate(subjects):
        train = np.array([s for s in subjects if s != test])
        gss = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed + k)
        _, v = next(gss.split(train, groups=train))
        folds.append({"test": [int(test)], "val": sorted(train[v].tolist())})
    return folds


def scored_epochs_in_baseline_order(records):
    """(record index, epoch position) of every labelled epoch, in the order notebook 03 used
    (recordings sorted by file name, epochs in time order)."""
    order = sorted(range(len(records)), key=lambda i: records[i]["rec"])
    pairs = [(i, j) for i in order for j in np.flatnonzero(records[i]["labels"] >= 0)]
    return np.array(pairs)


def group_kfold_folds(records, n_splits=10, seed=0, val_fraction=0.1):
    """Subject folds identical to notebook 03 for SC-78: GroupKFold over labelled epochs grouped by subject,
    then GroupShuffleSplit (random_state = seed + fold) over the training epochs for validation subjects."""
    pairs = scored_epochs_in_baseline_order(records)
    groups = np.array([records[i]["subject"] for i, _ in pairs])
    folds = []
    for k, (train, test) in enumerate(GroupKFold(n_splits=n_splits).split(np.zeros((len(groups), 1)), groups=groups)):
        inner = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed + k)
        _, v = next(inner.split(train, groups=groups[train]))
        folds.append({"test": sorted(np.unique(groups[test]).tolist()), "val": sorted(np.unique(groups[train][v]).tolist())})
    return folds


def epochwise_folds(records, n_splits=20, seed=0, val_fraction=0.1):
    """The leaky split of notebook 03 (section H2): every labelled epoch is its own group, so test, validation
    and training epochs come from the same nights. Returned as boolean masks per recording."""
    pairs = scored_epochs_in_baseline_order(records)
    ids = np.arange(len(pairs))
    folds = []
    for k, (train, test) in enumerate(GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
                                      .split(np.zeros((len(ids), 1)), groups=ids)):
        inner = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed + k)
        fit, val = (train[i] for i in next(inner.split(train, groups=ids[train])))
        masks = {}
        for name, idx in [("fit_mask", fit), ("val_mask", val), ("test_mask", test)]:
            m = [np.zeros(len(r["labels"]), dtype=bool) for r in records]
            for i, j in pairs[idx]:
                m[i][j] = True
            masks[name] = m
        folds.append(masks)
    return folds


# --------------------------------------------------------------------------------------------- cross-validation

def _masked(rec, mask):
    return dict(rec, labels=np.where(mask, rec["labels"], IGNORE))


def run_cv(name, cfg, seed, records, folds, runs_dir, log=print):
    """Cross-validated run of one configuration and one seed. Results are saved after every fold
    (runs_dir/<name>_seed<seed>.partial.npz) and the run resumes from there; the finished run is
    runs_dir/<name>_seed<seed>.npz and is reused only if configuration and folds are unchanged.
    Folds are either subject folds {"test": [...], "val": [...]} or epoch masks (epochwise_folds)."""
    runs_dir = Path(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    path, part = runs_dir / f"{name}_seed{seed}.npz", runs_dir / f"{name}_seed{seed}.partial.npz"
    fold_sig = json.dumps([{k: v for k, v in f.items() if k in ("test", "val")} or {"epochwise": len(folds)}
                           for f in folds])
    key = json.dumps({**cfg, "name": name, "seed": seed, "folds": fold_sig}, sort_keys=True)
    # `with` closes the npz file; an open handle would block part.unlink() on Windows
    if path.exists():
        with np.load(path) as z:
            if str(z["config"]) == key:
                return {"probs": z["probs"], "val_mf1": z["val_mf1"], "best_epoch": z["best_epoch"],
                        "history": json.loads(str(z["history"])), "minutes": float(z["minutes"])}
    offsets = np.cumsum([0] + [len(r["labels"]) for r in records])
    probs = np.full((offsets[-1], 5), np.nan)
    val_mf1, best_epoch, history, minutes_before = [], [], {}, 0.0
    if part.exists():
        with np.load(part) as z:
            if str(z["config"]) == key:
                probs, val_mf1, best_epoch = z["probs"], list(z["val_mf1"]), list(z["best_epoch"])
                history, minutes_before = json.loads(str(z["history"])), float(z["minutes"])
                log(f"  {name} seed {seed}: resuming after {len(val_mf1)} of {len(folds)} folds")
    t0 = time.time()
    for k in range(len(val_mf1), len(folds)):
        fold = folds[k]
        if "test" in fold:
            test_s, val_s = set(fold["test"]), set(fold["val"])
            assert not test_s & val_s
            train_recs = [r for r in records if r["subject"] not in test_s | val_s]
            val_recs = [r for r in records if r["subject"] in val_s]
            test_idx = [i for i, r in enumerate(records) if r["subject"] in test_s]
            model, hist, best, ep = train_fold(train_recs, val_recs, cfg, seed=1000 * seed + k)
            for i, p in zip(test_idx, predict_probs(model, [records[i] for i in test_idx], cfg["seq_len"])):
                probs[offsets[i]:offsets[i + 1]] = p
        else:
            train_recs = [_masked(r, m) for r, m in zip(records, fold["fit_mask"])]
            val_recs = [_masked(r, m) for r, m in zip(records, fold["val_mask"])]
            model, hist, best, ep = train_fold(train_recs, val_recs, cfg, seed=1000 * seed + k)
            for i, (p, m) in enumerate(zip(predict_probs(model, records, cfg["seq_len"]), fold["test_mask"])):
                probs[offsets[i]:offsets[i + 1]][m] = p[m]
        val_mf1.append(best)
        best_epoch.append(ep)
        history[str(k)] = hist
        minutes = minutes_before + (time.time() - t0) / 60
        log(f"  {name} seed {seed} fold {k:2d}: best epoch {ep:2d}, val MF1 {100 * best:.1f}, {minutes:.1f} min")
        np.savez(part, probs=probs, val_mf1=np.array(val_mf1), best_epoch=np.array(best_epoch),
                 history=json.dumps(history), minutes=minutes, config=key)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    minutes = minutes_before + (time.time() - t0) / 60
    np.savez(path, probs=probs, val_mf1=np.array(val_mf1), best_epoch=np.array(best_epoch),
             history=json.dumps(history), minutes=minutes, config=key)
    part.unlink(missing_ok=True)
    return {"probs": probs, "val_mf1": np.array(val_mf1), "best_epoch": np.array(best_epoch),
            "history": history, "minutes": minutes}


# --------------------------------------------------------------------------------------------- statistics

def per_subject_mf1(pred, y, subj):
    """Macro-F1 of every subject over the stages that subject actually has."""
    keep = y >= 0
    return pd.Series({s: f1_score(y[keep & (subj == s)], pred[keep & (subj == s)],
                                  labels=np.unique(y[keep & (subj == s)]), average="macro")
                      for s in np.unique(subj[keep])})


def paired(a, b, n_boot=10_000, seed=0):
    """a - b per subject (same subjects, two models): mean difference in points with a 95% bootstrap interval
    over subjects, number of subjects where a is better, and the Wilcoxon signed-rank p-value."""
    d = (pd.Series(a) - pd.Series(b)).dropna().to_numpy()
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"mean diff": 100 * d.mean(), "CI low": 100 * lo, "CI high": 100 * hi,
            "better": int((d > 0).sum()), "subjects": len(d), "p": wilcoxon(d).pvalue}


def holm(p):
    """Holm-Bonferroni adjusted p-values (same order as the input)."""
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj
