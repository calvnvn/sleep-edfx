"""Extract per-epoch spectral/EMG features for SC records, then run quick ablations
(modality, validation scheme, class weighting, temporal context) with a GBDT baseline."""
import os, sys, glob, time
import numpy as np
from scipy.signal import welch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from edfx_eda import read_header, read_signals, read_annotations, epochs_from_ann, trim_wake

ROOT = sys.argv[1]
OUT = sys.argv[2]
BANDS = [(0.5, 2), (2, 4), (4, 8), (8, 12), (12, 15), (15, 30)]


def eeg_feats(x):  # x: (n_ep, 3000)
    f, p = welch(x, fs=100, nperseg=400, axis=-1)
    tot = p[:, (f >= 0.5) & (f <= 30)].sum(1) + 1e-12
    rel = [p[:, (f >= a) & (f < b)].sum(1) / tot for a, b in BANDS]
    return np.column_stack([np.log(tot)] + [np.log(r + 1e-6) for r in rel] +
                           [np.log(np.std(x, 1) + 1e-6), np.log(np.abs(np.diff(x, axis=1)).mean(1) + 1e-6)])


def eog_feats(x):
    f, p = welch(x, fs=100, nperseg=400, axis=-1)
    slow = p[:, (f >= 0.3) & (f < 2)].sum(1); fast = p[:, (f >= 2) & (f < 8)].sum(1)
    return np.column_stack([np.log(slow + 1e-6), np.log(fast + 1e-6), np.log(fast / (slow + 1e-9) + 1e-6),
                            np.log(np.ptp(x, 1) + 1e-6)])


def emg_feats(x):  # x: (n_ep, 30) envelope @1Hz
    lx = x  # SC EMG envelope has per-recording offset and negative values: no log; robust per-record scaling is applied later
    return np.column_stack([lx.mean(1), lx.std(1), lx.min(1), lx.max(1)])


def main():
    Xs, ys, gs, rs = [], [], [], []
    files = sorted(glob.glob(os.path.join(ROOT, "sleep-cassette", "*-PSG.edf")))
    t0 = time.time()
    for k, psg in enumerate(files):
        base = os.path.basename(psg)[:7]
        hyp = glob.glob(os.path.join(ROOT, "sleep-cassette", base + "*-Hypnogram.edf"))[0]
        hdr = read_header(psg)
        sig = read_signals(psg, hdr, {"EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal", "EMG submental"})
        lab = epochs_from_ann(read_annotations(hyp), hdr["duration_s"])
        sleep = np.where((lab >= 1) & (lab <= 4))[0]
        a, b = max(0, sleep[0] - 60), min(len(lab), sleep[-1] + 61)
        idx = np.arange(a, b); idx = idx[lab[idx] <= 4]
        def ep(name, fs):
            s = sig[name]; n = int(fs * 30)
            return s[: (len(s) // n) * n].reshape(-1, n)[idx]
        F = np.hstack([eeg_feats(ep("EEG Fpz-Cz", 100)), eeg_feats(ep("EEG Pz-Oz", 100)),
                       eog_feats(ep("EOG horizontal", 100)), emg_feats(ep("EMG submental", 1))])
        F = (F - np.median(F, 0)) / (np.percentile(F, 75, 0) - np.percentile(F, 25, 0) + 1e-6)  # per-record robust z
        Xs.append(F.astype(np.float32)); ys.append(lab[idx]); gs.append(np.full(len(idx), int(base[3:5])))
        rs.append(np.full(len(idx), k))
        if k % 20 == 0:
            print(k, base, F.shape, f"{time.time()-t0:.0f}s", flush=True)
    np.savez_compressed(os.path.join(OUT, "sc_features.npz"), X=np.vstack(Xs), y=np.concatenate(ys),
                        g=np.concatenate(gs), r=np.concatenate(rs))
    print("done", time.time() - t0)


if __name__ == "__main__":
    main()
