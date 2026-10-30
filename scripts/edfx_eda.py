"""Minimal EDF/EDF+ reader + EDA for Sleep-EDF Expanded (no mne dependency)."""
import os, re, sys, json, glob
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.join(REPO, "data", "raw", "sleep-edfx")  # overridable from the command line (see __main__)
OUT = os.path.join(REPO, "results", "eda")


def read_header(path):
    with open(path, "rb") as f:
        h = f.read(256)
        ns = int(h[252:256])
        sh = f.read(ns * 256)
    def field(off, width):
        return [sh[off + i * width: off + (i + 1) * width].decode("latin-1").strip() for i in range(ns)]
    o = 0
    labels = field(o, 16); o += 16 * ns
    trans = field(o, 80); o += 80 * ns
    dims = field(o, 8); o += 8 * ns
    pmin = field(o, 8); o += 8 * ns
    pmax = field(o, 8); o += 8 * ns
    dmin = field(o, 8); o += 8 * ns
    dmax = field(o, 8); o += 8 * ns
    pref = field(o, 80); o += 80 * ns
    spr = field(o, 8); o += 8 * ns
    nrec = int(h[236:244]); dur = float(h[244:252])
    return dict(patient=h[8:88].decode("latin-1").strip(), recording=h[88:168].decode("latin-1").strip(),
                startdate=h[168:176].decode(), starttime=h[176:184].decode(), hdr_bytes=int(h[184:192]),
                reserved=h[192:236].decode().strip(), nrec=nrec, rec_dur=dur, ns=ns,
                labels=labels, dims=dims, pmin=[float(x) for x in pmin], pmax=[float(x) for x in pmax],
                dmin=[int(x) for x in dmin], dmax=[int(x) for x in dmax], prefilter=pref,
                spr=[int(x) for x in spr], fs=[int(x) / dur if dur else 0 for x in spr], duration_s=nrec * dur)


def read_signals(path, hdr, wanted):
    """Return dict label -> physical signal (float32) for wanted labels."""
    spr = hdr["spr"]; rec_len = sum(spr)
    raw = np.fromfile(path, dtype="<i2", offset=hdr["hdr_bytes"], count=hdr["nrec"] * rec_len)
    raw = raw.reshape(hdr["nrec"], rec_len)
    out, start = {}, 0
    for i, lab in enumerate(hdr["labels"]):
        if lab in wanted:
            d = raw[:, start:start + spr[i]].reshape(-1).astype(np.float32)
            g = (hdr["pmax"][i] - hdr["pmin"][i]) / (hdr["dmax"][i] - hdr["dmin"][i])
            out[lab] = (d - hdr["dmin"][i]) * g + hdr["pmin"][i]
        start += spr[i]
    return out


TAL = re.compile(rb"([+-]\d+(?:\.\d*)?)(?:\x15(\d+(?:\.\d*)?))?\x14(.*?)\x14")


def read_annotations(path):
    hdr = read_header(path)
    with open(path, "rb") as f:
        f.seek(hdr["hdr_bytes"]); data = f.read()
    anns = []
    for m in TAL.finditer(data):
        onset, dur, txt = m.group(1), m.group(2), m.group(3)
        for t in txt.split(b"\x14"):
            if t:
                anns.append((float(onset), float(dur) if dur else 0.0, t.decode("latin-1")))
    return anns


STAGE = {"Sleep stage W": 0, "Sleep stage 1": 1, "Sleep stage 2": 2, "Sleep stage 3": 3,
         "Sleep stage 4": 3, "Sleep stage R": 4, "Movement time": 5, "Sleep stage ?": 6}
NAMES = ["W", "N1", "N2", "N3", "REM", "MT", "?"]


def epochs_from_ann(anns, psg_dur):
    n = int(psg_dur // 30)
    lab = np.full(n, 6, dtype=np.int8)  # default unscored
    for on, du, t in anns:
        if t not in STAGE:
            continue
        a, b = int(on // 30), int((on + du) // 30)
        lab[a:min(b, n)] = STAGE[t]
    return lab


def trim_wake(lab, minutes=30):
    sleep = np.where((lab >= 1) & (lab <= 4))[0]
    if len(sleep) == 0:
        return lab[:0]
    k = minutes * 2
    return lab[max(0, sleep[0] - k): min(len(lab), sleep[-1] + k + 1)]


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # console Windows (cp1252) tidak crash pada karakter non-ASCII
    recs = []
    for sub in ["sleep-cassette", "sleep-telemetry"]:
        for psg in sorted(glob.glob(os.path.join(ROOT, sub, "*-PSG.edf"))):
            base = os.path.basename(psg)[:7]
            hyps = glob.glob(os.path.join(ROOT, sub, base + "*-Hypnogram.edf"))
            hdr = read_header(psg)
            anns = read_annotations(hyps[0])
            ann_end = max(on + du for on, du, t in anns)
            lab = epochs_from_ann(anns, hdr["duration_s"])
            lab_t = trim_wake(lab)
            subj = int(base[3:5]); night = int(base[5])
            m = re.search(r"(Male|Female)_(\d+)yr", hdr["patient"])
            recs.append(dict(rec=base, set=sub[6:], subject=("SC" if sub[6] == "c" else "ST") + f"{subj:02d}",
                             night=night, sex=m.group(1) if m else None, age=int(m.group(2)) if m else None,
                             hours=hdr["duration_s"] / 3600, ann_end_h=ann_end / 3600,
                             start=hdr["starttime"], labels=hdr["labels"], fs=hdr["fs"],
                             prefilter=hdr["prefilter"], dims=hdr["dims"],
                             full=np.bincount(lab, minlength=7).tolist(),
                             trim=np.bincount(lab_t, minlength=7).tolist(),
                             lab_trim=lab_t))
    # --- summaries
    def agg(rs, key):
        c = np.sum([r[key] for r in rs], axis=0)
        return c
    report = {}
    groups = {"SC-all(153)": [r for r in recs if r["set"] == "cassette"],
              "SC-20(39)": [r for r in recs if r["set"] == "cassette" and int(r["rec"][3:5]) < 20],
              "ST(44)": [r for r in recs if r["set"] == "telemetry"],
              "ALL(197)": recs}
    lines = []
    for g, rs in groups.items():
        for key in ["full", "trim"]:
            c = agg(rs, key); tot = c.sum(); sc = c[:5].sum()
            pct = " ".join(f"{NAMES[i]}={c[i]}({100*c[i]/tot:.1f}%)" for i in range(7))
            pct5 = " ".join(f"{NAMES[i]}={100*c[i]/sc:.1f}%" for i in range(5))
            lines.append(f"{g:12s} {key:4s} n_rec={len(rs)} subj={len(set(r['subject'] for r in rs))} "
                         f"epochs={tot} scored5={sc} hours={sum(r['hours'] for r in rs):.1f}\n   {pct}\n   5-class: {pct5}")
    # channel layouts
    layouts = {}
    for r in recs:
        k = (r["set"], tuple(f"{l}@{f:g}Hz" for l, f in zip(r["labels"], r["fs"])))
        layouts.setdefault(k, []).append(r["rec"])
    lines.append("\nCHANNEL LAYOUTS:")
    for (s, k), v in layouts.items():
        lines.append(f" [{s}] n={len(v)} e.g.{v[0]}: {k}")
    lines.append("\nPREFILTER/UNITS example SC: " + json.dumps(list(zip(recs[0]["labels"], recs[0]["dims"], recs[0]["prefilter"]))))
    st0 = groups["ST(44)"][0]
    lines.append("PREFILTER/UNITS example ST: " + json.dumps(list(zip(st0["labels"], st0["dims"], st0["prefilter"]))))
    # durations, annotation mismatch
    for g in ["SC-all(153)", "ST(44)"]:
        rs = groups[g]; h = np.array([r["hours"] for r in rs])
        mism = [(r["rec"], round(r["ann_end_h"] - r["hours"], 3)) for r in rs if abs(r["ann_end_h"] - r["hours"]) > 1/120]
        lines.append(f"{g}: hours mean={h.mean():.2f} min={h.min():.2f} max={h.max():.2f}; hypnogram_end != psg_end in {len(mism)} recs, e.g. {mism[:4]}")
    # subjects / nights
    from collections import Counter
    for g in ["SC-all(153)", "ST(44)"]:
        rs = groups[g]
        nights = Counter(Counter(r["subject"] for r in rs).values())
        ages = np.array([r["age"] for r in rs if r["night"] == 1 or True])
        sexes = Counter(r["sex"] for r in {r['subject']: r for r in rs}.values())
        lines.append(f"{g}: nights-per-subject={dict(nights)} age range={ages.min()}-{ages.max()} median={np.median(ages):.0f} sex={dict(sexes)}")
    single = [s for s, c in Counter(r["subject"] for r in groups["SC-all(153)"]).items() if c == 1]
    lines.append(f"SC subjects with single night: {single}")
    # per-record N1 share variability (trimmed, 5-class)
    for g in ["SC-all(153)", "ST(44)"]:
        n1 = np.array([r["trim"][1] / max(1, sum(r["trim"][:5])) for r in groups[g]])
        w = np.array([r["trim"][0] / max(1, sum(r["trim"][:5])) for r in groups[g]])
        n3 = np.array([r["trim"][3] / max(1, sum(r["trim"][:5])) for r in groups[g]])
        lines.append(f"{g} per-record (trimmed) N1% mean={100*n1.mean():.1f} min={100*n1.min():.1f} max={100*n1.max():.1f}; "
                     f"W% mean={100*w.mean():.1f}; N3% mean={100*n3.mean():.1f} min={100*n3.min():.1f} recs N3<1%: {int((n3<0.01).sum())}")
    # age vs N3
    rs = groups["SC-all(153)"]
    for lo, hi in [(25, 40), (40, 60), (60, 80), (80, 120)]:
        sel = [r for r in rs if lo <= r["age"] < hi]
        if sel:
            c = np.sum([r["trim"][:5] for r in sel], axis=0); c = c / c.sum()
            lines.append(f"SC age {lo}-{hi}: n_rec={len(sel)} " + " ".join(f"{NAMES[i]}={100*c[i]:.1f}%" for i in range(5)))
    # transition matrix SC trimmed (scored only)
    T = np.zeros((5, 5), int)
    for r in rs:
        l = r["lab_trim"]
        for a, b in zip(l[:-1], l[1:]):
            if a < 5 and b < 5:
                T[a, b] += 1
    P = T / T.sum(1, keepdims=True)
    lines.append("\nTRANSITION P(next|cur) SC-78 trimmed (rows W,N1,N2,N3,REM):")
    for i in range(5):
        lines.append("  " + NAMES[i].ljust(4) + " ".join(f"{p:.3f}" for p in P[i]))
    # epochs where label differs from both neighbours
    iso = sum(int(((l[1:-1] != l[:-2]) & (l[1:-1] != l[2:])).sum()) for l in [r["lab_trim"] for r in rs])
    lines.append(f"Self-transition (stay) rate overall: {np.trace(T)/T.sum():.3f}; isolated epochs (differ from both neighbours): {iso}")
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(OUT, "eda_summary.txt"), "w", encoding="utf-8") as f:
        f.write(txt)
    with open(os.path.join(OUT, "eda_records.json"), "w", encoding="utf-8") as f:
        json.dump([{k: v for k, v in r.items() if k != "lab_trim"} for r in recs], f, default=str)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        ROOT = sys.argv[1]
    if len(sys.argv) > 2:
        OUT = sys.argv[2]
    main()
