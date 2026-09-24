#!/usr/bin/env python3
"""Analyse the final MNIST runs: accuracy with confidence intervals, held-out
accuracy, per-digit results, confusions, learning curves. Writes SVG figures
and summary.json into results/final/. Standard library only."""
import csv
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FINAL = os.path.join(HERE, "results", "final")
FIG = os.path.join(FINAL, "figures")
TUNED_ON = 1000  # settings were chosen on test images 0..999

# Reference palette (light mode; figures carry their own light surface).
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
BLUE, ORANGE = "#2a78d6", "#eb6834"
SEQ = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
FONT = "font-family='system-ui, -apple-system, Segoe UI, sans-serif'"


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def load_run(name):
    js = os.path.join(FINAL, name + ".json")
    if not os.path.exists(js):
        return None
    with open(js) as f:
        r = json.load(f)
    with open(os.path.join(FINAL, name + "_test_predictions.csv")) as f:
        preds = [(int(x["index"]), int(x["label"]), int(x["predicted"]), int(x["spikes"])) for x in csv.DictReader(f)]
    held = [p for p in preds if p[0] >= TUNED_ON] or preds  # no untouched images: fall back to all
    k_all = sum(p[1] == p[2] for p in preds)
    k_held = sum(p[1] == p[2] for p in held)
    r["name"] = name
    r["n_test"] = len(preds)
    r["acc_all"] = k_all / len(preds)
    r["ci_all"] = wilson(k_all, len(preds))
    r["acc_heldout"] = k_held / len(held) if held else None
    r["ci_heldout"] = wilson(k_held, len(held))
    r["n_heldout"] = len(held)
    r["per_digit"] = [r["confusion"][y][y] / max(1, sum(r["confusion"][y])) for y in range(10)]
    conf = [(r["confusion"][y][g], y, g) for y in range(10) for g in range(10) if y != g]
    r["top_confusions"] = sorted(conf, reverse=True)[:6]
    silent = [p for p in preds if p[3] == 0]
    r["test_no_spikes"] = len(silent)
    train_csv = os.path.join(FINAL, name + "_train.csv")
    r["curve"] = []
    if os.path.exists(train_csv):
        with open(train_csv) as f:
            r["curve"] = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]
    return r


# ------------------------------------------------------------------ SVG ----

def svg_open(w, h, title):
    return [f"<svg xmlns='http://www.w3.org/2000/svg' width='{w}' height='{h}' viewBox='0 0 {w} {h}' {FONT}>",
            f"<title>{title}</title>",
            f"<rect width='{w}' height='{h}' fill='{SURFACE}'/>",
            f"<text x='16' y='26' font-size='15' font-weight='600' fill='{INK}'>{title}</text>"]


def nice_max(v):
    if v <= 0:
        return 1
    e = 10 ** math.floor(math.log10(v))
    for m in (1, 2, 2.5, 5, 10):
        if v <= m * e:
            return m * e
    return 10 * e


def line_chart(path, title, xs, ys, xlabel, ylabel, color=BLUE, w=640, h=300, ymax=None):
    L, R, T, B = 64, 24, 48, 48
    pw, ph = w - L - R, h - T - B
    xmax = max(xs)
    ymax = ymax or nice_max(max(ys) * 1.05)
    X = lambda x: L + pw * x / xmax
    Y = lambda y: T + ph * (1 - y / ymax)
    s = svg_open(w, h, title)
    for i in range(5):
        v = ymax * i / 4
        s.append(f"<line x1='{L}' x2='{L + pw}' y1='{Y(v):.1f}' y2='{Y(v):.1f}' stroke='{GRID}' stroke-width='1'/>")
        s.append(f"<text x='{L - 8}' y='{Y(v) + 4:.1f}' font-size='11' fill='{INK2}' text-anchor='end'>{v:g}</text>")
    for i in range(5):
        v = xmax * i / 4
        s.append(f"<text x='{X(v):.1f}' y='{T + ph + 18}' font-size='11' fill='{INK2}' text-anchor='middle'>{v / 1000:g}k</text>")
    s.append(f"<text x='{L + pw / 2}' y='{h - 8}' font-size='12' fill='{INK2}' text-anchor='middle'>{xlabel}</text>")
    s.append(f"<text x='14' y='{T + ph / 2}' font-size='12' fill='{INK2}' text-anchor='middle' transform='rotate(-90 14 {T + ph / 2})'>{ylabel}</text>")
    pts = " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in zip(xs, ys))
    s.append(f"<polyline points='{pts}' fill='none' stroke='{color}' stroke-width='2' stroke-linejoin='round'/>")
    s.append(f"<text x='{X(xs[-1]) - 4:.1f}' y='{Y(ys[-1]) - 8:.1f}' font-size='11' fill='{INK}' text-anchor='end'>{ys[-1]:.1f}</text>")
    s.append("</svg>")
    open(path, "w").write("\n".join(s))


def hbar_chart(path, title, rows, w=640, note=None):
    """rows: (label, value 0..1, lo, hi, emphasis)"""
    L, R, T = 250, 70, 48
    bh, gap = 22, 12
    h = T + len(rows) * (bh + gap) + 40
    pw = w - L - R
    X = lambda v: L + pw * v
    s = svg_open(w, h, title)
    for i in range(6):
        v = i / 5
        s.append(f"<line x1='{X(v):.1f}' x2='{X(v):.1f}' y1='{T - 6}' y2='{h - 34}' stroke='{GRID}' stroke-width='1'/>")
        s.append(f"<text x='{X(v):.1f}' y='{h - 18}' font-size='11' fill='{INK2}' text-anchor='middle'>{int(v * 100)}%</text>")
    for i, (label, v, lo, hi, emph) in enumerate(rows):
        y = T + i * (bh + gap)
        color = BLUE if emph else "#9ec5f4"
        s.append(f"<text x='{L - 10}' y='{y + bh / 2 + 4}' font-size='12' fill='{INK}' text-anchor='end'>{label}</text>")
        bw = max(0.0, X(v) - L)
        s.append(f"<path d='M{L},{y} h{max(0, bw - 4):.1f} a4,4 0 0 1 4,4 v{bh - 8} a4,4 0 0 1 -4,4 h{-max(0, bw - 4):.1f} z' fill='{color}'/>")
        if hi > lo:
            s.append(f"<line x1='{X(lo):.1f}' x2='{X(hi):.1f}' y1='{y + bh / 2}' y2='{y + bh / 2}' stroke='{INK}' stroke-width='1.5'/>")
        s.append(f"<text x='{X(max(v, hi)) + 6:.1f}' y='{y + bh / 2 + 4}' font-size='12' fill='{INK}'>{v * 100:.1f}%</text>")
    if note:
        s.append(f"<text x='16' y='{h - 2}' font-size='10' fill='{MUTED}'>{note}</text>")
    s.append("</svg>")
    open(path, "w").write("\n".join(s))


def vbar_digits(path, title, vals, w=640, h=300):
    L, R, T, B = 56, 16, 48, 44
    pw, ph = w - L - R, h - T - B
    slot = pw / 10
    bw = slot * 0.62
    Y = lambda v: T + ph * (1 - v)
    s = svg_open(w, h, title)
    for i in range(6):
        v = i / 5
        s.append(f"<line x1='{L}' x2='{L + pw}' y1='{Y(v):.1f}' y2='{Y(v):.1f}' stroke='{GRID}' stroke-width='1'/>")
        s.append(f"<text x='{L - 8}' y='{Y(v) + 4:.1f}' font-size='11' fill='{INK2}' text-anchor='end'>{int(v * 100)}%</text>")
    for d, v in enumerate(vals):
        x = L + d * slot + (slot - bw) / 2
        top = Y(v)
        hh = max(0.0, T + ph - top)
        s.append(f"<path d='M{x:.1f},{T + ph} v{-max(0, hh - 4):.1f} a4,4 0 0 1 4,-4 h{bw - 8:.1f} a4,4 0 0 1 4,4 v{max(0, hh - 4):.1f} z' fill='{BLUE}'/>")
        s.append(f"<text x='{x + bw / 2:.1f}' y='{top - 6:.1f}' font-size='11' fill='{INK}' text-anchor='middle'>{v * 100:.0f}</text>")
        s.append(f"<text x='{x + bw / 2:.1f}' y='{T + ph + 18}' font-size='13' fill='{INK}' text-anchor='middle'>{d}</text>")
    s.append(f"<text x='{L + pw / 2}' y='{h - 6}' font-size='12' fill='{INK2}' text-anchor='middle'>true digit</text>")
    s.append("</svg>")
    open(path, "w").write("\n".join(s))


def confusion_svg(path, title, conf, w=520):
    L, T, cell = 70, 70, 38
    h = T + 10 * cell + 40
    s = svg_open(w, h, title)
    s.append(f"<text x='{L + 5 * cell}' y='{T - 28}' font-size='12' fill='{INK2}' text-anchor='middle'>predicted</text>")
    s.append(f"<text x='20' y='{T + 5 * cell}' font-size='12' fill='{INK2}' text-anchor='middle' transform='rotate(-90 20 {T + 5 * cell})'>true digit</text>")
    for d in range(10):
        s.append(f"<text x='{L + d * cell + cell / 2}' y='{T - 8}' font-size='12' fill='{INK}' text-anchor='middle'>{d}</text>")
        s.append(f"<text x='{L - 10}' y='{T + d * cell + cell / 2 + 4}' font-size='12' fill='{INK}' text-anchor='end'>{d}</text>")
    for y in range(10):
        n = max(1, sum(conf[y]))
        for g in range(10):
            p = conf[y][g] / n
            step = min(len(SEQ) - 1, int(round(p * (len(SEQ) - 1) / 1.0))) if p > 0 else 0
            if 0 < p and step == 0:
                step = 1
            fill = SEQ[step]
            s.append(f"<rect x='{L + g * cell + 1}' y='{T + y * cell + 1}' width='{cell - 2}' height='{cell - 2}' rx='3' fill='{fill}'/>")
            if p >= 0.01:
                ink = "#ffffff" if step >= 4 else INK
                s.append(f"<text x='{L + g * cell + cell / 2}' y='{T + y * cell + cell / 2 + 4}' font-size='10' fill='{ink}' text-anchor='middle'>{p * 100:.0f}</text>")
    s.append(f"<text x='16' y='{h - 12}' font-size='10' fill='{MUTED}'>% of each true digit (rows sum to 100); cells under 1% shaded but not labelled</text>")
    s.append("</svg>")
    open(path, "w").write("\n".join(s))


def main():
    os.makedirs(FIG, exist_ok=True)
    names = ["stdp_100", "stdp_100_seed2", "stdp_400", "control_100"]
    runs = {n: load_run(n) for n in names}
    runs = {n: r for n, r in runs.items() if r}
    if "stdp_100" not in runs:
        sys.exit("results/final/stdp_100.json missing - run the experiment first")
    main_run = runs["stdp_100"]

    summary = {}
    for n, r in runs.items():
        summary[n] = {k: r[k] for k in ("neurons", "learning", "train_images", "acc_all", "ci_all", "acc_heldout",
                                        "ci_heldout", "n_heldout", "nearest_centroid_accuracy", "per_digit",
                                        "neurons_per_digit", "silent_neurons", "mean_test_spikes", "test_no_spikes",
                                        "train_seconds", "total_seconds")}
        summary[n]["top_confusions"] = [{"true": y, "predicted": g, "count": c} for c, y, g in r["top_confusions"]]
    json.dump(summary, open(os.path.join(FINAL, "summary.json"), "w"), indent=2)

    # Accuracy comparison (held-out 9000 images).
    label = {"stdp_400": "SNN, 400 neurons (STDP)", "stdp_100": "SNN, 100 neurons (STDP)",
             "stdp_100_seed2": "SNN, 100 neurons, seed 2", "control_100": "SNN, 100 neurons, no learning"}
    rows = []
    for n in ["stdp_400", "stdp_100", "stdp_100_seed2", "control_100"]:
        if n in runs:
            r = runs[n]
            rows.append((label[n], r["acc_heldout"], r["ci_heldout"][0], r["ci_heldout"][1], n.startswith("stdp")))
    nc = main_run["nearest_centroid_accuracy"]
    rows.append(("Nearest class mean, raw pixels", nc, nc, nc, False))
    rows.append(("Chance", 0.1, 0.1, 0.1, False))
    hbar_chart(os.path.join(FIG, "accuracy.svg"), "Test accuracy (9,000 held-out test images)", rows,
               note="Lines: 95% confidence interval. Nearest class mean is on all 10,000 test images.")

    vbar_digits(os.path.join(FIG, "per_digit.svg"), "Accuracy per digit (SNN, 100 neurons)", main_run["per_digit"])
    confusion_svg(os.path.join(FIG, "confusion.svg"), "Confusion matrix (SNN, 100 neurons)", main_run["confusion"])

    curve = main_run["curve"]
    if curve:
        xs = [c["seen"] for c in curve]
        line_chart(os.path.join(FIG, "curve_spikes.svg"), "Output spikes per image during training",
                   xs, [c["mean_spikes"] for c in curve], "training images seen", "spikes / image")
        line_chart(os.path.join(FIG, "curve_neurons.svg"), "Neurons that respond to an image during training",
                   xs, [c["mean_neurons_per_image"] for c in curve], "training images seen", "neurons / image")

    for n, r in runs.items():
        print(f"{n:16s} all {r['acc_all'] * 100:5.2f}%  held-out {r['acc_heldout'] * 100:5.2f}% "
              f"[{r['ci_heldout'][0] * 100:.1f}, {r['ci_heldout'][1] * 100:.1f}]  "
              f"centroid {r['nearest_centroid_accuracy'] * 100:.2f}%  no-spike {r['test_no_spikes']}")


if __name__ == "__main__":
    main()
