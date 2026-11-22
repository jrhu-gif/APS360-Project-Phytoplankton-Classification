"""Appendix Figure A2: HAB-Net (selected run) examples on the 2021 test set, larger serif fonts.

Run on the PC:  python plot_figA2.py
Uses the same picks as 05_compare.py (same seed, same groups), so the 16 images do not change.
Writes ..\\figures\\habnet_examples.png (overwrites the old version).
"""
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ---- same font as the report text (Times); falls back to Times look-alikes
for f in ["/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyretermes-regular.otf",
          "/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyretermes-bold.otf"]:
    if Path(f).exists():
        font_manager.fontManager.addfont(f)
plt.rcParams.update({"font.family": "serif",
                     "font.serif": ["Times New Roman", "TeX Gyre Termes", "Liberation Serif", "DejaVu Serif"],
                     "mathtext.fontset": "stix"})

GROUPS = ["HAB, correctly flagged", "HAB, missed", "Non-HAB, false HAB alert",
          "Labelled uncertain (confidence < τ)"]


def draw(rows, out):
    """rows: list of (group title, n, [(img, true, pred, conf), ...4])"""
    fig, axes = plt.subplots(len(rows), 4, figsize=(11, 7.6))
    fig.subplots_adjust(left=0.005, right=0.995, top=0.905, bottom=0.005, wspace=0.06, hspace=0.95)
    for g, (title, n, items) in enumerate(rows):
        for j in range(4):
            ax = axes[g, j]
            ax.axis("off")
            if j < len(items):
                img, t, p, c = items[j]
                ax.imshow(img, cmap="gray", vmin=0, vmax=255)
                ax.set_title(f"true: {t.replace('_', ' ')}\npred: {p.replace('_', ' ')} ({c:.2f})",
                             fontsize=11.5, pad=3, linespacing=1.15)
        # row header above the row, left-aligned
        axes[g, 0].annotate(f"{title}  (n = {n:,})", xy=(0, 1), xycoords="axes fraction",
                            xytext=(0, 30), textcoords="offset points", ha="left", va="bottom",
                            fontsize=14, fontweight="bold")
    fig.savefig(out, dpi=200)
    plt.close(fig)


def from_cache():
    """Same selection as 05_compare.py (runs on the PC)."""
    import config as C
    sel = (C.OUT_ROOT / "selected_model.txt").read_text(encoding="utf-8").strip()
    run = C.RUNS_DIR / sel.split("(")[-1].rstrip(")")
    res = json.loads((run / "results.json").read_text())
    with open(C.PREP_DIR / "label_mapping.csv", encoding="utf-8") as fh:
        lm = sorted((r for r in csv.DictReader(fh) if r["class_id"]), key=lambda r: int(r["class_id"]))
    cls = [r["name_2016_2019"] for r in lm]
    is_hab = np.array([int(r["hab"]) for r in lm], bool)
    p = np.load(run / "test_pred.npz")
    y, pred, conf = p["y"], p["pred"], p["conf"]
    tau = res["tau"]
    imgs = np.load(C.CACHE_DIR / "test_images.npy", mmap_mode="r")
    rng = np.random.default_rng(C.SEED)
    masks = [is_hab[y] & (pred == y), is_hab[y] & ~is_hab[pred],
             ~is_hab[y] & is_hab[pred] & (conf >= tau), conf < tau]
    rows = []
    for title, mask in zip(GROUPS, masks):
        idx = np.flatnonzero(mask)
        pick = rng.choice(idx, size=min(4, len(idx)), replace=False) if len(idx) else []
        rows.append((title, int(mask.sum()),
                     [(np.asarray(imgs[i]), cls[y[i]], cls[pred[i]], float(conf[i])) for i in pick]))
    return rows, C.FIG_DIR / "habnet_examples.png"


if __name__ == "__main__":
    rows, out = from_cache()
    draw(rows, out)
    print("saved", out)
