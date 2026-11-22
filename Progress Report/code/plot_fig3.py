"""Figure 3 of the progress report: HAB-Net (M80) learning curves with large fonts.

Run on the PC:  python plot_fig3.py
Reads  C:\\APS360\\outputs\\runs\\M80\\history.csv
Writes ..\\figures\\learning_curves_M80.png (overwrites the old version)
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager

# same font as the report text (Times); falls back to Times look-alikes
for f in ["/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyretermes-regular.otf",
          "/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyretermes-bold.otf"]:
    if Path(f).exists():
        font_manager.fontManager.addfont(f)

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--hist", default=r"C:\APS360\outputs\runs\M80\history.csv")
ap.add_argument("--out", default=str(HERE.parent / "figures" / "learning_curves_M80.png"))
args = ap.parse_args()

h = pd.read_csv(args.hist)
best = int(h.loc[h["val_macro_f1"].idxmax(), "epoch"])  # epoch kept (71)

plt.rcParams.update({"font.family": "serif",
                     "font.serif": ["Times New Roman", "TeX Gyre Termes", "Liberation Serif", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 16, "axes.titlesize": 18, "axes.labelsize": 18,
                     "xtick.labelsize": 16, "ytick.labelsize": 16, "legend.fontsize": 15})
fig, (a, b) = plt.subplots(1, 2, figsize=(12, 3.8))

a.plot(h.epoch, h.train_loss, lw=2, label="train")
a.plot(h.epoch, h.val_loss, lw=2, label="validation")
a.set(title="Loss", xlabel="epoch", ylabel="loss")
a.legend(frameon=False)

b.plot(h.epoch, h.train_acc, lw=2, label="train accuracy")
b.plot(h.epoch, h.val_acc, lw=2, label="val. accuracy")
b.plot(h.epoch, h.val_macro_f1, lw=2, label="val. macro-F1")
b.axvline(best, color="grey", ls="--", lw=1.5)
b.text(best - 1.5, 0.05, f"kept: epoch {best}", ha="right", va="bottom", fontsize=15, color="dimgrey")
b.set(title="Accuracy / macro-F1", xlabel="epoch", ylim=(0, 1.0))
b.legend(frameon=False, loc="center", bbox_to_anchor=(0.45, 0.42))

for ax in (a, b):
    ax.grid(alpha=0.3)
    ax.set_xlim(0, h.epoch.max() + 1)
fig.tight_layout()
fig.savefig(args.out, dpi=200)
print("saved", args.out, "| kept epoch", best)
