"""
05_compare.py  -  Compare RF (B1), ResNet-18 (B2) and HAB-Net (M) on the same 2021 test set.

Writes: outputs/comparison.csv, outputs/per_class_comparison.csv,
        figures/model_comparison.png, figures/habnet_examples.png
Usage:  python 05_compare.py
"""
import csv
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C

MODELS = {"RF (B1)": C.RF_DIR, "ResNet-18 (B2)": C.RUNS_DIR / "B2", "HAB-Net 40ep (M)": C.RUNS_DIR / "M",
          "HAB-Net 80ep wH=3 (M80)": C.RUNS_DIR / "M80", "HAB-Net 80ep wH=1 (M80w1)": C.RUNS_DIR / "M80w1"}
ROWS = [("val", "macro_f1", "Val macro-F1"), ("test_2021", "accuracy", "Test accuracy"),
        ("test_2021", "macro_f1", "Test macro-F1"), ("test_2021", "weighted_f1", "Test weighted-F1"),
        ("test_2021", "hab_recall", "Test HAB recall"), ("test_2021", "hab_precision", "Test HAB precision"),
        ("test_2021", "coverage", "Test coverage (conf >= tau)"),
        ("test_2021", "accuracy_confident", "Test accuracy when confident")]


def main():
    res = {m: json.loads((d / ("rf_results.json" if m.startswith("RF") else "results.json")).read_text())
           for m, d in MODELS.items() if d.exists()}
    names = list(res)

    # ---- headline table
    table = [[label] + [res[m][s][k] for m in names] for s, k, label in ROWS]
    table.append(["OOD false HAB alerts"] + [res[m]["ood_hab_alert_rate"] for m in names])
    table.append(["Parameters"] + [res[m].get("params", "-") for m in names])
    table.append(["Training time (min)"] + [round(res[m].get("train_time_s", res[m].get("runtime_s", 0)) / 60, 1)
                                            for m in names])
    with open(C.OUT_ROOT / "comparison.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["metric"] + names)
        w.writerows(table)
    print(f"{'metric':<30}" + "".join(f"{m.split('(')[-1][:-1]:>12}" for m in names))
    for r in table:
        print(f"{r[0]:<30}" + "".join(f"{v:>12,}" if isinstance(v, int) else f"{v:>12}" for v in r[1:]))

    # ---- per-class F1 side by side (with n)
    pc = {}
    for m in names:
        with open(MODELS[m] / "per_class_test.csv", encoding="utf-8") as fh:
            pc[m] = {r["class"]: r for r in csv.DictReader(fh)}
    base = pc[names[0]]
    with open(C.OUT_ROOT / "per_class_comparison.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["class", "hab", "n_train", "n_test"] + [f"F1 {m}" for m in names])
        for c, r in base.items():
            w.writerow([c, r["hab"], r["n_train"], r["n_test"]] + [pc[m][c]["f1"] for m in names])

    # ---- comparison figure
    keys = [("test_2021", "macro_f1", "Macro-F1"), ("test_2021", "hab_recall", "HAB recall"),
            ("test_2021", "hab_precision", "HAB precision"), ("test_2021", "accuracy", "Accuracy")]
    x = np.arange(len(keys))
    colors = ["#999999", "#4c72b0", "#e69f00", "#d55e00", "#009e73"]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for i, m in enumerate(names):
        v = [res[m][s][k] for s, k, _ in keys]
        wd = 0.8 / len(names)
        b = ax.bar(x + (i - (len(names) - 1) / 2) * wd, v, wd, label=m, color=colors[i % 5])
        ax.bar_label(b, fmt="%.2f", fontsize=7)
    ax.set_xticks(x, [k[2] for k in keys])
    ax.set(ylim=(0, 1.25), ylabel="2021 test set", title="Models on the independent 2021 test set (n = 56,890)")
    ax.legend(fontsize=6.5, ncol=min(len(names), 3), loc="upper center", frameon=False)
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "model_comparison.png", dpi=200)
    plt.close(fig)

    # ---- qualitative examples from HAB-Net on the 2021 test set
    # selected HAB-Net = highest VAL macro-F1 among runs with VAL HAB recall >= 0.95 (test set never used)
    cands = [m for m in names if m.startswith("HAB-Net") and res[m]["val"]["hab_recall"] >= 0.95]
    if not cands:
        return
    best = max(cands, key=lambda m: res[m]["val"]["macro_f1"])
    print(f"\nSelected HAB-Net (validation rule): {best}")
    (C.OUT_ROOT / "selected_model.txt").write_text(best + "\n", encoding="utf-8")
    with open(C.PREP_DIR / "label_mapping.csv", encoding="utf-8") as fh:
        lm = sorted((r for r in csv.DictReader(fh) if r["class_id"]), key=lambda r: int(r["class_id"]))
    cls = [r["name_2016_2019"] for r in lm]
    is_hab = np.array([int(r["hab"]) for r in lm], bool)
    p = np.load(MODELS[best] / "test_pred.npz")
    y, pred, conf = p["y"], p["pred"], p["conf"]
    tau = res[best]["tau"]
    imgs = np.load(C.CACHE_DIR / "test_images.npy", mmap_mode="r")
    rng = np.random.default_rng(C.SEED)
    groups = {"HAB, correctly flagged": is_hab[y] & (pred == y),
              "HAB, missed": is_hab[y] & ~is_hab[pred],
              "Non-HAB, false HAB alert": ~is_hab[y] & is_hab[pred] & (conf >= tau),
              "Labelled uncertain (conf < tau)": conf < tau}
    fig, axes = plt.subplots(len(groups), 4, figsize=(12, 1.25 * 1.6 * len(groups)))
    for g, (title, mask) in enumerate(groups.items()):
        idx = np.flatnonzero(mask)
        pick = rng.choice(idx, size=min(4, len(idx)), replace=False) if len(idx) else []
        for j in range(4):
            ax = axes[g, j]
            ax.axis("off")
            if j < len(pick):
                i = pick[j]
                ax.imshow(imgs[i], cmap="gray", vmin=0, vmax=255)
                ax.set_title(f"true: {cls[y[i]]}\npred: {cls[pred[i]]} ({conf[i]:.2f})", fontsize=7)
        axes[g, 0].text(-0.05, 0.5, f"{title}\n(n={mask.sum():,})", transform=axes[g, 0].transAxes,
                        ha="right", va="center", fontsize=9)
    fig.tight_layout()
    fig.subplots_adjust(left=0.17)
    fig.savefig(C.FIG_DIR / "habnet_examples.png", dpi=200)
    plt.close(fig)
    print(f"\nWrote comparison.csv, per_class_comparison.csv -> {C.OUT_ROOT}; figures -> {C.FIG_DIR}")


if __name__ == "__main__":
    main()
