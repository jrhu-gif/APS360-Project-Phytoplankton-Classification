"""
03_random_forest.py  -  Baseline B1: random forest on hand-crafted features.

Inputs : cache/{split}_feats.npy + prep/{split}.csv  (from steps 1-5)
Model  : RandomForest, class_weight="balanced", seed 42; n_trees chosen from
         {100, 200, 350, 500} by validation macro-F1 (proposal).
Uncertain: tau = confidence giving COVERAGE (95%) of validation predictions.
Writes :
  outputs/rf/rf_results.json       all headline numbers
  outputs/rf/per_class_test.csv    per-class P/R/F1 with n_train / n_test
  outputs/rf/test_pred.npz         test predictions (for comparing with HAB-Net)
  figures/rf_confusion_test.png    normalized confusion matrix, 2021 test
  figures/rf_feature_importance.png
Usage:  python 03_random_forest.py
"""
import csv
import json
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)

import config as C

DROP = {"scale", "seg_fail"}          # canvas artifact / constant


def load(split, names, keep):
    X = np.load(C.CACHE_DIR / f"{split}_feats.npy")[:, keep]
    with open(C.PREP_DIR / f"{split}.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    y = np.array([int(r["class_id"]) for r in rows])
    return X, y


def hab_metrics(y_true_hab, pred_hab):
    tp = int((y_true_hab & pred_hab).sum())
    return {"hab_recall": tp / max(1, y_true_hab.sum()),
            "hab_precision": tp / max(1, pred_hab.sum()),
            "n_hab": int(y_true_hab.sum())}


def main():
    t0 = time.time()
    C.RF_DIR.mkdir(parents=True, exist_ok=True)
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    names = json.loads((C.CACHE_DIR / "feature_names.json").read_text())
    keep = [i for i, n in enumerate(names) if n not in DROP]
    feat_names = [names[i] for i in keep]

    with open(C.PREP_DIR / "label_mapping.csv", encoding="utf-8") as fh:
        lm = [r for r in csv.DictReader(fh) if r["class_id"] != ""]
    lm.sort(key=lambda r: int(r["class_id"]))
    classes = [r["name_2016_2019"] for r in lm]
    is_hab = np.array([int(r["hab"]) for r in lm], bool)
    K = len(classes)

    Xtr, ytr = load("train", names, keep)
    Xva, yva = load("val", names, keep)
    Xte, yte = load("test", names, keep)
    Xood = np.load(C.CACHE_DIR / "ood_feats.npy")[:, keep]
    print(f"Features: {len(feat_names)} | train {len(ytr):,} val {len(yva):,} "
          f"test {len(yte):,} ood {len(Xood):,}")

    # ---- choose number of trees on validation macro-F1
    grid = {}
    best = None
    for n in (100, 200, 350, 500):
        rf = RandomForestClassifier(n_estimators=n, class_weight="balanced",
                                    n_jobs=-1, random_state=C.SEED)
        rf.fit(Xtr, ytr)
        f1 = f1_score(yva, rf.predict(Xva), average="macro")
        grid[n] = round(f1, 4)
        print(f"  n_trees={n:<4} val macro-F1 = {f1:.4f}")
        if best is None or f1 > best[1]:
            best = (n, f1, rf)
    n_best, _, rf = best

    # ---- uncertain threshold from validation coverage
    pva = rf.predict_proba(Xva)
    tau = float(np.quantile(pva.max(1), 1 - C.COVERAGE))

    def evaluate(X, y):
        p = rf.predict_proba(X)
        pred, conf = p.argmax(1), p.max(1)
        acc_mask = conf >= tau
        out = {"accuracy": accuracy_score(y, pred),
               "macro_f1": f1_score(y, pred, average="macro",
                                    labels=np.unique(y)),     # classes present
               "weighted_f1": f1_score(y, pred, average="weighted"),
               "coverage": float(acc_mask.mean()),
               "accuracy_confident": accuracy_score(y[acc_mask], pred[acc_mask])}
        out.update(hab_metrics(is_hab[y], is_hab[pred]))                       # no rejection
        r = hab_metrics(is_hab[y], is_hab[pred] & acc_mask)                    # alerts need conf >= tau
        out.update({"hab_recall_confident": r["hab_recall"],
                    "hab_precision_confident": r["hab_precision"]})
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}, pred, conf

    res_val, _, _ = evaluate(Xva, yva)
    res_test, pred, conf = evaluate(Xte, yte)
    pood = rf.predict_proba(Xood)
    ood_alert = float((is_hab[pood.argmax(1)] & (pood.max(1) >= tau)).mean())
    ood_uncertain = float((pood.max(1) < tau).mean())

    # ---- per-class table with n
    P, R, F, _ = precision_recall_fscore_support(yte, pred, labels=range(K), zero_division=0)
    n_train = np.bincount(ytr, minlength=K)
    n_test = np.bincount(yte, minlength=K)
    with open(C.RF_DIR / "per_class_test.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["class", "hab", "n_train", "n_test", "precision", "recall", "f1"])
        for k in np.argsort(-n_test):
            na = n_test[k] == 0
            w.writerow([classes[k], int(is_hab[k]), n_train[k], n_test[k],
                        "n/a" if na else f"{P[k]:.3f}", "n/a" if na else f"{R[k]:.3f}",
                        "n/a" if na else f"{F[k]:.3f}"])
    np.savez_compressed(C.RF_DIR / "test_pred.npz", y=yte, pred=pred, conf=conf)

    results = {"model": "RandomForest(class_weight=balanced)", "n_features": len(feat_names),
               "features": feat_names, "grid_val_macro_f1": grid, "n_trees": n_best,
               "tau": round(tau, 4), "coverage_target": C.COVERAGE,
               "val": res_val, "test_2021": res_test,
               "ood_hab_alert_rate": round(ood_alert, 4),
               "ood_uncertain_rate": round(ood_uncertain, 4),
               "runtime_s": round(time.time() - t0, 1)}
    (C.RF_DIR / "rf_results.json").write_text(json.dumps(results, indent=1))

    # ---- figures
    cm = confusion_matrix(yte, pred, labels=range(K)).astype(float)
    cm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    order = np.r_[np.flatnonzero(is_hab), np.flatnonzero(~is_hab)]       # HAB classes first
    fig, ax = plt.subplots(figsize=(11, 10))
    im = ax.imshow(cm[np.ix_(order, order)], cmap="Blues", vmin=0, vmax=1)
    labels = [f"{classes[k]} ({n_test[k]})" for k in order]
    ax.set_xticks(range(K), [classes[k] for k in order], rotation=90, fontsize=7)
    ax.set_yticks(range(K), labels, fontsize=7)
    nh = is_hab.sum()
    ax.axhline(nh - 0.5, color="#d55e00", lw=1)
    ax.axvline(nh - 0.5, color="#d55e00", lw=1)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class (n test images)")
    ax.set_title(f"Random forest, 2021 test: recall-normalized confusion "
                 f"(HAB classes above/left of orange line)", fontsize=10)
    fig.colorbar(im, fraction=0.03)
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "rf_confusion_test.png", dpi=200)
    plt.close(fig)

    imp = rf.feature_importances_
    top = np.argsort(imp)[::-1][:15][::-1]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.barh([feat_names[i] for i in top], imp[top], color="#4c72b0")
    ax.set_xlabel("Mean decrease in impurity")
    ax.set_title("Random forest: top 15 features", fontsize=10)
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "rf_feature_importance.png", dpi=200)
    plt.close(fig)

    # ---- summary
    t = res_test
    print(f"\nBest n_trees = {n_best} | tau = {tau:.3f} (keeps {C.COVERAGE:.0%} of val)")
    print(f"VAL  : acc {res_val['accuracy']:.3f}  macro-F1 {res_val['macro_f1']:.3f}  "
          f"HAB recall {res_val['hab_recall']:.3f}")
    print(f"TEST : acc {t['accuracy']:.3f}  macro-F1 {t['macro_f1']:.3f}  "
          f"weighted-F1 {t['weighted_f1']:.3f}")
    print(f"       HAB recall {t['hab_recall']:.3f}  precision {t['hab_precision']:.3f}  "
          f"(n HAB = {t['n_hab']:,})")
    print(f"       coverage {t['coverage']:.3f}  acc on confident {t['accuracy_confident']:.3f}")
    print(f"OOD  : false HAB alerts {ood_alert:.3%}  | labelled uncertain {ood_uncertain:.1%}")
    print(f"Done in {time.time() - t0:.0f}s -> {C.RF_DIR}, figures -> {C.FIG_DIR}")


if __name__ == "__main__":
    main()
