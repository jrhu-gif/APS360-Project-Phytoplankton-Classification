"""
04_train.py  -  Step 6: train and evaluate neural networks (HAB-Net "M", ResNet-18 "B2").

Data stay on the GPU (no DataLoader workers): cached uint8 images + morphology features,
class-balanced sampling, GPU augmentation (flips, +-10 deg rotation, 0.8-1.2 zoom, brightness),
train-only size jitter on morphology. AdamW + cosine LR, mixed precision, early stopping on
validation macro-F1. Evaluation matches 03_random_forest.py (same tau rule, same tables).

Loss (M):  L = CE(species) + lam * BCE(HAB, pos_weight = w_hab)      (B2: CE only)
Writes per run to outputs/runs/<run>/: history.csv, best.pt, results.json,
per_class_test.csv, test_pred.npz;  figure: figures/learning_curves_<run>.png
Usage:  python 04_train.py --runs M B2 M80    (add --smoke for a 1-epoch test on a subset)
"""
import argparse
import csv
import json
import math
import shutil
import time

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

import config as C
from models import build_model, n_params

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP = DEV.type == "cuda"
LINEAR_SIZE = {"img_w", "img_h", "perimeter", "eq_diameter"}   # scale x s (area scales x s^2)


# ------------------------------------------------------------------ data
def labels(split):
    with open(C.PREP_DIR / f"{split}.csv", encoding="utf-8") as fh:
        return np.array([int(r["class_id"]) for r in csv.DictReader(fh)])


class Morph:
    """Raw morphology -> (size jitter) -> log1p for size features -> z-score (train stats)."""
    def __init__(self, names, stats):
        st = stats["train_stats"]["morph"]
        ks = C.MORPH_FEATURES
        self.idx = [names.index(k) for k in ks]
        t = lambda v: torch.tensor(v, device=DEV)
        self.mean, self.std = t([st[k]["mean"] for k in ks]), t([st[k]["std"] for k in ks])
        self.log = t([st[k]["log"] for k in ks])
        self.lin, self.area = t([k in LINEAR_SIZE for k in ks]), t([k == "area" for k in ks])

    def raw(self, feats):
        return torch.tensor(feats[:, self.idx], dtype=torch.float32, device=DEV)

    def __call__(self, x, jitter=False):
        if jitter:
            s = torch.empty(len(x), 1, device=DEV).uniform_(*C.SIZE_JITTER)
            x = torch.where(self.lin, x * s, x)
            x = torch.where(self.area, x * s * s, x)
        x = torch.where(self.log, torch.log1p(x), x)
        return (x - self.mean) / self.std


def augment(x):
    """x: (B,1,H,W) in [0,1] on GPU."""
    B, _, H, W = x.shape
    r = lambda lo, hi: torch.empty(B, device=DEV).uniform_(lo, hi)
    x = torch.where((torch.rand(B, device=DEV) < 0.5)[:, None, None, None], x.flip(3), x)
    x = torch.where((torch.rand(B, device=DEV) < 0.5)[:, None, None, None], x.flip(2), x)
    a, z = r(-10, 10) * math.pi / 180, r(0.8, 1.2)
    c, s = torch.cos(a) / z, torch.sin(a) / z
    zero = torch.zeros_like(c)
    theta = torch.stack([torch.stack([c, -s * H / W, zero], 1),        # rotation in pixel space
                         torch.stack([s * W / H, c, zero], 1)], 1)     # for a non-square image
    grid = F.affine_grid(theta, (B, 1, H, W), align_corners=False)
    x = F.grid_sample(x, grid, padding_mode="border", align_corners=False)
    return (x * r(0.9, 1.1)[:, None, None, None]).clamp(0, 1)


@torch.no_grad()
def predict(model, imgs, mraw, morph, bs=256):
    """imgs: uint8 GPU tensor or numpy memmap. Returns softmax probs, HAB-head probs (or None)."""
    model.eval()
    P, Hp = [], []
    for i in range(0, len(mraw), bs):
        xb = imgs[i:i + bs]
        if isinstance(xb, np.ndarray):
            xb = torch.from_numpy(np.array(xb)).to(DEV, non_blocking=True)
        xb = xb.unsqueeze(1).float() / 255
        with torch.autocast(DEV.type, dtype=torch.float16, enabled=AMP):
            logits, hl = model(xb, morph(mraw[i:i + bs]))
        P.append(torch.softmax(logits.float(), 1).cpu())
        if hl is not None:
            Hp.append(torch.sigmoid(hl.float()).cpu())
    return torch.cat(P).numpy(), (torch.cat(Hp).numpy() if Hp else None)


# ------------------------------------------------------------------ metrics (same rules as the RF)
def metrics(y, P, Hp, is_hab, tau):
    pred, conf = P.argmax(1), P.max(1)
    ok = conf >= tau
    yh, ph = is_hab[y], is_hab[pred]
    rec = lambda t, p: float((t & p).sum() / max(1, t.sum()))
    pre = lambda t, p: float((t & p).sum() / max(1, p.sum()))
    out = {"accuracy": accuracy_score(y, pred),
           "macro_f1": f1_score(y, pred, average="macro", labels=np.unique(y)),
           "weighted_f1": f1_score(y, pred, average="weighted"),
           "hab_recall": rec(yh, ph), "hab_precision": pre(yh, ph), "n_hab": int(yh.sum()),
           "coverage": float(ok.mean()), "accuracy_confident": accuracy_score(y[ok], pred[ok]),
           "hab_recall_confident": rec(yh, ph & ok), "hab_precision_confident": pre(yh, ph & ok)}
    if Hp is not None:                                   # HAB head (M only), p >= 0.5
        out.update({"habhead_recall": rec(yh, Hp >= 0.5), "habhead_precision": pre(yh, Hp >= 0.5)})
    return {k: round(v, 4) if isinstance(v, float) else v for k, v in out.items()}


# ------------------------------------------------------------------ one run
def run(name, smoke):
    cfg = C.TRAIN[name]
    torch.manual_seed(C.SEED)
    np.random.seed(C.SEED)
    out = C.RUNS_DIR / (f"smoke_{name}" if smoke else name)
    out.mkdir(parents=True, exist_ok=True)
    stats = json.loads((C.CACHE_DIR / "cache_stats.json").read_text())
    names = stats["feature_names"]
    with open(C.PREP_DIR / "label_mapping.csv", encoding="utf-8") as fh:
        lm = sorted((r for r in csv.DictReader(fh) if r["class_id"]), key=lambda r: int(r["class_id"]))
    classes = [r["name_2016_2019"] for r in lm]
    is_hab = np.array([int(r["hab"]) for r in lm], bool)
    K = len(classes)
    morph = Morph(names, stats)
    lim = (lambda n: min(n, 2048)) if smoke else (lambda n: n)

    def split(s, gpu):
        y = labels(s)
        n = lim(len(y))
        im = np.load(C.CACHE_DIR / f"{s}_images.npy", mmap_mode="r")[:n]
        mr = morph.raw(np.load(C.CACHE_DIR / f"{s}_feats.npy")[:n])
        return (torch.from_numpy(np.array(im)).to(DEV) if gpu else im), mr, y[:n]

    Xtr, Mtr, ytr = split("train", True)
    Xva, Mva, yva = split("val", True)
    ytr_t = torch.tensor(ytr, device=DEV)
    hab_t = torch.tensor(is_hab, device=DEV)

    ts = stats["train_stats"]
    model = build_model(cfg, K, len(C.MORPH_FEATURES), ts["pixel_mean"], ts["pixel_std"]).to(DEV)
    epochs = 1 if smoke else cfg.get("epochs", C.EPOCHS)
    patience = cfg.get("patience", C.PATIENCE)
    steps = math.ceil(len(ytr) / C.BATCH)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=C.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * steps)
    scaler = torch.amp.GradScaler(DEV.type, enabled=AMP)
    torch.backends.cudnn.benchmark = True
    w = 1.0 / np.bincount(ytr, minlength=K)[ytr]                       # class-balanced sampling
    w_t = torch.tensor(w / w.sum(), device=DEV, dtype=torch.float32)
    pos_w = torch.tensor(cfg.get("w_hab", 1.0), device=DEV)
    print(f"\n=== {name}: {cfg['model']} | {n_params(model):,} params | device {DEV} | "
          f"train {len(ytr):,} val {len(yva):,} | {epochs} epochs x {steps} steps")

    hist, best, bad, t0 = [], -1.0, 0, time.time()
    for ep in range(1, epochs + 1):
        model.train()
        idx = torch.multinomial(w_t, len(ytr), replacement=True)
        tl, tc, tn = 0.0, 0, 0
        for b in range(steps):
            j = idx[b * C.BATCH:(b + 1) * C.BATCH]
            x = augment(Xtr[j].unsqueeze(1).float() / 255)   # no channels_last: 4.8x slower on this GPU
            y = ytr_t[j]
            with torch.autocast(DEV.type, dtype=torch.float16, enabled=AMP):
                logits, hl = model(x, morph(Mtr[j], jitter=True))
                loss = F.cross_entropy(logits.float(), y)
                if hl is not None:
                    loss = loss + cfg["lam"] * F.binary_cross_entropy_with_logits(
                        hl.float(), hab_t[y].float(), pos_weight=pos_w)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tl += loss.item() * len(j)
            tc += (logits.argmax(1) == y).sum().item()
            tn += len(j)
        Pva, _ = predict(model, Xva, Mva, morph)
        vloss = float(F.nll_loss(torch.log(torch.from_numpy(Pva).clamp_min(1e-8)), torch.from_numpy(yva)))
        vpred = Pva.argmax(1)
        gb = torch.cuda.max_memory_allocated() / 1e9 if AMP else 0.0
        row = {"epoch": ep, "train_loss": tl / tn, "train_acc": tc / tn, "val_loss": vloss,
               "val_acc": accuracy_score(yva, vpred),
               "val_macro_f1": f1_score(yva, vpred, average="macro", labels=np.unique(yva)),
               "val_hab_recall": float((is_hab[yva] & is_hab[vpred]).sum() / max(1, is_hab[yva].sum())),
               "lr": sched.get_last_lr()[0], "time_s": time.time() - t0, "gpu_peak_gb": gb}
        hist.append(row)
        print(f"  ep {ep:2d} | train loss {row['train_loss']:.3f} acc {row['train_acc']:.3f} | "
              f"val loss {vloss:.3f} acc {row['val_acc']:.3f} macro-F1 {row['val_macro_f1']:.3f} "
              f"HAB rec {row['val_hab_recall']:.3f} | {row['time_s']:.0f}s | GPU peak {gb:.1f} GB")
        if row["val_macro_f1"] > best:
            best, bad, best_ep = row["val_macro_f1"], 0, ep
            torch.save(model.state_dict(), out / "best.pt")
        else:
            bad += 1
            if bad >= patience:
                print(f"  early stop (no val macro-F1 gain for {patience} epochs)")
                break
    train_time = time.time() - t0
    with open(out / "history.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(hist[0].keys()))
        wr.writeheader()
        wr.writerows(hist)

    # ---- evaluate best checkpoint
    model.load_state_dict(torch.load(out / "best.pt", map_location=DEV))
    del Xtr
    Pva, Hva = predict(model, Xva, Mva, morph)
    tau = float(np.quantile(Pva.max(1), 1 - C.COVERAGE))
    Xte, Mte, yte = split("test", False)
    Pte, Hte = predict(model, Xte, Mte, morph)
    Xood, Mood, _ = split("ood", False)
    Pood, Hood = predict(model, Xood, Mood, morph)
    ood_conf = Pood.max(1) >= tau
    res = {"run": name, "config": cfg, "params": n_params(model), "epochs_trained": len(hist),
           "best_epoch": best_ep, "train_time_s": round(train_time, 1), "tau": round(tau, 4),
           "coverage_target": C.COVERAGE,
           "val": metrics(yva, Pva, Hva, is_hab, tau), "test_2021": metrics(yte, Pte, Hte, is_hab, tau),
           "ood_hab_alert_rate": round(float((is_hab[Pood.argmax(1)] & ood_conf).mean()), 4),
           "ood_uncertain_rate": round(float((~ood_conf).mean()), 4)}
    if Hood is not None:
        res["ood_habhead_alert_rate"] = round(float(((Hood >= 0.5) & ood_conf).mean()), 4)
    (out / "results.json").write_text(json.dumps(res, indent=1))

    pred = Pte.argmax(1)
    Pc, Rc, Fc, _ = precision_recall_fscore_support(yte, pred, labels=range(K), zero_division=0)
    n_tr, n_te = np.bincount(ytr, minlength=K), np.bincount(yte, minlength=K)
    with open(out / "per_class_test.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["class", "hab", "n_train", "n_test", "precision", "recall", "f1"])
        for k in np.argsort(-n_te):
            na = n_te[k] == 0
            wr.writerow([classes[k], int(is_hab[k]), n_tr[k], n_te[k]] +
                        (["n/a"] * 3 if na else [f"{Pc[k]:.3f}", f"{Rc[k]:.3f}", f"{Fc[k]:.3f}"]))
    np.savez_compressed(out / "test_pred.npz", y=yte, pred=pred, conf=Pte.max(1),
                        habp=Hte if Hte is not None else np.zeros(0))

    # ---- learning curves
    ep = [h["epoch"] for h in hist]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(ep, [h["train_loss"] for h in hist], label="train")
    ax[0].plot(ep, [h["val_loss"] for h in hist], label="validation")
    ax[0].set(xlabel="epoch", ylabel="loss", title=f"{name}: loss")
    ax[1].plot(ep, [h["train_acc"] for h in hist], label="train accuracy")
    ax[1].plot(ep, [h["val_acc"] for h in hist], label="val accuracy")
    ax[1].plot(ep, [h["val_macro_f1"] for h in hist], label="val macro-F1")
    ax[1].axvline(best_ep, color="grey", ls="--", lw=0.8)
    ax[1].set(xlabel="epoch", ylim=(0, 1), title=f"{name}: accuracy / macro-F1")
    for a in ax:
        a.legend(fontsize=8)
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig((out if smoke else C.FIG_DIR) / f"learning_curves_{name}.png", dpi=200)
    plt.close(fig)

    t = res["test_2021"]
    print(f"  best epoch {best_ep} | tau {tau:.3f} | train time {train_time / 60:.1f} min")
    print(f"  VAL : acc {res['val']['accuracy']:.3f} macro-F1 {res['val']['macro_f1']:.3f} "
          f"HAB recall {res['val']['hab_recall']:.3f}")
    print(f"  TEST: acc {t['accuracy']:.3f} macro-F1 {t['macro_f1']:.3f} weighted-F1 {t['weighted_f1']:.3f} "
          f"HAB recall {t['hab_recall']:.3f} precision {t['hab_precision']:.3f}")
    if "habhead_recall" in t:
        print(f"        HAB head: recall {t['habhead_recall']:.3f} precision {t['habhead_precision']:.3f}")
    print(f"  OOD : false HAB alerts {res['ood_hab_alert_rate']:.3%} | uncertain {res['ood_uncertain_rate']:.1%}")
    if smoke:
        shutil.rmtree(out)                       # smoke test leaves no files behind
        print(f"  smoke test OK ({name}); temporary outputs removed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["M", "B2"], choices=list(C.TRAIN))
    ap.add_argument("--smoke", action="store_true", help="1 epoch on 2,048 images per split")
    args = ap.parse_args()
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    for r in args.runs:
        run(r, args.smoke)
