"""
01_prepare_splits.py  -  Steps 1-3: label mapping, HAB lookup, splits.

Reads audit_out/index.csv (from 00_audit_data.py); no images are opened.
Writes to C:\\APS360\\outputs\\prep  (paths in config.py):
  label_mapping.csv   final 2016-19 <-> 2021 class names, class_id, HAB flag
  train.csv / val.csv stratified 85/15 split of 2016-19 (train capped at 1,500/class)
  test.csv            2021 images in the 38 retained classes
  ood.csv             2021 labelled images outside the retained classes + unclassifiable
  split_counts.csv    per-class counts per split (the "n" for per-class metrics)
  size_shift.csv      per-class median image size, 2016-19 vs 2021
  prep_summary.txt    headline numbers + checks

Usage:  python 01_prepare_splits.py          (standard library only)
"""
import csv
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import config as C

FIELDS = ["path", "dataset", "class", "class_id", "hab", "width", "height", "timestamp"]


def load_index(path):
    if not path.is_file():
        sys.exit(f"[error] index not found: {path}  (run 00_audit_data.py first)")
    with open(path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    bad = [r for r in rows if r["error"]]
    if bad:
        print(f"[warn] skipping {len(bad)} unreadable files listed in the audit")
    rows = [r for r in rows if not r["error"]]
    for r in rows:
        root = C.DATA_1619 if r["dataset"] == "2016_2019" else C.DATA_2021
        r["path"] = str(root / Path(r["rel_path"]))
        r["width"], r["height"] = int(r["width"]), int(r["height"])
    return rows


def write_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main():
    C.PREP_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_index(C.AUDIT_INDEX)
    by = defaultdict(lambda: defaultdict(list))          # by[dataset][class] -> rows
    for r in rows:
        by[r["dataset"]][r["class"]].append(r)
    c1619, c2021 = by["2016_2019"], by["2021"]

    # ---- Step 1-2: retained classes, ids, HAB flag, label mapping
    retained = sorted(c for c, v in c1619.items()
                      if len(v) >= C.MIN_IMAGES and c not in C.DROP_CLASSES)
    class_id = {c: i for i, c in enumerate(retained)}
    missing_hab = C.HAB_CLASSES - set(retained)
    if missing_hab:
        sys.exit(f"[error] HAB classes not among retained classes: {missing_hab}")

    mapping = []
    for c in sorted(set(c1619) | set(c2021) - {C.UNCLASSIFIABLE}):
        in1, in2 = c in c1619, c in c2021
        if in1 and in2:
            note = "exact folder-name match"
        elif in1:
            note = "absent from 2021"
        else:
            note = "only in 2021"
        if c in C.DROP_CLASSES:
            note += "; dropped (non-plankton)"
        elif in1 and c not in class_id:
            note += f"; dropped (<{C.MIN_IMAGES} images in 2016-19)"
        mapping.append({
            "name_2016_2019": c if in1 else "", "name_2021": c if in2 else "",
            "class_id": class_id.get(c, ""), "hab": int(c in C.HAB_CLASSES) if c in class_id else "",
            "n_2016_2019": len(c1619.get(c, [])), "n_2021": len(c2021.get(c, [])), "note": note,
        })
    write_csv(C.PREP_DIR / "label_mapping.csv", mapping, list(mapping[0].keys()))

    def tag(r):
        r["class_id"] = class_id.get(r["class"], -1)
        r["hab"] = int(r["class"] in C.HAB_CLASSES)
        return r

    # ---- Step 3: stratified split, then cap the training split only
    rng = random.Random(C.SEED)
    train, val, counts = [], [], []
    for c in retained:
        items = sorted(c1619[c], key=lambda r: r["rel_path"])   # fixed order -> reproducible
        rng.shuffle(items)
        n_val = round(C.VAL_FRAC * len(items))
        v, t = items[:n_val], items[n_val:]
        t_cap = t[:C.TRAIN_CAP] if len(t) > C.TRAIN_CAP else t    # already shuffled -> random cap
        train += [tag(r) for r in t_cap]
        val += [tag(r) for r in v]
        counts.append({"class": c, "class_id": class_id[c], "hab": int(c in C.HAB_CLASSES),
                       "n_2016_2019": len(items), "n_train_before_cap": len(t),
                       "n_train": len(t_cap), "n_val": len(v),
                       "n_test_2021": len(c2021.get(c, []))})

    test = [tag(r) for c in retained for r in c2021.get(c, [])]
    ood = [tag(r) for c, v in c2021.items() if c not in class_id for r in v]
    for name, split in (("train", train), ("val", val), ("test", test), ("ood", ood)):
        write_csv(C.PREP_DIR / f"{name}.csv", split, FIELDS)
    write_csv(C.PREP_DIR / "split_counts.csv", counts, list(counts[0].keys()))

    # ---- size shift between years (does "size" mean the same thing in 2021?)
    shift = []
    for c in retained:
        a, b = c1619[c], c2021.get(c, [])
        row = {"class": c, "hab": int(c in C.HAB_CLASSES), "n_2016_2019": len(a), "n_2021": len(b)}
        for m in ("width", "height"):
            ma = statistics.median(r[m] for r in a)
            mb = statistics.median(r[m] for r in b) if b else ""
            row[f"median_{m}_1619"], row[f"median_{m}_2021"] = ma, mb
            row[f"ratio_{m}"] = round(mb / ma, 2) if b else ""
        flag = [m for m in ("width", "height")
                if row[f"ratio_{m}"] != "" and len(b) >= 20
                and not 0.8 <= row[f"ratio_{m}"] <= 1.25]
        row["flag"] = ",".join(flag)
        shift.append(row)
    write_csv(C.PREP_DIR / "size_shift.csv", shift, list(shift[0].keys()))

    # ---- fit of images on the 96x256 canvas (no downscaling needed?)
    fits = sum(r["height"] <= C.CANVAS_H and r["width"] <= C.CANVAS_W for r in train)

    # ---- summary
    n_unc = len(c2021.get(C.UNCLASSIFIABLE, []))
    flagged = [s for s in shift if s["flag"]]
    hab_tr = sum(r["hab"] for r in train)
    hab_te = sum(r["hab"] for r in test)
    lines = [
        "PREP SUMMARY (steps 1-3)", "=" * 60,
        f"Retained classes: {len(retained)}  (HAB: {len(C.HAB_CLASSES)})",
        f"Train (capped):   {len(train):>7,}   [proposal: 27,033]  HAB images: {hab_tr:,}",
        f"Validation:       {len(val):>7,}   HAB images: {sum(r['hab'] for r in val):,}",
        f"Test 2021:        {len(test):>7,}   [proposal: 56,890]  HAB images: {hab_te:,}",
        f"OOD 2021:         {len(ood):>7,}   (labelled other: {len(ood) - n_unc:,}, unclassifiable: {n_unc:,})",
        "",
        "Retained classes with < 20 test images (report metrics as unreliable / n/a):",
        "  " + ", ".join(f"{s['class']} ({s['n_test_2021']})" for s in counts if s["n_test_2021"] < 20),
        "",
        f"Train images fitting {C.CANVAS_H}x{C.CANVAS_W} without downscaling: {fits / len(train):.0%}",
        "",
        f"Classes whose median size changed >25% between years (n_2021 >= 20): {len(flagged)}",
    ]
    for s in flagged:
        lines.append(f"  {s['class']:<36} width x{s['ratio_width']}  height x{s['ratio_height']}")
    lines += ["", f"Outputs: {C.PREP_DIR}"]
    text = "\n".join(lines)
    (C.PREP_DIR / "prep_summary.txt").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
