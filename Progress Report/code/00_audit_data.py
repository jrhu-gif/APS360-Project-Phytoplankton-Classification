"""
00_audit_data.py  -  Step 0: read-only audit of the two SYKE IFCB datasets.

Nothing in the data folders is modified. The script:
  1. walks both datasets (one sub-folder per class, PNG files inside)
  2. reads every PNG header (width, height, bit depth, colour type) and
     hashes every file (MD5) to find duplicates / train-test leakage
  3. writes:
       index.csv               one row per image (all later steps build on this)
       class_counts.csv        images per class, both years, + "retained" flag
       size_summary.csv        image-size percentiles (decides 128 vs 160 px)
       problems.txt            unreadable / non-grayscale / duplicate files
       summary.txt             counts checked against the published numbers

Usage (Windows, from the folder holding this script):
    python 00_audit_data.py
    python 00_audit_data.py --data-1619 "D:\\path\\SYKE_2016_2019" --data-2021 "D:\\path\\SYKE_2021"
    python 00_audit_data.py --decode      # also fully decodes every PNG (needs Pillow, slower)

Standard library only (Pillow only for --decode).
"""

import argparse
import csv
import hashlib
import os
import re
import struct
import sys
import time
import zlib
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# ---------------------------------------------------------------- settings
DEFAULT_1619 = r"C:\APS360\data\SYKE_2016_2019"
DEFAULT_2021 = r"C:\APS360\data\SYKE_2021"
DEFAULT_OUT = str(Path(__file__).resolve().parent.parent / "audit_out")   # project folder

MIN_IMAGES = 100            # proposal: drop classes with < 100 images (2016-19)
DROP_CLASSES = {"Beads"}    # non-plankton
UNCLASSIFIABLE = "Unclassifiable"

# Published numbers (dataset pages / proposal) used as sanity checks
EXPECTED = {
    "n_1619": 63074, "classes_1619": 50,
    "n_2021_labelled": 57207, "n_2021_unclassifiable": 94028,
    "retained_classes": 38, "n_2021_retained": 56890,
}

PNG_SIG = b"\x89PNG\r\n\x1a\n"
COLOR_TYPES = {0: "gray", 2: "rgb", 3: "palette", 4: "gray+alpha", 6: "rgba"}
TS_2021 = re.compile(r"D(\d{8})T(\d{6})")


# ---------------------------------------------------------------- helpers
def read_png(path, decode=False):
    """Return dict with header info + md5. Never raises; errors go in 'error'."""
    out = {"width": "", "height": "", "bit_depth": "", "color": "",
           "bytes": "", "md5": "", "error": ""}
    try:
        data = Path(path).read_bytes()
        out["bytes"] = len(data)
        out["md5"] = hashlib.md5(data).hexdigest()
        if data[:8] != PNG_SIG or data[12:16] != b"IHDR":
            out["error"] = "not a PNG"
            return out
        w, h = struct.unpack(">II", data[16:24])
        out["width"], out["height"] = w, h
        out["bit_depth"] = data[24]
        out["color"] = COLOR_TYPES.get(data[25], f"type{data[25]}")
        if not data.rstrip(b"\x00").endswith(b"IEND\xaeB`\x82"):
            out["error"] = "truncated (no IEND)"
        if decode and not out["error"]:
            from PIL import Image
            import io
            with Image.open(io.BytesIO(data)) as im:
                im.load()
    except Exception as e:  # noqa: BLE001 - audit must not crash on one file
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def normalize(name):
    """Lower-case and collapse spaces / underscores / hyphens."""
    return re.sub(r"[\s_\-]+", "_", name.strip().lower())


def list_dataset(root, dataset):
    """Yield (dataset, class_name, abs_path, rel_path) for every PNG."""
    root = Path(root)
    for cls_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for f in sorted(cls_dir.iterdir()):
            if f.is_file() and f.suffix.lower() == ".png":
                yield dataset, cls_dir.name, f, f.relative_to(root).as_posix()


def percentiles(values, ps=(5, 25, 50, 75, 95, 99)):
    v = sorted(values)
    if not v:
        return {p: "" for p in ps}
    return {p: v[min(len(v) - 1, int(round(p / 100 * (len(v) - 1))))] for p in ps}


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-1619", default=DEFAULT_1619,
                    help="folder with one sub-folder per class (2016-2019 set)")
    ap.add_argument("--data-2021", default=DEFAULT_2021,
                    help="folder with one sub-folder per class (2021 Uto set)")
    ap.add_argument("--out", default=DEFAULT_OUT, help="output folder")
    ap.add_argument("--workers", type=int, default=min(32, (os.cpu_count() or 8) * 2))
    ap.add_argument("--decode", action="store_true",
                    help="fully decode every PNG with Pillow (slower)")
    args = ap.parse_args()

    roots = {"2016_2019": Path(args.data_1619), "2021": Path(args.data_2021)}
    for k, r in roots.items():
        if not r.is_dir():
            sys.exit(f"[error] {k} folder not found: {r}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---- 1. list files
    t0 = time.time()
    files = []
    for ds, r in roots.items():
        files.extend(list_dataset(r, ds))
    print(f"Found {len(files):,} PNG files in {time.time() - t0:.1f}s; "
          f"reading headers with {args.workers} threads ...")

    # ---- 2. read headers + hash (I/O bound -> threads)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        infos = list(ex.map(lambda f: read_png(f[2], args.decode), files,
                            chunksize=256))
    print(f"Read {len(infos):,} files in {time.time() - t0:.1f}s")

    # ---- 3. index.csv
    rows = []
    for (ds, cls, _abs, rel), info in zip(files, infos):
        m = TS_2021.search(Path(rel).name)
        rows.append({
            "dataset": ds,
            "class": cls,
            "role": "unclassifiable" if cls == UNCLASSIFIABLE else "labelled",
            "rel_path": rel,
            "timestamp": f"{m.group(1)}T{m.group(2)}" if m else "",
            **info,
        })
    with open(out / "index.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---- 4. class counts + retained flag
    counts = {ds: Counter(r["class"] for r in rows if r["dataset"] == ds) for ds in roots}
    all_classes = sorted(set(counts["2016_2019"]) | set(counts["2021"]))
    retained = {c for c, n in counts["2016_2019"].items()
                if n >= MIN_IMAGES and c not in DROP_CLASSES}
    with open(out / "class_counts.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["class", "n_2016_2019", "n_2021", "retained"])
        for c in sorted(all_classes, key=lambda c: -counts["2016_2019"].get(c, 0)):
            w.writerow([c, counts["2016_2019"].get(c, 0), counts["2021"].get(c, 0),
                        int(c in retained)])

    # ---- 5. name-match check (final mapping table is written by 01_prepare_splits.py)
    c2021 = {c for c in counts["2021"] if c != UNCLASSIFIABLE}
    norm2021 = {normalize(c) for c in c2021}
    match_types = Counter(
        "exact" if c in c2021 else "normalized" if normalize(c) in norm2021 else "none (check by hand)"
        for c in counts["2016_2019"])

    # ---- 6. size summary
    ok = [r for r in rows if r["width"] != ""]
    size_rows = []
    for ds in roots:
        sub = [r for r in ok if r["dataset"] == ds and r["role"] == "labelled"]
        for name, vals in (("width", [r["width"] for r in sub]),
                           ("height", [r["height"] for r in sub]),
                           ("max_side", [max(r["width"], r["height"]) for r in sub])):
            p = percentiles(vals)
            size_rows.append({"dataset": ds, "measure": name, "n": len(vals),
                              **{f"p{k}": v for k, v in p.items()}})
    with open(out / "size_summary.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(size_rows[0].keys()))
        w.writeheader()
        w.writerows(size_rows)
    max_sides = [max(r["width"], r["height"]) for r in ok
                 if r["dataset"] == "2016_2019" and r["class"] in retained]
    frac_le = {s: sum(m <= s for m in max_sides) / max(1, len(max_sides))
               for s in (64, 128, 160, 224)}

    # ---- 7. problems: errors, non-gray, duplicates
    errors = [r for r in rows if r["error"]]
    nongray = [r for r in ok if r["color"] != "gray"]
    by_md5 = defaultdict(list)
    for r in rows:
        if r["md5"]:
            by_md5[r["md5"]].append(r)
    dup_groups = [g for g in by_md5.values() if len(g) > 1]
    cross_year = [g for g in dup_groups if len({r["dataset"] for r in g}) > 1]
    cross_class = [g for g in dup_groups if len({(r["dataset"], r["class"]) for r in g}) > 1]
    with open(out / "problems.txt", "w", encoding="utf-8") as fh:
        fh.write(f"Unreadable / invalid files: {len(errors)}\n")
        for r in errors:
            fh.write(f"  {r['dataset']}/{r['rel_path']}: {r['error']}\n")
        fh.write(f"\nNon-grayscale images: {len(nongray)} "
                 f"{dict(Counter(r['color'] for r in nongray))}\n")
        for r in nongray[:200]:
            fh.write(f"  {r['dataset']}/{r['rel_path']}: {r['color']}\n")
        fh.write(f"\nDuplicate groups (identical bytes): {len(dup_groups)}\n"
                 f"  of which span 2016-19 AND 2021 (leakage): {len(cross_year)}\n"
                 f"  of which have different labels: {len(cross_class)}\n")
        for g in sorted(dup_groups, key=lambda g: -len(g))[:200]:
            fh.write("  " + " | ".join(f"{r['dataset']}/{r['rel_path']}" for r in g) + "\n")

    # ---- 8. summary vs published numbers
    n1619 = sum(counts["2016_2019"].values())
    n21_unc = counts["2021"].get(UNCLASSIFIABLE, 0)
    n21_lab = sum(counts["2021"].values()) - n21_unc
    n21_ret = sum(n for c, n in counts["2021"].items() if c in retained)
    checks = [
        ("2016-19 images", n1619, EXPECTED["n_1619"]),
        ("2016-19 classes", len(counts["2016_2019"]), EXPECTED["classes_1619"]),
        ("2021 labelled images", n21_lab, EXPECTED["n_2021_labelled"]),
        ("2021 unclassifiable images", n21_unc, EXPECTED["n_2021_unclassifiable"]),
        ("retained classes (>=100, no Beads)", len(retained), EXPECTED["retained_classes"]),
        ("2021 images in retained classes", n21_ret, EXPECTED["n_2021_retained"]),
    ]
    lines = ["DATA AUDIT SUMMARY", "=" * 60]
    for name, got, exp in checks:
        flag = "OK " if got == exp else "!! "
        lines.append(f"{flag}{name:<38}{got:>8,}   (expected {exp:,})")
    lines += ["", "Class-name match 2016-19 -> 2021: " + ", ".join(f"{k}={v}" for k, v in match_types.items()),
              "Retained classes missing from 2021: "
              + (", ".join(sorted(c for c in retained if counts['2021'].get(c, 0) == 0)) or "none"),
              "", f"Unreadable files: {len(errors)}   Non-grayscale: {len(nongray)}",
              f"Duplicate groups: {len(dup_groups)}  (cross-year: {len(cross_year)}, "
              f"conflicting labels: {len(cross_class)})",
              "", "Image size, retained 2016-19 classes (longest side):"]
    p = percentiles(max_sides)
    lines.append("  " + "  ".join(f"p{k}={v}" for k, v in p.items()))
    lines.append("  share with longest side <= 64/128/160/224 px: "
                 + " / ".join(f"{frac_le[s]:.0%}" for s in (64, 128, 160, 224)))
    lines += ["", f"Outputs written to: {out.resolve()}"]
    text = "\n".join(lines)
    (out / "summary.txt").write_text(text + "\n", encoding="utf-8")
    print("\n" + text)


if __name__ == "__main__":   # required on Windows for clean multi-worker runs
    main()
