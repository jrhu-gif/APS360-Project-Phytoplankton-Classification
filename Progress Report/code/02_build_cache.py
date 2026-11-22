"""
02_build_cache.py  -  Step 5: segment, extract features, cache 96x256 images.

For every image in prep/{train,val,test,ood}.csv (same row order):
  * Otsu segmentation on the ORIGINAL image; all blobs >= 5% of the largest
    are kept as one organism (chains/colonies), n_parts = number of blobs
  * features: 8 morphology (HAB-Net) + gray-level + GLCM texture (random forest)
  * canvas: shrink-only resize into CANVAS_H x CANVAS_W, aspect kept,
    padded with the most frequent border (background) pixel value
Writes to C:\\APS360\\outputs\\cache:
  {split}_images.npy  uint8   (N, H, W)
  {split}_feats.npy   float32 (N, F)     column names in feature_names.json
  cache_stats.json    train pixel mean/std + morphology mean/std, seg failures
Usage:  python 02_build_cache.py [--splits train val test ood] [--workers 16]
Needs:  numpy, pillow, scikit-image
"""
import os
# One math thread per worker: 16 processes x ~20 OpenBLAS threads each exhausts
# the Windows paging file ("DLL load failed ... paging file is too small").
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
import json
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image
from skimage.feature import graycomatrix, graycoprops
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops

import config as C

GLCM_PROPS = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]
FEATURE_NAMES = (
    C.MORPH_FEATURES
    + ["n_parts", "major_axis", "minor_axis", "extent", "fill_frac",
       "obj_mean", "obj_std", "obj_min", "obj_max", "bg_mean",
       "img_mean", "img_std"]
    + [f"glcm_{p}" for p in GLCM_PROPS]
    + ["scale", "seg_fail"]
)


def process(path):
    """Return (canvas uint8 HxW, feature vector float32). Runs in a worker."""
    img = np.asarray(Image.open(path).convert("L"))
    h, w = img.shape
    f = dict.fromkeys(FEATURE_NAMES, 0.0)
    f.update(img_w=w, img_h=h, aspect=w / h,
             img_mean=img.mean(), img_std=img.std())

    # ---- segmentation: background brightness from the border decides polarity
    border = np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]])
    bg = float(np.median(border))
    try:
        t = threshold_otsu(img)
        mask = img <= t if bg > t else img > t   # Otsu: values <= t form the lower class
    except ValueError:                       # constant image
        mask = np.zeros_like(img, bool)
    # Chains (Skeletonema, Dolichospermum, colonies) are separate blobs:
    # keep every blob >= 5% of the largest (and >= 10 px), measure them as one organism.
    lab = label(mask)
    areas = np.bincount(lab.ravel())[1:]
    if areas.size:
        keep = np.flatnonzero(areas >= max(10, 0.05 * areas.max())) + 1
        obj = np.isin(lab, keep)
        f["n_parts"] = len(keep)
    else:                                    # fallback: whole image
        obj = np.ones_like(img, bool)
        f["seg_fail"] = 1.0
        f["n_parts"] = 1
    r = regionprops(obj.astype(np.uint8), intensity_image=img)[0]
    vals = img[obj]
    f.update(area=r.area, perimeter=r.perimeter,
             eq_diameter=r.equivalent_diameter_area,
             eccentricity=r.eccentricity, solidity=r.solidity,
             major_axis=r.axis_major_length, minor_axis=r.axis_minor_length,
             extent=r.extent, fill_frac=r.area / (h * w),
             obj_mean=vals.mean(), obj_std=vals.std(),
             obj_min=vals.min(), obj_max=vals.max(),
             bg_mean=img[~obj].mean() if (~obj).any() else bg)

    # ---- GLCM texture on the object's bounding box (32 gray levels, 4 angles)
    r0, c0, r1, c1 = r.bbox
    crop = (img[r0:r1, c0:c1] // 8).astype(np.uint8)
    if crop.shape[0] > 1 and crop.shape[1] > 1:
        g = graycomatrix(crop, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                         levels=32, symmetric=True, normed=True)
        for p in GLCM_PROPS:
            f[f"glcm_{p}"] = float(np.nan_to_num(graycoprops(g, p).mean()))

    # ---- canvas: shrink only, keep aspect, pad with most frequent value
    scale = min(1.0, C.CANVAS_W / w, C.CANVAS_H / h)
    f["scale"] = scale
    im = Image.fromarray(img)
    if scale < 1.0:
        im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))),
                       Image.Resampling.LANCZOS)
    small = np.asarray(im)
    pad = np.bincount(border, minlength=256).argmax()      # most frequent background value
    canvas = np.full((C.CANVAS_H, C.CANVAS_W), pad, np.uint8)
    sh, sw = small.shape
    y0, x0 = (C.CANVAS_H - sh) // 2, (C.CANVAS_W - sw) // 2
    canvas[y0:y0 + sh, x0:x0 + sw] = small
    return canvas, np.array([f[k] for k in FEATURE_NAMES], np.float32)


def build_split(split, ex):
    src = C.PREP_DIR / f"{split}.csv"
    with open(src, encoding="utf-8") as fh:
        paths = [r["path"] for r in csv.DictReader(fh)]
    n = len(paths)
    imgs = np.lib.format.open_memmap(C.CACHE_DIR / f"{split}_images.npy", mode="w+",
                                     dtype=np.uint8, shape=(n, C.CANVAS_H, C.CANVAS_W))
    feats = np.zeros((n, len(FEATURE_NAMES)), np.float32)
    t0 = time.time()
    for i, (canvas, fv) in enumerate(ex.map(process, paths, chunksize=64)):
        imgs[i] = canvas
        feats[i] = fv
        if (i + 1) % 20000 == 0:
            print(f"  {split}: {i + 1:,}/{n:,}  ({time.time() - t0:.0f}s)")
    imgs.flush()
    del imgs
    np.save(C.CACHE_DIR / f"{split}_feats.npy", feats)
    fails = int(feats[:, FEATURE_NAMES.index("seg_fail")].sum())
    shrunk = float((feats[:, FEATURE_NAMES.index("scale")] < 1).mean())
    print(f"  {split}: {n:,} images in {time.time() - t0:.0f}s | "
          f"seg failures {fails} | shrunk {shrunk:.0%}")
    return {"n": n, "seg_fail": fails, "shrunk_frac": round(shrunk, 4)}


def train_stats():
    imgs = np.load(C.CACHE_DIR / "train_images.npy", mmap_mode="r")
    feats = np.load(C.CACHE_DIR / "train_feats.npy")
    px = imgs.astype(np.float32) / 255.0
    morph = {}
    for k in C.MORPH_FEATURES:
        v = feats[:, FEATURE_NAMES.index(k)].astype(np.float64)
        if k in C.LOG_FEATURES:
            v = np.log1p(v)
        morph[k] = {"mean": float(v.mean()), "std": float(v.std() + 1e-8),
                    "log": k in C.LOG_FEATURES}
    return {"pixel_mean": float(px.mean()), "pixel_std": float(px.std()), "morph": morph}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=C.SPLITS)
    ap.add_argument("--workers", type=int, default=C.NUM_WORKERS)
    args = ap.parse_args()
    C.CACHE_DIR.mkdir(parents=True, exist_ok=True)

    need = sum(sum(1 for _ in open(C.PREP_DIR / f"{s}.csv", encoding="utf-8")) - 1
               for s in args.splits) * C.CANVAS_H * C.CANVAS_W
    free = shutil.disk_usage(C.CACHE_DIR).free
    print(f"Cache needs ~{need / 1e9:.1f} GB; free on disk: {free / 1e9:.1f} GB")
    if need > 0.9 * free:
        sys.exit("[error] not enough disk space")

    report = {"feature_names": FEATURE_NAMES, "canvas": [C.CANVAS_H, C.CANVAS_W]}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:   # one pool for all splits
        for s in args.splits:
            report[s] = build_split(s, ex)
    (C.CACHE_DIR / "feature_names.json").write_text(json.dumps(FEATURE_NAMES, indent=1))
    if (C.CACHE_DIR / "train_images.npy").exists():
        report["train_stats"] = train_stats()
    (C.CACHE_DIR / "cache_stats.json").write_text(json.dumps(report, indent=1))
    print(f"\nDone in {time.time() - t0:.0f}s -> {C.CACHE_DIR}")
    if "train_stats" in report:
        ts = report["train_stats"]
        print(f"Train pixel mean/std: {ts['pixel_mean']:.4f} / {ts['pixel_std']:.4f}")


if __name__ == "__main__":
    main()
