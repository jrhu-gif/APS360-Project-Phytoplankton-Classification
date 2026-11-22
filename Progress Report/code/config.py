"""Shared settings for every pipeline script. Change paths here only."""
from pathlib import Path

# ---- paths (local disk, outside OneDrive)
DATA_1619 = Path(r"C:\APS360\data\SYKE_2016_2019")
DATA_2021 = Path(r"C:\APS360\data\SYKE_2021")
OUT_ROOT = Path(r"C:\APS360\outputs")
AUDIT_INDEX = Path(__file__).resolve().parent.parent / "audit_out" / "index.csv"
PREP_DIR = OUT_ROOT / "prep"

# ---- data rules (proposal + feedback)
SEED = 42
VAL_FRAC = 0.15          # stratified, per class, round(0.15 * n)
TRAIN_CAP = 1500         # per class, training split only, applied after splitting
MIN_IMAGES = 100         # class must have >= 100 images in 2016-19
DROP_CLASSES = {"Beads"}
UNCLASSIFIABLE = "Unclassifiable"

# HAB classes (8; as marked in proposal Figure A1)
HAB_CLASSES = {
    "Aphanizomenon_flosaquae",
    "Dolichospermum-Anabaenopsis",
    "Dolichospermum-Anabaenopsis-coiled",
    "Nodularia_spumigena",
    "Oscillatoriales",
    "Snowella-Woronichinia",
    "Dinophysis_acuminata",
    "Prorocentrum_cordatum",
}

# ---- model input (decided 2026-11-18 from audit: images are wide, median 136x58 px)
CANVAS_H, CANVAS_W = 96, 256      # shrink-only (never enlarge), aspect kept, pad with mode value
CACHE_DIR = OUT_ROOT / "cache"
RF_DIR = OUT_ROOT / "rf"
FIG_DIR = Path(__file__).resolve().parent.parent / "figures"   # report figures (in project folder)
COVERAGE = 0.95          # "uncertain" threshold: keep 95% of validation predictions
RUNS_DIR = OUT_ROOT / "runs"

# ---- neural-network training (step 6). Tuning: pick by val macro-F1 with val HAB recall >= 0.95
TRAIN = {
    "M":  {"model": "habnet", "morph": True, "hab_head": True,          # primary model
           "lr": 1e-3, "w_hab": 3.0, "lam": 0.5},
    "B2": {"model": "resnet18", "morph": False, "hab_head": False,      # ImageNet ResNet-18, fine-tuned
           "lr": 3e-4},
    "M80": {"model": "habnet", "morph": True, "hab_head": True,         # M trained longer (M still improving at 40)
            "lr": 1e-3, "w_hab": 3.0, "lam": 0.5, "epochs": 80, "patience": 15},
    "M80w1": {"model": "habnet", "morph": True, "hab_head": True,       # tuning: HAB weight w_H = 1 vs 3
              "lr": 1e-3, "w_hab": 1.0, "lam": 0.5, "epochs": 80, "patience": 15},
}
EPOCHS, BATCH, PATIENCE, WEIGHT_DECAY = 40, 128, 8, 1e-4   # batch 256 overflowed 8 GB VRAM (4 min/epoch)
SIZE_JITTER = (0.8, 1.25)        # train-only random scale of size features (size shift between years)
SPLITS = ["train", "val", "test", "ood"]
NUM_WORKERS = 12                  # i7-12700KF (20 threads); leaves headroom for Jupyter/Colab

# ---- features (step 5). MORPH_FEATURES feed HAB-Net; RF uses all features.
MORPH_FEATURES = ["img_w", "img_h", "aspect", "area", "perimeter",
                  "eq_diameter", "eccentricity", "solidity"]
LOG_FEATURES = ["img_w", "img_h", "aspect", "area", "perimeter", "eq_diameter"]  # log1p before z-score
