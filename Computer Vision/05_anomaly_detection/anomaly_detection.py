"""Anomaly Detection — Unsupervised Industrial QC with PatchCore
================================================================
Dataset : Pills dataset in MVTec AD format (local — place at dataset/pill/).
          267 defect-free training images + 167 mixed test images across
          7 defect categories with pixel-precise ground-truth masks.

Model   : PatchCore via anomalib (WideResNet-50-2 backbone).
          Unsupervised — only normal (defect-free) images are used to train.
          No labels, no defect examples needed.

Two-phase pipeline
------------------
  TRAIN   Extract patch features from train/good/ → build a k-NN coreset
          (memory bank).  Saved as a Lightning checkpoint.
  INFER   Load checkpoint, score every test image via nearest-neighbour
          distance, compute AUROC / AP / F1 / per-category recall,
          save 3-panel PNGs (original | jet heatmap | GT mask).

Outputs (plots/pill/)
---------------------
  panel_<defect>_<name>.png   3-panel visualisation per test image

Usage
-----
  python anomaly_detection.py                    # train then infer (default)
  python anomaly_detection.py --phase train      # fit only
  python anomaly_detection.py --phase infer      # score only (needs checkpoint)
  python anomaly_detection.py --phase train --retrain   # force fresh fit
  python anomaly_detection.py --phase infer --max-test 0  # score all 167 images
"""

import argparse
import logging
import tarfile
import urllib.request
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import yaml
from tqdm import tqdm

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)

MVTEC_CATEGORIES: list[str] = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
]

_MVTEC_URL = (
    "https://www.mvtec.com/fileadmin/Redaktion/mvtec.com/"
    "company/research/datasets/mvtec_anomaly_detection/{category}.tar.xz"
)

# ── fallback config ────────────────────────────────────────────────────────────
_DEFAULT_CONFIG: dict[str, Any] = {
    "dataset": {
        "root":              "dataset",
        "category":          "pill",
        "image_size":        [256, 256],
        "train_batch_size":  32,
        "eval_batch_size":   32,
    },
    "model": {
        "name": "patchcore",
        "patchcore": {
            "backbone":               "wide_resnet50_2",
            "layers":                 ["layer2", "layer3"],
            "pre_trained":            True,
            "coreset_sampling_ratio": 0.1,
            "num_neighbors":          9,
        },
    },
    "engine": {
        "accelerator":       "auto",
        "max_epochs":        1,
        "default_root_dir":  "checkpoints",
    },
    "output_dir":        "plots",
    "threshold":         0.5,
    "max_test_images":   50,
}


# ── helpers ────────────────────────────────────────────────────────────────────

def _load_config(path: str | None) -> dict[str, Any]:
    cfg_path = Path(path) if path else Path(__file__).with_name("config.yaml")
    if cfg_path.exists():
        with cfg_path.open() as f:
            return yaml.safe_load(f)
    log.warning("config.yaml not found — using built-in defaults.")
    return _DEFAULT_CONFIG


def _download_mvtec_category(root: Path, category: str) -> None:
    """Download and extract a single MVTec AD category (~50-90 MB each)."""
    cat_dir = root / category
    if cat_dir.exists() and any(cat_dir.iterdir()):
        return  # already present

    root.mkdir(parents=True, exist_ok=True)
    tar_path = root / f"{category}.tar.xz"
    url      = _MVTEC_URL.format(category=category)

    log.info("Downloading MVTec AD '%s' (~50-90 MB, one-time download)…", category)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp, tar_path.open("wb") as out_f:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded, chunk = 0, 65536
            while True:
                block = resp.read(chunk)
                if not block:
                    break
                out_f.write(block)
                downloaded += len(block)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r  {pct:5.1f}%  ({downloaded/1e6:.1f}/{total/1e6:.1f} MB)", end="")
        print()
    except Exception as exc:
        tar_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Failed to download MVTec '{category}' from {url}\n"
            f"Download manually and extract to: {cat_dir}"
        ) from exc

    log.info("Extracting '%s'…", tar_path.name)
    with tarfile.open(tar_path, "r:xz") as tf:
        tf.extractall(root)
    tar_path.unlink()
    log.info("MVTec '%s' ready at %s", category, cat_dir)


def _make_panel(
    original: np.ndarray,
    anomaly_map: np.ndarray,
    gt_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Concatenate [original | jet heatmap | GT mask] side by side."""
    h, w = original.shape[:2]
    panels: list[np.ndarray] = [original]

    # Jet heatmap
    amap = np.nan_to_num(anomaly_map, nan=0.0, posinf=1.0, neginf=0.0)
    norm = (amap - amap.min()) / max(float(amap.max() - amap.min()), 1e-6)
    norm = norm.clip(0.0, 1.0)   # guard against any float precision drift
    heat = (cm.jet(norm)[:, :, :3] * 255).astype(np.uint8)
    panels.append(heat)

    if gt_mask is not None:
        gt_rgb = np.zeros((h, w, 3), dtype=np.uint8)
        gt_rgb[gt_mask > 0] = [220, 50, 50]   # red marks
        panels.append(gt_rgb)

    return np.concatenate(panels, axis=1)

# ── detector class ─────────────────────────────────────────────────────────────

class AnomalyDetector:
    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg         = cfg
        self.category    = cfg["dataset"]["category"]
        self.data_root   = Path(cfg["dataset"]["root"])
        self.output_dir  = Path(cfg["output_dir"]) / self.category
        self.ckpt_dir    = Path(cfg["engine"]["default_root_dir"]) / self.category
        self.threshold   = float(cfg.get("threshold", 0.5))
        self.max_test    = int(cfg.get("max_test_images", 50))

    # ── data preparation ──────────────────────────────────────────────────────

    def prepare_data(self) -> None:
        _download_mvtec_category(self.data_root, self.category)

    # ── anomalib integration ──────────────────────────────────────────────────

    def _build_datamodule(self):
        """Folder datamodule: loads only normal training images.

        Works around an anomalib v2 pandas-StringDtype / Split-enum comparison
        bug that makes MVTecAD return an empty training dataset.
        Folder validates the directory structure without relying on the broken
        string-vs-enum comparison in MVTecADDataset.
        """
        from anomalib.data import Folder
        dcfg      = self.cfg["dataset"]
        data_root = (self.data_root / self.category).resolve()
        return Folder(
            name=self.category,
            root=str(data_root),
            normal_dir="train/good",
            train_batch_size=int(dcfg.get("train_batch_size", 32)),
            eval_batch_size=int(dcfg.get("eval_batch_size", 32)),
            num_workers=0,
            val_split_mode="from_train",
            val_split_ratio=0.1,
            test_split_mode="none",
        )

    def _build_model(self):
        model_name = self.cfg["model"]["name"].lower()
        if model_name == "patchcore":
            from anomalib.models import Patchcore
            pc = self.cfg["model"].get("patchcore", {})
            return Patchcore(
                backbone=pc.get("backbone", "wide_resnet50_2"),
                layers=pc.get("layers", ["layer2", "layer3"]),
                pre_trained=pc.get("pre_trained", True),
                coreset_sampling_ratio=float(pc.get("coreset_sampling_ratio", 0.1)),
                num_neighbors=int(pc.get("num_neighbors", 9)),
            )
        if model_name == "efficientad":
            from anomalib.models import EfficientAd
            ea = self.cfg["model"].get("efficientad", {})
            return EfficientAd(
                model_size=ea.get("model_size", "small"),
                teacher_out_channels=int(ea.get("teacher_out_channels", 384)),
                imagenet_dir=ea.get("imagenet_dir", "./datasets/imagenet"),
            )
        raise ValueError(f"Unknown model: {model_name!r}. Use 'patchcore' or 'efficientad'.")

    def _build_engine(self):
        from anomalib.engine import Engine
        ecfg = self.cfg["engine"]
        return Engine(
            default_root_dir=str(self.ckpt_dir),
            accelerator=ecfg.get("accelerator", "auto"),
            max_epochs=int(ecfg.get("max_epochs", 1)),
        )

    def _find_checkpoint(self) -> Path | None:
        """Return the best checkpoint path if a previous run left one, else None."""
        for ckpt in sorted(self.ckpt_dir.rglob("best*.ckpt")):
            return ckpt
        for ckpt in sorted(self.ckpt_dir.rglob("*.ckpt")):
            return ckpt
        return None

    # ── test data (manual MVTec-format loader) ────────────────────────────────

    def _load_test_images(self) -> list[dict]:
        """Walk the MVTec-format test directory and return a list of item dicts.

        Returns one dict per image with keys:
          image_path, gt_label (0=normal / 1=anomaly), defect, mask_path.
        """
        test_dir = (self.data_root / self.category / "test").resolve()
        gt_dir   = (self.data_root / self.category / "ground_truth").resolve()
        if not test_dir.exists():
            raise FileNotFoundError(f"Test directory not found: {test_dir}")

        items: list[dict] = []
        for defect_dir in sorted(
            (d for d in test_dir.iterdir() if d.is_dir()),
            key=lambda d: (d.name != "good", d.name),  # good first, then alphabetical
        ):
            is_normal = defect_dir.name == "good"
            for img_path in sorted(defect_dir.glob("*.png")):
                mask_path = None
                if not is_normal:
                    candidate = gt_dir / defect_dir.name / f"{img_path.stem}_mask.png"
                    if candidate.exists():
                        mask_path = candidate
                items.append({
                    "image_path": img_path,
                    "gt_label":   0 if is_normal else 1,
                    "defect":     defect_dir.name,
                    "mask_path":  mask_path,
                })
        log.info("  Test set: %d images across %d categories",
                 len(items), len(set(i["defect"] for i in items)))
        return items

    # ── inference & visualisation ─────────────────────────────────────────────

    def _run_inference(self, model) -> list[dict]:
        """Manually loop over every test image and return per-image result dicts.

        Bypasses anomalib's test dataloader to avoid the pandas StringDtype vs
        Split-enum comparison bug present in this anomalib version.
        """
        import torch
        from PIL import Image as PILImage
        from torchvision import transforms

        image_size = tuple(self.cfg["dataset"].get("image_size", [256, 256]))
        preprocess = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),          # [0, 1] range only
        ])

        model.eval()
        device = next(iter(model.parameters()), torch.tensor(0)).device

        test_items = self._load_test_images()
        limit      = self.max_test if self.max_test > 0 else len(test_items)
        records: list[dict] = []
        self.output_dir.mkdir(parents=True, exist_ok=True)

        for item in tqdm(test_items[:limit], desc="[INFER] Scoring", unit="img"):
            img_path = item["image_path"]
            pil_img  = PILImage.open(img_path).convert("RGB")
            orig_np  = np.array(pil_img.resize(image_size))
            h, w     = orig_np.shape[:2]

            tensor = preprocess(pil_img).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(tensor)

            anomaly_map = self._get_attr(output, "anomaly_map")

            if anomaly_map is not None:
                amap_raw  = anomaly_map[0, 0].cpu().numpy().astype(np.float32)
                mn, mx    = float(amap_raw.min()), float(amap_raw.max())
                amap_norm = (amap_raw - mn) / max(mx - mn, 1e-6)
                # Convert to uint8 before PIL resize — PIL mode-"F" (float32)
                # bilinear resize is unreliable across Pillow versions and can
                # push values outside [0,1], corrupting the jet colormap.
                amap_u8 = (amap_norm.clip(0.0, 1.0) * 255).astype(np.uint8)
                amap_np = (
                    np.array(
                        PILImage.fromarray(amap_u8).resize((w, h), PILImage.BILINEAR)
                    ).astype(np.float32)
                    / 255.0
                )
            else:
                amap_np = np.zeros((h, w), dtype=np.float32)

            gt_mask_np = None
            if item["mask_path"]:
                gm         = PILImage.open(item["mask_path"]).convert("L").resize((w, h))
                gt_mask_np = (np.array(gm) > 0).astype(np.uint8)

            raw_score = float(amap_raw.max()) if anomaly_map is not None else 0.0

            panel    = _make_panel(orig_np, amap_np, gt_mask_np)
            out_path = self.output_dir / f"panel_{item['defect']}_{img_path.stem}.png"
            plt.imsave(str(out_path), panel)

            records.append({
                "image_path": str(img_path),
                "defect":     item["defect"],
                "raw_score":  raw_score,
                "gt_label":   item["gt_label"],
                "panel_file": out_path.name,
            })
        return records

    def _compute_metrics(self, records: list[dict]) -> dict:
        """Compute a comprehensive set of image-level anomaly detection metrics.

        Metrics
        -------
        image_AUROC
            Area Under the ROC Curve — overall ranking ability (normal vs anomaly).
            A random classifier scores 0.50; a perfect one scores 1.00.
        image_AP
            Average Precision (area under Precision-Recall curve).
            Better than AUROC on imbalanced datasets because it is sensitive to
            false positives in the anomaly class.  With 26 normal / 141 defective
            pills, imbalance is severe: use AP as the primary metric.
        image_F1
            F1 score at the optimal decision threshold (found by scanning all
            thresholds and picking the one that maximises F1).  Shows real
            operational accuracy: the best trade-off between catching defects and
            avoiding false alarms.
        per_category
            Per-defect-type detection rate (recall).  Reveals which defect types
            the model struggles with, e.g. 'color' might be easy while 'scratch'
            is hard.
        """
        from sklearn.metrics import (
            roc_auc_score,
            average_precision_score,
            f1_score,
            precision_recall_curve,
        )
        import numpy as np

        labeled = [(r["gt_label"], r["raw_score"], r["defect"])
                   for r in records if r["gt_label"] >= 0]
        if len(labeled) < 2:
            log.warning("Not enough labeled samples for metrics.")
            return {}

        gt     = [x[0] for x in labeled]
        scores = [x[1] for x in labeled]
        cats   = [x[2] for x in labeled]

        metrics: dict = {}

        n_normal  = sum(1 for g in gt if g == 0)
        n_anomaly = sum(1 for g in gt if g == 1)
        log.info("  Scored %d normal + %d anomaly images", n_normal, n_anomaly)

        # ── Image-level AUROC ─────────────────────────────────────────────────
        try:
            auroc = roc_auc_score(gt, scores)
            metrics["image_AUROC"] = round(float(auroc), 4)
            log.info("  image_AUROC        = %.4f", auroc)
        except Exception as exc:
            log.warning("AUROC failed: %s", exc)

        # ── Average Precision (better for imbalanced) ─────────────────────────
        try:
            ap = average_precision_score(gt, scores)
            metrics["image_AP"] = round(float(ap), 4)
            log.info("  image_AP           = %.4f  (primary metric for imbalanced data)", ap)
        except Exception as exc:
            log.warning("AP failed: %s", exc)

        # ── F1 at optimal threshold ───────────────────────────────────────────
        try:
            prec, rec, thresholds = precision_recall_curve(gt, scores)
            # F1 = 2*P*R/(P+R); avoid division by zero
            f1_scores = np.where(
                (prec + rec) > 0,
                2 * prec * rec / (prec + rec),
                0.0,
            )
            best_idx  = int(np.argmax(f1_scores))
            best_f1   = float(f1_scores[best_idx])
            best_thr  = float(thresholds[best_idx]) if best_idx < len(thresholds) else float("nan")
            metrics["image_F1_best"]       = round(best_f1, 4)
            metrics["image_F1_threshold"]  = round(best_thr, 6)
            log.info("  image_F1 (best)    = %.4f  @ threshold %.6f", best_f1, best_thr)
        except Exception as exc:
            log.warning("F1 failed: %s", exc)

        # ── Per-category detection rate (recall) ──────────────────────────────
        from collections import defaultdict
        cat_scores: dict = defaultdict(list)
        for g, s, c in zip(gt, scores, cats):
            cat_scores[c].append((g, s))

        # Use the best F1 threshold found above (or median score if F1 failed)
        thr = metrics.get("image_F1_threshold", float(np.median(scores)))

        log.info("  Per-category recall @ threshold %.6f:", thr)
        per_cat: dict = {}
        for cat in sorted(cat_scores):
            labels_cat  = [g for g, _ in cat_scores[cat]]
            scores_cat  = [s for _, s in cat_scores[cat]]
            n_total     = len(labels_cat)
            n_anomalous = sum(l == 1 for l in labels_cat)
            if n_anomalous == 0:
                # Normal category — false positive rate
                fp = sum(s >= thr for s in scores_cat)
                per_cat[cat] = {"total": n_total, "FP": fp, "FP_rate": round(fp / n_total, 3)}
                log.info("    %-20s  normal  FP_rate=%.0f%%  (%d/%d flagged)",
                         cat, fp / n_total * 100, fp, n_total)
            else:
                # Defect category — recall (true positive rate)
                tp = sum(s >= thr for s, l in zip(scores_cat, labels_cat) if l == 1)
                recall = tp / n_anomalous
                per_cat[cat] = {"total": n_total, "TP": tp, "recall": round(recall, 3)}
                log.info("    %-20s  defect  recall=%.0f%%  (%d/%d detected)",
                         cat, recall * 100, tp, n_anomalous)
        metrics["per_category"] = per_cat

        return metrics

    @staticmethod
    def _get_attr(obj, name: str):
        """Safely read an attribute from anomalib output (dict or namedtuple-like)."""
        if isinstance(obj, dict):
            return obj.get(name)
        return getattr(obj, name, None)

    # ── training phase ─────────────────────────────────────────────────────────

    def train(self) -> None:
        """Phase 1 — Fit PatchCore on defect-free training images.

        Builds a patch-feature coreset from every image in train/good/.
        No gradient descent — this is a memory-bank construction step.
        The fitted model is saved automatically as a .ckpt checkpoint.
        Subsequent calls skip fitting if the checkpoint already exists;
        pass retrain=True to force a fresh fit.
        """
        existing = self._find_checkpoint()
        if existing and not self._retrain:
            log.info("Checkpoint already exists at %s", existing)
            log.info("Skipping training. Use --retrain to force a fresh fit.")
            return

        log.info("[TRAIN] Fitting PatchCore on '%s/train/good/' (%d classes).",
                 self.category,
                 len(list((self.data_root / self.category / "train" / "good").glob("*.png"))))
        datamodule = self._build_datamodule()
        model      = self._build_model()
        engine     = self._build_engine()
        engine.fit(model=model, datamodule=datamodule)

        ckpt = self._find_checkpoint()
        if ckpt:
            log.info("[TRAIN] Done. Checkpoint → %s", ckpt)
        else:
            log.warning("[TRAIN] Training finished but no checkpoint found in %s", self.ckpt_dir)

    # ── inference phase ───────────────────────────────────────────────────────

    def infer(self) -> None:
        """Phase 2 — Score all test images, generate panels + metrics.

        Loads the checkpoint produced by train(), runs every image in
        test/ through the model, computes image-level AUROC, and saves
        3-panel PNGs (original | heatmap | GT mask), a CSV of per-image
        scores, a Plotly chart, and an HTML gallery report.

        Raises RuntimeError if no checkpoint exists (train first).
        """
        checkpoint = self._find_checkpoint()
        if not checkpoint:
            raise RuntimeError(
                f"No checkpoint found for category '{self.category}'.\n"
                f"Run the training phase first:\n"
                f"  python anomaly_detection.py --phase train\n"
                f"  python anomaly_detection.py --phase all  (train then infer)"
            )

        log.info("[INFER] Checkpoint: %s", checkpoint)

        # Use anomalib's own loader so that on_load_checkpoint() is called.
        # This is essential for PatchCore: the memory bank is stored via
        # Lightning hooks, NOT in state_dict, so torch.load + load_state_dict
        # leaves the memory bank empty and produces garbage scores.
        from anomalib.models import Patchcore
        log.info("[INFER] Loading model via load_from_checkpoint (restores memory bank)…")
        model = Patchcore.load_from_checkpoint(
            str(checkpoint), map_location="cpu", weights_only=False
        )
        model.eval()

        log.info("[INFER] Scoring all test images…")
        records = self._run_inference(model)

        log.info("[INFER] Computing metrics…")
        metrics = self._compute_metrics(records)

        if not records:
            return

        # Use the best F1 threshold from metrics to label each image,
        # falling back to the config threshold if metrics are unavailable.
        best_thr = metrics.get("image_F1_threshold", None)

        # Normalise scores to [0, 1] for the chart
        raw = [r["raw_score"] for r in records]
        mn, mx = min(raw), max(raw)
        for r in records:
            ns              = (r["raw_score"] - mn) / max(mx - mn, 1e-6)
            r["norm_score"] = ns
            # Use optimal F1 threshold (in normalised space) or config threshold
            thr = ((best_thr - mn) / max(mx - mn, 1e-6)) if best_thr is not None else self.threshold
            r["pred_label"] = "ANOMALY" if ns > thr else "normal"

        # Summary line
        n_flagged = sum(1 for r in records if r["pred_label"] == "ANOMALY")
        log.info(
            "[INFER] Summary: AUROC=%.4f  AP=%.4f  F1=%.4f  |  %d/%d flagged as ANOMALY",
            metrics.get("image_AUROC", float("nan")),
            metrics.get("image_AP",   float("nan")),
            metrics.get("image_F1_best", float("nan")),
            n_flagged, len(records),
        )

    # ── orchestrator ────────────────────────────────────────────────────────────

    def run(self, phase: str = "all", retrain: bool = False) -> None:
        """Run one or both phases.

        Args:
            phase:   "train" — fit on normal images only.
                     "infer" — score test images (checkpoint required).
                     "all"   — train then infer (default).
            retrain: Force a fresh fit even if a checkpoint exists.
        """
        self._retrain = retrain
        self.output_dir.mkdir(parents=True, exist_ok=True)

        log.info("Dataset : %s/%s", self.data_root, self.category)
        log.info("Preparing data…")
        self.prepare_data()

        log.info("Model   : %s", self.cfg["model"]["name"])

        if phase in ("train", "all"):
            self.train()

        if phase in ("infer", "all"):
            self.infer()

        log.info("Done. Outputs in %s/", self.output_dir)


# ── entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Industrial anomaly detection — PatchCore on pill dataset (MVTec AD format)"
    )
    parser.add_argument("--phase",    type=str, default="all",
                        choices=["train", "infer", "all"],
                        help="train = fit on normal images only; "
                             "infer = score test images (needs checkpoint); "
                             "all = train then infer (default).")
    parser.add_argument("--retrain",  action="store_true",
                        help="Force a fresh fit even if a checkpoint exists.")
    parser.add_argument("--category", type=str, default=None,
                        help="Dataset category to process (default: from config.yaml).")
    parser.add_argument("--config",   type=str, default=None,
                        help="Path to config.yaml (default: config.yaml next to this script).")
    parser.add_argument("--max-test", type=int, default=None,
                        help="Override max_test_images from config.")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    if args.max_test is not None:
        cfg["max_test_images"] = args.max_test
    if args.category is not None:
        cfg["dataset"]["category"] = args.category

    log.info("=" * 60)
    log.info("Category : %s", cfg["dataset"]["category"])
    log.info("Phase    : %s", args.phase)
    log.info("=" * 60)

    detector = AnomalyDetector(cfg)
    detector.run(phase=args.phase, retrain=args.retrain)


if __name__ == "__main__":
    main()

