# Industrial Anomaly Detection — PatchCore on Pills

Unsupervised defect detection for manufacturing quality control. Only defect-free images are needed to train — no defect labels, no annotated examples of what bad looks like. PatchCore learns the distribution of normal patches and flags anything that deviates, producing a pixel-level heat map that highlights exactly where the anomaly is.

## Dataset

A **pills dataset** in MVTec AD format, placed locally in `dataset/pill/`.

| Split | Contents |
|-------|----------|
| `train/good/` | **267 defect-free** pill images — the only images used for training |
| `test/good/` | 26 normal test images |
| `test/color/` | 25 images with colour anomalies |
| `test/combined/` | 17 images with multiple simultaneous defects |
| `test/contamination/` | 21 images with foreign-particle contamination |
| `test/crack/` | 26 images with surface cracks |
| `test/faulty_imprint/` | 19 images with wrong/missing imprint |
| `test/pill_type/` | 9 images with a completely wrong pill type |
| `test/scratch/` | 24 images with surface scratches |
| `ground_truth/` | Pixel-precise binary masks for every defect type |

**Total test images: 167** across 8 categories (including 26 normal).

## Algorithm

**PatchCore** via [anomalib](https://github.com/openvinotoolkit/anomalib) builds an unsupervised memory bank from patch features extracted by a frozen **WideResNet-50-2** backbone (pretrained on ImageNet).

**Training phase:**
1. Run every image in `train/good/` through WideResNet-50-2, tapping `layer2` and `layer3`.
2. Aggregate the spatial patch features from all training images.
3. Sub-sample to a compact **coreset** (default: 10%) using greedy approximation.
4. Fit a k-NN index on the coreset — this is the complete "memory bank".
5. Save the fitted model as a Lightning checkpoint for later inference.

**Inference phase:**
1. Load the checkpoint from training.
2. For each test image, compute the nearest-neighbour distance from its patch features to the memory bank → raw anomaly score + pixel heatmap.
3. Compute a comprehensive set of metrics (see below).
4. Save 3-panel PNGs: **original | jet heatmap | GT mask**.

## Metrics

Four complementary metrics are reported in the terminal after `--phase infer`:

| Metric | What it measures |
|--------|-----------------|
| **AUROC** | Ranking ability — can the model order defective above normal? (0.5 = random, 1.0 = perfect) |
| **AP (Average Precision)** | Primary metric for this dataset — more robust than AUROC on imbalanced data (26 normal vs 141 defective) |
| **F1 at optimal threshold** | Best achievable F1 by scanning all decision thresholds |
| **Per-category recall** | Which defect types are detected at the optimal threshold |

## How to run

```bash
# Train then infer in one shot (default)
python anomaly_detection.py

# Training phase only (fit PatchCore on 267 normal pills)
python anomaly_detection.py --phase train

# Inference phase only (requires an existing checkpoint)
python anomaly_detection.py --phase infer

# Force a fresh fit even if a checkpoint already exists
python anomaly_detection.py --phase train --retrain

# Limit test images for a quick preview (default: 50; 0 = all 167)
python anomaly_detection.py --phase infer --max-test 0
```

### Checkpointing

The fitted PatchCore model (memory bank + weights) is saved automatically under `checkpoints/pill/` after training. Running `--phase infer` skips feature extraction entirely — only the forward pass and scoring happen.

Use `--retrain` to rebuild from scratch (e.g. after adding new normal training images).

## Expected results

PatchCore with WideResNet-50-2 typically achieves **image-level AUROC ≥ 0.95** on MVTec AD pill. The per-category breakdown will show which defect types score well (e.g. `crack`, `color`) and which are harder (e.g. `contamination` — tiny particles are near the edge of the model's detection range).

## Code structure

```
anomaly_detection.py
├── _build_datamodule()       Folder datamodule — loads train/good/ only (no defect images needed)
├── _build_model()            PatchCore(WideResNet-50-2, layer2+layer3, 10% coreset)
├── _build_engine()           anomalib Engine (Lightning Trainer wrapper)
├── _find_checkpoint()        scans checkpoints/pill/ for a saved .ckpt
│
├── AnomalyDetector.train()   Phase 1 — engine.fit() → memory bank → .ckpt
├── AnomalyDetector.infer()   Phase 2 — load_from_checkpoint() → score all test images
│
├── _load_test_images()       walks dataset/pill/test/ → list of {image_path, gt_label, mask_path}
├── _run_inference(model)     PIL + torchvision preprocessing → model forward → heatmaps
│                             Note: Normalize() is intentionally omitted — the loaded model's
│                             pre_processor handles it internally (double-normalising causes
│                             all scores to collapse to 1.0)
└── _compute_metrics()        AUROC · AP · F1@best_threshold · per-category recall
```

## Output

| File | Description |
|------|-------------|
| `plots/pill/panel_<defect>_<stem>.png` | 3-panel: original image · jet heatmap · GT mask (red) |
| `checkpoints/pill/` | Saved PatchCore model (Lightning checkpoint, reused on `--phase infer`) |
