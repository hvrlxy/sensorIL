# Active Sensor Increment for Human Activity Recognition

## Problem Statement

Wearable sensor systems for Human Activity Recognition (HAR) are typically trained once on a fixed sensor configuration and deployed. When a new sensor becomes available — whether due to hardware upgrades, changed deployment context, or expanded monitoring needs — the system must be updated. Naively, this requires collecting new labeled data for all activities with the expanded sensor set, which is expensive and disruptive.

We address the **sensor increment problem**: given a trained $n$-sensor system and an unlabeled dataset collected with $n+1$ sensors, improve activity recognition with the new sensor while **minimizing the number of newly labeled samples required**.

The key observation is that the new sensor is **not equally useful for all activities**. Adding a thigh sensor to a wrist+ankle system provides redundant information for distinguishing treadmill walking variants (leg motion is identical across them) but may be critical for separating lying positions or distinguishing standing from sitting. Requesting re-annotation for all $m$ activities wastes effort on classes that won't benefit.

---

## Core Contribution: Benefit-Guided Active Annotation

The central question is: **for which activities should we request new labeled data?**

We answer this with a benefit score that captures two complementary signals:

$$\text{benefit}(A) = (1 - F_1(A)) + \text{discriminability}(A)$$

The additive formulation is deliberate — discriminability **boosts** the ranking of classes that are both confused and separable by the new sensor, rather than suppressing classes with only one signal.

### Confusion Signal

$1 - F_1(A)$ measures how much the existing $n$-sensor system struggles with class $A$, computed on a held-out validation set. Crucially, we use the **binary classifier output** rather than the confusion matrix: a window of class $A$ is "confused" if $A$'s classifier does not fire, regardless of what other classifiers predict. This correctly handles the case where $A$'s windows are absorbed by dominant classes rather than explicitly misclassified.

### New Sensor Discriminability

Even if an activity is confused under $n$ sensors, the new sensor only helps if it provides discriminative signal. We estimate this from the unlabeled $n+1$-sensor data using a **confidence-weighted blend** of two signals:

$$\text{discriminability}(A) = \alpha \cdot d_{\text{direct}}(A) + (1 - \alpha) \cdot d_{\text{opposition}}(A)$$

where $\alpha = \min(1, N_{\text{pseudo}} / N_{\min})$ controls the blend based on pseudo-label availability.

**Direct discriminability** measures how much the new sensor alone separates class $A$ from its confusion targets using pseudo-labeled FL windows:

$$d_{\text{direct}}(A) = \frac{1}{|\mathcal{N}(A)|} \sum_{B \in \mathcal{N}(A)} d\!\left(\bar{e}^{\text{new}}_A,\ \bar{e}^{\text{new}}_B\right)$$

**Opposition discriminability** handles the case where $A$ has no pseudo-labels (e.g. the activity is rare or absent from the unlabeled pool). Instead of measuring $A$'s own new-sensor embedding, we measure how well-separated $A$'s confusion targets are from each other in new-sensor space:

$$d_{\text{opposition}}(A) = \frac{1}{\binom{|\mathcal{N}(A)|}{2}} \sum_{B \neq C \in \mathcal{N}(A)} d\!\left(\bar{e}^{\text{new}}_B,\ \bar{e}^{\text{new}}_C\right)$$

If the new sensor separates $A$'s neighborhood well, it will also help discriminate $A$ itself — even without directly observing $A$ in the unlabeled data.

### Special Case: Completely Unrecognized Classes

When $F_1(A) = 0$ (the base model never correctly predicts $A$), discriminability cannot be estimated from either direct or opposition signals because $A$'s confusion targets may also be absent or unreliable. In this case we fall back to the confusion signal alone:

$$\text{benefit}(A) = 1 - F_1(A) = 1.0 \quad \text{if } F_1(A) = 0 \text{ and } \text{confusion\_score}(A) > 0$$

This handles activities where the new sensor covers a completely different body region — the gain is real but cannot be estimated from FL data alone.

---

## Incremental Fine-Tuning with Provable Independence

Once new labels are obtained for the selected classes, each targeted classifier is retrained with a larger input (including the new sensor). The central challenge is constructing an appropriate training set — what are the negatives for a targeted class when most labeled data does not include the new sensor?

We propose a **Positive-Unlabeled (PU) learning formulation**:

- **Certain positives**: new labeled windows of the targeted class ($n+1$ sensors)
- **Certain negatives**: new labeled windows of other targeted classes that are unrelated in the activity hierarchy
- **Uncertain negatives**: pseudo-negatives sampled from the unlabeled $n+1$-sensor pool, weighted by confidence $w = 1 - P(\text{class} \mid x)$

The uncertainty weighting is crucial: unlabeled windows where the base classifier is unsure whether they belong to the targeted class contribute less to the negative gradient, preventing the classifier from being misled by mislabeled pseudo-negatives.

To ensure diversity in the unlabeled negatives — avoiding dominance by the most common activities — we perform **stratified sampling via K-means clustering** ($K=40$) on the $n$-sensor embedding space, sampling proportionally from each cluster.

### Independence Guarantee

A critical design requirement is that retraining targeted classifiers must not degrade non-targeted classes. We achieve this through independent binary classification: each class has its own classifier, and **non-targeted classifiers are completely frozen** during incremental fine-tuning. Evaluation uses per-class binary F1 with multi-label ground truth (e.g. a Treadmill window is positive for both Treadmill and Walking), ensuring that a change in one classifier cannot mathematically affect another class's metric.

### Threshold Calibration

After retraining, per-class thresholds are calibrated on the validation set by grid search over $[0.2, 0.8]$, maximizing binary F1 per class. The $[0.2, 0.8]$ clip prevents degenerate thresholds that would cause all or no windows to fire.

---

## Evaluation Protocol

We evaluate three conditions:

| Condition | Sensors at test time | New labels |
|-----------|---------------------|------------|
| **Baseline** | $n$ sensors | 0 |
| **Proposed** | $n+1$ sensors | $K$ (auto-selected) |
| **Oracle** | $n+1$ sensors | $m$ (all classes) |

**Key insight**: the proposed approach can and often does **beat the oracle**, because the oracle retrains all $m$ classes including those that do not benefit from the new sensor (or actively degrade), while the proposed method selectively retrains only classes with positive benefit scores.

**Metrics**: independent per-class binary F1 with multi-label ground truth, macro F1 and weighted F1. Per-class F1 is computed independently — changing one classifier cannot affect another class's metric.

---

## Sensor Configuration Ablation

We ablate over all possible combinations of base sensors and new sensors from the 5-sensor set {LeftWrist, RightWrist, RightThigh, RightWaist, RightAnkle}:

| n_base | Configs | Description |
|--------|---------|-------------|
| 1 → +1 | 20 | $\binom{5}{1} \times 4$ |
| 2 → +1 | 30 | $\binom{5}{2} \times 3$ |
| 3 → +1 | 20 | $\binom{5}{3} \times 2$ |
| 4 → +1 | 5  | $\binom{5}{4} \times 1$ |

**Total**: 75 configurations. Each config is evaluated at annotation budgets $K \in \{5, 10, 15, \text{all}\}$.

---

## Benefit Score Validation

After the sensor ablation, we empirically validate the benefit score by computing Spearman correlation between benefit score and actual $\Delta F_1$ across all targeted classes and sensor configs. We compare four formulations:

1. **Current** (additive): $(1 - F_1) + \text{discriminability}$
2. **Product**: $(1 - F_1) \times \text{discriminability}$
3. **Confusion only**: $1 - F_1$
4. **Discriminability only**: discriminability

The formulation with highest Spearman $\rho$ is the most principled choice.

---

## Repository Structure

```
sensorIL/
├── configs/
│   └── pipeline_config.json          # Main config (sensors, paths, hyperparams)
├── scripts/
│   ├── simclr_encoder.py             # Frozen SimCLR encoder wrapper
│   ├── dataset.py                    # SensorDataset, UnlabeledFLDataset
│   ├── cooccurrence.py               # Activity hierarchy, multi-label encoding,
│   │                                 # get_multilabel(), are_related()
│   ├── train_base.py                 # ParallelBinaryClassifiers, BinaryClassifier,
│   │                                 # FocalLoss, train_base()
│   ├── detect_confusion.py           # Step 2: rank classes by F1, build confusion pairs
│   ├── estimate_benefit.py           # Step 3: benefit score with confidence-weighted
│   │                                 # blend of direct + opposition discriminability
│   ├── incremental_ft.py             # Step 5: PU learning with diverse FL negatives
│   ├── calibrate_thresholds.py       # Per-class threshold calibration [0.2, 0.8]
│   ├── evaluate.py                   # Independent per-class binary F1, combined table
│   ├── run_pipeline.py               # Full pipeline runner (Steps 1-6)
│   ├── ablation_budget.py            # Annotation budget vs F1 curve
│   ├── ablation_sensor.py            # Sensor config ablation (all 75 combos)
│   └── analyze_benefit_score.py      # Post-hoc benefit score correlation analysis
├── checkpoints/                      # Saved model checkpoints
│   ├── base_{name}.pt                # Base classifiers per sensor config
│   ├── incremental_{name}_k{K}.pt    # Incremental classifiers per config + budget
│   ├── oracle_{name}.pt              # Oracle classifiers per sensor config
│   ├── *_thresholds.pt               # Calibrated thresholds
│   ├── ablation_sensor_n{N}_results.json   # Sensor ablation results per n_base
│   ├── ablation_budget.json          # Budget ablation results
│   └── benefit_score_analysis.json   # Benefit score correlation analysis
└── logs/
    ├── ablation_sensor_n{N}_b{budgets}_{timestamp}.log  # Full stdout per run
    └── ablation_budget_{timestamp}.log
```

---

## How to Run

### Full pipeline (single sensor config)
```bash
cd ~/sensorIL
python scripts/run_pipeline.py --config configs/pipeline_config.json
```

### Skip base training (use existing checkpoint)
```bash
python scripts/run_pipeline.py --config configs/pipeline_config.json \
    --base-checkpoint checkpoints/base_classifiers.pt
```

### Sensor config ablation (run one set at a time)
```bash
# 1-base configs (20 combos) with budget ablation
python scripts/ablation_sensor.py \
    --config   configs/pipeline_config.json \
    --n-base   1 \
    --budgets  5,10,15,all

# 2-base configs (30 combos)
python scripts/ablation_sensor.py --config configs/pipeline_config.json --n-base 2 --budgets 5,10,15,all

# 3-base configs (20 combos)
python scripts/ablation_sensor.py --config configs/pipeline_config.json --n-base 3 --budgets 5,10,15,all

# 4-base configs (5 combos)
python scripts/ablation_sensor.py --config configs/pipeline_config.json --n-base 4 --budgets 5,10,15,all
```

### Benefit score correlation analysis (run after ablation)
```bash
python scripts/analyze_benefit_score.py \
    --results-dir checkpoints/ \
    --n-base 1 2 3 4
```

### Results location
- **Logs**: `logs/ablation_sensor_n{N}_b{budgets}_{timestamp}.log` — full stdout with all tables
- **JSON results**: `checkpoints/ablation_sensor_n{N}_results.json` — saved incrementally after each config
- **Benefit analysis**: `checkpoints/benefit_score_analysis.json`

### Config file (`configs/pipeline_config.json`)
Key fields:
```json
{
  "sensors": {
    "known_sensors": ["LeftWrist", "RightAnkle"],
    "new_sensor":    ["RightThigh"]
  },
  "data": {
    "labeled_dir":   "/path/to/lab/data",
    "unlabeled_dir": "/path/to/fl/data"
  },
  "model": {
    "encoder_path": "/path/to/simclr.pt"
  },
  "finetune": {
    "few_shot_samples_per_class": 40,
    "val_split":    0.2,
    "epochs":       100,
    "batch_size":   256,
    "lr":           1e-3,
    "weight_decay": 1e-4
  },
  "active_learning": {
    "pseudo_label_threshold": 0.7,
    "n_clusters":             40
  },
  "output": {
    "checkpoint_dir": "checkpoints/",
    "log_dir":        "logs/"
  }
}
```