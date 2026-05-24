# Active Sensor Increment for Human Activity Recognition

## Problem Statement

Wearable sensor systems for Human Activity Recognition (HAR) are typically trained once on a fixed sensor configuration and deployed. When a new sensor becomes available — whether due to hardware upgrades, changed deployment context, or expanded monitoring needs — the system must be updated. Naively, this requires collecting new labeled data for all activities with the expanded sensor set, which is expensive and disruptive.

We address the **sensor increment problem**: given a trained $n$-sensor system and an unlabeled dataset collected with $n+1$ sensors, improve activity recognition with the new sensor while **minimizing the number of newly labeled samples required**.

The key observation is that the new sensor is **not equally useful for all activities**. Adding a thigh sensor to a wrist+ankle system provides redundant information for distinguishing treadmill walking variants (leg motion is identical across them) but may be critical for separating lying positions or distinguishing standing from sitting. Requesting re-annotation for all $m$ activities wastes effort on classes that won't benefit.

---

## Technical Contributions

We make two distinct contributions that together explain why the proposed approach consistently outperforms the oracle (which retrains all classes with full new labels):

### Contribution 1: Benefit-Guided Active Annotation

We introduce a **benefit score** that identifies which activities will actually improve from adding the new sensor, without requiring any new labeled data. The oracle retrains everything — including classes where the new sensor hurts or adds noise. By selectively retraining only beneficial classes, the proposed approach avoids these degradations.

### Contribution 2: PU Learning with Diverse Unlabeled Negatives

Given the selected classes, we propose a training strategy that exploits the large unlabeled paired dataset as a source of diverse negatives. The oracle retrains with fully labeled data using standard supervised training. We instead use the unlabeled $n+1$-sensor pool as a rich negative source with uncertainty weighting — enabling better-calibrated classifiers with fewer labeled samples.

These contributions are complementary but independently valuable, as shown by the ablation:

| Condition | Selection | Training | Result |
|-----------|-----------|----------|--------|
| Baseline | — | $n$ sensors | lowest |
| Oracle | all $m$ classes | $n+1$, full labels | good but noisy |
| Ablation A | elbow only | standard (no PU) | tests Contribution 1 |
| Ablation B | all classes | PU + FL negatives | tests Contribution 2 |
| **Proposed** | **elbow** | **PU + FL negatives** | **best** |

---

## Contribution 1: Benefit Score

The benefit score combines two signals:

$$\text{benefit}(A) = (1 - F_1(A)) + \text{discriminability}(A)$$

The additive formulation is deliberate — discriminability **boosts** classes with new sensor evidence rather than suppressing classes with only one signal.

### Confusion Signal

$1 - F_1(A)$ measures how much the existing $n$-sensor system struggles with class $A$, computed on a held-out validation set. We use the **binary classifier output** rather than the confusion matrix: a window of class $A$ is "confused" if $A$'s classifier does not fire, regardless of what other classifiers predict. This correctly handles the case where $A$'s windows are absorbed by dominant classes rather than explicitly misclassified.

### New Sensor Discriminability

Even if an activity is confused under $n$ sensors, the new sensor only helps if it provides discriminative signal. We estimate this from the unlabeled $n+1$-sensor data using a **confidence-weighted blend** of two signals:

$$\text{discriminability}(A) = \alpha \cdot d_{\text{direct}}(A) + (1 - \alpha) \cdot d_{\text{opposition}}(A)$$

where $\alpha = \min(1, N_{\text{pseudo}} / N_{\min})$ controls the blend based on pseudo-label availability.

**Direct discriminability** measures how much the new sensor alone separates class $A$ from its confusion targets using pseudo-labeled FL windows:

$$d_{\text{direct}}(A) = \frac{1}{|\mathcal{N}(A)|} \sum_{B \in \mathcal{N}(A)} d\!\left(\bar{e}^{\text{new}}_A,\ \bar{e}^{\text{new}}_B\right)$$

**Opposition discriminability** handles the case where $A$ has few or no pseudo-labels (e.g. the activity is rare or absent from the unlabeled pool). Instead of measuring $A$'s own new-sensor embedding, we measure how well-separated $A$'s confusion targets are from each other in new-sensor space:

$$d_{\text{opposition}}(A) = \frac{1}{\binom{|\mathcal{N}(A)|}{2}} \sum_{B \neq C \in \mathcal{N}(A)} d\!\left(\bar{e}^{\text{new}}_B,\ \bar{e}^{\text{new}}_C\right)$$

If the new sensor separates $A$'s neighborhood well, it will also help discriminate $A$ itself — even without directly observing $A$ in the unlabeled data.

### Class Selection via Elbow Detection

The benefit score induces a ranking of all classes. Rather than using a fixed threshold, we use **elbow detection** on the sorted benefit score curve: the point of maximum curvature identifies the natural cutoff where marginal benefit drops sharply. This is self-adaptive — it selects fewer classes when the score distribution drops off quickly and more when there is a long tail of genuinely beneficial classes.

---

## Contribution 2: PU Learning with Diverse FL Negatives

Once new labels are obtained for the selected classes, we retrain each targeted classifier using the large unlabeled paired dataset as a negative source.

### Training Set Construction

For each targeted class $A$:

- **Certain positives**: new labeled windows of class $A$ ($n+1$ sensors)
- **Certain negatives**: new labeled windows of other targeted classes unrelated to $A$ in the activity hierarchy
- **Uncertain negatives**: pseudo-negatives sampled from a **shared negative pool** built from the unlabeled $n+1$-sensor data, weighted by $w = 1 - P(A \mid x)$

### PU Learning Formulation

The uncertain negatives use a **weighted focal loss** where FL windows that might belong to class $A$ contribute less to the negative gradient:

$$\mathcal{L} = \mathcal{L}_{\text{pos}} + \mathcal{L}_{\text{certain neg}} + \sum_j w_j \cdot \mathcal{L}_{\text{neg}}(x_j)$$

This handles the unknown label problem: if the base classifier is uncertain whether a FL window belongs to class $A$, it gets down-weighted rather than treated as a hard negative.

### Pseudo-Label Stratified Negative Sampling

Rather than blind K-means clustering on the embedding space, we use the base classifiers to **pseudo-label each FL window** by predicted activity, then build a shared negative pool by sampling proportionally from each pseudo-labeled group.

This has two key advantages:

1. **Semantic diversity**: the pool covers all activities present in the FL data (Walking, Standing, Treadmill variants, etc.), ensuring the retrained classifier sees realistic negative examples from across the activity distribution.
2. **Consistent negative distribution**: all targeted classifiers sample from the same shared pool, differing only in which groups are excluded. For class $A$, windows pseudo-labeled as $A$ or any hierarchically related class (parents and children) are excluded before sampling, ensuring negatives are always semantically unrelated to $A$.

The PU weight $w = 1 - P(A \mid x)$ is computed from the same pseudo-labeling pass used to build the pool, so no additional inference is required at sampling time.

### Independence Guarantee

Non-targeted classifiers are completely frozen — their weights, thresholds, and input dimensions are unchanged. Evaluation uses independent per-class binary F1 with multi-label ground truth, so changing one classifier mathematically cannot affect another class's metric.

---

## Evaluation Protocol

| Condition | Sensors | New labels | Description |
|-----------|---------|------------|-------------|
| **Baseline** | $n$ | 0 | existing system |
| **Proposed** | $n+1$ | $K$ (elbow) | our approach |
| **Oracle** | $n+1$ | $m$ (all) | upper bound |

Metrics: independent per-class binary F1, macro F1, weighted F1.

---

## Limitations

The approach works best when:
- The new sensor covers a **different body region** from the base sensors
- The base model has **room for improvement** (macro F1 < 0.75)
- The unlabeled FL data **covers the activity space**

Performance is limited when:
- The base model is already strong and the new sensor is redundant
- The val set is small, giving noisy F1 estimates for the confusion signal
- Targeted classes are semantic siblings where the new sensor doesn't discriminate between them (e.g. treadmill variants at the same speed)

---

## Sensor Configuration Ablation

We ablate over all 75 possible combinations of base sensors and new sensors from {LeftWrist, RightWrist, RightThigh, RightWaist, RightAnkle}, grouped by number of base sensors (1–4). Each config is evaluated at annotation budgets $K \in \{5, 10, 15, \text{elbow}\}$.

---

## Benefit Score Validation

We empirically validate the benefit score by computing Spearman correlation between benefit score and actual $\Delta F_1$ across all targeted classes and sensor configs, comparing four formulations:

| Formula | Expression |
|---------|-----------|
| Additive (ours) | $(1-F_1) + \text{discriminability}$ |
| Product | $(1-F_1) \times \text{discriminability}$ |
| Confusion only | $1 - F_1$ |
| Discriminability only | discriminability |

---

## Repository Structure

```
sensorIL/
├── configs/
│   └── pipeline_config.json
├── scripts/
│   ├── simclr_encoder.py             # Frozen SimCLR encoder wrapper
│   ├── dataset.py                    # SensorDataset, UnlabeledFLDataset
│   ├── cooccurrence.py               # Activity hierarchy, multi-label encoding
│   ├── train_base.py                 # ParallelBinaryClassifiers, train_base()
│   ├── detect_confusion.py           # Step 2: rank classes by F1, confusion pairs
│   ├── estimate_benefit.py           # Step 3: benefit score, elbow detection
│   ├── incremental_ft.py             # Step 5: PU learning, diverse FL negatives
│   ├── calibrate_thresholds.py       # Per-class threshold calibration [0.2, 0.8]
│   ├── evaluate.py                   # Independent per-class binary F1
│   ├── run_pipeline.py               # Full pipeline (Steps 1-6)
│   ├── ablation_budget.py            # Annotation budget vs F1 curve
│   ├── ablation_sensor.py            # Sensor config ablation (75 combos × 4 budgets)
│   └── analyze_benefit_score.py      # Benefit score correlation analysis
├── checkpoints/                      # Model checkpoints + JSON results
└── logs/                             # Full stdout logs per run
```

---

## How to Run

### Full pipeline
```bash
python scripts/run_pipeline.py --config configs/pipeline_config.json
```

### Sensor config + budget ablation
```bash
# Run one n_base at a time
python scripts/ablation_sensor.py --config configs/pipeline_config.json --n-base 1 --budgets 5,10,15,all
python scripts/ablation_sensor.py --config configs/pipeline_config.json --n-base 2 --budgets 5,10,15,all
python scripts/ablation_sensor.py --config configs/pipeline_config.json --n-base 3 --budgets 5,10,15,all
python scripts/ablation_sensor.py --config configs/pipeline_config.json --n-base 4 --budgets 5,10,15,all
```

### Benefit score analysis (after ablation)
```bash
python scripts/analyze_benefit_score.py --results-dir checkpoints/ --n-base 1 2 3 4
```

### Config reference
```json
{
  "sensors": {"known_sensors": ["LeftWrist", "RightAnkle"], "new_sensor": ["RightThigh"]},
  "data": {"labeled_dir": "/path/to/lab", "unlabeled_dir": "/path/to/fl"},
  "model": {"encoder_path": "/path/to/simclr.pt"},
  "finetune": {"few_shot_samples_per_class": 40, "val_split": 0.2,
               "epochs": 100, "batch_size": 256, "lr": 1e-3, "weight_decay": 1e-4},
  "active_learning": {"pseudo_label_threshold": 0.7, "n_clusters": 40},
  "output": {"checkpoint_dir": "checkpoints/", "log_dir": "logs/"}
}
```