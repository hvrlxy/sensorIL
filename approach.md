# Active Sensor Increment for Human Activity Recognition

## Problem Statement

Wearable sensor systems for Human Activity Recognition (HAR) are typically trained once on a fixed sensor configuration and deployed. When a new sensor becomes available — whether due to hardware upgrades, changed deployment context, or expanded monitoring needs — the system must be updated. Naively, this requires collecting new labeled data for all activities with the expanded sensor set, which is expensive and disruptive.

We address the **sensor increment problem**: given a trained $n$-sensor system and an unlabeled dataset collected with $n+1$ sensors, improve activity recognition with the new sensor while **minimizing the number of newly labeled samples required**.

The key observation is that the new sensor is **not equally useful for all activities**. Adding a thigh sensor to a wrist+ankle system provides redundant information for distinguishing treadmill walking variants (leg motion is identical across them) but may be critical for separating lying positions or distinguishing standing from sitting. Requesting re-annotation for all $m$ activities wastes effort on classes that won't benefit.

---

## Core Contribution: Benefit-Guided Active Annotation

The central question is: **for which activities should we request new labeled data?**

We answer this with a benefit score that captures two complementary signals:

$$\text{benefit}(A) = \underbrace{(1 - F_1(A))}_{\text{confusion signal}} \times \underbrace{\text{discriminability}(A)}_{\text{new sensor signal}}$$

### Confusion Signal

$1 - F_1(A)$ measures how much the existing $n$-sensor system struggles with class $A$.

### New Sensor Discriminability

Even if an activity is confused under $n$ sensors, the new sensor only helps if it provides discriminative signal. We estimate this directly from the unlabeled $n+1$-sensor data: for each confused class $A$ and its confused neighbors $\mathcal{N}(A)$, we measure how much the new sensor alone separates them:

$$\text{discriminability}(A) = \frac{1}{|\mathcal{N}(A)|} \sum_{B \in \mathcal{N}(A)} d\!\left(\bar{e}^{\text{new}}_A,\ \bar{e}^{\text{new}}_B\right)$$

where $\bar{e}^{\text{new}}_A$ is the mean embedding of the new sensor stream for class $A$, using pseudo-labels from the base classifier on the unlabeled data. This is computed **without any new labels** — the unlabeled paired data is sufficient to estimate the discriminative value of the new sensor.

Classes with positive benefit score are selected for re-annotation, and the system requests the minimum necessary labels.

---

## Incremental Fine-Tuning with Provable Independence

Once new labels are obtained for the selected classes, each targeted classifier is retrained with a larger input (including the new sensor). The central challenge is constructing an appropriate training set — **what are the negatives for a targeted class when most labeled data does not include the new sensor?**

We propose a **Positive-Unlabeled (PU) learning formulation**:

- **Certain positives**: new labeled windows of the targeted class ($n+1$ sensors)
- **Certain negatives**: new labeled windows of other targeted classes that are unrelated in the activity hierarchy
- **Uncertain negatives**: pseudo-negatives sampled from the unlabeled $n+1$-sensor pool, weighted by confidence $w = 1 - P(\text{class} \mid x)$

The uncertainty weighting is crucial: unlabeled windows where the base classifier is unsure whether they belong to the targeted class contribute less to the negative gradient, preventing the classifier from being misled by mislabeled pseudo-negatives.

To ensure diversity in the unlabeled negatives — avoiding dominance by the most common activities — we perform **stratified sampling via K-means clustering** on the $n$-sensor embedding space, sampling proportionally from each cluster.

### Independence Guarantee

A critical design requirement is that retraining targeted classifiers must not degrade non-targeted classes. We achieve this through independent binary classification: each class has its own classifier, and **non-targeted classifiers are completely frozen** during incremental fine-tuning. Evaluation uses per-class binary F1, ensuring that a change in one classifier cannot mathematically affect another class's metric. This gives a strong guarantee: activities that do not benefit from the new sensor experience zero performance change.

---

## What the Approach Does Not Do

To be precise about the scope:

- We do **not** claim the new sensor helps all activities — only those identified by the benefit score
- We do **not** assume any particular encoder architecture — the approach is compatible with any frozen feature extractor
- We do **not** require the unlabeled data to cover all activities — pseudo-labeling naturally assigns low confidence to activities poorly represented in the unlabeled pool, reducing their influence

---

## Annotation Cost vs Performance

The benefit score induces a natural ranking of activities by expected gain from re-annotation. By varying the number of classes selected ($K$), we obtain an **annotation efficiency curve** showing macro F1 as a function of labeling cost. Empirically, the top $K$ classes identified by the benefit score achieve near-oracle performance at a fraction of the full annotation cost, with the optimal operating point typically around 25–35% of all classes.

---

## Sensor Configuration Generalization

The benefit of adding a new sensor depends fundamentally on what information it provides relative to the existing sensor set. We characterize this through a **systematic ablation** over all 75 possible sensor increment configurations from a 5-sensor body-worn IMU system, varying both the number of base sensors (1–4) and the identity of the new sensor. Key findings:

- The absolute performance gain from adding one sensor **decreases as the base sensor set grows** — consistent with diminishing marginal returns on information
- Gains are largest when the new sensor covers a **new body region** not represented in the base set
- The benefit score reliably predicts which configurations produce meaningful gains, enabling the system to automatically recommend whether a new sensor is worth deploying at all

