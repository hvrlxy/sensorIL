"""
hitl_simulation.py
==================
Human-in-the-Loop simulation layer.

In a real deployment, these functions would receive actual user input
(e.g. via a UI). In experiments, they simulate that input using ground-truth
labels and the parameters in hparams.json under "hitl_simulation".

Separating simulation logic here means:
  - E2/E3 scripts are cleaner and contain only pipeline orchestration.
  - The simulation strategy can be changed (e.g. adding label noise) without
    touching E2/E3.
  - It is clear which parts of the code represent user interactions vs
    automated decisions.

Simulation modes (set in hparams.json hitl_simulation.confirmation_source)
---------------------------------------------------------------------------
  "ground_truth"   Perfect oracle — confirms/denies based on dataset labels.
                   This is what all published experiments use.

  "noisy"          Like ground_truth but with label_noise_rate probability
                   of flipping a confirmed co-occurrence. Reserved for
                   future robustness experiments.

The functions return the same types a real UI would return, so swapping
ground_truth for real input only requires replacing this module.
"""
from __future__ import annotations

import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scripts.misc.config_loader import Config


# ─────────────────────────────────────────────────────────────────────────────
# Co-occurrence confirmation
# ─────────────────────────────────────────────────────────────────────────────

def simulate_cooccurrence_confirmation(
    new_activity: str,
    cooc_results: dict,
    trained_activities: list[str],
    cfg: "Config",
    rng: random.Random | None = None,
) -> tuple[list[str], list[str], list[str]]:
    """
    Simulate the user confirming which fired co-occurrences are real.

    Parameters
    ----------
    new_activity      : activity being added
    cooc_results      : output of check_cooccurrence — {activity: {fires, ...}}
    trained_activities: list of already-trained activities
    cfg               : Config object (reads cooccurrence_graph and hitl_simulation)
    rng               : random.Random instance (for noisy mode reproducibility)

    Returns
    -------
    confirmed : list of correctly confirmed co-occurrences (TP)
    missed    : list of ground-truth co-occurrences not fired (FN)
                  — only returned if no TPs (i.e. complete miss)
    false_pos : list of fired co-occurrences that are not ground-truth (FP)
    """
    sim_cfg    = cfg.HITL_SIMULATION
    source     = sim_cfg.get("confirmation_source", "ground_truth")
    noise_rate = float(sim_cfg.get("label_noise_rate", 0.0))

    gt_cooc   = set(cfg.get_cooccurrences_for(new_activity, trained_activities))
    fired     = {a for a, r in cooc_results.items() if r["fires"]}

    if source == "noisy" and noise_rate > 0.0:
        if rng is None:
            rng = random.Random()
        # Randomly flip some confirmed TPs to unconfirmed (user misses them)
        noisy_fired = set()
        for a in fired:
            if rng.random() < noise_rate:
                pass  # simulate user missing this one
            else:
                noisy_fired.add(a)
        fired = noisy_fired

    confirmed = sorted(fired & gt_cooc)
    false_pos = sorted(fired - gt_cooc)
    # Missed = GT not fired, but only if no TP (full miss scenario)
    missed    = sorted(gt_cooc - fired) if not (fired & gt_cooc) else []

    _print_cooc_summary(new_activity, gt_cooc, fired, confirmed, missed, false_pos)
    return confirmed, missed, false_pos


def _print_cooc_summary(activity, gt_cooc, fired, confirmed, missed, false_pos):
    print(f"\n  Co-occurrence GT   : {sorted(gt_cooc)}")
    print(f"  Fired heads        : {sorted(fired)}")
    print(f"  Confirmed (TP)     : {confirmed}")
    print(f"  Missed    (FN)     : {missed}")
    print(f"  False pos (FP)     : {false_pos}")


# ─────────────────────────────────────────────────────────────────────────────
# Retraining trigger decision
# ─────────────────────────────────────────────────────────────────────────────

def should_retrain_for_fn(
    activity: str,
    cfg: "Config",
) -> bool:
    """
    Decide whether to retrain a head for a missed co-occurrence (FN).

    Returns False for ambiguous activities where retraining is unreliable.
    """
    if activity in cfg.AMBIGUOUS_COOCCURRENCE:
        print(f"  Skipping retrain for '{activity}' (ambiguous co-occurrence)")
        return False
    return True
