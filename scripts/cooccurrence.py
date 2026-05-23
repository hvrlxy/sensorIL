"""
cooccurrence.py

Co-occurrence hierarchy for multi-label encoding.

COOCCURRENCE[class] = list of parent classes
A window of class A should also be labeled as all ancestors.

Provides:
  - get_ancestors(class_name): all ancestor classes
  - get_multilabel(class_name, all_classes): binary vector
  - get_specificity(class_name): number of ancestors (more = more specific)
  - are_related(class_a, class_b): True if one is ancestor of other
"""

COOCCURRENCE = {
    # ── Seed activities ───────────────────────────────────────────────────────
    "Walking":                                      [],
    "Sitting_Still":                                [],
    "Standing_Still":                               [],

    # ── Walking variants ──────────────────────────────────────────────────────
    "Treadmill_2mph_Lab":                           ["Walking"],
    "Treadmill_3mph_Free_Walk_Lab":                 ["Walking"],
    "Treadmill_3mph_Hands_Pockets_Lab":             ["Walking"],
    "Treadmill_3mph_Conversation_Lab":              ["Walking"],
    "Treadmill_3mph_Phone_Lab":                     ["Walking"],
    "Treadmill_3mph_Drink_Lab":                     ["Walking"],
    "Treadmill_3mph_Briefcase_Lab":                 ["Walking"],
    "Treadmill_4mph_Lab":                           ["Walking"],
    "Treadmill_5_5mph_Lab":                         ["Walking"],
    "Walking_Up_Stairs":                            ["Walking"],
    "Walking_Down_Stairs":                          ["Walking"],

    # ── Sitting variants ──────────────────────────────────────────────────────
    "Sitting_With_Movement":                        ["Sitting_Still"],
    "Sit_Recline_Talk_Lab":                         ["Sitting_Still", "Sitting_With_Movement"],
    "Sit_Recline_Web_Browse_Lab":                   ["Sitting_Still", "Sitting_With_Movement"],
    "Sit_Typing_Lab":                               ["Sitting_Still", "Sitting_With_Movement"],
    "Sit_Writing_Lab":                              ["Sitting_Still", "Sitting_With_Movement"],
    "Machine_Chest_Press_Lab":                      ["Sitting_Still", "Sitting_With_Movement"],
    "Machine_Leg_Press_Lab":                        ["Sitting_Still", "Sitting_With_Movement"],

    # ── Standing variants ─────────────────────────────────────────────────────
    "Standing_With_Movement":                       [],
    "Stand_Conversation_Lab":                       ["Standing_Still", "Standing_With_Movement"],
    "Stand_Shelf_Load_Lab":                         ["Standing_Still", "Standing_With_Movement"],
    "Stand_Shelf_Unload_Lab":                       ["Standing_Still", "Standing_With_Movement"],
    "Organizing_Shelf_Cabinet":                     ["Standing_Still", "Standing_With_Movement"],
    "Arm_Curls_Lab":                                ["Standing_Still", "Standing_With_Movement"],
    "Chopping_Food_Lab":                            ["Standing_Still", "Standing_With_Movement"],
    "Folding_Clothes":                              ["Standing_Still"],
    "Washing_Dishes_Lab":                           ["Standing_Still"],

    # ── Chores ────────────────────────────────────────────────────────────────
    "Sweeping":                                     ["Walking", "Standing_With_Movement"],
    "Vacuuming":                                    ["Walking", "Standing_With_Movement"],
    "Playing_Frisbee":                              ["Walking", "Standing_With_Movement"],

    # ── Cycling ───────────────────────────────────────────────────────────────
    "Stationary_Biking_300_Lab":                    [],

    # ── Lying ─────────────────────────────────────────────────────────────────
    "Lying_On_Back_Lab":                            [],
    "Lying_On_Left_Side_Lab":                       [],
    "Lying_On_Right_Side_Lab":                      [],
    "Lying_On_Stomach_Lab":                         [],
    "Ab_Crunches_Lab":                              [],
    "Push_Up_Lab":                                  [],
}


def get_ancestors(class_name):
    """Return all ancestor classes (direct and transitive)."""
    ancestors = set()
    parents   = COOCCURRENCE.get(class_name, [])
    for parent in parents:
        ancestors.add(parent)
        ancestors.update(get_ancestors(parent))
    return ancestors


def get_all_related(class_name):
    """Return class itself + all ancestors."""
    return {class_name} | get_ancestors(class_name)


def get_multilabel(class_name, all_classes):
    """
    Return binary label vector for a window of class_name.
    Label is 1 for class_name and all its ancestors.

    Args:
        class_name  : the ground truth class of this window
        all_classes : ordered list of all class names

    Returns: list of 0/1 of length len(all_classes)
    """
    positive_classes = get_all_related(class_name)
    return [1 if c in positive_classes else 0 for c in all_classes]


def get_specificity(class_name):
    """Number of ancestors — more ancestors = more specific class."""
    return len(get_ancestors(class_name))


def are_related(class_a, class_b):
    """True if class_a is an ancestor of class_b or vice versa."""
    return (class_a in get_ancestors(class_b) or
            class_b in get_ancestors(class_a))


def predict_most_specific(scores, class_names, threshold=0.5,
                          low_clip=0.2, high_clip=0.8):
    """
    Independent threshold-based prediction with hierarchy-aware tiebreak.

    Steps:
      1. Clip scores to [low_clip, high_clip]
      2. Find all classes that fire (score > threshold)
      3. If none fire: fallback to highest score
      4. If one fires: return it
      5. If multiple fire:
           - Group into related clusters (connected by ancestor/descendant)
           - Within each cluster: pick most specific (most ancestors)
           - Across clusters: pick highest score winner from each cluster
           - Final: highest score among cluster winners

    This ensures non-related classes never compete via specificity —
    only related classes (e.g. Walking vs Treadmill) use hierarchy tiebreak.

    Args:
        scores      : (n_classes,) array of sigmoid scores
        class_names : list of class names
        threshold   : firing threshold (default 0.5)
        low_clip    : minimum score (default 0.2)
        high_clip   : maximum score (default 0.8)

    Returns: predicted class index
    """
    import numpy as np
    scores = np.clip(scores, low_clip, high_clip)

    fired = [i for i in range(len(class_names)) if scores[i] > threshold]

    if len(fired) == 0:
        return int(scores.argmax())

    if len(fired) == 1:
        return fired[0]

    # Group fired classes into related clusters
    # Two classes are in the same cluster if they are related (ancestor/descendant)
    clusters = []
    assigned = set()

    for i in fired:
        if i in assigned:
            continue
        cluster = {i}
        for j in fired:
            if j != i and are_related(class_names[i], class_names[j]):
                cluster.add(j)
                assigned.add(j)
        assigned.add(i)
        clusters.append(cluster)

    # Within each cluster: pick most specific (most ancestors), tiebreak by score
    cluster_winners = []
    for cluster in clusters:
        winner = max(cluster,
                     key=lambda i: (get_specificity(class_names[i]), scores[i]))
        cluster_winners.append(winner)

    # Across clusters: pick by highest score
    return max(cluster_winners, key=lambda i: scores[i])


if __name__ == "__main__":
    # Quick sanity check
    print("Ancestors of Treadmill_3mph_Drink_Lab:",
          get_ancestors("Treadmill_3mph_Drink_Lab"))
    print("Ancestors of Sit_Typing_Lab:",
          get_ancestors("Sit_Typing_Lab"))
    print("Specificity of Walking:", get_specificity("Walking"))
    print("Specificity of Treadmill_3mph_Drink_Lab:",
          get_specificity("Treadmill_3mph_Drink_Lab"))
    print("are_related(Walking, Treadmill_3mph_Drink_Lab):",
          are_related("Walking", "Treadmill_3mph_Drink_Lab"))
    print("are_related(Walking, Sit_Typing_Lab):",
          are_related("Walking", "Sit_Typing_Lab"))

    classes = ["Walking", "Treadmill_3mph_Drink_Lab", "Sit_Typing_Lab"]
    print("\nMulti-label for Treadmill_3mph_Drink_Lab:",
          get_multilabel("Treadmill_3mph_Drink_Lab", classes))