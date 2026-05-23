"""
Evaluate all 9 sensor increment configurations.
For each config: E1 (known only) / Imputed (known+translated) / Oracle (all sensors)
Uses grouped 7-class evaluation. Reports per-config and aggregate results.
"""
import sys, os
sys.path.insert(0, os.path.expanduser("~/sensorIL/scripts"))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

SEED     = 42
N_RUNS   = 5
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LAB_DIR  = "/mnt/storage/hitl_experiments/paaws_tuned"
PARTICIPANT = "DS_11"
OUT_DIR  = "output/multi_sensor_eval"

torch.manual_seed(SEED); np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

# ── Lab sensor order ──────────────────────────────────────────────────────────
LAB_SENSOR_ORDER = ["LeftAnkle","LeftThigh","LeftWaist","LeftWrist",
                    "RightAnkle","RightThigh","RightWaist","RightWrist"]
LAB_IDX = {s: i for i, s in enumerate(LAB_SENSOR_ORDER)}
WRIST_L  = LAB_IDX["LeftWrist"]   # 3
ANKLE_R  = LAB_IDX["RightAnkle"]  # 4
THIGH_R  = LAB_IDX["RightThigh"]  # 5

# ── Sensor configs ─────────────────────────────────────────────────────────────
# Each entry: (config_name, known_lab_indices, target_lab_index, translator_path)
CONFIGS = [
    # Scenario 1: 1→1
    ("Wrist→Ankle",   [WRIST_L],         ANKLE_R, "output/translators/wrist_to_ankle/translator.pt"),
    ("Wrist→Thigh",   [WRIST_L],         THIGH_R, "output/translators/wrist_to_thigh/translator.pt"),
    ("Ankle→Wrist",   [ANKLE_R],         WRIST_L, "output/translators/ankle_to_wrist/translator.pt"),
    ("Ankle→Thigh",   [ANKLE_R],         THIGH_R, "output/translators/ankle_to_thigh/translator.pt"),
    ("Thigh→Wrist",   [THIGH_R],         WRIST_L, "output/translators/thigh_to_wrist/translator.pt"),
    ("Thigh→Ankle",   [THIGH_R],         ANKLE_R, "output/translators/thigh_to_ankle/translator.pt"),
    # Scenario 2: 2→1
    ("Wrist+Ankle→Thigh", [WRIST_L,ANKLE_R], THIGH_R, "output/translators/wrist_ankle_to_thigh/translator.pt"),
    ("Wrist+Thigh→Ankle", [WRIST_L,THIGH_R], ANKLE_R, "output/translators/wrist_thigh_to_ankle/translator.pt"),
    ("Ankle+Thigh→Wrist", [ANKLE_R,THIGH_R], WRIST_L, "output/translators/ankle_thigh_to_wrist/translator.pt"),
]

# ── Activity groups ───────────────────────────────────────────────────────────
GROUPS = {
    "Lying":        ["Lying_On_Back_Lab","Lying_On_Left_Side_Lab",
                     "Lying_On_Right_Side_Lab","Lying_On_Stomach_Lab"],
    "Sitting":      ["Sit_Recline_Talk_Lab","Sit_Recline_Web_Browse_Lab",
                     "Sit_Typing_Lab","Sit_Writing_Lab",
                     "Sitting_Still","Sitting_With_Movement"],
    "Standing":     ["Stand_Conversation_Lab","Stand_Shelf_Load_Lab",
                     "Stand_Shelf_Unload_Lab","Standing_Still",
                     "Standing_With_Movement"],
    "Walking":      ["Treadmill_2mph_Lab","Walking",
                     "Walking_Down_Stairs","Walking_Up_Stairs"],
    "Running/Fast": ["Treadmill_4mph_Lab","Treadmill_5_5mph_Lab"],
    "Cycling":      ["Cycling_Active_Pedaling_Regular_Bicycle",
                     "Stationary_Biking_300_Lab"],
}

# ── Load data once ────────────────────────────────────────────────────────────
print("Loading data and encoders...")
from scripts.misc.helpers import create_dataset_file_split
from scripts.misc.config_loader import cfg
from scripts.misc.encoder import load_encoders_from_cfg, extract_all_features
from scripts.misc.signal_translator import load_translator, impute_with_translator

np_train, np_val, np_test, label_dict = create_dataset_file_split(
    LAB_DIR, [PARTICIPANT], cfg.SEED)
X_tr, y_tr = np_train[0], np_train[1]
X_vl, y_vl = np_val[0],   np_val[1]
X_te, y_te = np_test[0],  np_test[1]
print(f"  Train={X_tr.shape}  Val={X_vl.shape}  Test={X_te.shape}")

encoders = load_encoders_from_cfg(cfg)

# 41-class mapping
ALL_ACTS    = sorted(label_dict.keys())
int2cls     = {label_dict[a]: i for i, a in enumerate(ALL_ACTS)}
eval_labels = list(range(len(ALL_ACTS)))

def to_cls(y_int):
    return np.array([int2cls[int(v)] for v in y_int])

# ── Feature helpers ───────────────────────────────────────────────────────────
def get_stream_name(lab_idx):
    return LAB_SENSOR_ORDER[lab_idx]

def extract_feats(X_raw, lab_indices):
    """Extract and L2-normalize SimCLR features for given lab sensor indices."""
    names = [get_stream_name(i) for i in lab_indices]
    Z = extract_all_features(X_raw[:, :, lab_indices, :], encoders,
                              cfg.STREAM_TO_ENCODER, names, batch_size=256)
    norms = np.linalg.norm(Z, axis=2, keepdims=True)
    Z = np.where(norms > 1e-6, Z / norms.clip(1e-6), Z)  # keep zeros as zeros
    return Z.reshape(len(Z), -1)

def impute_and_extract(X_raw, translator, known_lab_idx, target_lab_idx):
    """Impute target stream, z-score normalize, extract L2-normalized SimCLR."""
    known_names = [get_stream_name(i) for i in known_lab_idx]
    target_name = get_stream_name(target_lab_idx)
    all_names   = known_names + [target_name]
    X_known = X_raw[:, :, known_lab_idx, :]
    X_imp   = impute_with_translator(
        translator, X_known,
        known_indices=list(range(len(known_lab_idx))),
        n_streams_total=len(known_lab_idx)+1,
        encoders=encoders, stream_names=known_names,
        stream_to_encoder=cfg.STREAM_TO_ENCODER)
    # Z-score normalize imputed target stream per window
    # Removes amplitude bias before SimCLR — makes distribution closer to real
    n_streams = X_imp.shape[2]
    target_pos = n_streams - 1  # imputed target is always last
    imp_tgt = X_imp[:, :, target_pos, :]  # (N, T, C)
    mu  = imp_tgt.mean(axis=1, keepdims=True)
    std = imp_tgt.std(axis=1, keepdims=True).clip(1e-6)
    X_imp[:, :, target_pos, :] = (imp_tgt - mu) / std
    Z = extract_all_features(X_imp, encoders, cfg.STREAM_TO_ENCODER,
                              all_names, batch_size=256)
    norms = np.linalg.norm(Z, axis=2, keepdims=True)
    Z = np.where(norms > 1e-6, Z / norms.clip(1e-6), Z)
    return Z.reshape(len(Z), -1)

# ── Classifier ────────────────────────────────────────────────────────────────
def train_eval(Z_tr, Z_vl, Z_te, y_tr, y_vl, y_te, tag,
               epochs=200, lr=1e-3, batch_size=256, patience=20, seed=SEED):
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr = torch.from_numpy(Z_tr.astype('float32')).to(DEVICE)
    Xvl = torch.from_numpy(Z_vl.astype('float32')).to(DEVICE)
    Xte = torch.from_numpy(Z_te.astype('float32')).to(DEVICE)
    ytr = torch.from_numpy(to_cls(y_tr)).long().to(DEVICE)
    yvl = torch.from_numpy(to_cls(y_vl)).long().to(DEVICE)
    D   = Xtr.shape[1]
    model = nn.Sequential(
        nn.Linear(D, 1024), nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(512, len(ALL_ACTS))
    ).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_vl, best_state, wait = float('inf'), None, 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), batch_size):
            idx = perm[i:i+batch_size]
            loss = F.cross_entropy(model(Xtr[idx]), ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            vl = F.cross_entropy(model(Xvl), yvl).item()
        if vl < best_vl:
            best_vl = vl
            best_state = {k: v.cpu().clone() for k,v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience: break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(Xte).argmax(dim=1).cpu().numpy()
    y_cls = to_cls(y_te)
    macro = float(f1_score(y_cls, pred, average='macro', labels=eval_labels, zero_division=0))
    f1s   = {ALL_ACTS[gi]: float(f1_score(y_cls==gi, pred==gi, zero_division=0))
             for gi in eval_labels}
    return f1s, macro, model

def eval_only(model, Z_te, y_te):
    Xte = torch.from_numpy(Z_te.astype('float32')).to(DEVICE)
    model.eval()
    with torch.no_grad():
        pred = model(Xte).argmax(dim=1).cpu().numpy()
    y_cls = to_cls(y_te)
    macro = float(f1_score(y_cls, pred, average='macro', labels=eval_labels, zero_division=0))
    f1s   = {ALL_ACTS[gi]: float(f1_score(y_cls==gi, pred==gi, zero_division=0))
             for gi in eval_labels}
    return f1s, macro

# ── Sibling translators for full 3-stream Task 2 (1→1 configs only) ──────────
# For each 1→1 config: maps to the other translator from the same known sensor
# so we can impute ALL missing streams from just the E1 sensor
SIBLING_TRANSLATORS = {
    "Wrist→Ankle":  ("output/translators/wrist_to_thigh/translator.pt",  THIGH_R),
    "Wrist→Thigh":  ("output/translators/wrist_to_ankle/translator.pt",  ANKLE_R),
    "Ankle→Wrist":  ("output/translators/ankle_to_thigh/translator.pt",  THIGH_R),
    "Ankle→Thigh":  ("output/translators/ankle_to_wrist/translator.pt",  WRIST_L),
    "Thigh→Wrist":  ("output/translators/thigh_to_ankle/translator.pt",  ANKLE_R),
    "Thigh→Ankle":  ("output/translators/thigh_to_wrist/translator.pt",  WRIST_L),
}

results = {}  # config_name → results dict

for cfg_name, known_idx, target_idx, translator_path in CONFIGS:
    if not os.path.exists(translator_path):
        print(f"\n[SKIP] {cfg_name} — translator not found: {translator_path}")
        continue

    print(f"\n{'='*65}")
    print(f"CONFIG: {cfg_name}")
    print(f"{'='*65}")

    translator = load_translator(translator_path)
    all_idx    = known_idx + [target_idx]  # all sensor indices for oracle

    print("  Extracting features...")
    # E1: known sensors only (SimCLR)
    Z_e1_tr  = extract_feats(X_tr, known_idx)
    Z_e1_vl  = extract_feats(X_vl, known_idx)
    Z_e1_te  = extract_feats(X_te, known_idx)
    # Oracle: all sensors (SimCLR)
    Z_ora_tr = extract_feats(X_tr, all_idx)
    Z_ora_vl = extract_feats(X_vl, all_idx)
    Z_ora_te = extract_feats(X_te, all_idx)
    # Imputed: known + translated target (L2-normalized SimCLR)
    Z_imp_tr = impute_and_extract(X_tr, translator, known_idx, target_idx)
    Z_imp_vl = impute_and_extract(X_vl, translator, known_idx, target_idx)
    Z_imp_te = impute_and_extract(X_te, translator, known_idx, target_idx)
    print(f"  Feature dims: E1={Z_e1_tr.shape[1]}  "
          f"Imp={Z_imp_tr.shape[1]}  Ora={Z_ora_tr.shape[1]}")

    # Task 2 imputed: real known streams + imputed target SimCLR
    # Oracle model trained on real streams — keep known streams real
    def make_t2_imp_feats(X_raw, translator, known_lab_idx, target_lab_idx, all_lab_idx):
        known_names = [get_stream_name(i) for i in known_lab_idx]
        X_known_raw = X_raw[:, :, known_lab_idx, :]
        X_imp = impute_with_translator(
            translator, X_known_raw,
            known_indices=list(range(len(known_lab_idx))),
            n_streams_total=len(known_lab_idx)+1,
            encoders=encoders, stream_names=known_names,
            stream_to_encoder=cfg.STREAM_TO_ENCODER)
        # Z-score normalize imputed target before SimCLR
        imp_tgt = X_imp[:, :, -1, :]
        mu  = imp_tgt.mean(axis=1, keepdims=True)
        std = imp_tgt.std(axis=1, keepdims=True).clip(1e-6)
        imp_tgt_norm = (imp_tgt - mu) / std
        # Build full array with real known + normalized imputed target
        X_full = X_raw.copy()
        X_full[:, :, target_lab_idx, :] = imp_tgt_norm
        return extract_feats(X_full, all_lab_idx)

    Z_imp_te_t2 = make_t2_imp_feats(X_te, translator, known_idx, target_idx, all_idx)

    mf1s = {"E1": [], "Imputed": [], "Oracle": [],
            "T2_Oracle": [], "T2_Zero": [], "T2_Imputed": [],
            "T2_FullOra": [], "T2_FullZero": [], "T2_FullImp": []}
    f1s_runs = {"E1": [], "Imputed": [], "Oracle": []}

    # Zero-masked test features for Task 2
    # Use mean training embedding as neutral placeholder (zero SimCLR is OOD)
    Z_all_tr = extract_all_features(
        X_tr[:, :, all_idx, :], encoders,
        cfg.STREAM_TO_ENCODER,
        [get_stream_name(i) for i in all_idx], batch_size=256)  # (N, S, D)
    Z_all_tr = Z_all_tr / np.linalg.norm(Z_all_tr, axis=2, keepdims=True).clip(1e-6)
    mean_emb = Z_all_tr.mean(axis=0)  # (S, D) — mean embedding per stream
    target_pos = len(known_idx)  # position of target stream in all_idx

    Z_all_te = extract_all_features(
        X_te[:, :, all_idx, :], encoders,
        cfg.STREAM_TO_ENCODER,
        [get_stream_name(i) for i in all_idx], batch_size=256)  # (N, S, D)
    Z_all_te = Z_all_te / np.linalg.norm(Z_all_te, axis=2, keepdims=True).clip(1e-6)
    Z_zero_arr = Z_all_te.copy()
    Z_zero_arr[:, target_pos, :] = mean_emb[target_pos]
    Z_zero_te = Z_zero_arr.reshape(len(Z_zero_arr), -1)

    # Full 3-sensor Task 2 (1→1 only): oracle trained on all 3 streams
    all3_idx = [WRIST_L, ANKLE_R, THIGH_R]
    do_full  = cfg_name in SIBLING_TRANSLATORS
    if do_full:
        sib_path, sib_idx = SIBLING_TRANSLATORS[cfg_name]
        if os.path.exists(sib_path):
            sib_translator = load_translator(sib_path)
            Z_full_ora_tr = extract_feats(X_tr, all3_idx)
            Z_full_ora_vl = extract_feats(X_vl, all3_idx)
            Z_full_ora_te = extract_feats(X_te, all3_idx)
            # Full zero: mask both missing streams
            missing_idx = [i for i in all3_idx if i not in known_idx]
            Z_all3_tr = extract_all_features(
                X_tr[:, :, all3_idx, :], encoders, cfg.STREAM_TO_ENCODER,
                [get_stream_name(i) for i in all3_idx], batch_size=256)
            Z_all3_tr = Z_all3_tr / np.linalg.norm(Z_all3_tr, axis=2, keepdims=True).clip(1e-6)
            mean_emb3 = Z_all3_tr.mean(axis=0)  # (3, D)
            Z_all3_te = extract_all_features(
                X_te[:, :, all3_idx, :], encoders, cfg.STREAM_TO_ENCODER,
                [get_stream_name(i) for i in all3_idx], batch_size=256)
            Z_all3_te = Z_all3_te / np.linalg.norm(Z_all3_te, axis=2, keepdims=True).clip(1e-6)
            Z_fz = Z_all3_te.copy()
            for mi in missing_idx:
                mi_pos = all3_idx.index(mi)
                Z_fz[:, mi_pos, :] = mean_emb3[mi_pos]
            Z_full_zero_te = Z_fz.reshape(len(Z_fz), -1)
            # Full imputed: SimCLR of both imputed streams (matches oracle model dim)
            X_imp1 = impute_with_translator(
                translator, X_te[:,:,known_idx,:],
                known_indices=list(range(len(known_idx))),
                n_streams_total=len(known_idx)+1,
                encoders=encoders,
                stream_names=[get_stream_name(i) for i in known_idx],
                stream_to_encoder=cfg.STREAM_TO_ENCODER)
            X_imp2 = impute_with_translator(
                sib_translator, X_te[:,:,known_idx,:],
                known_indices=list(range(len(known_idx))),
                n_streams_total=len(known_idx)+1,
                encoders=encoders,
                stream_names=[get_stream_name(i) for i in known_idx],
                stream_to_encoder=cfg.STREAM_TO_ENCODER)
            # Build full 3-stream array with both imputed streams
            X_full_imp = X_te.copy()
            X_full_imp[:, :, target_idx, :] = X_imp1[:, :, -1, :]
            X_full_imp[:, :, sib_idx, :]    = X_imp2[:, :, -1, :]
            Z_full_imp_te = extract_feats(X_full_imp, all3_idx)
            print(f"  Full-imp dim: {Z_full_imp_te.shape[1]}")
        else:
            print(f"  [SKIP full-imp] sibling not found: {sib_path}")
            do_full = False

    for run in range(N_RUNS):
        seed = SEED + run
        print(f"  Run {run+1}/{N_RUNS}...", end=" ", flush=True)
        f_e1,  m_e1,  _         = train_eval(Z_e1_tr,  Z_e1_vl,  Z_e1_te,  y_tr, y_vl, y_te, f"E1-{run}",  seed=seed)
        f_imp, m_imp, _         = train_eval(Z_imp_tr, Z_imp_vl, Z_imp_te, y_tr, y_vl, y_te, f"Imp-{run}", seed=seed)
        f_ora, m_ora, model_ora = train_eval(Z_ora_tr, Z_ora_vl, Z_ora_te, y_tr, y_vl, y_te, f"Ora-{run}", seed=seed)
        # Task 2: current config oracle model
        _, m_ora2  = eval_only(model_ora, Z_ora_te,     y_te)
        _, m_zero2 = eval_only(model_ora, Z_zero_te,    y_te)
        _, m_imp2  = eval_only(model_ora, Z_imp_te_t2,  y_te)
        # Task 2 full (1→1 only): 3-stream oracle model
        if do_full:
            _, m_full_ora, model_full = train_eval(
                Z_full_ora_tr, Z_full_ora_vl, Z_full_ora_te,
                y_tr, y_vl, y_te, f"FullOra-{run}", seed=seed)
            _, m_full_zero = eval_only(model_full, Z_full_zero_te, y_te)
            _, m_full_imp  = eval_only(model_full, Z_full_imp_te,  y_te)
        else:
            m_full_ora = m_full_zero = m_full_imp = float('nan')

        print(f"T1: E1={m_e1:.3f} Imp={m_imp:.3f} Ora={m_ora:.3f} | "
              f"T2: Zero={m_zero2:.3f} Imp={m_imp2:.3f}"
              + (f" | T2-Full: Zero={m_full_zero:.3f} Imp={m_full_imp:.3f}" if do_full else ""))
        mf1s["E1"].append(m_e1);           f1s_runs["E1"].append(f_e1)
        mf1s["Imputed"].append(m_imp);     f1s_runs["Imputed"].append(f_imp)
        mf1s["Oracle"].append(m_ora);      f1s_runs["Oracle"].append(f_ora)
        mf1s["T2_Oracle"].append(m_ora2)
        mf1s["T2_Zero"].append(m_zero2)
        mf1s["T2_Imputed"].append(m_imp2)
        mf1s["T2_FullOra"].append(m_full_ora)
        mf1s["T2_FullZero"].append(m_full_zero)
        mf1s["T2_FullImp"].append(m_full_imp)

    results[cfg_name] = {
        "macro_means": {k: float(np.nanmean(v)) for k,v in mf1s.items()},
        "macro_stds":  {k: float(np.nanstd(v))  for k,v in mf1s.items()},
        "f1s_runs":    f1s_runs,
        "scenario":    "1→1" if len(known_idx)==1 else "2→1",
        "has_full":    do_full,
    }
    r = results[cfg_name]["macro_means"]
    s = results[cfg_name]["macro_stds"]
    print(f"  T1: E1={r['E1']:.4f}±{s['E1']:.4f}  "
          f"Imp={r['Imputed']:.4f}±{s['Imputed']:.4f}  "
          f"Ora={r['Oracle']:.4f}±{s['Oracle']:.4f}  "
          f"Δimp={r['Imputed']-r['E1']:+.4f}")
    print(f"  T2: Ora={r['T2_Oracle']:.4f}  "
          f"Zero={r['T2_Zero']:.4f}  "
          f"Imp={r['T2_Imputed']:.4f}  "
          f"Δimp={r['T2_Imputed']-r['T2_Zero']:+.4f}")
    if do_full:
        print(f"  T2-Full: Ora={r['T2_FullOra']:.4f}  "
              f"Zero={r['T2_FullZero']:.4f}  "
              f"Imp={r['T2_FullImp']:.4f}  "
              f"Δimp={r['T2_FullImp']-r['T2_FullZero']:+.4f}")

# ── Print aggregate summary ───────────────────────────────────────────────────
print(f"\n\n{'='*85}")
print("AGGREGATE RESULTS — TASK 1 (Sensor Increment)")
print(f"{'='*85}")
print(f"{'Config':<26} {'Scen':>5} {'E1':>8} {'Imputed':>9} {'Oracle':>9} {'Δ imp':>8}")
print(f"{'─'*75}")
for scenario in ["1→1", "2→1"]:
    for cfg_name, r in results.items():
        if r["scenario"] != scenario: continue
        m = r["macro_means"]; s = r["macro_stds"]
        print(f"  {cfg_name:<24} {scenario:>5}  "
              f"{m['E1']:.4f}±{s['E1']:.3f}  "
              f"{m['Imputed']:.4f}±{s['Imputed']:.3f}  "
              f"{m['Oracle']:.4f}±{s['Oracle']:.3f}  "
              f"{m['Imputed']-m['E1']:+.4f}")
    print()

print(f"\n{'='*85}")
print("AGGREGATE RESULTS — TASK 2 (Missing Sensor)")
print(f"{'='*85}")
print(f"{'Config':<26} {'Scen':>5} {'Oracle':>9} {'Zero':>9} {'Imputed':>9} {'Δ imp-zero':>11}")
print(f"{'─'*80}")
for scenario in ["1→1", "2→1"]:
    for cfg_name, r in results.items():
        if r["scenario"] != scenario: continue
        m = r["macro_means"]; s = r["macro_stds"]
        print(f"  {cfg_name:<24} {scenario:>5}  "
              f"{m['T2_Oracle']:.4f}±{s['T2_Oracle']:.3f}  "
              f"{m['T2_Zero']:.4f}±{s['T2_Zero']:.3f}  "
              f"{m['T2_Imputed']:.4f}±{s['T2_Imputed']:.3f}  "
              f"{m['T2_Imputed']-m['T2_Zero']:+.4f}")
    print()

print(f"\n{'='*85}")
print("AGGREGATE RESULTS — TASK 2 FULL IMPUTATION (1→1 only: impute all missing streams)")
print(f"{'='*85}")
print(f"{'Config':<26} {'Scen':>5} {'3S-Ora':>9} {'3S-Zero':>9} {'3S-Imp':>9} {'Δ imp-zero':>11}")
print(f"{'─'*80}")
for cfg_name, r in results.items():
    if not r.get("has_full"): continue
    m = r["macro_means"]; s = r["macro_stds"]
    print(f"  {cfg_name:<24} {'1→1':>5}  "
          f"{m['T2_FullOra']:.4f}±{s['T2_FullOra']:.3f}  "
          f"{m['T2_FullZero']:.4f}±{s['T2_FullZero']:.3f}  "
          f"{m['T2_FullImp']:.4f}±{s['T2_FullImp']:.3f}  "
          f"{m['T2_FullImp']-m['T2_FullZero']:+.4f}")
print()

os.makedirs(OUT_DIR, exist_ok=True)

# ── Summary bar plot ──────────────────────────────────────────────────────────
def plot_summary(results, out_path):
    configs   = list(results.keys())
    scenarios = [results[c]["scenario"] for c in configs]
    x = np.arange(len(configs)); w = 0.2

    fig, axes = plt.subplots(2, 1, figsize=(max(14, len(configs)*1.6), 10))

    for ax, task, keys, labels, colors, title in [
        (axes[0], "T1",
         ["E1", "Imputed", "Oracle"],
         ["E1", "Imputed", "Oracle"],
         ["#90CAF9", "#A5D6A7", "#FFCC80"],
         "Task 1: Sensor Increment"),
        (axes[1], "T2",
         ["T2_Oracle", "T2_Zero", "T2_Imputed"],
         ["Oracle", "Zero-masked", "Imputed"],
         ["#FFCC80", "#EF9A9A", "#A5D6A7"],
         "Task 2: Missing Sensor"),
    ]:
        offsets = [-w, 0, w]
        for key, label, color, offset in zip(keys, labels, colors, offsets):
            means = [results[c]["macro_means"][key] for c in configs]
            stds  = [results[c]["macro_stds"][key]  for c in configs]
            bars  = ax.bar(x+offset, means, w, yerr=stds, capsize=3,
                           label=label, color=color, alpha=0.85,
                           edgecolor='black', lw=0.5)

        # Δ imp annotation (vs E1 for T1, vs Zero for T2)
        base_key = "E1" if task=="T1" else "T2_Zero"
        imp_key  = "Imputed" if task=="T1" else "T2_Imputed"
        for xi, cfg in enumerate(configs):
            base = results[cfg]["macro_means"][base_key]
            imp  = results[cfg]["macro_means"][imp_key]
            color = '#2E7D32' if imp >= base else '#C62828'
            ax.text(xi, max(results[cfg]["macro_means"][k] for k in keys) + 0.05,
                    f'{imp-base:+.2f}', ha='center', fontsize=7,
                    color=color, fontweight='bold')

        # Scenario separator
        s1_end = sum(1 for s in scenarios if s=="1→1") - 0.5
        ax.axvline(s1_end, color='gray', lw=1.5, ls='--', alpha=0.7)
        ax.text(s1_end/2, 1.08, '1→1', ha='center', fontsize=9, color='gray')
        ax.text((s1_end+len(configs))/2, 1.08, '2→1', ha='center', fontsize=9, color='gray')

        ax.set_xticks(x)
        ax.set_xticklabels(configs, rotation=20, ha='right', fontsize=8)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel('Macro F1 (grouped)', fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    fig.suptitle(f'All Sensor Configurations ({N_RUNS} runs, mean±std)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")

# ── Per-group breakdown ───────────────────────────────────────────────────────
def plot_per_group(results, out_path):
    groups = list(GROUPS.keys())
    configs = list(results.keys())
    fig, axes = plt.subplots(1, len(groups), figsize=(4*len(groups), 5), sharey=True)
    colors = {"E1": "#90CAF9", "Imputed": "#A5D6A7", "Oracle": "#FFCC80"}
    for gi, (ax, gname) in enumerate(zip(axes, groups)):
        x = np.arange(len(configs)); w = 0.25
        for ci, (cond, offset) in enumerate(zip(["E1","Imputed","Oracle"], [-w, 0, w])):
            means = [float(np.nanmean([r.get(gname, np.nan)
                     for r in results[cfg]["f1s_runs"][cond]]))
                     for cfg in configs]
            stds  = [float(np.nanstd([r.get(gname, np.nan)
                     for r in results[cfg]["f1s_runs"][cond]]))
                     for cfg in configs]
            ax.bar(x+offset, means, w, yerr=stds, capsize=2,
                   color=colors[cond], alpha=0.85, edgecolor='black', lw=0.3,
                   label=cond if gi==0 else None)
        ax.set_title(gname, fontsize=9, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(configs, rotation=45, ha='right', fontsize=6)
        ax.set_ylim(0, 1.1); ax.grid(axis='y', alpha=0.3)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    axes[0].set_ylabel('Macro F1', fontsize=10)
    axes[0].legend(fontsize=8)
    fig.suptitle('Per-Group F1 Across All Sensor Configurations', fontsize=11, fontweight='bold')
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")

plot_per_group(results, f"{OUT_DIR}/per_group_breakdown.png")
print("\nDone.")