import sys, numpy as np
sys.path.insert(0, 'scripts')
from scripts.misc.helpers import create_dataset_file_split
from scripts.misc.config_loader import cfg
from scripts.misc.encoder import load_encoders_from_cfg, extract_all_features
from scripts.misc.signal_translator import load_translator, impute_with_translator

encoders   = load_encoders_from_cfg(cfg)
translator = load_translator('output/translator/translator.pt')
_, _, np_test, ld = create_dataset_file_split(
    '/mnt/storage/hitl_experiments/paaws_tuned', ['DS_11'], cfg.SEED)
X_te, y_te = np_test[0], np_test[1]

# Real thigh embeddings
Z_real = extract_all_features(X_te[:,:,[5],:], encoders,
    cfg.STREAM_TO_ENCODER, ['RightThigh'], batch_size=256)[:,0,:]  # (N, 96)

# Imputed thigh embeddings
X_imp = impute_with_translator(translator, X_te[:,:,[3,4],:],
    known_indices=[0,1], n_streams_total=3, encoders=encoders,
    stream_names=['LeftWrist','RightAnkle'],
    stream_to_encoder=cfg.STREAM_TO_ENCODER)
Z_imp = extract_all_features(X_imp[:,:,[2],:], encoders,
    cfg.STREAM_TO_ENCODER, ['RightThigh'], batch_size=256)[:,0,:]

dist    = np.abs(Z_real - Z_imp).mean()
cosine  = (Z_real * Z_imp).sum(axis=1) / (
    np.linalg.norm(Z_real, axis=1) * np.linalg.norm(Z_imp, axis=1) + 1e-8)

print(f'Overall  L1={dist:.4f}  cos={cosine.mean():.4f}')
print()
for act in ['Walking', 'Sitting_Still', 'Lying_On_Back_Lab',
            'Treadmill_3mph_Free_Walk_Lab', 'Standing_Still']:
    if act not in ld: continue
    mask = y_te == ld[act]
    d = np.abs(Z_real[mask] - Z_imp[mask]).mean()
    c = cosine[mask].mean()
    print(f'  {act:<40} L1={d:.4f}  cos={c:.4f}  n={mask.sum()}')
