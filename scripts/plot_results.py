"""
plot_results.py - Publication-quality plots from sensor ablation results.

Usage:
    python scripts/plot_results.py \
        --results-dir checkpoints/ \
        --output-dir  plots/ \
        --n-base 1 2 3
"""

import os, json, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams.update({
    'font.family'      : 'DejaVu Sans',
    'font.size'        : 11,
    'axes.spines.top'  : False,
    'axes.spines.right': False,
    'axes.grid'        : True,
    'axes.grid.axis'   : 'y',
    'grid.alpha'       : 0.3,
    'grid.linestyle'   : '--',
    'figure.dpi'       : 300,
})

SHORT = {'LeftWrist':'LW','RightWrist':'RW','RightThigh':'RT',
         'RightWaist':'RWa','RightAnkle':'RA'}
SENSOR_FULL = {'LW':'LeftWrist','RW':'RightWrist','RT':'RightThigh',
               'RWa':'RightWaist','RA':'RightAnkle'}

BUD_COLORS = {'5':'#B5D4F4','10':'#6AAEE8','15':'#2B7FC9',
              'all':'#0C447C','oracle':'#D85A30'}
BUD_LABELS = {'5':'Top-5','10':'Top-10','15':'Top-15',
              'all':'Elbow','oracle':'Oracle'}


# ── Data helpers ──────────────────────────────────────────────────────────────

def load_data(results_dir, n_base_list):
    data = []
    for n in n_base_list:
        p = os.path.join(results_dir, f'ablation_sensor_n{n}_results.json')
        if os.path.exists(p):
            data.extend(json.load(open(p)))
        else:
            print(f'  [skip] {p}')
    print(f'Loaded {len(data)} configs')
    return data

def get_br(cfg, budget):
    return next((b for b in cfg['budget_results'] if b['budget']==budget), None)

def overall_delta(cfg, budget):
    br = get_br(cfg, budget)
    return (br['proposed']['macro_f1'] - cfg['baseline']['macro_f1']) if br else None

def oracle_delta(cfg, _=None):
    return cfg['oracle']['macro_f1'] - cfg['baseline']['macro_f1']

def targeted_delta(cfg, budget):
    br = get_br(cfg, budget)
    if not br: return None
    targets = set(br['target_classes'])
    if not targets: return None
    pc   = br['per_class']
    tgts = [c for c in targets if c in pc]
    if not tgts: return None
    return float(np.mean([pc[c]['proposed'] for c in tgts]) -
                 np.mean([pc[c]['baseline'] for c in tgts]))

def targeted_oracle_delta(cfg, budget='all'):
    br = get_br(cfg, budget)
    if not br: return None
    targets = set(br['target_classes'])
    if not targets: return None
    pc   = br['per_class']
    tgts = [c for c in targets if c in pc]
    if not tgts: return None
    return float(np.mean([pc[c]['oracle']   for c in tgts]) -
                 np.mean([pc[c]['baseline'] for c in tgts]))

def get_deltas(configs, cond, prop_fn, orac_fn):
    if cond == 'oracle':
        vals = [orac_fn(r) for r in configs]
    else:
        vals = [prop_fn(r, cond) for r in configs]
    return [v for v in vals if v is not None]

def draw_bars(ax, configs, conds, prop_fn, orac_fn):
    means, errs = [], []
    for cond in conds:
        vals = get_deltas(configs, cond, prop_fn, orac_fn)
        means.append(float(np.mean(vals)) if vals else 0)
        errs.append(float(np.std(vals)/np.sqrt(len(vals))) if len(vals)>1 else 0)

    x      = np.arange(len(conds))
    colors = [BUD_COLORS[c] for c in conds]
    bars   = ax.bar(x, means, color=colors, yerr=errs, capsize=3,
                    error_kw={'linewidth':1,'ecolor':'#444'},
                    width=0.65, zorder=3)

    for i, (bar, val) in enumerate(zip(bars, means)):
        yoff = errs[i] + 0.004
        ypos = val + yoff if val >= 0 else val - yoff - 0.012
        ax.text(bar.get_x()+bar.get_width()/2, ypos, f'{val:+.3f}',
                ha='center', va='bottom' if val>=0 else 'top',
                fontsize=7.5, fontweight='bold')

    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([BUD_LABELS[c] for c in conds], fontsize=9)
    if configs:
        ax.text(0.98, 0.97, f'n={len(configs)}',
                transform=ax.transAxes, ha='right', va='top',
                fontsize=8, color='gray')


# ── Plot 1: Budget comparison overall + targeted ───────────────────────────

def plot_budget_comparison(all_data, out_dir):
    budgets = ['5','10','15','all']
    conds   = budgets + ['oracle']
    n_bases = sorted(set(r['n_base'] for r in all_data))

    fig, axes = plt.subplots(2, len(n_bases),
                             figsize=(4.5*len(n_bases), 7),
                             sharey='row')

    row_specs = [
        (overall_delta,  oracle_delta,          'Δ Overall Macro F1\n(all 40 classes)'),
        (targeted_delta, targeted_oracle_delta,  'Δ Targeted Macro F1\n(re-labeled classes only)'),
    ]

    for col, n_base in enumerate(n_bases):
        configs = [r for r in all_data if r['n_base']==n_base]
        for row, (prop_fn, orac_fn, ylabel) in enumerate(row_specs):
            ax = axes[row][col]
            draw_bars(ax, configs, conds, prop_fn, orac_fn)
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=10)
            if row == 0:
                ax.set_title(f'n_base = {n_base}', fontsize=12, fontweight='bold')

    patches = [mpatches.Patch(color=BUD_COLORS[c], label=BUD_LABELS[c]) for c in conds]
    fig.legend(handles=patches, loc='lower center', ncol=len(conds),
               bbox_to_anchor=(0.5,-0.02), fontsize=10,
               title='Annotation condition', title_fontsize=10)
    fig.suptitle('Macro F1 gain by annotation budget and base sensor count',
                 fontsize=13, fontweight='bold', y=1.01)
    fig.tight_layout()
    path = os.path.join(out_dir, '1_budget_comparison.pdf')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')


# ── Plot 2a/2b: Per new sensor budget comparison ───────────────────────────

def plot_per_sensor_budget(all_data, out_dir, metric='overall'):
    sensors   = ['LW','RW','RT','RWa','RA']
    budgets   = ['5','10','15','all']
    conds     = budgets + ['oracle']
    n_bases   = sorted(set(r['n_base'] for r in all_data))

    if metric == 'overall':
        prop_fn   = overall_delta
        orac_fn   = oracle_delta
        ylabel    = 'Δ Overall Macro F1'
        fname     = '2a_per_sensor_overall.pdf'
        title     = 'Overall macro F1 gain by new sensor and annotation budget'
    else:
        prop_fn   = targeted_delta
        orac_fn   = targeted_oracle_delta
        ylabel    = 'Δ Targeted Macro F1'
        fname     = '2b_per_sensor_targeted.pdf'
        title     = 'Targeted macro F1 gain by new sensor and annotation budget'

    fig, axes = plt.subplots(len(sensors), len(n_bases),
                             figsize=(4.5*len(n_bases), 3.5*len(sensors)),
                             sharey='row')

    for row, sensor in enumerate(sensors):
        for col, n_base in enumerate(n_bases):
            ax = axes[row][col]
            configs = [r for r in all_data
                       if r['n_base']==n_base
                       and SHORT[r['new_sensor'][0]]==sensor]

            if not configs:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                        transform=ax.transAxes, color='gray', fontsize=10)
                ax.set_xticks([])
            else:
                draw_bars(ax, configs, conds, prop_fn, orac_fn)

            if col == 0:
                ax.set_ylabel(f'+{SENSOR_FULL[sensor]}\n{ylabel}', fontsize=9)
            if row == 0:
                ax.set_title(f'n_base = {n_base}', fontsize=11, fontweight='bold')

    patches = [mpatches.Patch(color=BUD_COLORS[c], label=BUD_LABELS[c]) for c in conds]
    fig.legend(handles=patches, loc='lower center', ncol=len(conds),
               bbox_to_anchor=(0.5,-0.01), fontsize=10,
               title='Annotation condition', title_fontsize=10)
    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.005)
    fig.tight_layout()
    path = os.path.join(out_dir, fname)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')


# ── Main ──────────────────────────────────────────────────────────────────────


def plot_per_config_activities(all_data, out_dir):
    """
    For each sensor config, generate one PDF with grouped bar plots showing
    baseline, proposed (elbow), and oracle F1 for each of the 40 activities.
    Targeted activities are highlighted.
    Saved under out_dir/n_base_{N}/
    """
    ACT_COLORS = {
        'baseline' : '#888780',
        'proposed' : '#1D9E75',
        'oracle'   : '#D85A30',
    }

    def shorten(name):
        return (name.replace('_Lab','').replace('Treadmill_','Tread_')
                    .replace('_',' '))

    for r in all_data:
        n_base = r['n_base']
        name   = r['name']
        br     = get_br(r, 'all')
        if not br:
            continue

        targets  = set(br['target_classes'])
        pc       = br['per_class']
        acts     = list(pc.keys())
        n_acts   = len(acts)

        baseline = [pc[a]['baseline'] for a in acts]
        proposed = [pc[a]['proposed'] for a in acts]
        oracle   = [pc[a]['oracle']   for a in acts]
        is_tgt   = [a in targets      for a in acts]

        fig, ax = plt.subplots(figsize=(max(16, n_acts*0.55), 5))

        x     = np.arange(n_acts)
        w     = 0.26
        bars_b = ax.bar(x - w, baseline, w, color=ACT_COLORS['baseline'],
                        label='Baseline', zorder=3)
        bars_p = ax.bar(x,     proposed, w, color=ACT_COLORS['proposed'],
                        label='Proposed (elbow)', zorder=3)
        bars_o = ax.bar(x + w, oracle,   w, color=ACT_COLORS['oracle'],
                        label='Oracle', zorder=3)

        # Highlight targeted activities
        for i, tgt in enumerate(is_tgt):
            if tgt:
                ax.axvspan(x[i] - w*1.8, x[i] + w*1.8,
                           alpha=0.08, color='green', zorder=0)

        ax.set_xticks(x)
        ax.set_xticklabels([shorten(a) for a in acts],
                           rotation=45, ha='right', fontsize=7.5)
        ax.set_ylabel('F1 Score')
        ax.set_ylim(0, 1.12)
        ax.set_title(f'{name}  (n_base={n_base})', fontsize=11, fontweight='bold')

        # Legend + targeted marker
        handles, labels = ax.get_legend_handles_labels()
        tgt_patch = mpatches.Patch(color='green', alpha=0.2, label='Targeted (retrained)')
        ax.legend(handles=handles+[tgt_patch], fontsize=9,
                  loc='upper right', ncol=4)

        fig.tight_layout()

        sub_dir = os.path.join(out_dir, f'n_base_{n_base}')
        os.makedirs(sub_dir, exist_ok=True)
        path = os.path.join(sub_dir, f'{name}.pdf')
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)

    print(f'  Saved per-config activity plots under {out_dir}/n_base_{{1,2,3}}/')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', type=str, required=True)
    parser.add_argument('--output-dir',  type=str, default='plots')
    parser.add_argument('--n-base',      type=int, nargs='+', default=[1,2,3])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    all_data = load_data(args.results_dir, args.n_base)

    print('\nGenerating plots...')
    plot_budget_comparison(all_data, args.output_dir)
    plot_per_sensor_budget(all_data, args.output_dir, metric='overall')
    plot_per_sensor_budget(all_data, args.output_dir, metric='targeted')
    plot_per_config_activities(all_data, args.output_dir)
    print('Done.')

if __name__ == '__main__':
    main()


# ── Plot 3: Per-activity performance per config ───────────────────────────────