"""r07_fig_compact.py — Composite robustness figure (Figure D.1) for the compact robustness figure (thesis Figure 3.3).
One image summarising all 32 variant models (runs r01–r04 must have been executed first).
Style matched to the existing figR1–figR4 thesis figures (matplotlib defaults, black dashed
'Thesis Table 4.1' reference)."""
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pathlib import Path
BASE = str(Path(__file__).resolve().parent.parent / 'results')
PERIODS = ['2007-2014', '2015-2018', '2019-2022', '2023-2026']
LAYERS = ['operational', 'institutional', 'technology', 'meta']
LABELS = {
    'operational': 'Operational core',
    'institutional': 'Institutional & strategic',
    'technology': 'Technology enablers',
    'meta': 'Meta-research',
}
COLORS = {
    'operational': '#1f77b4',
    'institutional': '#ff7f0e',
    'technology': '#2ca02c',
    'meta': '#9467bd',
}
THESIS = {  # Table 4.1
    'operational': [58.9, 57.9, 49.0, 32.2],
    'institutional': [25.3, 30.6, 30.3, 35.7],
    'technology': [15.5, 8.8, 16.8, 24.2],
    'meta': [0.3, 2.7, 3.9, 7.9],
}

rows = list(csv.DictReader(open(f'{BASE}/layer_period_all_runs.csv', encoding='utf-8-sig')))
runs = sorted(set(r['run_id'] for r in rows))
assert len(runs) == 32, len(runs)

# share[layer][period] -> list over runs
share = {l: {p: [] for p in PERIODS} for l in LAYERS}
per_run = {rid: {l: {} for l in LAYERS} for rid in runs}
for r in rows:
    share[r['layer']][r['period']].append(float(r['share_pct']))
    per_run[r['run_id']][r['layer']][r['period']] = float(r['share_pct'])

summ = list(csv.DictReader(open(f'{BASE}/runs_summary.csv', encoding='utf-8-sig')))
deltas = {l: [] for l in LAYERS}
key = {'operational': 'op_delta', 'institutional': 'inst_delta',
       'technology': 'tech_delta', 'meta': 'meta_delta'}
for s in summ:
    for l in LAYERS:
        deltas[l].append(float(s[key[l]]))
thesis_delta = {l: THESIS[l][3] - THESIS[l][0] for l in LAYERS}

plt.rcParams.update({'font.size': 13, 'axes.titlesize': 15, 'figure.facecolor': 'white'})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 4.8), gridspec_kw={'width_ratios': [1.05, 1]})

# ---- Panel (a): layer trajectories, mean ± 1 sd across the 32 variant models ----
x = np.arange(4)
for l in LAYERS:
    m = np.array([np.mean(share[l][p]) for p in PERIODS])
    sd = np.array([np.std(share[l][p]) for p in PERIODS])
    ax1.plot(x, m, color=COLORS[l], lw=2, marker='o', ms=5, label=LABELS[l], zorder=3)
    ax1.fill_between(x, np.maximum(m - sd, 0), m + sd, color=COLORS[l], alpha=0.15, lw=0, zorder=1)
    ax1.plot(x, THESIS[l], color=COLORS[l], lw=1.6, ls='--', marker='s', ms=5,
             dashes=(4, 2), zorder=2)
ax1.set_xticks(x)
ax1.set_xticklabels(PERIODS)
ax1.set_ylabel('Topic mass (%)')
ax1.set_title('(a) Layer trajectories: 32 variant models vs retained model')
ax1.grid(alpha=0.3)
leg1 = ax1.legend(loc='upper right', fontsize=11, framealpha=0.9)
# linestyle key
from matplotlib.lines import Line2D
style_handles = [
    Line2D([0], [0], color='0.25', lw=2, marker='o', ms=5, label='Mean of 32 variants (band: ±1 sd)'),
    Line2D([0], [0], color='0.25', lw=1.6, ls='--', dashes=(4, 2), marker='s', ms=5, label='Retained model (Table 4.1)'),
]
ax1.add_artist(leg1)
ax1.legend(handles=style_handles, loc='lower left', fontsize=10.5, framealpha=0.9)

# ---- Panel (b): first-to-last-period change per run ----
rng = np.random.default_rng(42)
ypos = {l: i for i, l in enumerate(reversed(LAYERS))}
for l in LAYERS:
    y = ypos[l] + rng.uniform(-0.16, 0.16, size=len(deltas[l]))
    ax2.scatter(deltas[l], y, s=34, color=COLORS[l], alpha=0.75, edgecolors='white',
                linewidths=0.6, zorder=3)
    ax2.scatter([thesis_delta[l]], [ypos[l]], marker='D', s=90, color='black', zorder=4)
ax2.axvline(0, color='0.35', lw=1.2, zorder=2)
ax2.set_yticks([ypos[l] for l in LAYERS])
ax2.set_yticklabels([LABELS[l] for l in LAYERS])
ax2.set_xlabel('Change in layer share, 2023–2026 vs 2007–2014 (pp)')
ax2.set_title('(b) First-to-last-period change, all 32 models')
ax2.grid(axis='x', alpha=0.3)
ax2.set_ylim(-0.55, 3.55)
ax2.scatter([], [], marker='D', s=70, color='black', label='Retained model (Table 4.1)')
ax2.legend(loc='upper left', fontsize=11, framealpha=0.9)

fig.tight_layout()
fig.savefig(str(Path(BASE) / 'figures' / 'figD1_robustness_summary.png'), dpi=200, bbox_inches='tight')
print('saved')

# sanity prints
for l in LAYERS:
    d = np.array(deltas[l])
    print(l, 'neg:', int((d < 0).sum()), 'pos:', int((d > 0).sum()), 'thesis delta: %.1f' % thesis_delta[l])
