"""Figure 2 (fig_diagnosis_compound.pdf/.png): the untrained model, Rubric-RL and ProRubric on the
coverage-appropriateness plane (medical, 4B).

x = rubric coverage g (g_pro), y = appropriateness c (c_pro), both DeepSeek-V4-Pro, three-seed means.
The drawn numbers are those of Table tab:consensus_ablations / tab:ablations_full (output of
analysis/tables/ablations.py); pass --ablations-json to check them against that output.

  python3 plot_diagnosis_compound.py [--out-dir DIR] [--ablations-json out/ablations.json]
"""
import argparse
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ap = argparse.ArgumentParser(description='Figure 2: coverage-appropriateness plane.')
ap.add_argument('--out-dir', default='.', help='directory for the .pdf/.png (default: current directory)')
ap.add_argument('--ablations-json', default=None, help='ablations.py output to check the drawn numbers against')
args = ap.parse_args()

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 7.2, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5,
    'axes.linewidth': 0.6, 'mathtext.fontset': 'cm',
})
C_ADD, T_ADD = '#d95f38', '#9a3412'   # additive (Rubric-RL family)
C_OURS, T_OURS = '#1d4ed8', '#1e40af'  # ProRubric
C_NON = '#64748b'                      # other non-additive arms
C_REF = '#475569'
BASE_C, BASE_G = 54.1, 26.6

fig, ax = plt.subplots(figsize=(2.75, 2.05), dpi=300)
base_line, = ax.plot([25.3, 38.3], [BASE_C, BASE_C], color='#94a3b8', lw=0.75, ls='--', zorder=1)
ax.scatter(BASE_G, BASE_C, s=60, marker='*', facecolors=C_REF, edgecolors='black', lw=0.5, zorder=6)
ax.annotate('Untrained', (BASE_G, BASE_C), textcoords='offset points', xytext=(6, 2),
            fontsize=5.8, color='#334155', ha='left', va='bottom', fontweight='bold')
ax.annotate('', xy=(31.8, 26.3), xytext=(BASE_G + 0.15, BASE_C - 1.6),
            arrowprops=dict(arrowstyle='->', color=T_ADD, lw=0.7, ls=(0, (3, 2)),
                            connectionstyle='arc3,rad=0.35'), zorder=2)
ax.text(27.3, 45.0, 'training', color=T_ADD, fontsize=5.6, style='italic')

# only the core contrast; every other arm is in the ablation table
ax.scatter(32.1, 26.0, s=36, marker='o', facecolors='white', edgecolors=C_ADD, lw=1.1, zorder=4)
ax.annotate('Rubric-RL', (32.1, 26.0), textcoords='offset points', xytext=(0, -6),
            fontsize=6.0, color=T_ADD, ha='center', va='top', fontweight='bold')
ax.scatter(35.0, 36.8, s=48, marker='s', facecolors=C_OURS, edgecolors='#1e3a8a', lw=0.7, zorder=6)
ax.annotate('ProRubric', (35.0, 36.8), textcoords='offset points', xytext=(0, 6),
            fontsize=6.0, color=T_OURS, ha='center', va='bottom', fontweight='bold')

ax.annotate('', xy=(34.85, 35.6), xytext=(32.3, 26.8),
            arrowprops=dict(arrowstyle='->', color=C_OURS, lw=1.1), zorder=3)
ax.text(33.2, 31.4, '+10.8', color=C_OURS, fontsize=6.6, fontweight='bold', ha='right', va='center')

ax.set_xlabel(r'Rubric coverage $g$', labelpad=2, fontsize=7.0)
ax.set_ylabel(r'Appropriateness $c$', labelpad=2, fontsize=7.0)
ax.set_xlim(25.3, 38.3)
ax.set_ylim(22.0, 58.0)
ax.grid(True, linestyle=':', alpha=0.35, color='#cbd5e1')

os.makedirs(args.out_dir, exist_ok=True)
for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(args.out_dir, f'fig_diagnosis_compound.{ext}'), bbox_inches='tight', dpi=300)
print('wrote fig_diagnosis_compound.pdf/.png to', args.out_dir)

if args.ablations_json:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _check import check
    check(args.ablations_json, [('Base', 'c_pro', BASE_C, 1), ('Base', 'g_pro', BASE_G, 1),
                                ('Rubric-RL', 'c_pro', 26.0, 1), ('Rubric-RL', 'g_pro', 32.1, 1),
                                ('ProRubric', 'c_pro', 36.8, 1), ('ProRubric', 'g_pro', 35.0, 1)])
