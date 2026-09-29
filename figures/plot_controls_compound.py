"""Figure 3 (fig_controls_compound.pdf/.png): controls for length, KL and domain.

Full text width (5.5 in). Panels (a) and (b) draw three-seed means of Table tab:ablations_full
(tables/consensus_analysis.tex, output of analysis/tables/ablations.py): (a) c_pro against mean
response length; (b) x = g_pro, y = c_pro over the KL sweep beta in {0, 0.005, 0.01, 0.04}.
Pass --ablations-json to check those numbers against that output.
Panel (c) draws the four-domain appropriateness probe (Table tab:probe: three-seed mean change
vs. the untrained model, sd error bars); those numbers are not produced by the scripts here.

  python3 plot_controls_compound.py [--out-dir DIR] [--ablations-json out/ablations.json]
"""
import argparse
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ap = argparse.ArgumentParser(description='Figure 3: length, KL and domain controls.')
ap.add_argument('--out-dir', default='.', help='directory for the .pdf/.png (default: current directory)')
ap.add_argument('--ablations-json', default=None, help='ablations.py output to check panels (a) and (b) against')
args = ap.parse_args()

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 7.0,
    'axes.labelsize': 7.0,
    'axes.titlesize': 7.2,
    'xtick.labelsize': 6.3,
    'ytick.labelsize': 6.3,
    'legend.fontsize': 6.0,
    'axes.linewidth': 0.6,
    'mathtext.fontset': 'cm',
    'axes.unicode_minus': True,
})

C_RUBRIC = '#d95f38'  # Rubric-RL
C_OURS   = '#1d4ed8'  # ProRubric
C_REF    = '#475569'  # untrained model
T_RUBRIC = '#9a3412'  # text on Rubric-RL
T_OURS   = '#1e40af'  # text on ProRubric
T_REF    = '#334155'
GRID = dict(linestyle=':', alpha=0.35, color='#cbd5e1')
BASE_C = 54.1
MINUS = '−'


def signed(v, fmt='{:.0f}'):
    s = fmt.format(abs(v))
    return s if float(s) == 0 else ('+' if v > 0 else MINUS) + s


fig = plt.figure(figsize=(5.5, 1.85), dpi=300)
gs = fig.add_gridspec(1, 3, wspace=0.12, width_ratios=[1, 1, 1.02])
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1], sharey=ax1)
ax3 = fig.add_subplot(gs[2])

# ---------------------------------------------------------------- (a) length
ax1.axhline(BASE_C, color='#94a3b8', ls='--', lw=0.7, zorder=1)
ax1.scatter(2.97, BASE_C, s=55, marker='*', facecolors=C_REF, edgecolors='black', lw=0.5, zorder=5)
ax1.annotate('Untrained (3.0k)', (2.97, BASE_C), textcoords='offset points', xytext=(0, -5),
             fontsize=5.8, color=T_REF, ha='left', va='top')

ax1.scatter(11.7, 26.0, s=34, marker='o', facecolors=C_RUBRIC, edgecolors='black', lw=0.5, zorder=4)
ax1.annotate('Rubric-RL\n(11.7k)', (11.7, 26.0), textcoords='offset points', xytext=(0, -5),
             fontsize=5.8, color=T_RUBRIC, ha='center', va='top', fontweight='bold')

ax1.scatter(6.8, 27.1, s=34, marker='o', facecolors='white', edgecolors=C_RUBRIC, lw=1.0, zorder=4)
ax1.annotate('Length-matched\n(6.8k)', (6.8, 27.1), textcoords='offset points', xytext=(0, -5),
             fontsize=5.8, color=T_RUBRIC, ha='center', va='top')
ax1.annotate('', xy=(7.33, 27.0), xytext=(11.25, 26.1),
             arrowprops=dict(arrowstyle='->', color=T_RUBRIC, lw=0.7, ls=(0, (3, 2))))
ax1.text(9.3, 27.5, f'{MINUS}42%', fontsize=5.8, color=T_RUBRIC, ha='center', va='bottom')

ax1.scatter(9.8, 36.8, s=42, marker='s', facecolors=C_OURS, edgecolors='#1e3a8a', lw=0.6, zorder=5)
ax1.annotate('ProRubric\n(9.8k)', (9.8, 36.8), textcoords='offset points', xytext=(-6, 0),
             fontsize=5.8, color=T_OURS, ha='right', va='center', fontweight='bold')

ax1.set_xlabel('Response length (1,000 chars)', labelpad=2)
ax1.set_ylabel(r'Appropriateness $c$', labelpad=2)
ax1.set_title('(a) Length', pad=3)
ax1.set_xlim(1.5, 13.8)
ax1.set_ylim(18.0, 58.0)
ax1.grid(True, **GRID)

# ---------------------------------------------------------------- (b) KL sweep
# beta = 0, 0.005, 0.01, 0.04 ; x = g_pro, y = c_pro
rub_g, rub_c = [32.1, 28.8, 27.7, 28.4], [26.0, 25.8, 27.9, 49.3]
our_g, our_c = [35.0, 30.2, 30.5, 28.4], [36.8, 36.9, 48.0, 53.6]

ax2.axhline(BASE_C, color='#94a3b8', ls='--', lw=0.7, zorder=1)
ax2.scatter(26.6, BASE_C, s=55, marker='*', facecolors=C_REF, edgecolors='black', lw=0.5, zorder=5)
ax2.plot(rub_g, rub_c, color=C_RUBRIC, lw=1.0, ls='--', marker='o', ms=3.6,
         mec='black', mew=0.4, zorder=3)
ax2.plot(our_g, our_c, color=C_OURS, lw=1.1, marker='s', ms=3.9,
         mec='#1e3a8a', mew=0.5, zorder=4)

for (g, c), b, off, ha, va in [((35.0, 36.8), '0', (0, -4), 'center', 'top'),
                                ((30.2, 36.9), '0.005', (2, -4), 'left', 'top'),
                                ((30.5, 48.0), '0.01', (4, 0), 'left', 'center'),
                                ((28.4, 53.6), '0.04', (3, 3), 'left', 'bottom')]:
    ax2.annotate(rf'$\beta{{=}}{b}$', (g, c), textcoords='offset points', xytext=off,
                 fontsize=5.4, color='#64748b', ha=ha, va=va)

ax2.annotate('', xy=(29.25, 47.2), xytext=(29.25, 28.8),
             arrowprops=dict(arrowstyle='<->', color=T_OURS, lw=0.8), zorder=5)
ax2.text(29.45, 31.3, '+20.1', fontsize=6.0, color=T_OURS, fontweight='bold',
         ha='left', va='center', zorder=6)

ax2.set_xlabel(r'Rubric coverage $g$', labelpad=2)
ax2.set_title(r'(b) KL coefficient $\beta$', pad=3)
ax2.set_xlim(25.5, 36.6)
ax2.tick_params(labelleft=False)
ax2.grid(True, **GRID)

# Direct labels instead of a legend: (a) labels every point; (b) names each line at its
# right end; (c) carries a two-swatch key in the empty lower-right corner.
ax2.annotate('ProRubric', (35.0, 36.8), textcoords='offset points', xytext=(0, 4),
             fontsize=5.8, color=T_OURS, ha='right', va='bottom', fontweight='bold')
ax2.annotate('Rubric-RL', (32.1, 26.0), textcoords='offset points', xytext=(0, -5),
             fontsize=5.8, color=T_RUBRIC, ha='right', va='top', fontweight='bold')
ax2.text(36.4, BASE_C + 0.8, 'Untrained', fontsize=5.8, color=T_REF, ha='right', va='bottom')

# ---------------------------------------------------------------- (c) domains
domains = ['Medicine', 'Science', 'Dialogue', 'Writing']
x = np.arange(len(domains))
w = 0.34
# Table tab:probe: change in appropriateness vs. the untrained model, three-seed mean and sd
d_rub = [-28.9, -11.2, -7.08, +16.0]
s_rub = [2.3, 0.4, 2.82, 3.1]
d_our = [-15.6, -3.7, -0.25, +24.3]
s_our = [4.5, 2.7, 0.50, 1.4]

ax3.axhline(0, color='#334155', lw=0.6, zorder=2)
b1 = ax3.bar(x - w / 2, d_rub, w, color=C_RUBRIC, edgecolor='#7c2d12', lw=0.4, zorder=3, yerr=s_rub, error_kw=dict(elinewidth=0.6, capsize=1.6, capthick=0.6, ecolor='#334155'))
b2 = ax3.bar(x + w / 2, d_our, w, color=C_OURS, edgecolor='#1e3a8a', lw=0.4, zorder=3, yerr=s_our, error_kw=dict(elinewidth=0.6, capsize=1.6, capthick=0.6, ecolor='#334155'))
for bars, vals, sds, col in [(b1, d_rub, s_rub, T_RUBRIC), (b2, d_our, s_our, T_OURS)]:
    for r, v, sd in zip(bars, vals, sds):
        off = -(sd + 1.0) if v < 0 else sd + 1.0
        ax3.text(r.get_x() + r.get_width() / 2 - (0.04 if bars is b1 and v > 0 else 0), v + off, signed(v), ha='center',
                 va='top' if v < 0 else 'bottom', fontsize=5.6, color=col, fontweight='bold')

from matplotlib.patches import Patch
ax3.legend(handles=[Patch(fc=C_RUBRIC, ec='#7c2d12', lw=0.4, label='Rubric-RL'),
                    Patch(fc=C_OURS, ec='#1e3a8a', lw=0.4, label='ProRubric')],
           loc='lower right', frameon=False, handlelength=1.0, handleheight=0.8,
           handletextpad=0.4, labelspacing=0.3, borderaxespad=0.3)
ax3.set_xticks(x)
ax3.set_xticklabels(domains)
ax3.tick_params(axis='x', length=0, pad=3)
ax3.yaxis.tick_right()
ax3.yaxis.set_label_position('right')
ax3.set_ylabel(r'$\Delta c$ vs. untrained', labelpad=3)
ax3.set_title('(c) Other domains', pad=3)
ax3.set_ylim(-37.0, 31.0)
ax3.set_xlim(-0.55, 3.55)
ax3.grid(True, axis='y', **GRID)

fig.subplots_adjust(left=0.07, right=0.93, top=0.90, bottom=0.19)

os.makedirs(args.out_dir, exist_ok=True)
for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(args.out_dir, f'fig_controls_compound.{ext}'), bbox_inches='tight',
                pad_inches=0.02, dpi=300)
print('wrote fig_controls_compound.pdf/.png to', args.out_dir)

if args.ablations_json:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _check import check
    exp = [('Base', 'c_pro', BASE_C, 1), ('Base', 'len', 2.97, 1000.0), ('Base', 'g_pro', 26.6, 1),
           ('Rubric-RL', 'len', 11.7, 1000.0), ('Rubric-RL', 'c_pro', 26.0, 1),
           ('Length-matched', 'len', 6.8, 1000.0), ('Length-matched', 'c_pro', 27.1, 1),
           ('ProRubric', 'len', 9.8, 1000.0), ('ProRubric', 'c_pro', 36.8, 1)]
    for fam, gs, cs in (('Rubric-RL', rub_g, rub_c), ('ProRubric', our_g, our_c)):
        for arm, g, c in zip([fam] + ['%s, beta=%s' % (fam, b) for b in ('0.005', '0.01', '0.04')], gs, cs):
            exp += [(arm, 'g_pro', g, 1), (arm, 'c_pro', c, 1)]
    check(args.ablations_json, exp)
