"""Appendix figure fig:pairwise_winrates (fig1_length_utility.pdf/.png).

(a) Appropriateness c against mean response length for seven medical 4B arms: filled markers
    c_pro (DeepSeek-V4-Pro), open markers c_lite (Doubao-lite). Three-seed means of Table
    tab:ablations_full (output of analysis/tables/ablations.py); pass --ablations-json to check.
(b) Rubric-free pairwise win rates of ProRubric when both policies regenerate under the same
    2,048-token budget (DeepSeek-V4-Pro; vs. Rubric-RL: three-seed mean with sd bars; other bars:
    seed 42 with bootstrap intervals). These come from the rubric-free pairwise evaluation
    (eval/pairwise.py, appendix on rubric-free comparisons), not from the scripts here.

  python3 plot_length_utility.py [--out-dir DIR] [--ablations-json out/ablations.json]
"""
import argparse
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ap = argparse.ArgumentParser(description='Appendix figure: length vs. appropriateness; equal-budget pairwise.')
ap.add_argument('--out-dir', default='.', help='directory for the .pdf/.png (default: current directory)')
ap.add_argument('--ablations-json', default=None, help='ablations.py output to check panel (a) against')
args = ap.parse_args()

# 1:1 scale for 5.5in width (ICLR textwidth)
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 7.8,
    'axes.labelsize': 7.8,
    'axes.titlesize': 8.5,
    'xtick.labelsize': 6.8,
    'ytick.labelsize': 6.8,
    'legend.fontsize': 6.5,
    'axes.linewidth': 0.7,
    'mathtext.fontset': 'cm',
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.5, 2.3), dpi=300,
                               gridspec_kw={'width_ratios': [1.12, 1.0], 'wspace': 0.32})

# ==========================================
# Panel (a): Length vs. Consensus Score
# ==========================================
methods = ['Base', 'Atomic', 'Length-matched', 'raw-AND', 'ProRubric', 'Graded', '+ Appr.']

# Three-seed means from tab:ablations_full
length_k        = np.array([3.0, 11.7, 6.8, 10.4, 9.8, 11.5, 9.2])   # 1,000 characters
deepseek_scores = np.array([54.1, 26.0, 27.1, 34.9, 36.8, 38.3, 43.1])  # c, strict judge
lite_scores     = np.array([78.0, 69.8, 68.2, 73.9, 75.9, 78.7, 77.7])  # c, lenient judge

colors = {'Base': '#475569', 'Atomic': '#d95f38', 'Length-matched': '#9a3412', 'raw-AND': '#64748b',
          'ProRubric': '#1d4ed8', 'Graded': '#0891b2', '+ Appr.': '#0d9488'}
markers = {'Base': '*', 'Atomic': 'o', 'Length-matched': 'v', 'raw-AND': 's',
           'ProRubric': 'D', 'Graded': '^', '+ Appr.': 'P'}

ax1.axhline(54.1, color='#94a3b8', linestyle='--', linewidth=0.8, zorder=1)
ax1.text(3.55, 55.0, '54.1', fontsize=6.2, color='#475569', ha='left', style='italic')

for m, x, y in zip(methods, length_k, deepseek_scores):
    s = 48 if m not in ('ProRubric', '+ Appr.') else 60
    ax1.scatter(x, y, color=colors[m], marker=markers[m], s=s, edgecolor='black', linewidth=0.5, zorder=4)
for m, x, y in zip(methods, length_k, lite_scores):
    s = 32 if m != 'ProRubric' else 45
    ax1.scatter(x, y, facecolors='none', edgecolors=colors[m], marker=markers[m], s=s, linewidth=1.1, zorder=3)

ax1.annotate('', xy=(7.2, 27.0), xytext=(11.3, 26.1),
             arrowprops=dict(arrowstyle="->", color='#9a3412', lw=1.0, ls='--'))
ax1.text(9.2, 24.8, 'Length cut only\n(+1.1)', color='#9a3412', fontsize=5.8, ha='center', va='top', fontweight='bold')
ax1.annotate('', xy=(9.95, 35.6), xytext=(11.5, 27.3),
             arrowprops=dict(arrowstyle="->", color='#1d4ed8', lw=1.2, connectionstyle='arc3,rad=0.2'), zorder=3)
ax1.text(11.9, 29.8, '+10.8', color='#1e40af', fontsize=6.2, ha='left', va='center', fontweight='bold')

ax1.text(3.0, 50.5, 'Base\n(3.0k)', fontsize=6.2, ha='center', va='top', color='#334155', fontweight='bold')
ax1.text(11.7, 22.8, 'Rubric-RL\n(11.7k)', fontsize=6.2, ha='center', va='top', color='#9a3412', fontweight='bold')
ax1.text(6.8, 30.6, 'Length-matched', fontsize=6.0, ha='center', va='bottom', color='#9a3412')
ax1.text(9.35, 36.8, 'ProRubric', fontsize=6.5, ha='right', va='center', color='#1e40af', fontweight='bold')
ax1.text(9.1, 46.0, '+Appr.', fontsize=6.2, ha='center', color='#0f766e', fontweight='bold')
ax1.text(10.2, 32.6, 'raw-AND', fontsize=6.0, ha='right', va='top', color='#475569')
ax1.text(12.0, 38.3, 'Graded', fontsize=6.0, ha='left', va='center', color='#0e7490')

ax1.set_xlabel('Mean Length (1,000 chars)', labelpad=2, fontsize=7.2)
ax1.set_ylabel('Appropriateness $c$', labelpad=2, fontsize=7.2)
ax1.set_title('(a) Length vs. Appropriateness', pad=5, fontweight='bold')
ax1.set_xlim(2.0, 14.2)
ax1.set_ylim(14.0, 85.0)
ax1.grid(True, linestyle=':', alpha=0.4)

legend_elements_a = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#334155', markeredgecolor='k', markersize=5, label='DeepSeek-V4-Pro'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='none', markeredgecolor='#334155', markeredgewidth=1.1, markersize=5, label='Doubao-lite'),
]
ax1.legend(handles=legend_elements_a, loc='center right', bbox_to_anchor=(1.0, 0.66), framealpha=0.92, edgecolor='#cbd5e1',
           handletextpad=0.2, borderpad=0.25)

# ==========================================
# Panel (b): Fixed-Budget Pairwise Win Rates
# ==========================================
comparisons = [
    'vs. Rubric-RL',
    'vs. raw-AND',
    'vs. Graded',
    'vs. Base'
]

# vs. Rubric-RL: three-seed mean, sd error bar; other bars: seed 42, bootstrap CI (rubric-free pairwise, 2,048-token budget)
med_win_rates = [83.1, 86.5, 49.2, 63.5]
med_errors    = [[10.4, 3.7, 4.7, 4.6], [10.4, 3.4, 6.3, 4.5]]

sci_win_rates = [78.7, 89.0, np.nan, 99.4]
sci_errors    = [[11.4, 5.9, 0, 1.3], [11.4, 5.1, 0, 0.6]]

y_pos = np.arange(len(comparisons))[::-1]
bar_height = 0.36

rects1 = ax2.barh(y_pos + bar_height/2, med_win_rates, bar_height, 
                  xerr=med_errors, capsize=2, color='#2563eb', alpha=0.88, 
                  label='Medicine', edgecolor='#1e3a8a', linewidth=0.5)
rects2 = ax2.barh(y_pos - bar_height/2, [w if not np.isnan(w) else 0 for w in sci_win_rates], 
                  bar_height, xerr=sci_errors, capsize=2, color='#ea580c', alpha=0.88, 
                  label='Science', edgecolor='#9a3412', linewidth=0.5)

# Reference 50% line (Parity)
ax2.axvline(50, color='#64748b', linestyle='--', linewidth=0.9, zorder=2)
ax2.text(51.5, 3.52, 'Parity (50%)', color='#475569', fontsize=5.6, fontweight='bold', ha='left', va='center',
         bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='#cbd5e1', lw=0.6), zorder=6)

# Value annotations inside bars
for idx, (rect, val) in enumerate(zip(rects1, med_win_rates)):
    err_left = med_errors[0][idx]
    ax2.text(val - err_left - 3.0, rect.get_y() + rect.get_height()/2, 
             f'{val:.0f}%', ha='right', va='center', 
             fontsize=6.2, color='white', fontweight='bold')

for idx, (rect, val) in enumerate(zip(rects2, sci_win_rates)):
    if not np.isnan(val) and val > 0:
        err_left = sci_errors[0][idx]
        ax2.text(val - err_left - 3.0, rect.get_y() + rect.get_height()/2, 
                 f'{val:.0f}%', ha='right', va='center', 
                 fontsize=6.2, color='white', fontweight='bold')
    elif np.isnan(sci_win_rates[idx]):
        ax2.text(6, rect.get_y() + rect.get_height()/2, 'not evaluated', ha='left', va='center', fontsize=5.8, color='#64748b', style='italic')

ax2.set_yticks(y_pos)
ax2.set_yticklabels(comparisons, fontsize=6.8)
ax2.set_xlabel('ProRubric Win Rate (%)', labelpad=2, fontsize=7.2)
ax2.set_title('(b) Win Rate at a 2k-Token Budget', pad=5, fontweight='bold')
ax2.set_xlim(0, 105)
ax2.set_ylim(-1.3, 3.65)
ax2.grid(True, axis='x', linestyle=':', alpha=0.4)

ax2.legend(loc='lower center', ncol=2, framealpha=0.92, edgecolor='#cbd5e1',
           handletextpad=0.2, borderpad=0.25, fontsize=6.2)

plt.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.16)

os.makedirs(args.out_dir, exist_ok=True)
plt.savefig(os.path.join(args.out_dir, 'fig1_length_utility.pdf'), bbox_inches='tight')
plt.savefig(os.path.join(args.out_dir, 'fig1_length_utility.png'), bbox_inches='tight', dpi=300)
print('wrote fig1_length_utility.pdf/.png to', args.out_dir)

if args.ablations_json:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _check import check
    arm = {'Base': 'Base', 'Atomic': 'Rubric-RL', 'Length-matched': 'Length-matched', 'raw-AND': 'raw-AND',
           'ProRubric': 'ProRubric', 'Graded': 'Graded', '+ Appr.': 'ProRubric + appr. criterion'}
    exp = []
    for m, L, d, l in zip(methods, length_k, deepseek_scores, lite_scores):
        exp += [(arm[m], 'len', L, 1000.0), (arm[m], 'c_pro', d, 1), (arm[m], 'c_lite', l, 1)]
    check(args.ablations_json, exp)
