"""Figure `fig:training_dynamics` (training_dynamics.pdf/.png): appropriateness c vs training step.

Medical 4B dynamics replicates (Rubric-RL / ProRubric, seeds 42/43/44; the recipes of Table
`tab:consensus_ablations` retrained keeping a checkpoint every 50 steps, not the table's checkpoints). c =
HealthBench consensus on the fixed 300-prompt sample, one judge deployment for the whole trajectory. Step 0 = the
shared untrained model. Three-seed mean +- sd (ddof=1).

With --summary, the values are read from score_checkpoints.py's summary.json (base.consensus_c.c and
seeds[SEED].points["ARM/STEP"].consensus_c.c). Without it, the per-seed values embedded below (the paper's numbers,
the same summary) are plotted.

    python3 plot_training_dynamics.py [--summary OUT/summary.json] [--arm rubric-rl=Rubric-RL ...] [--out-dir .]
"""
import argparse, json, os, statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 7.2, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5,
    'axes.linewidth': 0.6, 'mathtext.fontset': 'cm',
})
C_ADD, T_ADD = '#d95f38', '#9a3412'    # additive (Rubric-RL)
C_OURS, T_OURS = '#1d4ed8', '#1e40af'  # ProRubric
C_REF = '#475569'

BASE_C = 55.83  # untrained model, same 300 prompts, same judge deployment
STEPS = [50, 100, 150, 200, 250, 300]
# per-seed c (s42, s43, s44) at each step
C = {
    'Rubric-RL': [(34.17, 37.94, 35.56), (34.72, 33.72, 34.83), (31.06, 31.39, 28.61),
                  (27.44, 30.11, 25.89), (26.17, 24.94, 24.89), (25.78, 22.44, 23.22)],
    'ProRubric': [(48.78, 51.67, 50.44), (40.22, 39.78, 43.11), (43.83, 40.67, 39.39),
                  (41.06, 39.06, 40.11), (36.94, 38.39, 34.50), (34.17, 36.83, 35.00)],
}
STYLE = {  # color, text color, marker, filled, size, label y-offset (points)
    'Rubric-RL': (C_ADD, T_ADD, 'o', False, 3.6, 0),
    'ProRubric': (C_OURS, T_OURS, 's', True, 3.6, -3),
}
DEFAULT_ARMS = ['rubric-rl=Rubric-RL', 'prorubric=ProRubric']


def from_summary(path, arms, seeds):
    """(base c, {label: [(c per seed) per step]}) from score_checkpoints.py's summary.json."""
    s = json.load(open(path))
    seeds = seeds or sorted(s['seeds'])
    out = {}
    for key, label in arms:
        out[label] = [tuple(s['seeds'][sd]['points']['%s/%d' % (key, st)]['consensus_c']['c'] for sd in seeds) for st in STEPS]
    return s['base']['consensus_c']['c'], out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--summary', help="summary.json of score_checkpoints.py (default: the embedded paper values)")
    ap.add_argument('--arm', action='append', metavar='KEY=LABEL',
                    help='arm key in the summary and its label (default: rubric-rl=Rubric-RL, prorubric=ProRubric)')
    ap.add_argument('--seeds', help='comma-separated seeds to average (default: all in the summary)')
    ap.add_argument('--out-dir', default='.')
    a = ap.parse_args()
    base_c, cvals = BASE_C, C
    if a.summary:
        arms = [x.split('=', 1) for x in (a.arm or DEFAULT_ARMS)]
        base_c, cvals = from_summary(a.summary, arms, a.seeds.split(',') if a.seeds else None)

    fig, ax = plt.subplots(figsize=(2.75, 1.85), dpi=300)
    ax.axhline(base_c, color='#94a3b8', lw=0.75, ls='--', zorder=1)
    ax.scatter(0, base_c, s=55, marker='*', facecolors=C_REF, edgecolors='black', lw=0.5, zorder=6)
    ax.annotate('Untrained', (0, base_c), textcoords='offset points', xytext=(5, 2),
                fontsize=5.8, color='#334155', ha='left', va='bottom', fontweight='bold')

    for arm, seeds in cvals.items():
        col, tcol, mk, filled, ms, dy = STYLE[arm]
        mean = [statistics.mean(v) for v in seeds]
        sd = [statistics.stdev(v) for v in seeds]  # ddof=1
        xs, ys = [0] + STEPS, [base_c] + mean
        lo = [base_c] + [m - s for m, s in zip(mean, sd)]
        hi = [base_c] + [m + s for m, s in zip(mean, sd)]
        ax.fill_between(xs, lo, hi, color=col, alpha=0.13, lw=0, zorder=2)
        ax.plot(xs, ys, color=col, lw=1.0, zorder=3)
        ax.errorbar(STEPS, mean, yerr=sd, fmt='none', ecolor=col, elinewidth=0.6, capsize=1.3,
                    capthick=0.6, zorder=3)
        ax.plot(STEPS, mean, ls='', marker=mk, ms=ms, mfc=(col if filled else 'white'),
                mec=('black' if filled else col), mew=(0.4 if filled else 0.9), zorder=4)
        ax.annotate(arm, (STEPS[-1], mean[-1]), textcoords='offset points', xytext=(5, dy),
                    fontsize=6.0, color=tcol, ha='left', va='center', fontweight='bold')

    ax.set_xlabel('Training step', labelpad=2, fontsize=7.0)
    ax.set_ylabel(r'Appropriateness $c$', labelpad=2, fontsize=7.0)
    ax.set_xticks([0] + STEPS)
    ax.set_xlim(-12, 312)
    ax.set_ylim(19.0, 60.0)
    ax.grid(True, linestyle=':', alpha=0.35, color='#cbd5e1')

    os.makedirs(a.out_dir, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(a.out_dir, f'training_dynamics.{ext}'), bbox_inches='tight', dpi=300)
    print('wrote training_dynamics.pdf/.png to', a.out_dir)


if __name__ == '__main__':
    main()
