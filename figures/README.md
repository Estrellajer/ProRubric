# Figure scripts

These scripts draw the paper's result figures with matplotlib. The plotting code is unchanged from
the version used for the paper. The numbers each figure draws are written in the script, next to
the table they come from.

| Script | Output | Paper figure | Numbers drawn |
|---|---|---|---|
| `plot_diagnosis_compound.py` | `fig_diagnosis_compound.pdf/.png` | Fig. 2 (`fig:diagnosis_compound`) | Untrained, Rubric-RL and ProRubric on the (g_pro, c_pro) plane. Three-seed means from `tab:consensus_ablations`. |
| `plot_controls_compound.py` | `fig_controls_compound.pdf/.png` | Fig. 3 (`fig:controls_compound`) | Three panels, listed below the table. |
| `plot_length_utility.py` | `fig1_length_utility.pdf/.png` | Appendix `fig:pairwise_winrates` | Two panels, listed below the table. |

Panels of `plot_controls_compound.py`:

- **(a)** c_pro against mean response length, from `tab:ablations_full`.
- **(b)** The KL sweep, plotting (g_pro, c_pro) for beta in {0, 0.005, 0.01, 0.04}, from
  `tab:ablations_full`.
- **(c)** Change in appropriateness against the untrained model, per domain. These numbers come
  from the appropriateness probe (`tab:probe`).

Panels of `plot_length_utility.py`:

- **(a)** c_pro (filled markers) and c_lite (open markers) against length for seven methods, from
  `tab:ablations_full`.
- **(b)** Rubric-free pairwise win rates of ProRubric at a 2,048-token budget. These come from
  the rubric-free pairwise evaluation (`eval/pairwise.py`).

## Checking the drawn numbers

Every value a figure takes from the ablation tables can be checked against the JSON written by
`analysis/tables/ablations.py`. Pass that JSON with `--ablations-json`:

```bash
python3 plot_diagnosis_compound.py --out-dir out --ablations-json ../analysis/tables/out/ablations.json
python3 plot_controls_compound.py  --out-dir out --ablations-json ../analysis/tables/out/ablations.json
python3 plot_length_utility.py     --out-dir out --ablations-json ../analysis/tables/out/ablations.json
```

The check compares each drawn value with the table value at one decimal. It prints every
mismatch and exits with a non-zero status if there is any. The method names it looks up are the
ones used in `analysis/tables/arms.example.json`.

The probe panel (Fig. 3c) and the pairwise panel (appendix figure, panel b) are not checked.
Their numbers come from evaluations that the table scripts do not produce.

## Fonts

The scripts ask for Times New Roman and fall back to DejaVu Serif, which ships with matplotlib.
No font files are included in this package.

`--out-dir` defaults to the current directory. Dependencies: `matplotlib`, `numpy`.
