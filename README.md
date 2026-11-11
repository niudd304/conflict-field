# Conflict Field — Artifact

Reference implementation and released experimental results for
*Conflict Field: Label-Free Graph Data Quality Auditing via Node–Edge Consistency*.

The repository contains the final code (single driver + data preparation +
ablation utilities) and the complete outputs of every experiment reported in
the paper: per-dataset metric summaries (CSV/JSON), per-seed records, and the
diagnostic figures.

---

## 1. Repository layout

```
.
├── run_all.sh              every command that backs the paper, in order
├── requirements.txt
├── src/
│   ├── run_main.py         main experiments: Tables I–II, verdict, diagnostic figures
│   ├── run_ablation.py     synthetic homophily sweep, GNN-depth sweep
│   ├── prepare_data.py     downloads and lays out the 11 benchmarks
│   └── make_figures.py     restacks 1×4 diagnostic panels into the paper's 2×2 format
└── results/                released outputs of the paper's runs
    ├── main/                              main method (Tables I–II, verdict)
    ├── ablation/
    │   ├── binary_deg/                    Table III row 1: binary edge verdict
    │   ├── soft_deg/                      Table III row 2: + soft expectation
    │   ├── soft_sym/                      Table III row 3: + symmetric aggregation
    │   ├── soft_deg_background/           2×2 grid completion (not in Table III)
    │   ├── label_free_scorer/             Table III: label-free edge scorer
    │   ├── val_candidate_selection/       Table III: validation-tuned selection
    │   ├── synthetic_sweep.json           synthetic homophily control
    │   └── depth.json                     GNN-depth control
    └── large_scale/                       Penn94 + genius (Scalability section)
```

Each experiment directory contains:

| file | contents |
|---|---|
| `global_summary_table_v2.csv` | per-dataset means and standard deviations, one row per dataset |
| `global_summary_results_v2.json` | all scalar metrics per seed and averaged; per-node curve arrays are stripped for size |
| `figures/<dataset>_conflict.png` | the 1×4 diagnostic panel |

---

## 2. Environment

```bash
pip install -r requirements.txt
```

Tested with Python 3.10–3.13, `torch>=2.0` (CPU or CUDA), `torch-geometric>=2.3`,
`scikit-learn`, `scipy`, `pandas`, `matplotlib`, `tabulate`.

No GPU is required. All timings quoted in Table II were taken single-threaded on
one CPU core; the timings you reproduce will depend on the core, not on the
presence of a GPU.

---

## 3. Data

```bash
python src/prepare_data.py
```

Datasets land in `src/data/{hetero,planetoid}/` by default (both scripts resolve
their data roots relative to their own location, and both roots can be
redirected — see below).

* The eight heterophilous benchmarks come from the public release of Platonov et
  al., *A Critical Look at the Evaluation of GNNs on Heterophilous Graphs*
  (NeurIPS 2022 D&B). Each `.npz` ships ten official 60/10/30 splits; we rotate
  over them (`seed % 10`), so `seed` and `split` are the same axis.
* Cora / Citeseer / Pubmed are pulled by PyG. Because the paper reports a random
  60/20/20 protocol for these three, the masks are **regenerated** from `seed`
  rather than using the public 140-label split.
* The graph is deduplicated and symmetrised before any statistic is computed.
  Homophily `h` is recomputed from the same arrays the experiments read — we do
  not copy published homophily figures, because they move with preprocessing.

The two large-scale graphs of the Scalability section are **not** downloaded by
`prepare_data.py`; fetch them manually into one directory and point
`CF_EXT_ROOT` at it:

| graph | source |
|---|---|
| `genius.mat` | `github.com/CUAI/Non-Homophily-Large-Scale` → `data/genius.mat` |
| `Penn94.mat`, `penn94_splits.npy` | same repository → `data/facebook100/Penn94.mat`, `data/splits/fb100-Penn94-splits.npy` |

Both roots of the standard suite can be redirected with `CF_HETERO_ROOT` and
`CF_PLANETOID_ROOT`.

---

## 4. Quick start

```bash
# Full reproduction of every table (~25 min on CPU, 11 graphs × 5 seeds):
bash run_all.sh

# Wiring check only (~2 min, 3 graphs):
bash run_all.sh quick

# Main configuration only:
CF_CONFLICT_MODE=curate2 python src/run_main.py --outdir results/main
```

---

## 5. The score, as implemented

The paper's §III is three equations; the code computes exactly these, in this
order, at `CF_CONFLICT_MODE=curate2` (the default of `run_all.sh`).

**Step 1 — edge posterior.** A logistic regression on three structural features
(cosine similarity of the edge endpoints' representations, log degree difference,
Jaccard overlap) is fitted on training-split edges and returns `p_uv`, the
probability that the edge joins two same-label nodes.

**Step 2 — soft conflict (Eq. `conflict_edge`).**

```
c_uv = 1[ŷ_u = ŷ_v] · (1 − p_uv) + 1[ŷ_u ≠ ŷ_v] · p_uv
```

This is the posterior expectation of the binary indicator `1[ŷ_u=ŷ_v] ≠ ê_uv`:
replacing the sampled 0/1 edge verdict by its probability removes Bernoulli
sampling noise and adds no information.

**Step 3 — symmetric aggregation (Eq. `conflict_node`) and the neighbour
background (Eqs. `background`, `final`).**

```
C(v)  = (1/d_v) · Σ_{u~v} c_uv / √d_u
M(v)  = (1/d_v) · Σ_{u~v} m_e / √d_u,     m_e = (Σ_{e'∋u} c_e' + Σ_{e'∋v} c_e' − 2 c_e) / (d_u + d_v − 2)
C̃(v) = ½ ( C(v) + M(v) )
```

`M(v)` is the conflict level of the edges *around* `v`'s edges — a local
background. Where conflict is sparse, `M ≈ C` and the formula is a mild
smoother; where conflict is pervasive, `M` is large and the score compresses the
common part, leaving the excess. `M` reuses the accumulation already performed
for `C`, so the term costs two `bincount` reductions rather than a second
neighbourhood traversal.

**Step 4 — audit-time verdict (Eqs. `identity`, `qhat`).** Three label-free
statistics are recorded per run: `p_e_obs = mean(ê)` (the rate at which the
predictor calls edges consistent, *not* an accuracy), `C̄ = mean(C)`, and
`s = mean(1[ŷ_u=ŷ_v] · ê)`. Then

```
q̂ = C̄ − p_e + 2 s
verdict = Trust  if (p_e > 0.5 and q̂ > 1/C) else Unreliable
```

The identity is exact edge by edge and does not assume independence between the
edge predictor and the same-class indicator.

**Degenerate case.** With `--edge-mode unsup` the scorer is a cosine threshold
with `p_uv = (cos+1)/2`, whose mean is 0.85–0.98 on every graph in the suite.
Every edge is then called consistent, `ê ≡ 1`, and `C(v) ≡ 1 − NC(v)`. The score
collapses onto neighbourhood consistency; Table III quantifies what that costs.

---

## 6. Configuration

Everything that changes the score is an environment variable, so no code edit is
needed to reproduce an ablation.

| Variable | Values | Default | Effect |
|---|---|---|---|
| `CF_CONFLICT_MODE` | `hard`, `soft`, `curate`, `curate2`, `nbr2`–`nbr6` | `hard` | Score family. `curate2` is the paper's main method; `hard` is the binary variant of Table III row 1; `nbr6` is the validation-tuned candidate selection. |
| `CF_AGG` | `sym`, `mean` | `sym` | Node-level aggregation: `1/√d_u` weighting or plain degree normalisation. |
| `CF_COMBINE` | `avg`, `none` | `avg` | `avg` = `½(C+M)`; `none` drops the background term. |
| `CF_SEEDS` | e.g. `0,1,2,3,4` | `0,1,2,3,4` | Seeds (and, for the `.npz` datasets, official splits). |
| `CF_EDGE_BALANCED` | `0`, `1` | `0` | Adds `class_weight='balanced'` to the edge logistic regression. |
| `CF_EDGE_CALIB` | `raw`, `rank`, `z`, `otsu` | `raw` | Label-free recalibration of `p_uv` in the `unsup` branch only. |
| `CF_EXT_ROOT` | path | unset | Directory holding `genius.mat` / `Penn94.mat` for the large-scale run. |
| `CF_HETERO_ROOT`, `CF_PLANETOID_ROOT` | path | `data/…` | Data locations. |

---

## 7. Where each paper item comes from

| Paper item | Released directory / column |
|---|---|
| Table I — Conflict AUC and all baselines | `results/main/…` → `AUC`, `auc_msp`, `auc_entropy`, `auc_nc`, `auc_mc` |
| Table II — accuracies, timings, speed-up | `results/main/…` → `base_acc`, `acc_at_80_*`, `t_conflict`, `t_mc_dropout` |
| Verdict (Trust / Unreliable) | `results/main/…` → `p_e_obs`, `q_hat`, `boundary` (`= 1/C`) |
| Table III rows 1–4 (score operators) | `results/ablation/{binary_deg, soft_deg, soft_sym, main}` → `AUC` |
| Table III — label-free edge scorer | `results/ablation/label_free_scorer` → `AUC` |
| Table III — validation-tuned selection | `results/ablation/val_candidate_selection` → `AUC` |
| Synthetic homophily control | `results/ablation/synthetic_sweep.json` |
| GNN-depth control | `results/ablation/depth.json` |
| Threshold protocol (validation vs oracle) | `results/main/…` → `acc_at_80_val` vs `acc_at_80_oracle` |
| Scalability, two million-edge graphs | `results/large_scale/…` → `AUC`, `t_conflict`, `t_mc_dropout`, `p_e_obs`, `q_hat` |
| Four-panel figures | `results/main/figures/<dataset>_conflict.png` (+ `src/make_figures.py`) |

---

## 8. Evaluation protocol

**Label isolation.** The edge scorer is fitted on training-split edges only; the
score is computed from the trained model's predictions and embeddings alone; the
abstention threshold is the coverage-matched quantile of the **validation** risk
scores, applied unchanged to test. No test label is read at any of those stages.
Test labels are touched only at the final evaluation, as a referee for AUC and
Acc@80 %.

One qualification, stated in the paper as well: the edge scorer *does* consume
training labels. These are the labels the GNN already trains on, so no extra
annotation is needed, but the pipeline is not label-free in the stronger sense.
`--edge-mode unsup` is the fully label-free variant.

**AUC.** The target is `err = 1[ŷ ≠ y]` on test nodes, and the score is a ranking
score, so AUC measures ranking quality only.

**Abstention thresholds.** Two are reported and must not be mixed:

* `acc_at_80_val` — threshold from the validation quantile. This is the number in
  Table II.
* `acc_at_80_oracle` — threshold taken from the test scores themselves; the
  leakage upper bound, reported only in the threshold-protocol ablation.

**Seeds.** Every number is the mean over seeds 0–4; standard deviations sit next
to the means in the CSV. For the eight `.npz` datasets each seed selects a
different official split, so the variance mixes seed and split variance; this is
intentional and conservative.

**Degenerate edge predictor.** On very small, strongly heterophilous graphs the
training split can contain edges of a single class, and the logistic regression
collapses to a constant. The script detects this, reports `p_e_*` as `NaN`
rather than a meaningless accuracy, and records the success rate in
`edge_fit_rate`. This is precisely the condition the verdict is designed to
catch, so such a row should be read as `Unreliable` rather than as a loss.

---

## 9. Reported results

Eleven benchmarks, five seeds, one backbone (2-layer GCN, 64 hidden). Mean
diagnostic AUC over misclassified test nodes:

| Configuration | Directory | Mean AUC |
|---|---|---|
| Binary edge verdict, degree-normalised | `results/ablation/binary_deg` | 0.575 |
| + soft expectation | `results/ablation/soft_deg` | 0.607 |
| + symmetric aggregation | `results/ablation/soft_sym` | 0.653 |
| + neighbour background (**main method**) | `results/main` | **0.657** |
| Main method, label-free edge scorer | `results/ablation/label_free_scorer` | 0.620 |
| Main method + validation-tuned candidate selection | `results/ablation/val_candidate_selection` | 0.692 |
| Penn94 (41.5k nodes, 1.36M edges) | `results/large_scale` | 0.542 |
| genius (422k nodes, 0.92M edges) | `results/large_scale` | 0.447 |

Both large graphs return the `Unreliable` verdict (`q̂ = 0.38 / 0.27` against
`1/C = 0.5`): they sit in the regime the theory marks as out of range, and the
audit says so.

On the abstention side the main method reaches `+3.2` points over the unfiltered
model at 80 % coverage on average (9 of 11 graphs positive), and the audit costs
`4.6`–`21.6×` less than twenty-pass MC-Dropout, the margin growing with graph
size (213 ms against 2.50 s on Tolokers, single CPU core).

---

## 10. License

MIT. See `LICENSE`. Third-party datasets retain their own licenses and terms;
see `src/prepare_data.py` for their sources.
