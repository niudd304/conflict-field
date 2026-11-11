"""
Ablation experiments for Conflict Field.

Covers the two ablations that the paper declares but the main pipeline does not
produce by itself:

  Ablation #2  Synthetic-graph control: fix the class count C, sweep homophily h
               independently, and check that the phase transition is driven by h
               rather than by dataset confounders.

  Ablation #4  GNN-depth ablation: vary the number of propagation layers from
               two to four and observe how over-smoothing shifts the transition.

Both reuse the data loading / model training / conflict aggregation code from
``run_main.py`` so that the definition of C(v) is identical to the
main table.

Usage
-----
    python run_ablation.py --which synthetic --outdir results_ablation
    python run_ablation.py --which depth     --outdir results_ablation
    python run_ablation.py --which all       --outdir results_ablation
"""

import os
import json
import time
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sklearn.metrics import roc_auc_score

import run_main as R


# ============================================================
# Ablation #2: controlled synthetic graphs (fix C, sweep h)
# ============================================================
def gen_synthetic(N=2000, C=5, h=0.6, d=32, avg_deg=10, noise=0.85, seed=0, imbalance=0.0):
    """Generate a homophily-controlled attributed graph.

    Node labels are drawn from a class distribution controlled by ``imbalance``
    (0 = uniform, 1 = a single class). Features are class centres plus isotropic
    Gaussian noise, so ``noise`` controls how hard the task is independently of
    the structure.

    For every one of the ``avg_deg`` out-edges of a node, the target class is the
    same as the source with probability ``h`` and a uniformly random class
    otherwise. Sampling per edge (rather than rounding ``h * avg_deg``) keeps
    ``h`` continuous: with ``avg_deg = 10`` a rounding scheme can only realise
    homophilies that are multiples of 0.1, which is too coarse to locate a
    boundary near ``1/C = 0.2``.
    """
    rng = np.random.default_rng(seed)

    if imbalance > 0:
        p = (1.0 - imbalance) * np.ones(C) / C
        p[0] += imbalance
    else:
        p = np.ones(C) / C
    y = rng.choice(C, size=N, p=p)

    centres = rng.normal(0, 1, (C, d))
    x = centres[y] + rng.normal(0, noise, (N, d))

    # group node ids by class so that same-class sampling is O(1)
    by_class = [np.where(y == c)[0] for c in range(C)]

    src, dst = [], []
    for v in range(N):
        same = rng.random(avg_deg) < h
        n_same = int(same.sum())
        n_diff = avg_deg - n_same

        if n_same:
            pool = by_class[y[v]]
            pool = pool[pool != v]
            if len(pool):
                dst += list(rng.choice(pool, n_same, replace=len(pool) < n_same))
                src += [v] * n_same
        if n_diff:
            others = [c for c in range(C) if c != y[v]]
            tgt_cls = rng.choice(others, n_diff)
            for c in tgt_cls:
                pool = by_class[c]
                if len(pool):
                    dst.append(int(rng.choice(pool)))
                    src.append(v)

    src = np.asarray(src); dst = np.asarray(dst)
    # symmetrise + drop self loops
    src, dst = np.concatenate([src, dst]), np.concatenate([dst, src])
    keep = src != dst
    src, dst = src[keep], dst[keep]

    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
    data = Data(x=torch.tensor(x, dtype=torch.float32),
                edge_index=edge_index,
                y=torch.tensor(y, dtype=torch.long))

    perm = rng.permutation(N)
    n_tr, n_va = int(0.6 * N), int(0.2 * N)
    for name, idx in [("train_mask", perm[:n_tr]),
                      ("val_mask", perm[n_tr:n_tr + n_va]),
                      ("test_mask", perm[n_tr + n_va:])]:
        m = torch.zeros(N, dtype=torch.bool); m[idx] = True
        setattr(data, name, m)
    return data


def eval_on_data(data, C_num, seed=0, edge_mode="sup"):
    """Train a GCN on ``data`` and return (base_acc, AUC, dE, p_e, h_true)."""
    device = data.x.device
    y_np = data.y.cpu().numpy()
    pred = R.get_predictions(data, C_num, y_np, seed, edge_mode=edge_mode)
    C_metric, err, _ = R.compute_C_and_err(
        pred["y_pred"], pred["e_pred"], pred["src"], pred["dst"], pred["deg"], y_np)

    tm = data.test_mask.cpu().numpy()
    err_t, C_t = err[tm], C_metric[tm]
    base_acc = float(1.0 - err_t.mean())
    auc = float(roc_auc_score(err_t, C_t)) if len(np.unique(err_t)) > 1 else float("nan")
    if err_t.sum() > 0 and (1 - err_t).sum() > 0:
        dE = float(C_t[err_t == 1].mean() - C_t[err_t == 0].mean())
    else:
        dE = float("nan")
    h_true = float((y_np[pred["src"]] == y_np[pred["dst"]]).mean())
    return base_acc, auc, dE, pred["p_e_test"], h_true


def run_synthetic_sweep(outdir, C=5, N=1200, h_grid=None, seeds=(0, 1, 2),
                        noises=(2.2,), imbalances=(0.0, 0.6)):
    """Sweep h under two class-imbalance levels.

    The paper's flip mechanism requires majority-class collapse, which cannot
    occur when classes are balanced, so the sweep is reported for a balanced and
    an imbalanced class distribution side by side. ``noise = 2.2`` is chosen so
    that the base accuracy stays in a range where errors actually occur across
    the whole sweep (with lower noise the graph is solved perfectly for
    ``h > 0.3`` and the AUC becomes undefined).
    """
    h_grid = h_grid if h_grid is not None else [
        0.02, 0.08, 0.14, 0.20, 0.28, 0.40, 0.55, 0.75]
    all_rows = {}
    print(f"\n[Ablation #2] synthetic sweep: C={C}, N={N}, boundary 1/C = {1.0/C:.2f}")
    for noise in noises:
        for imb in imbalances:
            tag = f"noise{noise}_imb{imb}"
            rows = []
            print(f"\n  -- noise={noise}, imbalance={imb} --")
            print(f"  {'h_nom':>7}{'h_true':>8}{'base':>7}{'AUC':>7}{'dE':>8}{'p_e':>7}")
            for h in h_grid:
                accs, aucs, des, pes, hs = [], [], [], [], []
                for s in seeds:
                    dev = "cuda" if torch.cuda.is_available() else "cpu"
                    data = gen_synthetic(N=N, C=C, h=h, noise=noise,
                                         seed=s, imbalance=imb).to(dev)
                    base, auc, dE, pe, h_true = eval_on_data(data, C, seed=s)
                    accs.append(base); aucs.append(auc); des.append(dE)
                    pes.append(pe); hs.append(h_true)
                row = dict(h_nominal=h, h=float(np.nanmean(hs)),
                           base_acc=float(np.nanmean(accs)),
                           AUC=float(np.nanmean(aucs)), dE=float(np.nanmean(des)),
                           p_e=float(np.nanmean(pes)) if not np.all(np.isnan(pes)) else float("nan"),
                           AUC_std=float(np.nanstd(aucs)), dE_std=float(np.nanstd(des)),
                           boundary_1_over_C=1.0 / C)
                rows.append(row)
                print(f"  {h:>7.2f}{row['h']:>8.3f}{row['base_acc']:>7.3f}{row['AUC']:>7.3f}"
                      f"{row['dE']:>8.3f}{row['p_e']:>7.3f}")
            all_rows[tag] = rows

    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "ablation_synthetic.json"), "w") as f:
        json.dump({"C": C, "N": N, "seeds": list(seeds),
                   "noises": list(noises), "imbalances": list(imbalances),
                   "runs": all_rows}, f, indent=2)
    print(f"\n[saved] {outdir}/ablation_synthetic.json")
    return all_rows


# ============================================================
# Ablation #4: GNN depth (2 vs 3 vs 4 layers)
# ============================================================
class GCNDeep(torch.nn.Module):
    """Stack of ``n_layers`` GCNConv layers; used only for the depth ablation."""

    def __init__(self, in_dim, hidden, out_dim, n_layers=2, dropout=0.5):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden))
        for _ in range(n_layers - 2):
            self.convs.append(GCNConv(hidden, hidden))
        self.convs.append(GCNConv(hidden, out_dim))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


def train_deep(data, C, n_layers, seed, epochs=200, hidden=64, patience=50):
    torch.manual_seed(seed); np.random.seed(seed)
    model = GCNDeep(data.num_features, hidden, C, n_layers=n_layers).to(data.x.device)
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    best_val, best_out, wait = -1.0, None, 0
    for _ in range(epochs):
        model.train(); opt.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            va = (out[data.val_mask].argmax(1) == data.y[data.val_mask]).float().mean().item()
            if va > best_val:
                best_val, best_out, wait = va, out.clone(), 0
            else:
                wait += 1
                if wait >= patience:
                    break
    return model, best_out


def run_depth_ablation(outdir, datasets, depths=(2, 3, 4), seeds=(0, 1)):
    rows = []
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[Ablation #4] GNN depth: {depths} layers on {len(datasets)} datasets")
    print(f"{'dataset':<19}{'h':>6}{'depth':>7}{'base':>7}{'AUC':>7}{'dE':>8}{'p_e':>7}")
    for ds in datasets:
        for L in depths:
            accs, aucs, des, pes = [], [], [], []
            for s in seeds:
                data = R.load_dataset(ds, split_seed=s).to(dev)
                y_np = data.y.cpu().numpy()
                C_num = int(y_np.max()) + 1

                model, logits = train_deep(data, C_num, L, s)
                probs = F.softmax(logits, dim=1).cpu().numpy()
                y_pred = probs.argmax(1)

                # recompute edge consistency with the same helper used in the main run
                pred = R.get_predictions(data, C_num, y_np, s)   # for edge scores / p_e
                pred["y_pred"] = y_pred
                C_metric, err, _ = R.compute_C_and_err(
                    y_pred, pred["e_pred"], pred["src"], pred["dst"], pred["deg"], y_np)

                tm = data.test_mask.cpu().numpy()
                err_t, C_t = err[tm], C_metric[tm]
                accs.append(1.0 - err_t.mean())
                if len(np.unique(err_t)) > 1:
                    aucs.append(roc_auc_score(err_t, C_t))
                if err_t.sum() > 0 and (1 - err_t).sum() > 0:
                    des.append(C_t[err_t == 1].mean() - C_t[err_t == 0].mean())
                pes.append(pred["p_e_test"])

            h = float((y_np[data.edge_index[0].cpu().numpy()] ==
                       y_np[data.edge_index[1].cpu().numpy()]).mean())
            row = dict(dataset=ds, h=h, depth=L,
                       base_acc=float(np.mean(accs)),
                       AUC=float(np.nanmean(aucs)) if aucs else float("nan"),
                       dE=float(np.nanmean(des)) if des else float("nan"),
                       p_e=float(np.nanmean(pes)))
            rows.append(row)
            print(f"{ds:<19}{h:>6.3f}{L:>7d}{row['base_acc']:>7.3f}{row['AUC']:>7.3f}"
                  f"{row['dE']:>8.3f}{row['p_e']:>7.3f}")

    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "ablation_depth.json"), "w") as f:
        json.dump({"depths": list(depths), "seeds": list(seeds), "rows": rows}, f, indent=2)
    print(f"[saved] {outdir}/ablation_depth.json")
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--which", default="all", choices=["synthetic", "depth", "all"])
    p.add_argument("--outdir", default="results_ablation")
    p.add_argument("--datasets", default="cora,citeseer,pubmed,texas,cornell,chameleon_filtered,actor")
    args = p.parse_args()

    t0 = time.time()
    if args.which in ("synthetic", "all"):
        run_synthetic_sweep(args.outdir)
    if args.which in ("depth", "all"):
        run_depth_ablation(args.outdir, [d.strip() for d in args.datasets.split(",")])
    print(f"\n[done] ablation finished in {time.time() - t0:.1f}s")
