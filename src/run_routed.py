"""
run_routed.py -- The routed audit: the paper's final method (Sec. "Routing").

Computes, for every dataset and seed, the full battery of audit scores under a
single protocol and applies the paper's routing rule:

    route = model channel (mdis)   if 0.5 < p_e_obs < 1.0
    route = graph channel (curate2) otherwise

Channels scored:
  curate2    : edge-conflict channel (the original main method, unchanged)
  msp        : 1 - max softmax
  ent        : predictive entropy
  nc         : neighbourhood consistency
  mc         : MC-Dropout, 20 stochastic forward passes (mean predictive entropy)
  mdis       : model-disagreement channel -- a second GCN trained on the same
               split with a different initialisation (seed + 100); risk is
               1 - sum_c P0(c) * P1(c), the overlap deficit of the two members
  mdis_drop2 : cheap surrogate -- disagreement of two dropout passes of the
               same model (no second training)
  routed     : the routed audit under the rule above

The routing variable p_e_obs = mean(e_pred) is the rate at which the edge
scorer calls training-visible edges consistent; it is label-free at audit
time and is the same scalar already computed by the verdict.

Outputs results/routed/final_table.json (per-seed records + mean/std) and
results/routed/final_table.log (per-dataset summary).

Usage:
    python src/run_routed.py                    # all 13 datasets, seeds 0-4
    python src/run_routed.py --datasets cora    # subset
    python src/run_routed.py --seeds 0,1,2      # subset
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("CF_CONFLICT_MODE", "curate2")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

import run_main as RM

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = [0, 1, 2, 3, 4]
DSETS = ["cora", "citeseer", "pubmed", "chameleon_filtered", "squirrel_filtered",
         "texas", "cornell", "wisconsin", "actor", "tolokers", "amazon_ratings",
         "penn94", "genius"]
HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "..", "results", "routed")


def auc_of(e, s):
    try:
        return float(roc_auc_score(e, np.asarray(s, np.float64)))
    except Exception:
        return float("nan")


def two_dropout_passes(model, data):
    """Disagreement of two stochastic forward passes of the same model
    (a 2-pass MC surrogate). Risk = 1 - overlap of the two distributions."""
    model.eval()
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()
    with torch.no_grad():
        p1 = F.softmax(model(data.x, data.edge_index), 1)
        p2 = F.softmax(model(data.x, data.edge_index), 1)
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.eval()
    return (1.0 - (p1 * p2).sum(1)).cpu().numpy()


def run_seed(name, seed):
    data = RM.load_dataset(name, split_seed=seed).to(DEV)
    y_np = data.y.cpu().numpy()
    C_num = int(y_np.max()) + 1
    te = data.test_mask.cpu().numpy()

    t0 = time.time()
    pred = RM.get_predictions(data, C_num, y_np, seed, edge_mode="sup",
                              edge_space="auto", edge_feat="full")
    t_pred = time.time() - t0

    model = pred["model"]
    probs = pred["probs"]
    y_pred = pred["y_pred"]
    err = (y_pred != y_np).astype(int)
    src, dst, deg = pred["src"], pred["dst"], pred["deg"]
    N = pred["N"]

    # ---- graph channel: soft conflict + symmetric aggregation + background ----
    same = (y_pred[src] == y_pred[dst]).astype(np.float32)
    p = np.asarray(pred["p_uv"], np.float32)
    c_e = (same * (1.0 - p) + (1.0 - same) * p).astype(np.float32)
    Cs, Cn = RM.neighbor_conflict_pair(c_e, src, dst, deg, N, agg="sym")
    curate2 = (0.5 * (Cs + Cn)).astype(np.float32)

    # ---- model channel: second GCN, same split, different initialisation ----
    t0 = time.time()
    _, out2 = RM.train_gcn(data, C_num, seed + 100)
    P2 = F.softmax(out2, 1).cpu().numpy()
    mdis = (1.0 - (probs * P2).sum(1)).astype(np.float32)
    t_second_train = time.time() - t0

    mdis_drop2 = two_dropout_passes(model, data).astype(np.float32)

    t0 = time.time()
    mc_risk, _ = RM.mc_dropout_risk(model, data)
    t_mc = time.time() - t0

    # ---- the routing rule (parameter-free; zero is theory-given) ----
    p_e_obs = float(pred["e_pred"].mean())
    route = "model" if 0.5 < p_e_obs < 1.0 else "graph"
    risk = mdis if route == "model" else curate2

    # verdict scalars (unchanged from the main pipeline)
    s_obs = float((same * pred["e_pred"]).mean())
    cbar = float(Cs.mean())
    qhat = cbar - p_e_obs + 2.0 * s_obs

    return dict(
        seed=seed, route=route, p_e_obs=p_e_obs, cbar=cbar, s_obs=s_obs, qhat=qhat,
        h=float((y_np[src] == y_np[dst]).mean()),
        base_acc=float((y_pred[te] == y_np[te]).mean()),
        N=int(N), E=int(pred["E"]), C=int(C_num),
        t_pred=t_pred, t_second_train=t_second_train, t_mc=t_mc,
        aucs=dict(curate2=auc_of(err[te], curate2[te]),
                  msp=auc_of(err[te], (1.0 - pred["conf"])[te]),
                  ent=auc_of(err[te], pred["entropy"][te]),
                  nc=auc_of(err[te], RM.neighborhood_inconsistency(y_pred, src, dst, deg)[te]),
                  mc=auc_of(err[te], mc_risk[te]),
                  mdis=auc_of(err[te], mdis[te]),
                  mdis_drop2=auc_of(err[te], mdis_drop2[te]),
                  routed=auc_of(err[te], risk[te])),
        verdict=("trust" if (p_e_obs > 0.5 and qhat > 1.0 / C_num) else "unreliable"),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", type=str, default=",".join(DSETS))
    ap.add_argument("--seeds", type=str, default=",".join(map(str, SEEDS)))
    args = ap.parse_args()
    dsets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    os.makedirs(OUTDIR, exist_ok=True)
    print(f"[routed] device={DEV} torch={torch.__version__}", flush=True)

    out, t_all = {}, time.time()
    keys = ["curate2", "msp", "ent", "nc", "mc", "mdis", "mdis_drop2", "routed"]
    for name in dsets:
        per = [run_seed(name, s) for s in seeds]
        mean = {k: float(np.nanmean([x["aucs"][k] for x in per])) for k in keys}
        std = {k: float(np.nanstd([x["aucs"][k] for x in per])) for k in keys}
        out[name] = dict(
            per_seed=per, mean=mean, std=std,
            routes=[x["route"] for x in per],
            stable=len(set(x["route"] for x in per)) == 1,
        )
        print(f"{name:<20s} h={per[0]['h']:.3f} C={per[0]['C']} N={per[0]['N']} "
              f"E={per[0]['E']} p_e={np.mean([x['p_e_obs'] for x in per]):.3f} "
              f"routes={out[name]['routes']} stable={out[name]['stable']}", flush=True)
        print("    " + "  ".join(f"{k}={mean[k]:.4f}+-{std[k]:.4f}" for k in keys),
              flush=True)
        print(f"    [t] pred={np.mean([x['t_pred'] for x in per]):.2f}s "
              f"2nd_train={np.mean([x['t_second_train'] for x in per]):.1f}s "
              f"mc20={np.mean([x['t_mc'] for x in per]):.2f}s "
              f"elapsed={time.time() - t_all:.0f}s", flush=True)

    routed_mean = np.mean([out[k]["mean"]["routed"] for k in dsets])
    msp_mean = np.mean([out[k]["mean"]["msp"] for k in dsets])
    mc_mean = np.mean([out[k]["mean"]["mc"] for k in dsets])
    print(f"\n[routed] MEAN routed={routed_mean:.4f} msp={msp_mean:.4f} "
          f"mc={mc_mean:.4f}", flush=True)
    print(f"[routed] routes stable on "
          f"{sum(out[k]['stable'] for k in dsets)}/{len(dsets)} datasets", flush=True)

    path = os.path.join(OUTDIR, "final_table.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"[routed] saved -> {path}", flush=True)


if __name__ == "__main__":
    main()
