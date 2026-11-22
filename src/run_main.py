"""
run_all_experiments.py (v2 - Protocol-Fixed Version)
================================================================================
Fix log (the four protocol weaknesses declared and addressed in the paper):
  P1/(2) abstention protocol switched to validation-tuned thresholds: pick the
      80%-coverage threshold tau from val-node conflict scores, apply it
      unchanged to test; the test-tuned oracle variant is kept as the
      "test-tuning inflation" control of Ablation #5.
  (1) record the true AUC (roc_auc(err, C) on test) and p_e_test on test edges,
      with skill gain skill_gain = p_e_test - max(h, 1-h).
  (3) restore three abstention baselines -- MSP / Predictive Entropy /
      MC-Dropout (20 passes) -- all under the identical val-threshold protocol,
      each reporting its own AUC.
  (4) 5 seeds (0-4) for every metric, reported as mean+-std; sweep curves are
      averaged at matched target points.
================================================================================
"""
import os
# Force-override potentially stale environment variables
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import sys, json, warnings, time, argparse, gc, copy
warnings.filterwarnings("ignore")
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import coalesce, to_undirected
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# ============================================================
# 0. Experiment configuration
# ============================================================
SEEDS = [int(s) for s in os.environ.get("CF_SEEDS", "0,1,2,3,4").split(",")]
N_DROPOUT_PASSES = 20
ABST_COVERAGES = [0.6, 0.7, 0.8, 0.9]

# Diagnostic switches (all off by default; they do not change existing results):
# CF_EDGE_BALANCED=1  -> adds class_weight='balanced' to the edge classifier, countering
#                        the collapse caused by the prior h dominating the label distribution
#                        (on strongly heterophilous graphs h~0.06, predicting the majority class alone yields 94%).
EDGE_BALANCED = os.environ.get("CF_EDGE_BALANCED", "0") == "1"
# CF_EDGE_CALIB=<raw|rank|z|otsu>  -> label-free recalibration of edge probabilities
#   (diagnostics for the unsup branch only). Motivation: the unsup p_uv=(cos+1)/2 has
#   mean 0.85-0.98 on every graph, so a 0.5 threshold means "declare every edge
#   same-class" and p_e degenerates to h; the true optimum sits at 0.92-0.95. The
#   absolute value of p has no calibration meaning -- only its ranking does -- so a
#   label-free recalibration maps p back into a usable range:
#     rank -- rank normalisation; threshold 0.5 = the median (monotone in the soft score,
#     z    -- z-score then sigmoid (adapts to mean and variance)
#     otsu -- histogram Otsu binarisation; fully adaptive, no assumption on h
EDGE_CALIB = os.environ.get("CF_EDGE_CALIB", "raw")
# CF_CONFLICT_MODE=soft -> soft conflict values (posterior expectation); default hard
#                        keeps existing results
# CF_CONFLICT_MODE=nbr  -> soft conflict values + neighbour-edge conflict
#                        (one O(|E|) rewrite, see Section 3)
CONFLICT_MODE = os.environ.get("CF_CONFLICT_MODE", "hard")
# The two operators of curate2 (for ablations; the defaults are the paper's method):
CURATE_AGG = os.environ.get("CF_AGG", "sym")          # sym | mean
CURATE_COMBINE = os.environ.get("CF_COMBINE", "avg")  # avg | none
# Neighbour-rewrite strength alpha: q_e=(1-alpha)c_e+alpha*(mean neighbour-edge conflict).
#                        alpha=0 falls back to the original method entirely.
NBR_ALPHAS = (-1.0, -0.5, 0.25, 0.5, 1.0, 2.0)
# Multiplicative contrast strength of nbr2: C*(1-b*M), M the neighbour conflict background.
NBR_PROD_BETAS = (0.5, 1.0, 1.5)
# nbr3: the mechanism is hard-gated by the mean conflict level Cbar=mean_v C(v)
# (uses only the conflict field itself: no labels, no h). A low Cbar -> conflict is
# sparse and neighbour conflict is mostly noise -> smooth; a high Cbar -> conflict is
# everywhere -> multiplicative contrast. On the 11 benchmarks Cbar forms two clusters,
# 0.08-0.16 and 0.54-0.90; a threshold inside that gap is insensitive to its exact value.
NBR_TAU = float(os.environ.get("CF_NBR_TAU", "0.35"))
NBR_PROD_BETA = float(os.environ.get("CF_NBR_BETA", "1.5"))
# Fall back to alpha=0 when the val gain is below this margin (prevents selection noise
# from degrading results).
NBR_MARGIN = float(os.environ.get("CF_NBR_MARGIN", "0.005"))

# ============================================================
# 1. Self-contained data loading (local paths, never downloads at run time)
# ============================================================
def get_fixed_split(num_nodes, seed=42):
    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(num_nodes, generator=rng)
    n = num_nodes
    tr, va = int(0.6 * n), int(0.2 * n)
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[perm[:tr]] = True
    val_mask[perm[tr:tr + va]] = True
    test_mask[perm[tr + va:]] = True
    return train_mask, val_mask, test_mask

def load_npz_generic(npz_path, split_idx):
    obj = np.load(npz_path, allow_pickle=True)
    X = obj["node_features"].astype(np.float32)
    y = obj["node_labels"].astype(np.int64)
    if y.ndim > 1: y = y.squeeze(-1)
    edges = obj["edges"]

    if edges.shape[0] == 2 and edges.shape[1] != 2:
        ei = torch.tensor(edges, dtype=torch.long)
    elif edges.shape[1] == 2 and edges.shape[0] != 2:
        ei = torch.tensor(edges.T, dtype=torch.long)
    else:
        ei = torch.tensor(edges, dtype=torch.long)

    src, dst = ei[0].numpy(), ei[1].numpy()
    ue = set(zip(np.minimum(src, dst).tolist(), np.maximum(src, dst).tolist()))
    ei = torch.tensor([[e[0] for e in ue], [e[1] for e in ue]], dtype=torch.long)
    ei = coalesce(to_undirected(ei))

    num_splits = obj["train_masks"].shape[0]
    real_split = split_idx % num_splits  # 10 official splits, rotated by seed

    return Data(
        x=torch.tensor(X, dtype=torch.float), edge_index=ei,
        y=torch.tensor(y, dtype=torch.long),
        train_mask=torch.tensor(obj["train_masks"][real_split], dtype=torch.bool),
        val_mask=torch.tensor(obj["val_masks"][real_split], dtype=torch.bool),
        test_mask=torch.tensor(obj["test_masks"][real_split], dtype=torch.bool),
    )

def _symmetrize_edge_index(ei):
    """Deduplicate and symmetrise (same convention as load_npz_generic: h is computed
    on the deduplicated undirected graph)."""
    src, dst = ei[0].numpy(), ei[1].numpy()
    ue = set(zip(np.minimum(src, dst).tolist(), np.maximum(src, dst).tolist()))
    ei = torch.tensor([[e[0] for e in ue], [e[1] for e in ue]], dtype=torch.long)
    return coalesce(to_undirected(ei))


def load_external(name, split_seed, root):
    """Three public large-scale graphs (NeurIPS'21 non-homophily benchmark + Platonov release):

    genius        421,961 nodes / 922,868 edges   h~0.59   C=2  Lim & Benson 2020
                  (2023 updated release; no official split -> random 60/20/20, seed-controlled)
    penn94         41,554 nodes / 1,362,229 edges h~0.47   C=2  Traud et al. 2012
                  (5 official splits, rotated by seed; gender label, 0 = unlabeled)
    roman_empire   22,662 nodes /  66,566 edges   h~0.20   C=18 Platonov et al. 2022
                  (10 official splits, rotated by seed)
    """
    import scipy.io as sio

    if name == "genius":
        d = sio.loadmat(os.path.join(root, "genius.mat"))
        X = np.asarray(d["node_feat"]).astype(np.float32)
        y = np.asarray(d["label"]).squeeze().astype(np.int64)
        ei = torch.tensor(np.asarray(d["edge_index"]), dtype=torch.long)
        ei = _symmetrize_edge_index(ei)
        tr, vl, te = get_fixed_split(len(y), seed=split_seed)
        return Data(x=torch.tensor(X), edge_index=ei, y=torch.tensor(y),
                    train_mask=tr, val_mask=vl, test_mask=te)

    if name == "penn94":
        from sklearn.preprocessing import label_binarize
        d = sio.loadmat(os.path.join(root, "Penn94.mat"))
        A = d["A"].tocsr()
        metadata = d["local_info"].astype(np.float64)
        y = (metadata[:, 1] - 1).astype(np.int64)          # gender; 0-1 = -1 (unlabeled)
        ei = torch.tensor(np.vstack(A.nonzero()), dtype=torch.long)
        ei = _symmetrize_edge_index(ei)
        # features: year + per-column one-hot of the remaining metadata (Lim et al.)
        vals = np.hstack((metadata[:, :1], metadata[:, 2:]))
        feats = []
        for col in range(vals.shape[1]):
            c = vals[:, col]
            feats.append(label_binarize(c, classes=np.unique(c)).astype(np.float32))
        X = np.hstack(feats)
        N = len(y)
        labeled = torch.tensor(y >= 0)
        sp = np.load(os.path.join(root, "penn94_splits.npy"), allow_pickle=True)
        # 5 official splits: each element is {'train': [...], 'valid': [...], 'test': [...]}
        if sp.dtype == object and len(sp) == 5:
            item = sp[split_seed % len(sp)]
            trm = torch.zeros(N, dtype=torch.bool); trm[torch.tensor(item["train"])] = True
            vlm = torch.zeros(N, dtype=torch.bool); vlm[torch.tensor(item["valid"])] = True
            tem = torch.zeros(N, dtype=torch.bool); tem[torch.tensor(item["test"])] = True
            trm &= labeled; vlm &= labeled; tem &= labeled
            return Data(x=torch.tensor(X), edge_index=ei, y=torch.tensor(y),
                        train_mask=trm, val_mask=vlm, test_mask=tem)
        # fallback: random 60/20/20 (over labeled nodes only)
        idx = np.where(y >= 0)[0]
        rng = np.random.RandomState(1000 + split_seed)
        perm = rng.permutation(idx)
        n = len(idx); ntr, nvl = int(0.6 * n), int(0.2 * n)
        trm = torch.zeros(N, dtype=torch.bool); trm[perm[:ntr]] = True
        vlm = torch.zeros(N, dtype=torch.bool); vlm[perm[ntr:ntr+nvl]] = True
        tem = torch.zeros(N, dtype=torch.bool); tem[perm[ntr+nvl:]] = True
        return Data(x=torch.tensor(X), edge_index=ei, y=torch.tensor(y),
                    train_mask=trm, val_mask=vlm, test_mask=tem)

    if name == "roman_empire":
        obj = np.load(os.path.join(root, "roman_empire.npz"))
        X = obj["node_features"].astype(np.float32)
        y = obj["node_class"].astype(np.int64) if "node_class" in obj.files \
            else obj["node_labels"].astype(np.int64)
        if y.ndim > 1: y = y.squeeze(-1)
        edges = obj["edges"]
        ei = torch.tensor(edges if edges.shape[0] == 2 else edges.T, dtype=torch.long)
        ei = _symmetrize_edge_index(ei)
        ns = obj["train_masks"].shape[0]
        k = split_seed % ns
        return Data(x=torch.tensor(X), edge_index=ei, y=torch.tensor(y),
                    train_mask=torch.tensor(obj["train_masks"][k], dtype=torch.bool),
                    val_mask=torch.tensor(obj["val_masks"][k], dtype=torch.bool),
                    test_mask=torch.tensor(obj["test_masks"][k], dtype=torch.bool))

    raise ValueError(f"Unknown external dataset: {name}")


def load_dataset(name, split_seed=0):
    # Data paths: by default read data/ produced by prepare_data.py in this directory;
    # overridable through environment variables.
    _here = os.path.dirname(os.path.abspath(__file__))
    HETERO_ROOT = os.environ.get("CF_HETERO_ROOT", os.path.join(_here, "data", "hetero"))
    PLANETOID_ROOT = os.environ.get("CF_PLANETOID_ROOT", os.path.join(_here, "data", "planetoid"))
    PATHS = {
        "chameleon_filtered": os.path.join(HETERO_ROOT, "chameleon_filtered.npz"),
        "squirrel_filtered": os.path.join(HETERO_ROOT, "squirrel_filtered.npz"),
        "texas": os.path.join(HETERO_ROOT, "texas.npz"),
        "cornell": os.path.join(HETERO_ROOT, "cornell.npz"),
        "wisconsin": os.path.join(HETERO_ROOT, "wisconsin.npz"),
        "actor": os.path.join(HETERO_ROOT, "actor.npz"),
        "tolokers": os.path.join(HETERO_ROOT, "tolokers.npz"),
        "amazon_ratings": os.path.join(HETERO_ROOT, "amazon_ratings.npz"),
    }

    if name in ["cora", "citeseer", "pubmed"]:
        from torch_geometric.datasets import Planetoid
        d = Planetoid(root=PLANETOID_ROOT, name=name.capitalize())[0].clone()
        # paper protocol: random 60/20/20 split; seed controls split randomness
        tr, vl, te = get_fixed_split(d.num_nodes, seed=split_seed)
        d.train_mask, d.val_mask, d.test_mask = tr, vl, te
        return d

    elif name in ("genius", "penn94", "roman_empire"):
        EXT_ROOT = os.environ.get("CF_EXT_ROOT", os.path.join(_here, "data", "external"))
        return load_external(name, split_seed, EXT_ROOT)

    elif name in PATHS:
        return load_npz_generic(PATHS[name], split_seed)
    else:
        raise ValueError(f"Unknown dataset: {name}")

# ============================================================
# 2. GCN model and predictions (raw features + Jaccard)
# ============================================================
class GCN(torch.nn.Module):
    def __init__(self, in_dim, hidden, out_dim, dropout=0.5):
        super().__init__()
        self.c1 = GCNConv(in_dim, hidden); self.c2 = GCNConv(hidden, out_dim)
        self.dropout = dropout
    def encode(self, x, ei):
        """Hidden representation z_v after the first GCN layer (dim = hidden).

        The paper's Sec. III defines edge consistency on the model's own latent
        embedding, so the audit must measure similarity on z, not on the raw input
        features x. z has fixed dimension 64 regardless of the input dimension d --
        this is where O(|E|*d) -> O(|E|*64) comes from.
        """
        return F.relu(self.c1(x, ei))
    def forward(self, x, ei):
        h = self.encode(x, ei)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.c2(h, ei)

def train_gcn(data, C, seed, epochs=200, hidden=64, lr=0.01, weight_decay=5e-4):
    torch.manual_seed(seed); np.random.seed(seed)
    model = GCN(data.num_features, hidden, C).to(data.x.device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val, best_out = -1.0, None
    wait, patience = 0, 50

    for epoch in range(epochs):
        model.train(); opt.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            val_acc = (out[data.val_mask].argmax(1) == data.y[data.val_mask]).float().mean().item()
            if val_acc > best_val:
                best_val = val_acc; best_out = out.clone(); wait = 0
            else:
                wait += 1
                if wait >= patience: break
    return model, best_out

def mc_dropout_risk(model, data, n_passes=N_DROPOUT_PASSES, force_cpu=True):
    """MC-Dropout: put only the Dropout layers into train mode, take n_passes stochastic
    forward passes; the risk score is the entropy of the mean predictive distribution
    (the standard selective-prediction baseline).

    force_cpu: unify timing on CPU. The Conflict Field pipeline is vectorised numpy/CPU;
    timing MC-Dropout on a GPU would compare across different hardware and make the
    speed-up meaningless. The forward pass runs on CPU and the entropy matches the GPU
    run (same model, same weights, same dropout-mask distribution).
    """
    on_cuda = next(model.parameters()).is_cuda
    if force_cpu and on_cuda:
        m = copy.deepcopy(model).cpu()
        x, ei = data.x.cpu(), data.edge_index.cpu()
    else:
        m = model
        x, ei = data.x, data.edge_index

    m.eval()
    for mod in m.modules():
        if isinstance(mod, torch.nn.Dropout):
            mod.train()

    t0 = time.perf_counter()
    prob_sum = 0.0
    with torch.no_grad():
        for _ in range(n_passes):
            out = m(x, ei)
            prob_sum = prob_sum + F.softmax(out, dim=1)
    p_mean = prob_sum / n_passes
    ent = -(p_mean * torch.log(p_mean + 1e-12)).sum(dim=1)
    t_elapsed = time.perf_counter() - t0

    for mod in m.modules():
        if isinstance(mod, torch.nn.Dropout):
            mod.eval()
    return ent.cpu().numpy(), t_elapsed

def get_predictions(data, C_num, y_np, seed=0, edge_mode="sup",
                    edge_space="auto", edge_feat="full"):
    """Train a GCN and compute edge-consistency scores.

    edge_mode:
      "sup"   -- logistic regression fitted on training-edge labels (main configuration;
                 no labels needed at audit time)
      "unsup" -- pure cosine threshold, touches no labels (Ablation #1 control)
    edge_space:
      "auto"  -- use the latent embedding when d > 64 (project to 64), else raw features.
                 Default. Complexity O(|E|*min(d, 64)): avoids blowing up memory on
                 high-dimensional graphs without adding dimensions on low-dimensional ones.
      "emb"   -- force the latent embedding (the literal Sec. III definition)
      "raw"   -- force raw input features (legacy implementation)
    edge_feat:
      "full"    -- cos + log1p(|deg diff|) + binary overlap (3 features)
      "cos_deg" -- cos + log1p(|deg diff|) (2 features, cheapest)
      "cos"     -- cos only
    """
    N = data.num_nodes
    model, logits = train_gcn(data, C_num, seed)
    probs = F.softmax(logits, dim=1).cpu().numpy()
    y_pred = probs.argmax(1)
    conf = probs.max(1)
    entropy = -(probs * np.log(probs + 1e-12)).sum(1)

    # ---------------- audit-stage timing ----------------
    # The paper reports only this segment: how long it takes to audit one graph once
    # the model is trained. The one-off edge-scorer fit is timed separately as t_fit
    # and never mixed into the audit cost.
    # the one-off edge-scorer fit is timed separately as t_fit and never mixed into the audit cost.
    t0 = time.perf_counter()
    src, dst = data.edge_index[0].cpu().numpy(), data.edge_index[1].cpu().numpy()

    if edge_space == "auto":
        # project only when it matters: switch to the embedding when d > hidden(64).
        # Cora (d=1433) / Squirrel (d=2089) use emb; Tolokers (d=1) stays on raw.
        use_emb = data.num_features > 64
    else:
        use_emb = (edge_space == "emb")

    if use_emb:
        # Sec. III: edge consistency is computed on the model's own latent embedding
        # (dim=64, independent of the feature dimension d)
        with torch.no_grad():
            Z = model.encode(data.x, data.edge_index).cpu().numpy().astype(np.float32)
    else:
        Z = data.x.cpu().numpy().astype(np.float32)   # low-dim graph: raw features are cheaper

    Zn = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
    t_embed = time.perf_counter() - t0

    t0 = time.perf_counter()
    cos = (Zn[src] * Zn[dst]).sum(1)
    t_sim = time.perf_counter() - t0

    deg = (np.bincount(src, minlength=N) + np.bincount(dst, minlength=N)).astype(np.float32)
    feats = [cos]
    t_jac = 0.0
    if edge_feat in ("cos_deg", "full", "pred"):
        feats.append(np.log1p(np.abs(deg[src] - deg[dst])))
    if edge_feat in ("full", "pred"):
        # Binary overlap. Chunked to avoid a dense E x d intermediate: the naive
        # x_bin[src]*x_bin[dst] costs 4.6e9 multiplications and an 18 GB temporary on
        # Squirrel -- that is why the old implementation broke its O(|E|) claim and
        # took 4.16 s on Squirrel.
        x_bin = (Z > 0).astype(np.float32)
        row_nnz = x_bin.sum(1)
        jac = np.empty(len(src), dtype=np.float32)
        CH = 262144
        t0 = time.perf_counter()
        for i in range(0, len(src), CH):
            s, d = src[i:i + CH], dst[i:i + CH]
            inter = (x_bin[s] * x_bin[d]).sum(1)
            jac[i:i + CH] = inter / (row_nnz[s] + row_nnz[d] - inter + 1e-9)
        t_jac = time.perf_counter() - t0
        feats.append(jac)
    if edge_feat == "pred":
        # Prediction-aware edge features (all label-free: model outputs only, no labels
        # read at audit time).
        # Motivation: on homophilous graphs the embedding cosine collapses to a constant
        # (same-class neighbours are naturally similar) and e_pred loses information.
        # 1) raw-feature cosine: an input-space view complementing the embedding
        Xr = data.x.cpu().numpy().astype(np.float32)
        Xrn = Xr / (np.linalg.norm(Xr, axis=1, keepdims=True) + 1e-9)
        feats.append((Xrn[src] * Xrn[dst]).sum(1))
        # 2) JS divergence of the endpoint predictive distributions: misclassified nodes differ
        P = np.clip(probs, 1e-12, 1.0)
        M = 0.5 * (P[src] + P[dst])
        kl_u = (P[src] * (np.log(P[src]) - np.log(M))).sum(1)
        kl_v = (P[dst] * (np.log(P[dst]) - np.log(M))).sum(1)
        feats.append((0.5 * (kl_u + kl_v)).astype(np.float32))
        # 3) endpoint confidence gap
        feats.append(np.abs(conf[src] - conf[dst]).astype(np.float32))

    X_edge = np.stack(feats, 1)
    e_true_all = (y_np[src] == y_np[dst]).astype(int)

    train_mask_np = data.train_mask.cpu().numpy()
    val_mask_np = data.val_mask.cpu().numpy()
    test_mask_np = data.test_mask.cpu().numpy()
    mask_tr = train_mask_np[src] & train_mask_np[dst]
    mask_va = val_mask_np[src] & val_mask_np[dst]
    mask_te = test_mask_np[src] & test_mask_np[dst]

    p_uv = np.clip((cos + 1) / 2, 0, 1)
    p_e_train, p_e_val, p_e_test = float("nan"), float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    edge_fit_ok = False
    t_fit = t_pred = 0.0
    if edge_mode == "sup" and mask_tr.sum() >= 20:
        y_edge_tr = e_true_all[mask_tr]
        # both classes must appear among training edges, otherwise the LR collapses
        # to a constant predictor (happens on small / extremely heterophilous graphs)
        if len(np.unique(y_edge_tr)) == 2:
            if mask_tr.sum() > 50000:
                idx = rng.choice(np.where(mask_tr)[0], 50000, replace=False)
                X_fit, y_fit = X_edge[idx], e_true_all[idx]
            else:
                X_fit, y_fit = X_edge[mask_tr], y_edge_tr

            t0 = time.perf_counter()
            lr = LogisticRegression(max_iter=1000, C=1.0, n_jobs=1,
                                    class_weight=("balanced" if EDGE_BALANCED else None))
            lr.fit(X_fit, y_fit)
            t_fit = time.perf_counter() - t0

            # in-sample (train edges; reference only, optimistically biased)
            e_pred_tr = (lr.predict_proba(X_fit)[:, 1] >= 0.5).astype(int)
            p_e_train = float((e_pred_tr == y_fit).mean())

            t0 = time.perf_counter()
            p_uv = lr.predict_proba(X_edge)[:, 1]
            t_pred = time.perf_counter() - t0
            edge_fit_ok = True

    # ---- label-free recalibration (only meaningful for the unsup cosine pseudo-
    # probabilities; the sup LR output is already calibrated) ----
    if EDGE_CALIB != "raw" and edge_mode == "unsup" and len(p_uv) > 100:
        if EDGE_CALIB == "rank":
            # rank normalisation: the absolute value of p is untrustworthy, keep only
            # the ranking; a 0.5 threshold is the median.
            order = np.argsort(p_uv, kind="stable")
            r = np.empty(len(p_uv), dtype=np.float32)
            r[order] = np.arange(len(p_uv), dtype=np.float32)
            p_uv = ((r + 0.5) / len(p_uv)).astype(np.float32)
        elif EDGE_CALIB == "z":
            mu, sd = float(p_uv.mean()), float(p_uv.std() + 1e-9)
            p_uv = (1.0 / (1.0 + np.exp(-(p_uv - mu) / sd))).astype(np.float32)
        elif EDGE_CALIB == "otsu":
            # histogram Otsu: maximise between-class variance; fully adaptive, no
            # assumption on h
            hist, edges = np.histogram(p_uv, bins=256, range=(0.0, 1.0))
            hist = hist.astype(np.float64)
            w0 = np.cumsum(hist)
            w1 = w0[-1] - w0
            centers = 0.5 * (edges[:-1] + edges[1:])
            m0 = np.cumsum(hist * centers) / np.maximum(w0, 1e-9)
            m1 = (np.sum(hist * centers) - np.cumsum(hist * centers)) / np.maximum(w1, 1e-9)
            var = w0 * w1 * (m0 - m1) ** 2
            thr = float(centers[int(np.argmax(var))])
            p_uv = ((p_uv - thr) / max(1e-3, float(p_uv.max() - thr)) * 0.5 + 0.5)
            p_uv = np.clip(p_uv, 0.0, 1.0).astype(np.float32)

    # p_e is reported only when the edge predictor actually works: sup requires a
    # successful fit, unsup uses the cosine threshold. Otherwise it stays NaN -- the
    # "accuracy" of a degenerate predictor is an illusion (e.g. the majority-class rate 1-h).
    # the "accuracy" of a degenerate predictor is an illusion (e.g. the majority-class rate 1-h).
    if edge_mode == "unsup" or edge_fit_ok:
        if mask_va.sum() >= 20:
            p_e_val = float(((p_uv[mask_va] >= 0.5).astype(int) == e_true_all[mask_va]).mean())
        if mask_te.sum() >= 20:
            p_e_test = float(((p_uv[mask_te] >= 0.5).astype(int) == e_true_all[mask_te]).mean())

    e_pred = (p_uv >= 0.5).astype(int)
    t_audit = t_embed + t_sim + t_jac + t_pred
    return {
        "model": model, "probs": probs,          # full distributions needed for score fusion
        "y_pred": y_pred, "conf": conf, "entropy": entropy,
        "e_pred": e_pred, "p_uv": p_uv,          # continuous edge verdicts, for soft-conflict experiments
        "e_true_all": e_true_all, "p_e_train": p_e_train,
        "p_e_val": p_e_val, "p_e_test": p_e_test, "edge_mode": edge_mode,
        "edge_space": ("emb" if use_emb else "raw") , "edge_space_arg": edge_space,
        "edge_feat": edge_feat,
        "edge_fit_ok": edge_fit_ok, "n_train_edges": int(mask_tr.sum()),
        # t_edge keeps its legacy field name; its meaning is now "audit stage",
        # excluding the one-off fit
        "t_edge": t_audit, "t_audit": t_audit, "t_fit": t_fit,
        "t_embed": t_embed, "t_sim": t_sim, "t_jac": t_jac, "t_pred": t_pred,
        "src": src, "dst": dst, "deg": deg, "N": N, "E": len(src)
    }

# ============================================================
# 3. Computation and experiments
# ============================================================
def _edge_disagree(y_pred, e_pred, src, dst, p_uv=None, mode="hard"):
    """Edge-level disagreement c_uv (hard: a 0/1 indicator; soft: the Bernoulli posterior expectation)."""
    same_pred = (y_pred[src] == y_pred[dst]).astype(np.float32)
    if mode == "soft" and p_uv is not None:
        p = np.asarray(p_uv, dtype=np.float32)
        return (same_pred * (1.0 - p) + (1.0 - same_pred) * p).astype(np.float32)
    return (same_pred != e_pred).astype(np.float32)


def _edge_cos(A, src, dst, budget=4_000_000):
    """Per-edge cosine, chunked to avoid a dense |E| x d intermediate (which would eat tens of GB on Squirrel)."""
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
    out = np.empty(len(src), dtype=np.float32)
    d = A.shape[1]
    ch = max(1024, int(budget // max(d, 1)))
    for i in range(0, len(src), ch):
        s, t = src[i:i + ch], dst[i:i + ch]
        out[i:i + ch] = (An[s] * An[t]).sum(1)
    return out


def _node_agg(c_e, src, dst, deg, N, agg="mean"):
    if agg == "max":
        C = np.zeros(N, dtype=np.float32)
        np.maximum.at(C, src, c_e)
        np.maximum.at(C, dst, c_e)
        return C
    if agg == "sym":
        # neighbour side sqrt, self side linear: C(v)=sum_u c_uv/(d_v*sqrt(d_u)).
        # Equivalent to weighting each edge by 1/sqrt(d_u d_v), aggregating, then
        # dividing by sqrt(d_v) (i.e. D^{-1} A D^{-1/2}).
        # Compared with plain degree division: a hub neighbour no longer leaks in with
        # weight 1/d nor amplifies itself by d; the cost is one extra per-edge division
        # (O(|E|), no new memory).
        w = c_e / np.sqrt(np.maximum(deg[src] * deg[dst], 1.0))
        num = np.bincount(src, weights=w, minlength=N) + \
            np.bincount(dst, weights=w, minlength=N)
        return (num / np.sqrt(np.maximum(deg, 1.0))).astype(np.float32)
    C = np.bincount(src, weights=c_e, minlength=N) + \
        np.bincount(dst, weights=c_e, minlength=N)
    return (C / np.maximum(deg, 1)).astype(np.float32)


def neighbor_inv_denom(deg, src, dst):
    """1/(deg_u+deg_v-2): purely topological, shared by the hard and soft base conflict values to avoid a redundant division."""
    du, dv = deg[src], deg[dst]
    return (1.0 / np.maximum(du + dv - 2.0, 1.0)).astype(np.float32)


def neighbor_conflict_pair(c_e, src, dst, deg, N, inv_denom=None, agg="mean"):
    """Propagate each edge's conflict value one step onto its "neighbour edges";
    returns (C_self, C_nbr).

    The neighbour edges of e=(u,v) are the other edges sharing an endpoint with it:
        m_e = (Σ_{e'∋u} c_e' + Σ_{e'∋v} c_e' − 2c_e) / (deg_u + deg_v − 2)
    the mean conflict level around this edge. After rewriting, node-level aggregation yields C_nbr.

    Key fact: the rewrite q_e=(1-alpha)c_e+alpha*m_e is linear in alpha, so the whole
        C(α) = (1−α)·C_self + α·C_nbr
    alpha family needs only these two vectors; sweeping alpha is an O(N) blend --
    edges are never re-walked.
    Cost: 2 bincounts (the src/dst aggregation of m_e), O(|E|) time and O(N+|E|) memory;
    no matrix products, no O(E*d) intermediates.

    agg: node-level aggregation operator, same as _node_agg ("mean" divides by degree /
    "sym" symmetric normalisation).
    """
    C_self = _node_agg(c_e, src, dst, deg, N, agg=agg)
    if agg == "mean":
        S = C_self * deg                  # sum of c_e over edges incident to u; numerator reused, no extra bincount
    else:
        S = np.bincount(src, weights=c_e, minlength=N) + \
            np.bincount(dst, weights=c_e, minlength=N)
    if inv_denom is None:
        inv_denom = neighbor_inv_denom(deg, src, dst)
    m = ((S[src] + S[dst] - 2.0 * c_e) * inv_denom).astype(np.float32)
    C_nbr = _node_agg(m, src, dst, deg, N, agg=agg)
    return C_self, C_nbr


def compute_C_and_err(y_pred, e_pred, src, dst, deg, y_np, p_uv=None, mode="hard", agg="mean",
                      alpha=0.0):
    """Algorithm 1, phase 2: aggregate edge-level disagreement into the node-level
    conflict score C(v), and return the true error vector.

    mode="hard": c_uv = I(yhat_u=yhat_v) XOR ehat_uv, a 0/1 edge verdict (the original implementation)
    mode="soft": c_uv = E[same quantity], the 0/1 verdict replaced by its posterior
                 probability p_uv:
                 c_uv = I(ŷu=ŷv)·(1−p_uv) + I(ŷu≠ŷv)·p_uv
                 the hard indicator is a Bernoulli sample of the same quantity; taking
                 the expectation only removes sampling noise and introduces no new
                 information (p_uv comes from the fitted edge classifier and remains
                 label-free at audit time).
    agg="mean": C(v) = sum_u c_uv / |N(v)| (the original, degree normalisation)
    agg="max":  C(v) = max_u c_uv (worst neighbouring-edge conflict; a single highly
                suspicious edge exposes the node -- suited to graphs where conflict is
                sparse and the mean gets diluted by many zeros)
    alpha:      neighbour-rewrite strength (see neighbor_conflict_pair); default 0 = the original method.

    Returns (C, err, t_agg); t_agg is the wall-clock time of the aggregation stage,
    cited in the Scalability section.
    """
    t0 = time.perf_counter()
    N = len(y_np)
    disagree = _edge_disagree(y_pred, e_pred, src, dst, p_uv=p_uv, mode=mode)

    if alpha != 0.0 and agg != "max":
        C_self, C_nbr = neighbor_conflict_pair(disagree, src, dst, deg, N)
        C = ((1.0 - alpha) * C_self + alpha * C_nbr).astype(np.float32)
    else:
        C = _node_agg(disagree, src, dst, deg, N, agg=agg)
    t_agg = time.perf_counter() - t0

    err = (y_pred != y_np).astype(int)
    return C, err, t_agg

def neighborhood_inconsistency(y_pred, src, dst, deg):
    """Zero-cost baseline: neighbourhood predicted-label agreement. Risk = 1 - (fraction
    of neighbours whose predicted label equals the node's). Uses only predicted labels
    -- no features, no edge classifier; the Ablation #3 control in the paper."""
    N = len(y_pred)
    same = (y_pred[src] == y_pred[dst]).astype(np.float32)
    agree = np.bincount(src, weights=same, minlength=N) + \
            np.bincount(dst, weights=same, minlength=N)
    return 1.0 - agree / np.maximum(deg, 1)

def exp_A_significance(C, err, test_mask):
    C_t, err_t = C[test_mask], err[test_mask]
    tau = np.median(C_t)
    high, low = err_t[C_t > tau], err_t[C_t <= tau]
    n_h, n_l = len(high), len(low)
    if n_h == 0 or n_l == 0: return {"RD": 0, "RR": 1, "z": 0, "p_z": 1}

    p_h, p_l = high.mean(), low.mean()
    p_pool = (p_h*n_h + p_l*n_l) / (n_h + n_l)
    se = np.sqrt(p_pool*(1-p_pool)*(1/n_h + 1/n_l) + 1e-12)
    z = (p_h - p_l) / se
    p_val = 1 - stats.norm.cdf(z)

    return {"z": z, "p_z": p_val, "RD": p_h - p_l, "RR": p_h/max(p_l, 1e-9)}

def exp_B_scan_pe(pred, y_np, test_mask, pe_targets=None):
    if pe_targets is None: pe_targets = np.linspace(0.5, 0.9, 9)
    src, dst, deg = pred["src"], pred["dst"], pred["deg"]
    y_pred, e_pred = pred["y_pred"], pred["e_pred"]
    e_true = pred["e_true_all"]
    rng = np.random.default_rng(0)
    results = []

    for pe_t in pe_targets:
        p_e_now = (e_pred == e_true).mean()
        if p_e_now < pe_t: continue
        gamma = (p_e_now - pe_t) / (2*p_e_now - 1) if (2*p_e_now - 1) > 1e-6 else 0
        flip_mask = rng.random(len(e_pred)) < gamma
        e_mod = e_pred.copy(); e_mod[flip_mask] = 1 - e_mod[flip_mask]

        C, err, _ = compute_C_and_err(y_pred, e_mod, src, dst, deg, y_np)
        C_t, err_t = C[test_mask], err[test_mask]
        dE = C_t[err_t == 1].mean() - C_t[err_t == 0].mean() if err_t.sum() > 0 and (1-err_t).sum() > 0 else np.nan
        auc = roc_auc_score(err_t, C_t) if len(np.unique(err_t)) > 1 else np.nan
        results.append({"p_e": float(pe_t), "dE": float(dE), "auc": float(auc)})
    return results

def exp_C_scan_pn(pred, y_np, test_mask, pn_targets=None, C_num=None):
    if C_num is None: C_num = int(y_np.max()) + 1
    src, dst, deg, e_pred = pred["src"], pred["dst"], pred["deg"], pred["e_pred"]
    y_pred_init = pred["y_pred"].copy()
    p_n_now = (y_pred_init == y_np).mean()
    rng = np.random.default_rng(1)
    results = []

    if pn_targets is None:
        start_pn = max(0.05, p_n_now * 0.3)
        pn_targets = np.linspace(start_pn, p_n_now, 10)

    for pn_t in pn_targets:
        if pn_t > p_n_now: continue
        gamma = (p_n_now - pn_t) / max(p_n_now, 1e-6)
        y_pred_cur = y_pred_init.copy()
        correct_nodes = np.where(y_pred_cur == y_np)[0]
        n_corrupt = int(gamma * len(correct_nodes))

        if n_corrupt > 0:
            weights = 1.0 / (deg[correct_nodes] + 1.0)
            weights = weights / weights.sum()
            corrupt = rng.choice(correct_nodes, n_corrupt, replace=False, p=weights)
            for v in corrupt:
                new_c = rng.integers(0, C_num - 1)
                if new_c >= y_pred_cur[v]: new_c += 1
                y_pred_cur[v] = new_c

        C, err, _ = compute_C_and_err(y_pred_cur, e_pred, src, dst, deg, y_np)
        C_t, err_t = C[test_mask], err[test_mask]
        dE = C_t[err_t == 1].mean() - C_t[err_t == 0].mean() if err_t.sum() > 0 and (1-err_t).sum() > 0 else np.nan
        auc = roc_auc_score(err_t, C_t) if len(np.unique(err_t)) > 1 else np.nan
        results.append({"p_n": float((y_pred_cur == y_np).mean()), "dE": float(dE), "auc": float(auc)})
    return results

# ============================================================
# [abstention, fixed] validation-tuned thresholds + test-oracle control
# ============================================================
def abstention_eval(risk_val, correct_val, risk_test, correct_test,
                    coverages=ABST_COVERAGES):
    """The protocol shared by every method:
    - val protocol (leak-free): tau = the coverage quantile of risk scores on val
      nodes, applied unchanged to test;
    - test oracle (leakage control): tau taken from the test scores themselves,
      reproducing the "test tuning inflates accuracy" claim of Ablation #5."""
    out = {}
    n_val, n_test = len(risk_val), len(risk_test)
    for cov in coverages:
        k_val = max(1, int(round(cov * n_val)))
        tau = np.sort(risk_val)[k_val - 1]
        keep = risk_test <= tau
        acc_val = float(correct_test[keep].mean()) if keep.sum() > 0 else np.nan
        cov_actual = float(keep.mean())

        k_test = max(1, int(round(cov * n_test)))
        tau_o = np.sort(risk_test)[k_test - 1]
        keep_o = risk_test <= tau_o
        acc_oracle = float(correct_test[keep_o].mean()) if keep_o.sum() > 0 else np.nan

        out[str(cov)] = {"acc_val": acc_val, "cov_actual": cov_actual,
                         "acc_oracle": acc_oracle}
    return out

def method_auc(risk_test, correct_test):
    """Discrimination AUC of a risk score against error (higher is better)."""
    err_t = (1 - correct_test).astype(int)
    if len(np.unique(err_t)) < 2: return np.nan
    return float(roc_auc_score(err_t, risk_test))

def exp_D_all_methods(C, pred, y_np, data, test_mask, val_mask, t_edge=0.0, t_agg=0.0):
    """All methods (conflict / msp / entropy / mc_dropout / neighborhood) run under
    one identical protocol.

    A single "budget protocol" is used: rank by risk score and keep a fixed cov
    fraction of nodes.
    Both the val-tuned (leak-free) and test-tuned (oracle leakage control) numbers
    are reported.
    """
    correct_t = (pred["y_pred"][test_mask] == y_np[test_mask]).astype(float)
    correct_v = (pred["y_pred"][val_mask] == y_np[val_mask]).astype(float)

    C_v = C[val_mask]
    C_t = C[test_mask]

    results, aucs = {}, {}
    methods = {
        "conflict": (C_v, C_t),
        "msp": (1.0 - pred["conf"][val_mask], 1.0 - pred["conf"][test_mask]),
        "entropy": (pred["entropy"][val_mask], pred["entropy"][test_mask]),
    }
    # neighbourhood-consistency baseline (zero cost, predicted labels only)
    nc = neighborhood_inconsistency(pred["y_pred"], pred["src"], pred["dst"], pred["deg"])
    methods["neighborhood"] = (nc[val_mask], nc[test_mask])

    # MC-Dropout (its true wall-clock time is recorded; cited in Scalability)
    mc_risk, t_mc = mc_dropout_risk(pred["model"], data)
    methods["mc_dropout"] = (mc_risk[val_mask], mc_risk[test_mask])

    for name, (r_v, r_t) in methods.items():
        results[name] = abstention_eval(r_v, correct_v, r_t, correct_t)
        aucs[name] = method_auc(r_t, correct_t)

    # fused audit score (dataset-level lambda): only the components are produced here.
    # Lambda is chosen in run_one_dataset from the mean val-AUC curves over 5 seeds --
    # far more stable than choosing per seed (val noise on small graphs gets averaged
    # out); the audit stage remains label-free.
    try:
        from scipy.stats import rankdata
        rc = rankdata(C) / len(C)
        rm = rankdata(1.0 - pred["conf"]) / len(pred["conf"])
        aucs["fused_parts"] = {
            "rc_v": rc[val_mask].tolist(), "rm_v": rm[val_mask].tolist(),
            "rc_t": rc[test_mask].tolist(), "rm_t": rm[test_mask].tolist(),
            "cor_v": correct_v.tolist(), "cor_t": correct_t.tolist(),
        }
    except Exception as _e:
        print("  [warn] fused parts failed:", _e)
        aucs["fused_parts"] = None

    # end-to-end conflict-score time: edge-feature/classifier stage (t_edge) +
    # node-level aggregation stage (t_agg)
    t_conflict = float(t_edge) + float(t_agg)
    return results, aucs, t_mc, t_conflict

def risk_coverage_curve(risk_test, correct_test, n_pts=10):
    """Coverage-accuracy curve keeping the top x fraction of nodes by ascending risk (for plotting)."""
    order = np.argsort(risk_test)
    cum_correct = np.cumsum(correct_test[order])
    coverages = np.linspace(0.1, 1.0, n_pts)
    accs = []
    for cov in coverages:
        n_keep = max(1, int(cov * len(risk_test)))
        accs.append(cum_correct[n_keep - 1] / n_keep)
    return coverages, np.array(accs)

# ============================================================
# Plotting (1x4 diagnostic panels; split into 2x2 by make_figures.py)
# ============================================================
def plot_single_dataset(ds_name, res_out, save_dir="figures"):
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    m = res_out["mean"]
    fig.suptitle(f"Conflict Theory Analysis: {ds_name.upper()} (h={m['h']:.3f})", fontsize=16)

    ax = axes[0]
    pe, dE = [r["p_e"] for r in res_out["exp_B_mean"]], [r["dE"] for r in res_out["exp_B_mean"]]
    ax.plot(1 - 2*np.array(pe), dE, 'o-', color='tab:blue')
    ax.axhline(0, color='k', ls='--', lw=0.5); ax.axvline(0, color='k', ls='--', lw=0.5)
    ax.set_xlabel(r"$1-2p_e$"); ax.set_ylabel(r"$\Delta E[C]$"); ax.set_title("Thm 2: Linearity")

    ax = axes[1]
    auc = [r["auc"] for r in res_out["exp_B_mean"]]
    ax.plot(pe, auc, 'o-', color='tab:orange')
    ax.axhline(0.5, color='k', ls='--', lw=0.5); ax.axvline(0.5, color='r', ls='--', lw=0.5)
    ax.set_xlabel(r"$p_e$"); ax.set_ylabel("AUC(C, err)"); ax.set_title("Predictive Power")

    ax = axes[2]
    pn, dE_c = [r["p_n"] for r in res_out["exp_C_mean"]], [r["dE"] for r in res_out["exp_C_mean"]]
    ax.plot(pn, dE_c, 'o-', color='tab:green')
    ax.axhline(0, color='k', ls='--', lw=0.5)
    ax.set_xlabel(r"$p_n$"); ax.set_ylabel(r"$\Delta E[C]$"); ax.set_title("Thm 3: Node Quality")

    ax = axes[3]
    colors = {"conflict": "tab:red", "msp": "tab:blue", "entropy": "tab:green", "mc_dropout": "tab:purple"}
    labels = {"conflict": "Conflict Field", "msp": "MSP", "entropy": "Entropy", "mc_dropout": "MC-Dropout"}
    for name in ["conflict", "msp", "entropy", "mc_dropout"]:
        cov, acc = res_out["rc_curves"][name]["cov"], res_out["rc_curves"][name]["acc"]
        ax.plot(cov, acc, 'o-', ms=3, color=colors[name], label=labels[name])
        # mark the actual operating point of the val-threshold protocol at 80% coverage
        op = res_out["abstention_mean"][name].get("0.8", None)
        if op is not None:
            ax.plot(op["cov_actual"], op["acc_val"], '*', ms=12, color=colors[name])
    ax.set_xlabel("Coverage"); ax.set_ylabel("Accuracy"); ax.set_title("Abstention (val-tuned * = 80%)")
    ax.legend(fontsize=8)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{ds_name}_conflict.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  [fig] {save_path}")

# ============================================================
# Curve-aggregation utilities
# ============================================================
def average_curves_on_targets(curve_lists, key):
    """Average multiple curves at matched target points (missing points ignored)."""
    if not curve_lists: return []
    all_targets = sorted({round(r[key], 6) for run in curve_lists for r in run})
    out = []
    for t in all_targets:
        dEs, aucs = [], []
        for run in curve_lists:
            for r in run:
                if abs(r[key] - t) < 1e-6:
                    if not np.isnan(r["dE"]): dEs.append(r["dE"])
                    if not np.isnan(r["auc"]): aucs.append(r["auc"])
                    break
        if dEs:
            out.append({key: t, "dE": float(np.mean(dEs)),
                        "auc": float(np.mean(aucs)) if aucs else np.nan})
    return out

def interpolate_curves(curve_lists, key, grid):
    """Per-seed p_n targets differ; interpolate onto a common grid and average."""
    acc = {g: [] for g in grid}
    for run in curve_lists:
        if not run: continue
        xs = np.array([r[key] for r in run])
        dEs = np.array([r["dE"] for r in run])
        ok = ~np.isnan(dEs)
        if ok.sum() < 2: continue
        for g in grid:
            if xs.min() - 1e-9 <= g <= xs.max() + 1e-9:
                acc[g].append(float(np.interp(g, xs[ok], dEs[ok])))
    return [{key: float(g), "dE": float(np.mean(v))} for g, v in acc.items() if v]

# ============================================================
# One dataset, one seed
# ============================================================
def run_one_seed(ds_name, seed, device="cpu", edge_mode="sup",
                 edge_space="emb", edge_feat="full"):
    data = load_dataset(ds_name, split_seed=seed).to(device)
    N = data.num_nodes
    y_np = data.y.cpu().numpy()
    C_num = int(y_np.max()) + 1
    test_mask = data.test_mask.cpu().numpy()
    val_mask = data.val_mask.cpu().numpy()

    pred = get_predictions(data, C_num, y_np, seed, edge_mode=edge_mode,
                           edge_space=edge_space, edge_feat=edge_feat)
    cbar_chosen, mech_chosen = float("nan"), ""
    p_e_obs, q_hat = float("nan"), float("nan")
    if CONFLICT_MODE in ("curate", "curate2") and pred.get("p_uv") is not None:
        # [zero-selection deployment mode] for the scenario "model trained, new graph
        # arrives, no val/test labels available".
        # The fundamental difference from earlier versions: no candidate set, no
        # val-based selection, no guard threshold.
        #   1) base conflict fixed to soft: c_uv = I(same)*(1-p_uv) + I(diff)*p_uv;
        #      the posterior expectation of the hard indicator -- it removes only
        #      Bernoulli sampling noise and introduces no new information;
        #   2) aggregation fixed to deg: C(v)=sum_u c_uv/d_v, range [0,1], the
        #      "neighbour conflict fraction";
        #      the dimension matches the paper's Eq. (expect_cv) and the boundary
        #      h*=1/C carries over directly;
        #   3) combination fixed to Ctilde(v)=C(v)*(1-M(v)), M the mean conflict of
        #      neighbouring edges (the background).
        #      beta=1 is hard-coded, nothing is selected: when M is small the expression
        #      is ~C (automatic smoothing on homophilous graphs);
        #      when M is large it suppresses the pervasive conflict and keeps only the
        #      excess over the background (automatic contrast on heterophilous graphs).
        src, dst, deg = pred["src"], pred["dst"], pred["deg"]
        same = (pred["y_pred"][src] == pred["y_pred"][dst]).astype(np.float32)
        p = np.asarray(pred["p_uv"], np.float32)
        c_e = (same * (1.0 - p) + (1.0 - same) * p).astype(np.float32)
        t0 = time.perf_counter()
        if CONFLICT_MODE == "curate2":
            # Variant: sym aggregation (neighbour evidence discounted by 1/sqrt(d_u))
            # and arithmetic smoothing as the combination. The formula is still unique;
            # only the choice of two operators differs. No val, no labels.
            # Ablation switches (defaults are the main method):
            #   CF_AGG=sym|mean      node-level aggregation operator; mean falls back to
            #                        degree normalisation
            #   CF_COMBINE=avg|none  none drops the neighbour background and uses C_self only
            C_self, C_nbr = neighbor_conflict_pair(c_e, src, dst, deg, N, agg=CURATE_AGG)
            if CURATE_COMBINE == "none":
                C_metric = C_self.astype(np.float32)
            else:
                C_metric = (0.5 * (C_self + C_nbr)).astype(np.float32)
        else:
            C_self, C_nbr = neighbor_conflict_pair(c_e, src, dst, deg, N)
            C_metric = (C_self * np.clip(1.0 - C_nbr, 0.0, None)).astype(np.float32)
        t_agg = time.perf_counter() - t0
        chosen_mode = CONFLICT_MODE
        err = (pred["y_pred"] != y_np).astype(int)

        # ---- on-the-spot diagnostics: model outputs and graph structure only, ----
        # ---- no labels read at audit time ----
        # p_e is, per Sec. III of the paper, "the probability that the predictor calls
        # an edge consistent" -- not an accuracy.
        p_e_obs = float(pred["e_pred"].mean())
        cbar_chosen = float(C_self.mean())
        # identity (exact edge by edge; no independence assumption between ehat and
        # the same-class indicator):
        #   C̄ = q + p_e − 2·E[I·ê]  ⇒  q = C̄ − p_e + 2·s
        s_obs = float((same * pred["e_pred"]).mean())
        q_hat = cbar_chosen - p_e_obs + 2.0 * s_obs
        # boundary decision: comparing qhat with 1/C decides whether "high conflict =
        # misclassified" or "high conflict = locally rare".
        # The criterion is valid only when p_e > 1/2 (stated in Sec. III).
        mech_chosen = "trust" if (p_e_obs > 0.5 and q_hat > 1.0 / C_num) else "unreliable"
    elif CONFLICT_MODE == "nbr3" and pred.get("p_uv") is not None:
        # [third use of neighbour conflict: mechanism set by the conflict field's own mean level]
        # reads only conflict values: no h, no labels, no per-dataset tuning:
        #   Cbar <= tau -> conflict sparse, neighbour conflict mostly noise ->
        #                  smoothing Ctilde=(C+M)/2
        #   Cbar >  tau -> conflict everywhere, neighbour conflict is the background ->
        #                  contrast Ctilde=C*(1-M)
        # The only val-driven choice is hard vs soft base conflict (as in the original
        # protocol), plus a 0.005 guard: if the gated score is worse than the plain C
        # of {hard, soft}, fall back to C.
        # plus a 0.005 guard: if the gated score is worse than the plain C of {hard, soft}, fall back to C.
        args = (pred["y_pred"], pred["e_pred"], pred["src"], pred["dst"], pred["deg"], y_np)
        cands, t_agg = {}, 0.0
        cbar_chosen, mech_chosen = float("nan"), ""
        inv_denom = neighbor_inv_denom(pred["deg"], pred["src"], pred["dst"])
        info = {}
        for base in ("hard", "soft"):
            c_e = _edge_disagree(pred["y_pred"], pred["e_pred"], pred["src"], pred["dst"],
                                 p_uv=pred["p_uv"], mode=base)
            t0 = time.perf_counter()
            C_self, C_nbr = neighbor_conflict_pair(c_e, pred["src"], pred["dst"], pred["deg"], N,
                                                   inv_denom=inv_denom)
            t_agg += time.perf_counter() - t0
            cbar = float(C_self.mean())          # mean conflict level, observable at audit time
            mech = "smooth" if cbar <= NBR_TAU else "contrast"
            if mech == "smooth":
                s = 0.5 * (C_self + C_nbr)
            else:
                s = C_self * np.clip(1.0 - NBR_PROD_BETA * C_nbr, 0.0, None)
            cands[f"{base}|{mech}"] = s.astype(np.float32)
            cands[f"{base}|C"] = C_self
            info[base] = (cbar, mech)
        err = (pred["y_pred"] != y_np).astype(int)
        C_metric, chosen_mode = cands["soft|C"], "soft|C"
        ev = err[val_mask]
        if 0 < int(ev.sum()) < len(ev):
            try:
                scores = {k: float(roc_auc_score(ev, v[val_mask])) for k, v in cands.items()}
                # the mechanism is fixed by Cbar, not chosen per dataset; val does two
                # things only:
                # (1) pick hard vs soft base conflict (as in the original protocol);
                # (2) enable the neighbour term only if it genuinely helps on val
                #     (gain >= NBR_MARGIN), otherwise fall back to C.
                # (2) is a switch, not a selection: one threshold shared by all datasets,
                # no extra information (no h, no labels, no graph size).
                gated = max((k for k in scores if "|C" not in k), key=lambda k: scores[k])
                ref = max(("hard|C", "soft|C"), key=lambda k: scores[k])
                chosen_mode = gated if scores[gated] > scores[ref] + NBR_MARGIN else ref
                C_metric = cands[chosen_mode]
            except ValueError:
                pass
        base_chosen = chosen_mode.split("|")[0]
        cbar_chosen, mech_chosen = info[base_chosen]
        base_chosen = chosen_mode.split("|")[0]
        cbar_chosen, mech_chosen = info[base_chosen]
    elif CONFLICT_MODE == "nbr2" and pred.get("p_uv") is not None:
        # [second use of neighbour conflict: switch mechanisms instead of mixing scores]
        # For each base conflict c_e, compute the neighbour background M(v) once (node-level
        # aggregation of the mean conflict of neighbouring edges),
        # then produce three verdicts in parallel:
        #   C            -- the original method (conflict on own edges)
        #   lin+0.5      -- smoothing: neighbour conflict as a second observation for denoising
        #   prod{b}      -- multiplicative contrast: C*(1-b*M), treating M as the background
        #                   conflict; only the part above the background is suspicious
        #                   (the key regime when conflict is pervasive on heterophilous graphs)
        #   max          -- worst neighbouring edge when conflict is sparse
        # All are O(N) combinations of the two vectors (C_self, M); edges are walked only
        # by 2 bincounts, so sweeping adds no cost.
        # Selection still uses val AUC only, and falls back to the C of {hard, soft}
        # when the gain is below NBR_MARGIN.
        args = (pred["y_pred"], pred["e_pred"], pred["src"], pred["dst"], pred["deg"], y_np)
        cands, t_agg = {}, 0.0
        inv_denom = neighbor_inv_denom(pred["deg"], pred["src"], pred["dst"])
        for base in ("hard", "soft"):
            c_e = _edge_disagree(pred["y_pred"], pred["e_pred"], pred["src"], pred["dst"],
                                 p_uv=pred["p_uv"], mode=base)
            t0 = time.perf_counter()
            C_self, C_nbr = neighbor_conflict_pair(c_e, pred["src"], pred["dst"], pred["deg"], N,
                                                   inv_denom=inv_denom)
            t_agg += time.perf_counter() - t0
            cands[f"{base}|C"] = C_self
            cands[f"{base}|lin"] = (0.5 * (C_self + C_nbr)).astype(np.float32)
            for b in NBR_PROD_BETAS:
                cands[f"{base}|prod{b:.1f}"] = \
                    (C_self * np.clip(1.0 - b * C_nbr, 0.0, None)).astype(np.float32)
            t0 = time.perf_counter()
            cands[f"{base}|max"] = _node_agg(c_e, pred["src"], pred["dst"], pred["deg"], N, agg="max")
            t_agg += time.perf_counter() - t0
        err = (pred["y_pred"] != y_np).astype(int)
        C_metric, chosen_mode = cands["soft|C"], "soft|C"
        ev = err[val_mask]
        if 0 < int(ev.sum()) < len(ev):
            try:
                scores = {k: float(roc_auc_score(ev, v[val_mask])) for k, v in cands.items()}
                ref = max(("hard|C", "soft|C"), key=lambda k: scores[k])
                best = max(scores, key=scores.get)
                chosen_mode = best if scores[best] > scores[ref] + NBR_MARGIN else ref
                C_metric = cands[chosen_mode]
            except ValueError:
                pass
    elif CONFLICT_MODE == "nbr6" and pred.get("p_uv") is not None:
        # [nbr2 + node-level aggregation operator only: no model internals, no raw features, no classification confidence]
        # candidates = {hard, soft} x {deg, sym} x {C, lin, prod x 3, max}
        #   deg: C(v) = sum_u c_uv / d_v                        (original, arithmetic mean)
        #   sym: C(v) = sum_u c_uv / (d_v * sqrt(d_u))          (neighbour evidence discounted by 1/sqrt(d_u))
        # sym tames hub neighbours: a high-degree neighbour conflicts with someone no matter
        # what, so its evidence per neighbour is discounted by
        # 1/sqrt(d_u); the self side still divides by d_v, keeping the dimension "mean
        # conflict" rather than a sum.
        # sym helps on low-degree graphs with skewed degree distributions
        # (tolokers / squirrel / wisconsin / cornell / texas); other graphs select back
        # to deg on val, with no change in results.
        src, dst, deg = pred["src"], pred["dst"], pred["deg"]
        same = (pred["y_pred"][src] == pred["y_pred"][dst]).astype(np.float32)
        p = np.asarray(pred["p_uv"], np.float32)
        bases = {
            "hard": (same != pred["e_pred"]).astype(np.float32),
            "soft": (same * (1 - p) + (1 - same) * p).astype(np.float32),
        }
        cands, t_agg = {}, 0.0
        inv_denom = neighbor_inv_denom(deg, src, dst)
        for ag in ("deg", "sym"):
            for bname, c_e in bases.items():
                t0 = time.perf_counter()
                C_self, C_nbr = neighbor_conflict_pair(c_e, src, dst, deg, N,
                                                       inv_denom=inv_denom, agg=ag)
                cands[f"{bname}|{ag}|C"] = C_self
                cands[f"{bname}|{ag}|lin"] = (0.5 * (C_self + C_nbr)).astype(np.float32)
                for b in NBR_PROD_BETAS:
                    cands[f"{bname}|{ag}|prod{b:.1f}"] = \
                        (C_self * np.clip(1.0 - b * C_nbr, 0.0, None)).astype(np.float32)
                t_agg += time.perf_counter() - t0
        for bname in ("hard", "soft"):
            t0 = time.perf_counter()
            cands[f"{bname}|deg|max"] = _node_agg(bases[bname], src, dst, deg, N, agg="max")
            t_agg += time.perf_counter() - t0
        err = (pred["y_pred"] != y_np).astype(int)
        C_metric, chosen_mode = cands["soft|deg|C"], "soft|deg|C"
        ev = err[val_mask]
        if 0 < int(ev.sum()) < len(ev):
            try:
                scores = {k: float(roc_auc_score(ev, v[val_mask])) for k, v in cands.items()}
                ref = max(("hard|deg|C", "soft|deg|C"), key=lambda k: scores[k])
                best = max(scores, key=scores.get)
                chosen_mode = best if scores[best] > scores[ref] + NBR_MARGIN else ref
                C_metric = cands[chosen_mode]
            except ValueError:
                pass
    elif CONFLICT_MODE == "nbr5" and pred.get("p_uv") is not None:
        # [nbr2 + two conflict-side changes, still introducing no classification confidence (MSP untouched)]
        # (1) aggregation operator: deg (divide by degree) vs sym (D^{-1/2} A D^{-1/2}),
        #     chosen on val.
        #     "Divide by degree" is more robust on homophilous high-dim graphs,
        #     "symmetric" on low-degree / heterophilous ones; neither dominates.
        # (2) base conflict extended with model internals: the edge-verdict probability
        #     becomes (1+cos)/2
        #     with cos from the hidden representation z after the first GCN layer -- no
        #     fitting, still "do the two endpoints agree".
        # The guard still references the original method (deg-aggregated C); fall back
        # when the gain is below NBR_MARGIN.
        src, dst, deg = pred["src"], pred["dst"], pred["deg"]
        same = (pred["y_pred"][src] == pred["y_pred"][dst]).astype(np.float32)
        p = np.asarray(pred["p_uv"], np.float32)
        t_agg = 0.0
        t0 = time.perf_counter()
        with torch.no_grad():
            Z = pred["model"].encode(data.x, data.edge_index).cpu().numpy().astype(np.float32)
        s_emb = np.clip((_edge_cos(Z, src, dst) + 1.0) * 0.5, 0.0, 1.0).astype(np.float32)
        t_agg += time.perf_counter() - t0
        bases = {
            "hard": (same != pred["e_pred"]).astype(np.float32),
            "soft": (same * (1 - p) + (1 - same) * p).astype(np.float32),
            "emb": (same * (1 - s_emb) + (1 - same) * s_emb).astype(np.float32),
        }
        cands = {}
        inv_denom = neighbor_inv_denom(deg, src, dst)
        for ag in ("deg", "sym"):
            for bname, c_e in bases.items():
                t0 = time.perf_counter()
                C_self, C_nbr = neighbor_conflict_pair(c_e, src, dst, deg, N,
                                                       inv_denom=inv_denom, agg=ag)
                cands[f"{bname}|{ag}|C"] = C_self
                cands[f"{bname}|{ag}|lin"] = (0.5 * (C_self + C_nbr)).astype(np.float32)
                for b in NBR_PROD_BETAS:
                    cands[f"{bname}|{ag}|prod{b:.1f}"] = \
                        (C_self * np.clip(1.0 - b * C_nbr, 0.0, None)).astype(np.float32)
                t_agg += time.perf_counter() - t0
        for bname in ("hard", "soft", "emb"):
            t0 = time.perf_counter()
            cands[f"{bname}|deg|max"] = _node_agg(bases[bname], src, dst, deg, N, agg="max")
            t_agg += time.perf_counter() - t0
        err = (pred["y_pred"] != y_np).astype(int)
        C_metric, chosen_mode = cands["soft|deg|C"], "soft|deg|C"
        ev = err[val_mask]
        if 0 < int(ev.sum()) < len(ev):
            try:
                scores = {k: float(roc_auc_score(ev, v[val_mask])) for k, v in cands.items()}
                ref = max(("hard|deg|C", "soft|deg|C"), key=lambda k: scores[k])
                best = max(scores, key=scores.get)
                chosen_mode = best if scores[best] > scores[ref] + NBR_MARGIN else ref
                C_metric = cands[chosen_mode]
            except ValueError:
                pass
    elif CONFLICT_MODE == "nbr4" and pred.get("p_uv") is not None:
        # [two more conflict-side changes of nbr2, still using no classification confidence (MSP untouched)]
        # (1) aggregation operator: symmetric normalisation D^{-1/2} A D^{-1/2} (the same
        #     operator for C and the neighbour background M)
        # (2) base conflict extended to model internals and raw features:
        #     the edge-verdict probability becomes (1+cos)/2 -- the cosine of the hidden z
        #     and of the input features x.
        #     Neither needs fitting; both remain in the "conflict" family (do the endpoints
        #     agree), only the metric space changes.
        src, dst, deg = pred["src"], pred["dst"], pred["deg"]
        same = (pred["y_pred"][src] == pred["y_pred"][dst]).astype(np.float32)
        p = np.asarray(pred["p_uv"], np.float32)
        t0 = time.perf_counter()
        with torch.no_grad():
            Z = pred["model"].encode(data.x, data.edge_index).cpu().numpy().astype(np.float32)
        s_emb = np.clip((_edge_cos(Z, src, dst) + 1.0) * 0.5, 0.0, 1.0).astype(np.float32)
        bases = {
            "hard": (same != pred["e_pred"]).astype(np.float32),
            "soft": (same * (1 - p) + (1 - same) * p).astype(np.float32),
            "emb": (same * (1 - s_emb) + (1 - same) * s_emb).astype(np.float32),
        }
        if data.num_features * len(src) <= 4e8:
            # skip the raw space on high-dim large graphs (Squirrel/Pubmed): O(|E|*d) would
            # break the complexity claim
            Xnp = data.x.cpu().numpy().astype(np.float32)
            s_raw = np.clip((_edge_cos(Xnp, src, dst) + 1.0) * 0.5, 0.0, 1.0).astype(np.float32)
            bases["raw"] = (same * (1 - s_raw) + (1 - same) * s_raw).astype(np.float32)
        t_agg = time.perf_counter() - t0
        cands = {}
        inv_denom = neighbor_inv_denom(deg, src, dst)
        for bname, c_e in bases.items():
            t0 = time.perf_counter()
            C_self, C_nbr = neighbor_conflict_pair(c_e, src, dst, deg, N,
                                                   inv_denom=inv_denom, agg="sym")
            cands[f"{bname}|C"] = C_self
            cands[f"{bname}|lin"] = (0.5 * (C_self + C_nbr)).astype(np.float32)
            for b in NBR_PROD_BETAS:
                cands[f"{bname}|prod{b:.1f}"] = \
                    (C_self * np.clip(1.0 - b * C_nbr, 0.0, None)).astype(np.float32)
            t_agg += time.perf_counter() - t0
        for bname in ("hard", "soft"):
            t0 = time.perf_counter()
            cands[f"{bname}|max"] = _node_agg(bases[bname], src, dst, deg, N, agg="max")
            t_agg += time.perf_counter() - t0
        err = (pred["y_pred"] != y_np).astype(int)
        C_metric, chosen_mode = cands["soft|C"], "soft|C"
        ev = err[val_mask]
        if 0 < int(ev.sum()) < len(ev):
            try:
                scores = {k: float(roc_auc_score(ev, v[val_mask])) for k, v in cands.items()}
                ref = max(("hard|C", "soft|C"), key=lambda k: scores[k])
                best = max(scores, key=scores.get)
                chosen_mode = best if scores[best] > scores[ref] + NBR_MARGIN else ref
                C_metric = cands[chosen_mode]
            except ValueError:
                pass
    elif CONFLICT_MODE == "sym" and pred.get("p_uv") is not None:
        # [aggregation operator only: symmetric normalisation instead of degree division]
        # C(v) = sum_{u in N(v)} c_uv / sqrt(d_u d_v), i.e. the GCN's own D^{-1/2} A D^{-1/2}.
        # No new selection dimension: base conflict is still hard-vs-soft chosen on val AUC
        # (as in the original protocol);
        # no per-dataset mechanism choice, so the degrees of freedom match the original method.
        cands, t_agg = {}, 0.0
        for base in ("hard", "soft"):
            c_e = _edge_disagree(pred["y_pred"], pred["e_pred"], pred["src"], pred["dst"],
                                 p_uv=pred["p_uv"], mode=base)
            t0 = time.perf_counter()
            cands[base] = _node_agg(c_e, pred["src"], pred["dst"], pred["deg"], N, agg="sym")
            t_agg += time.perf_counter() - t0
        err = (pred["y_pred"] != y_np).astype(int)
        chosen = "soft"
        ev = err[val_mask]
        if 0 < int(ev.sum()) < len(ev):
            try:
                chosen = max(cands, key=lambda k: float(roc_auc_score(ev, cands[k][val_mask])))
            except ValueError:
                pass
        C_metric, chosen_mode = cands[chosen], f"{chosen}|sym"
    elif CONFLICT_MODE in ("nbr", "auto") and pred.get("p_uv") is not None:
        # candidate set = {hard, soft} x {alpha=0} union {alpha in NBR_ALPHAS}, plus
        # soft_max (alpha=0).
        # Since C(alpha)=(1-alpha)C_self+alpha*C_nbr is linear in alpha, each base conflict
        # needs only 2 bincounts
        # to obtain C_self / C_nbr; the whole alpha family is an O(N) blend -- sweeping
        # never re-walks edges.
        # The selection protocol matches threshold tuning: val AUC only, one selection,
        # test never touched;
        # if the best candidate beats {hard, soft} by less than NBR_MARGIN, fall back --
        # guaranteeing no degradation from selection noise.
        args = (pred["y_pred"], pred["e_pred"], pred["src"], pred["dst"], pred["deg"], y_np)
        cands, t_agg = {}, 0.0
        inv_denom = neighbor_inv_denom(pred["deg"], pred["src"], pred["dst"])
        for base in ("hard", "soft"):
            c_e = _edge_disagree(pred["y_pred"], pred["e_pred"], pred["src"], pred["dst"],
                                 p_uv=pred["p_uv"], mode=base)
            t0 = time.perf_counter()
            C_self, C_nbr = neighbor_conflict_pair(c_e, pred["src"], pred["dst"], pred["deg"], N,
                                                   inv_denom=inv_denom)
            t_agg += time.perf_counter() - t0
            alphas = (0.0,) + NBR_ALPHAS if CONFLICT_MODE == "nbr" else (0.0,)
            for a in alphas:
                cands[f"{base}|{a:+.2f}"] = \
                    ((1.0 - a) * C_self + a * C_nbr).astype(np.float32)
        Cm, _, tx = compute_C_and_err(*args, p_uv=pred["p_uv"], mode="soft", agg="max")
        t_agg += tx
        cands["soft_max|+0.00"] = Cm
        err = (pred["y_pred"] != y_np).astype(int)
        C_metric, chosen_mode = cands["soft|+0.00"], "soft|+0.00"
        ev = err[val_mask]
        if 0 < int(ev.sum()) < len(ev):
            try:
                scores = {k: float(roc_auc_score(ev, v[val_mask])) for k, v in cands.items()}
                ref = max((scores["hard|+0.00"], "hard|+0.00"),
                          (scores["soft|+0.00"], "soft|+0.00"))[1]
                best = max(scores, key=scores.get)
                chosen_mode = best if scores[best] > scores[ref] + NBR_MARGIN else ref
                C_metric = cands[chosen_mode]
            except ValueError:
                pass
    else:
        alpha = 0.0
        if CONFLICT_MODE.startswith("nbr@"):
            alpha = float(CONFLICT_MODE.split("@")[1])
        C_metric, err, t_agg = compute_C_and_err(
            pred["y_pred"], pred["e_pred"], pred["src"], pred["dst"], pred["deg"], y_np,
            p_uv=pred.get("p_uv"), mode=("soft" if alpha else CONFLICT_MODE), alpha=alpha)
        chosen_mode = CONFLICT_MODE

    h = float((y_np[pred["src"]] == y_np[pred["dst"]]).mean())
    p_n = float((pred["y_pred"][test_mask] == y_np[test_mask]).mean())
    E_C = float(C_metric[test_mask].mean())

    resA = exp_A_significance(C_metric, err, test_mask)
    resB = exp_B_scan_pe(pred, y_np, test_mask)
    resC = exp_C_scan_pn(pred, y_np, test_mask, C_num=C_num)

    # [true AUC] AUC of C against node error on test
    auc_natural = method_auc(C_metric[test_mask], (pred["y_pred"][test_mask] == y_np[test_mask]).astype(float))

    # [abstention] all methods under one identical protocol
    abst, abst_aucs, t_mc, t_conflict = exp_D_all_methods(
        C_metric, pred, y_np, data, test_mask, val_mask,
        t_edge=pred.get("t_edge", 0.0), t_agg=t_agg)
    base_acc = float((pred["y_pred"][test_mask] == y_np[test_mask]).mean())

    out = {
        "seed": seed,
        "N": N, "E": pred["E"], "C": C_num,
        "h": h,
        "conflict_mode": chosen_mode,
        "conflict_cbar": cbar_chosen, "conflict_mech": mech_chosen,
        "p_e_obs": p_e_obs, "q_hat": q_hat, "boundary": 1.0 / C_num,
        "p_e_train": pred["p_e_train"], "p_e_val": pred["p_e_val"], "p_e_test": pred["p_e_test"],
        "edge_fit_ok": pred["edge_fit_ok"], "n_train_edges": pred["n_train_edges"],
        "skill_gain_val": pred["p_e_val"] - max(h, 1 - h),
        "skill_gain": pred["p_e_test"] - max(h, 1 - h),
        "p_n": p_n, "E_C": E_C, "AUC": auc_natural,
        "RD": float(resA["RD"]), "RR": float(resA["RR"]), "p_z": float(resA["p_z"]),
        "base_acc": base_acc,
        "acc_at_80_val": abst["conflict"]["0.8"]["acc_val"],
        "cov_at_80_val": abst["conflict"]["0.8"]["cov_actual"],
        "acc_at_80_oracle": abst["conflict"]["0.8"]["acc_oracle"],
        "acc_at_80_msp": abst["msp"]["0.8"]["acc_val"],
        "acc_at_80_entropy": abst["entropy"]["0.8"]["acc_val"],
        "acc_at_80_nc": abst["neighborhood"]["0.8"]["acc_val"],
        "acc_at_80_mc": abst["mc_dropout"]["0.8"]["acc_val"],
        "auc_msp": abst_aucs["msp"], "auc_entropy": abst_aucs["entropy"],
        "auc_nc": abst_aucs["neighborhood"], "auc_mc": abst_aucs["mc_dropout"],
        "t_conflict": t_conflict, "t_mc_dropout": t_mc,
        "t_fit": pred["t_fit"], "t_embed": pred["t_embed"],
        "t_sim": pred["t_sim"], "t_jac": pred["t_jac"], "t_pred": pred["t_pred"],
        "exp_B": resB, "exp_C": resC,
        "abstention": abst, "abstention_aucs": abst_aucs,
    }
    return out, pred, C_metric, y_np, test_mask, data

# ============================================================
# One dataset, 5-seed aggregation
# ============================================================
SCALAR_KEYS = ["h", "p_e_train", "p_e_val", "p_e_test", "p_e_obs", "q_hat", "boundary",
               "skill_gain_val", "skill_gain",
               "p_n", "E_C", "AUC", "RD", "RR", "p_z", "base_acc", "acc_at_80_val",
               "cov_at_80_val", "acc_at_80_oracle", "acc_at_80_msp", "acc_at_80_entropy",
               "acc_at_80_nc", "acc_at_80_mc", "acc_at_80_fused",
               "auc_msp", "auc_entropy", "auc_nc", "auc_mc", "auc_fused", "fused_lam",
               "t_conflict", "t_mc_dropout",
               "t_fit", "t_embed", "t_sim", "t_jac", "t_pred"]

def run_one_dataset(ds_name, outdir="results_v2", device="cpu", edge_mode="sup",
                    edge_space="emb", edge_feat="full"):
    seed_outs = []
    seed_parts = []
    first_pred = first_C = first_y = first_test = first_data = None
    for seed in SEEDS:
        out, pred, C_metric, y_np, test_mask, data = run_one_seed(
            ds_name, seed, device, edge_mode=edge_mode,
            edge_space=edge_space, edge_feat=edge_feat)
        seed_outs.append(out)
        seed_parts.append(out["abstention_aucs"].get("fused_parts"))
        if first_pred is None:
            first_pred, first_C, first_y, first_test, first_data = pred, C_metric, y_np, test_mask, data
        print(f"    seed {seed}: base={out['base_acc']:.3f} acc@80(val)={out['acc_at_80_val']:.3f} "
              f"AUC={out['AUC']:.3f} p_e_test={out['p_e_test']:.3f}")

    # ---- fused audit score: dataset-level lambda (selected on the mean val-AUC curves
    # over 5 seeds; more stable on small graphs) ----
    lams = np.linspace(0, 1, 21)
    if all(p is not None for p in seed_parts):
        parts_conv = []
        for p in seed_parts:
            p = dict(p)
            for k in ("rc_v", "rm_v", "rc_t", "rm_t", "cor_v", "cor_t"):
                p[k] = np.asarray(p[k])
            parts_conv.append(p)
        val_mat = np.array([
            [method_auc(lam * p["rm_v"] + (1 - lam) * p["rc_v"], p["cor_v"]) for lam in lams]
            for p in parts_conv], dtype=float)
        val_mean = np.nanmean(val_mat, axis=0)
        i_best = int(np.nanargmax(val_mean))
        best_lam = float(lams[i_best])
        # harmless guard: fall back to lambda=1 when the best lambda does not clearly
        # beat the pure confidence term on mean val AUC
        a_conf = np.nanmean([method_auc(p["rm_v"], p["cor_v"]) for p in parts_conv])
        if not np.isnan(a_conf) and val_mean[i_best] < a_conf + 0.005:
            best_lam = 1.0
        for o, p in zip(seed_outs, parts_conv):
            r_v = best_lam * p["rm_v"] + (1 - best_lam) * p["rc_v"]
            r_t = best_lam * p["rm_t"] + (1 - best_lam) * p["rc_t"]
            ev = abstention_eval(r_v, p["cor_v"], r_t, p["cor_t"])
            o["abstention"]["fused"] = ev
            o["auc_fused"] = float(method_auc(r_t, p["cor_t"]))
            o["acc_at_80_fused"] = float(ev["0.8"]["acc_val"])
            o["fused_lam"] = best_lam
    else:
        for o in seed_outs:
            o["abstention"]["fused"] = o["abstention"]["conflict"]
            o["auc_fused"] = np.nan
            o["acc_at_80_fused"] = np.nan
            o["fused_lam"] = np.nan

    mean, std = {}, {}
    for k in SCALAR_KEYS:
        vals = [o[k] for o in seed_outs if o[k] is not None and not (isinstance(o[k], float) and np.isnan(o[k]))]
        mean[k] = float(np.mean(vals)) if vals else np.nan
        std[k] = float(np.std(vals)) if vals else np.nan

    exp_B_mean = average_curves_on_targets([o["exp_B"] for o in seed_outs], "p_e")
    grid = np.linspace(0.35, 0.95, 10)
    exp_C_mean = interpolate_curves([o["exp_C"] for o in seed_outs], "p_n", grid)

    abst_mean = {}
    for m in ["conflict", "msp", "entropy", "neighborhood", "mc_dropout", "fused"]:
        abst_mean[m] = {}
        for cov in [str(c) for c in ABST_COVERAGES]:
            abst_mean[m][cov] = {
                kk: float(np.mean([o["abstention"][m][cov][kk] for o in seed_outs]))
                for kk in ["acc_val", "cov_actual", "acc_oracle"]
            }

    # coverage-accuracy curves (seed 0 ordering, representative figure; operating points
    # averaged over seeds)
    correct_t = (first_pred["y_pred"][first_test] == first_y[first_test]).astype(float)
    rc_curves = {}
    conf_t = first_pred["conf"][first_test]; ent_t = first_pred["entropy"][first_test]
    C_t = first_C[first_test]
    mc_risk, _ = mc_dropout_risk(first_pred["model"], first_data)
    cov_c, acc_c = risk_coverage_curve(C_t, correct_t)
    cov_m, acc_m = risk_coverage_curve(1 - conf_t, correct_t)
    cov_e, acc_e = risk_coverage_curve(ent_t, correct_t)
    cov_mc, acc_mc = risk_coverage_curve(mc_risk[first_test], correct_t)
    nc_all = neighborhood_inconsistency(first_pred["y_pred"], first_pred["src"],
                                        first_pred["dst"], first_pred["deg"])
    cov_n, acc_n = risk_coverage_curve(nc_all[first_test], correct_t)
    rc_curves["conflict"] = {"cov": cov_c.tolist(), "acc": acc_c.tolist()}
    rc_curves["msp"] = {"cov": cov_m.tolist(), "acc": acc_m.tolist()}
    rc_curves["entropy"] = {"cov": cov_e.tolist(), "acc": acc_e.tolist()}
    rc_curves["neighborhood"] = {"cov": cov_n.tolist(), "acc": acc_n.tolist()}
    rc_curves["mc_dropout"] = {"cov": cov_mc.tolist(), "acc": acc_mc.tolist()}

    res_out = {
        "dataset": ds_name, "N": seed_outs[0]["N"], "E": seed_outs[0]["E"],
        "C": seed_outs[0]["C"], "seeds": SEEDS,
        "mean": mean, "std": std,
        "exp_B_mean": exp_B_mean, "exp_C_mean": exp_C_mean,
        "abstention_mean": abst_mean, "rc_curves": rc_curves,
        "per_seed": seed_outs,
    }
    plot_single_dataset(ds_name, res_out, save_dir=os.path.join(outdir, "figures"))

    summary = {
        "dataset": ds_name, "N": seed_outs[0]["N"], "E": seed_outs[0]["E"], "K": seed_outs[0]["C"],
        "h": mean["h"],
        "p_e_test": mean["p_e_test"], "p_e_test_std": std["p_e_test"],
        "skill_gain": mean["skill_gain"],
        "p_n": mean["p_n"], "E_C": mean["E_C"], "AUC": mean["AUC"], "AUC_std": std["AUC"],
        "RD": mean["RD"], "RR": mean["RR"],
        "base_acc": mean["base_acc"], "base_acc_std": std["base_acc"],
        "acc_at_80_val": mean["acc_at_80_val"], "acc_at_80_val_std": std["acc_at_80_val"],
        "acc_at_80_oracle": mean["acc_at_80_oracle"],
        "acc_at_80_msp": mean["acc_at_80_msp"],
        "acc_at_80_entropy": mean["acc_at_80_entropy"],
        "acc_at_80_nc": mean["acc_at_80_nc"],
        "acc_at_80_mc": mean["acc_at_80_mc"],
        "auc_msp": mean["auc_msp"], "auc_entropy": mean["auc_entropy"],
        "auc_nc": mean["auc_nc"], "auc_mc": mean["auc_mc"],
        "t_conflict": mean["t_conflict"], "t_mc_dropout": mean["t_mc_dropout"],
        "t_fit": mean["t_fit"], "t_embed": mean["t_embed"], "t_sim": mean["t_sim"],
        "t_jac": mean["t_jac"], "t_pred": mean["t_pred"],
        "edge_mode": edge_mode, "edge_space": edge_space, "edge_feat": edge_feat,
        # on how many seeds the edge predictor truly fit (<1 means it degenerates on this
        # dataset and p_e stays NaN)
        "edge_fit_rate": float(np.mean([bool(o["edge_fit_ok"]) for o in seed_outs])),
    }
    return summary, res_out

# ============================================================
# main
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str,
                        default="cora,citeseer,pubmed,cornell,texas,wisconsin,amazon_ratings,actor,chameleon_filtered,squirrel_filtered,tolokers")
    parser.add_argument("--outdir", type=str, default="results_v2")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--edge-mode", type=str, default="sup", choices=["sup", "unsup"],
                        help="sup: LR fitted on training-edge labels (main); unsup: pure cosine threshold (Ablation #1)")
    parser.add_argument("--edge-space", type=str, default="auto", choices=["auto", "emb", "raw"],
                        help="auto: 64-dim embedding when d>64, else raw features (default); "
                             "emb: force embedding; raw: force raw features (legacy O(|E|*d))")
    parser.add_argument("--edge-feat", type=str, default="full", choices=["cos", "cos_deg", "full", "pred"],
                        help="edge-classifier features: cos / cos+degree diff / plus binary overlap")
    args = parser.parse_args()

    ds_names = [d.strip() for d in args.datasets.split(",")]
    os.makedirs(args.outdir, exist_ok=True)
    summary_rows, detailed = [], {}

    print(f"\n{'='*80}\nExperiment: {len(ds_names)} datasets x {len(SEEDS)} seeds | "
          f"edge_mode={args.edge_mode} edge_space={args.edge_space} "
          f"edge_feat={args.edge_feat} | MC-Dropout {N_DROPOUT_PASSES} passes | "
          f"device={args.device}\n{'='*80}\n")

    for name in ds_names:
        print(f"[{time.strftime('%H:%M:%S')}] === {name} ===", flush=True)
        t0 = time.time()
        try:
            summary, res_out = run_one_dataset(name, outdir=args.outdir,
                                               device=args.device, edge_mode=args.edge_mode,
                                               edge_space=args.edge_space,
                                               edge_feat=args.edge_feat)
            detailed[name] = res_out
            summary_rows.append(summary)
            print(f"[{time.strftime('%H:%M:%S')}] {name} done in {time.time()-t0:.1f}s\n")
        except Exception as e:
            import traceback
            print(f"\n[error] {name} failed: {e}")
            traceback.print_exc()
            print("\n")

        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    if not summary_rows: exit(0)

    df = pd.DataFrame(summary_rows)
    cols_fmt3 = ["h", "p_e_test", "skill_gain", "p_n", "E_C", "AUC", "RD",
                 "base_acc", "acc_at_80_val", "acc_at_80_oracle",
                 "acc_at_80_msp", "acc_at_80_entropy", "acc_at_80_nc", "acc_at_80_mc",
                 "auc_msp", "auc_entropy", "auc_nc", "auc_mc", "t_conflict", "t_mc_dropout",
                 "t_fit", "t_embed", "t_sim", "t_jac", "t_pred"]
    df_out = df.copy()
    for c in cols_fmt3:
        df_out[c] = df_out[c].map("{:.3f}".format)
    df_out["RR"] = df_out["RR"].map("{:.2f}".format)
    df_out["base_acc_pm"] = df.apply(lambda r: f"{r['base_acc']:.3f}±{r['base_acc_std']:.3f}", axis=1)
    df_out["acc80_pm"] = df.apply(lambda r: f"{r['acc_at_80_val']:.3f}±{r['acc_at_80_val_std']:.3f}", axis=1)

    print("\n" + "="*120 + "\n v2 summary (val-tuned thresholds + true AUC + test-edge p_e)\n" + "="*120)
    show_cols = ["dataset", "h", "p_e_test", "skill_gain", "AUC", "auc_msp", "E_C",
                 "base_acc_pm", "acc80_pm", "acc_at_80_msp", "acc_at_80_nc",
                 "acc_at_80_mc", "t_conflict", "t_mc_dropout", "t_fit"]
    try: print(df_out[show_cols].to_markdown(index=False, tablefmt="grid"))
    except: print(df_out[show_cols].to_string(index=False))

    print("\n" + "="*80 + "\n Radar coordinates (E[C], base_acc) mean+-std:\n" + "="*80)
    for r in summary_rows:
        print(f"({r['E_C']:.3f}, {r['base_acc']:.3f}) pm ({r['E_C']:.3f},{r['base_acc_std']:.3f}) % {r['dataset']}")

    with open(os.path.join(args.outdir, "global_summary_results_v2.json"), "w") as f:
        json.dump(detailed, f, indent=2, default=float)
    df.to_csv(os.path.join(args.outdir, "global_summary_table_v2.csv"), index=False)
    print(f"\n[done] JSON -> {args.outdir}/global_summary_results_v2.json")
    print(f"[done] CSV  -> {args.outdir}/global_summary_table_v2.csv")
