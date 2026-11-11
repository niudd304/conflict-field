"""
run_all_experiments.py (v2 - Protocol-Fixed Version)
================================================================================
修复清单（对应论文声明的四项硬伤）：
  P1/② abstention 协议改为「验证集调阈值」：在 val 节点上按冲突分定 80% 覆盖率
      的阈值 tau，再原样应用到 test；同时保留 test 上直接调阈值的 oracle 版本
      作为 Ablation #5 的「test 调优虚高」对照。
  ①   记录真实 AUC（test 上 roc_auc(err, C)）与 test 边上的 p_e_test，
      并计算技能增益 skill_gain = p_e_test - max(h, 1-h)。
  ③   加回三个 abstention 基线：MSP / Predictive Entropy / MC-Dropout(20次)，
      三者与 Conflict Field 走完全相同的 val 调阈值协议，并各自输出 AUC。
  ④   5 个 seed（0-4）重复，全部指标输出 mean±std；扫描曲线按目标点取均值。
================================================================================
"""
import os
# 强制覆盖可能损坏的环境变量
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
# 0. 实验配置
# ============================================================
SEEDS = [int(s) for s in os.environ.get("CF_SEEDS", "0,1,2,3,4").split(",")]
N_DROPOUT_PASSES = 20
ABST_COVERAGES = [0.6, 0.7, 0.8, 0.9]

# 诊断实验开关（默认全关，不改变既有复现结果）：
# CF_EDGE_BALANCED=1  → 边分类器加 class_weight='balanced'，抵消先验 h 主导标签分布
#                        造成的塌缩（强异配图 h≈0.06 时全预测多数类就有 94% 准确率）
EDGE_BALANCED = os.environ.get("CF_EDGE_BALANCED", "0") == "1"
# CF_EDGE_CALIB=<raw|rank|z|otsu>  → 边概率的 label-free 重标定（只影响 unsup 分支的诊断）
#   动机：unsup 的 p_uv=(cos+1)/2 在所有图上均值 0.85~0.98，阈值 0.5 等于「全判同类」，
#   p_e 恰好退化为 h。真实最优阈值在 0.92~0.95。这说明 p 的绝对值无校准意义、只有排序有意义，
#   因此用不依赖标签的重标定把 p 拉回可用区间：
#     rank -- 秩归一化，阈值 0.5 即中位数（假设 h=0.5，对 soft 单调等价、不影响其排序）
#     z    -- z-score 后 sigmoid（同上，自适应均值方差）
#     otsu -- 直方图 Otsu 二值化，完全自适应，不假设 h
EDGE_CALIB = os.environ.get("CF_EDGE_CALIB", "raw")
# CF_CONFLICT_MODE=soft → 软冲突值（后验期望），默认 hard 保持既有结果
# CF_CONFLICT_MODE=nbr  → 软冲突值 + 边的邻居冲突值（一次 O(|E|) 重写，见 §3）
CONFLICT_MODE = os.environ.get("CF_CONFLICT_MODE", "hard")
# curate2 的两个算子（消融用；默认即论文主方法）
CURATE_AGG = os.environ.get("CF_AGG", "sym")          # sym | mean
CURATE_COMBINE = os.environ.get("CF_COMBINE", "avg")  # avg | none
# 邻居重写强度 α：q_e=(1−α)c_e+α·(邻居边平均冲突)。α=0 完全退回原方法。
NBR_ALPHAS = (-1.0, -0.5, 0.25, 0.5, 1.0, 2.0)
# nbr2 的乘性对比强度：C·(1−b·M)，M 为邻居冲突本底
NBR_PROD_BETAS = (0.5, 1.0, 1.5)
# nbr3：机制由「平均冲突水平」C̄=mean_v C(v) 硬门控（只用冲突场自身信息，无标签、不用 h）
# C̄ 低 → 冲突稀疏、邻居冲突是噪声 → 平滑；C̄ 高 → 冲突遍地 → 乘性对比。
# 11 个数据集上 C̄ 的两簇是 0.08–0.16 与 0.54–0.90，阈值落在这段空白里对取值不敏感。
NBR_TAU = float(os.environ.get("CF_NBR_TAU", "0.35"))
NBR_PROD_BETA = float(os.environ.get("CF_NBR_BETA", "1.5"))
# val 上提升不足该阈值就退回 α=0（防止选择噪声把结果改坏）
NBR_MARGIN = float(os.environ.get("CF_NBR_MARGIN", "0.005"))

# ============================================================
# 1. 独立的数据加载逻辑（本地路径，绝不联网）
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
    real_split = split_idx % num_splits  # 官方 10 个划分，seed 轮换

    return Data(
        x=torch.tensor(X, dtype=torch.float), edge_index=ei,
        y=torch.tensor(y, dtype=torch.long),
        train_mask=torch.tensor(obj["train_masks"][real_split], dtype=torch.bool),
        val_mask=torch.tensor(obj["val_masks"][real_split], dtype=torch.bool),
        test_mask=torch.tensor(obj["test_masks"][real_split], dtype=torch.bool),
    )

def _symmetrize_edge_index(ei):
    """去重并双向化（与 load_npz_generic 相同的口径：h 按去重无向图计）。"""
    src, dst = ei[0].numpy(), ei[1].numpy()
    ue = set(zip(np.minimum(src, dst).tolist(), np.maximum(src, dst).tolist()))
    ei = torch.tensor([[e[0] for e in ue], [e[1] for e in ue]], dtype=torch.long)
    return coalesce(to_undirected(ei))


def load_external(name, split_seed, root):
    """三个公开大规模图（NeurIPS'21 非同配基准 + Platonov release）：

    genius        421,961 节点 / 922,868 边   h≈0.59   C=2  Lim & Benson 2020
                  （2023 更新版数据；无官方划分 → 随机 60/20/20，seed 控制）
    penn94         41,554 节点 / 1,362,229 边 h≈0.47   C=2  Traud et al. 2012
                  （官方 5 个划分，seed 轮换；性别标签，0 为未标注）
    roman_empire   22,662 节点 /  66,566 边   h≈0.20   C=18 Platonov et al. 2022
                  （官方 10 个划分，seed 轮换）
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
        y = (metadata[:, 1] - 1).astype(np.int64)          # 性别；0-1=-1 未标注
        ei = torch.tensor(np.vstack(A.nonzero()), dtype=torch.long)
        ei = _symmetrize_edge_index(ei)
        # 特征：year + 其余元数据列逐列 one-hot（Lim et al. 的处理）
        vals = np.hstack((metadata[:, :1], metadata[:, 2:]))
        feats = []
        for col in range(vals.shape[1]):
            c = vals[:, col]
            feats.append(label_binarize(c, classes=np.unique(c)).astype(np.float32))
        X = np.hstack(feats)
        N = len(y)
        labeled = torch.tensor(y >= 0)
        sp = np.load(os.path.join(root, "penn94_splits.npy"), allow_pickle=True)
        # 官方 5 个划分：元素为 {'train': [...], 'valid': [...], 'test': [...]}
        if sp.dtype == object and len(sp) == 5:
            item = sp[split_seed % len(sp)]
            trm = torch.zeros(N, dtype=torch.bool); trm[torch.tensor(item["train"])] = True
            vlm = torch.zeros(N, dtype=torch.bool); vlm[torch.tensor(item["valid"])] = True
            tem = torch.zeros(N, dtype=torch.bool); tem[torch.tensor(item["test"])] = True
            trm &= labeled; vlm &= labeled; tem &= labeled
            return Data(x=torch.tensor(X), edge_index=ei, y=torch.tensor(y),
                        train_mask=trm, val_mask=vlm, test_mask=tem)
        # 兜底：随机 60/20/20（只在 labeled 节点上切）
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
    # 数据路径：默认读取本目录下 prepare_data.py 生成的 data/，可用环境变量覆盖
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
        # 论文协议：随机 60/20/20 切分，seed 控制切分随机性
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
# 2. GCN 模型与预测（原始特征 + Jaccard）
# ============================================================
class GCN(torch.nn.Module):
    def __init__(self, in_dim, hidden, out_dim, dropout=0.5):
        super().__init__()
        self.c1 = GCNConv(in_dim, hidden); self.c2 = GCNConv(hidden, out_dim)
        self.dropout = dropout
    def encode(self, x, ei):
        """第一层 GCN 之后的隐藏表征 z_v（dim=hidden）。

        论文 §III 的边一致性明确由「模型自身的 latent embedding」计算，
        因此审计阶段应当在 z 上做相似度，而不是在原始输入特征 x 上做。
        z 的维度是 64，与输入特征维度 d 无关 —— 这正是 O(|E|·d) -> O(|E|·64) 的来源。
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
    """MC-Dropout：仅将 Dropout 层置于 train 模式，做 n_passes 次随机前向，
    风险分数 = 平均预测分布的预测熵（标准选择性预测基线）。

    force_cpu: 把计时口径统一到 CPU。Conflict Field 全流程是 numpy/CPU 向量化，
    若 MC-Dropout 在 GPU 上计时，两者不在同一硬件上，比出来的加速比没有意义。
    前向在 CPU 上做，熵值与 GPU 一致（同一模型、同一权重、同一 dropout 掩码分布）。
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
    """训练 GCN 并计算边一致性分数。

    edge_mode:
      "sup"   -- 用训练边标签拟合 LR（主配置；审计阶段无需任何标签）
      "unsup" -- 纯余弦阈值，完全不碰标签（Ablation #1 对照）
    edge_space:
      "auto"  -- d > 64 时用 latent embedding（降维到 64），否则用原始特征。默认。
                 复杂度 O(|E|·min(d, 64))：既不让高维图爆内存，也不给低维图无谓加维。
      "emb"   -- 强制用 latent embedding（论文 §III 的字面定义）
      "raw"   -- 强制用原始输入特征（旧实现）
    edge_feat:
      "full"    -- cos + log1p(|deg差|) + 二值重叠度（3 特征）
      "cos_deg" -- cos + log1p(|deg差|)（2 特征，最省）
      "cos"     -- 仅 cos
    """
    N = data.num_nodes
    model, logits = train_gcn(data, C_num, seed)
    probs = F.softmax(logits, dim=1).cpu().numpy()
    y_pred = probs.argmax(1)
    conf = probs.max(1)
    entropy = -(probs * np.log(probs + 1e-12)).sum(1)

    # ---------------- 审计阶段计时 ----------------
    # 论文 §V 只应报这一段：拿到训练好的模型之后，审计一张图需要多少时间。
    # 边预测器的拟合是一次性开销，单独记为 t_fit，不与审计成本混算。
    t0 = time.perf_counter()
    src, dst = data.edge_index[0].cpu().numpy(), data.edge_index[1].cpu().numpy()

    if edge_space == "auto":
        # 只在降维有意义时才用 embedding：d > hidden(64) 才换空间。
        # Cora(d=1433)/Squirrel(d=2089) 走 emb，Tolokers(d=1) 留在 raw。
        use_emb = data.num_features > 64
    else:
        use_emb = (edge_space == "emb")

    if use_emb:
        # 论文 §III：边一致性由模型自身的 latent embedding 计算（dim=64，与特征维 d 无关）
        with torch.no_grad():
            Z = model.encode(data.x, data.edge_index).cpu().numpy().astype(np.float32)
    else:
        Z = data.x.cpu().numpy().astype(np.float32)   # 低维图：原始输入特征本身更省

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
        # 二值重叠度。分块计算，避免 E x d 的稠密中间量：
        # 旧写法 x_bin[src]*x_bin[dst] 在 Squirrel 上要 4.6e9 次乘法 + 18GB 临时内存，
        # 这是旧实现「O(|E|)」声明失效、Squirrel 耗时 4.16s 的真正原因。
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
        # 预测感知边特征（全部 label-free：只用模型自身输出，审计阶段不读任何标签）。
        # 动机：同质图上 embedding cos 塌缩为常数（同类邻居天然相似），e_pred 失去信息。
        # 1) 原始特征余弦：embedding 之外补一个输入空间的视角
        Xr = data.x.cpu().numpy().astype(np.float32)
        Xrn = Xr / (np.linalg.norm(Xr, axis=1, keepdims=True) + 1e-9)
        feats.append((Xrn[src] * Xrn[dst]).sum(1))
        # 2) 端点预测分布的 JS 散度：错分节点与正确邻居的预测分布不同
        P = np.clip(probs, 1e-12, 1.0)
        M = 0.5 * (P[src] + P[dst])
        kl_u = (P[src] * (np.log(P[src]) - np.log(M))).sum(1)
        kl_v = (P[dst] * (np.log(P[dst]) - np.log(M))).sum(1)
        feats.append((0.5 * (kl_u + kl_v)).astype(np.float32))
        # 3) 端点置信度差
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
        # 训练边必须两类都有，否则 LR 退化为常数预测（小图/极端异配时会发生）
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

            # in-sample（train 边，仅供参考，有虚高偏差）
            e_pred_tr = (lr.predict_proba(X_fit)[:, 1] >= 0.5).astype(int)
            p_e_train = float((e_pred_tr == y_fit).mean())

            t0 = time.perf_counter()
            p_uv = lr.predict_proba(X_edge)[:, 1]
            t_pred = time.perf_counter() - t0
            edge_fit_ok = True

    # ---- label-free 重标定（只对 unsup 的余弦伪概率有意义；sup 的 LR 输出已校准）----
    if EDGE_CALIB != "raw" and edge_mode == "unsup" and len(p_uv) > 100:
        if EDGE_CALIB == "rank":
            # 秩归一化：p 的绝对值不可信，只保留排序。阈值 0.5 即中位数。
            order = np.argsort(p_uv, kind="stable")
            r = np.empty(len(p_uv), dtype=np.float32)
            r[order] = np.arange(len(p_uv), dtype=np.float32)
            p_uv = ((r + 0.5) / len(p_uv)).astype(np.float32)
        elif EDGE_CALIB == "z":
            mu, sd = float(p_uv.mean()), float(p_uv.std() + 1e-9)
            p_uv = (1.0 / (1.0 + np.exp(-(p_uv - mu) / sd))).astype(np.float32)
        elif EDGE_CALIB == "otsu":
            # 直方图 Otsu：最大化类间方差，完全自适应、不假设 h
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

    # p_e 只有在边预测器真正有效时才报告：sup 需拟合成功，unsup 用余弦阈值。
    # 否则留 NaN —— 退化预测器的 "精度" 是假象（例如全预测多数类的 1-h）。
    if edge_mode == "unsup" or edge_fit_ok:
        if mask_va.sum() >= 20:
            p_e_val = float(((p_uv[mask_va] >= 0.5).astype(int) == e_true_all[mask_va]).mean())
        if mask_te.sum() >= 20:
            p_e_test = float(((p_uv[mask_te] >= 0.5).astype(int) == e_true_all[mask_te]).mean())

    e_pred = (p_uv >= 0.5).astype(int)
    t_audit = t_embed + t_sim + t_jac + t_pred
    return {
        "model": model, "probs": probs,          # T2 融合需要完整概率分布
        "y_pred": y_pred, "conf": conf, "entropy": entropy,
        "e_pred": e_pred, "p_uv": p_uv,          # 连续边判定概率，供软冲突值实验使用
        "e_true_all": e_true_all, "p_e_train": p_e_train,
        "p_e_val": p_e_val, "p_e_test": p_e_test, "edge_mode": edge_mode,
        "edge_space": ("emb" if use_emb else "raw") , "edge_space_arg": edge_space,
        "edge_feat": edge_feat,
        "edge_fit_ok": edge_fit_ok, "n_train_edges": int(mask_tr.sum()),
        # t_edge 保留旧字段名，语义已改为「审计阶段」，不含一次性拟合
        "t_edge": t_audit, "t_audit": t_audit, "t_fit": t_fit,
        "t_embed": t_embed, "t_sim": t_sim, "t_jac": t_jac, "t_pred": t_pred,
        "src": src, "dst": dst, "deg": deg, "N": N, "E": len(src)
    }

# ============================================================
# 3. 计算与实验
# ============================================================
def _edge_disagree(y_pred, e_pred, src, dst, p_uv=None, mode="hard"):
    """边级分歧 c_uv（hard: 0/1 指示；soft: Bernoulli 后验期望）。"""
    same_pred = (y_pred[src] == y_pred[dst]).astype(np.float32)
    if mode == "soft" and p_uv is not None:
        p = np.asarray(p_uv, dtype=np.float32)
        return (same_pred * (1.0 - p) + (1.0 - same_pred) * p).astype(np.float32)
    return (same_pred != e_pred).astype(np.float32)


def _edge_cos(A, src, dst, budget=4_000_000):
    """逐边余弦，分块计算避免 |E|×d 的稠密中间量（Squirrel 上会吃掉几十 GB）。"""
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
        # 邻居侧 √、自身侧线性：C(v)=Σ_u c_uv/(d_v·√d_u)。
        # 等价于先按 1/√(d_u d_v) 给边加权、聚合后再除以 √d_v（即 D^{-1}AD^{-1/2}）。
        # 与"除以度"相比：hub 邻居不再以 1/d 的微弱权重混入，也不再以 d 倍放大自己；
        # 代价是多一次逐边除法（O(|E|)，无新内存）。
        w = c_e / np.sqrt(np.maximum(deg[src] * deg[dst], 1.0))
        num = np.bincount(src, weights=w, minlength=N) + \
            np.bincount(dst, weights=w, minlength=N)
        return (num / np.sqrt(np.maximum(deg, 1.0))).astype(np.float32)
    C = np.bincount(src, weights=c_e, minlength=N) + \
        np.bincount(dst, weights=c_e, minlength=N)
    return (C / np.maximum(deg, 1)).astype(np.float32)


def neighbor_inv_denom(deg, src, dst):
    """1/(deg_u+deg_v−2)：只与拓扑有关，hard/soft 两种基冲突值共用，避免重复除一次。"""
    du, dv = deg[src], deg[dst]
    return (1.0 / np.maximum(du + dv - 2.0, 1.0)).astype(np.float32)


def neighbor_conflict_pair(c_e, src, dst, deg, N, inv_denom=None, agg="mean"):
    """把每条边的冲突值再往它的「邻居边」传一步，返回 (C_self, C_nbr)。

    每条边的邻居边 = 与它共享端点的其它边。对边 e=(u,v)：
        m_e = (Σ_{e'∋u} c_e' + Σ_{e'∋v} c_e' − 2c_e) / (deg_u + deg_v − 2)
    即「这条边周围的平均冲突水平」。重写后回到节点级得到 C_nbr。

    关键：重写 q_e=(1−α)c_e+α·m_e 对 α 是线性的，故
        C(α) = (1−α)·C_self + α·C_nbr
    整族 α 只需上面这两个向量，扫参数是 O(N) 混合 —— 不重复走边。
    成本：2 次 bincount（m_e 的 src/dst 聚合），O(|E|) 时间、O(N+|E|) 内存，
    没有矩阵乘、没有 O(E·d) 中间量。

    agg: 节点级聚合算子，与 _node_agg 一致（"mean" 除以度 / "sym" 对称归一化）。
    """
    C_self = _node_agg(c_e, src, dst, deg, N, agg=agg)
    if agg == "mean":
        S = C_self * deg                  # Σ_{e∋u} c_e，复用已算好的分子，不再 bincount
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
    """Algorithm 1 第 2 阶段：把边级分歧聚合为节点级冲突分 C(v)，并给出真实误差向量。

    mode="hard": c_uv = I(ŷu=ŷv) XOR ê_uv，边判定取 0/1（原始实现）
    mode="soft": c_uv = E[同上]，把边判定换成其后验概率 p_uv：
                 c_uv = I(ŷu=ŷv)·(1−p_uv) + I(ŷu≠ŷv)·p_uv
                 硬指标是同一量的 Bernoulli 抽样，取期望只去掉抽样噪声，
                 不引入任何新信息（p_uv 来自已拟合的边分类器，审计期仍 label-free）。
    agg="mean": C(v) = Σ_u c_uv / |N(v)|（原实现，度归一化）
    agg="max":  C(v) = max_u c_uv（最坏邻边冲突；某一条高度可疑的边即可暴露节点，
                适合冲突稀疏、均值被大量 0 稀释的情形）
    alpha:      边的邻居冲突值重写强度（见 neighbor_conflict_pair），默认 0 = 原方法。

    返回 (C, err, t_agg)，t_agg 为聚合阶段墙钟耗时，供 Scalability 一节引用。
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
    """零成本基线：邻域预测标签一致度。风险 = 1 - (邻居中预测标签与自身相同的比例)。
    只用预测标签，不用特征、不用边分类器，是论文 §IV Ablation #3 的对照基线。"""
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
# 【② P1 修复】abstention：验证集调阈值 + test oracle 对照
# ============================================================
def abstention_eval(risk_val, correct_val, risk_test, correct_test,
                    coverages=ABST_COVERAGES):
    """对所有方法统一的协议：
    - val 协议（无泄漏）：tau = val 节点风险分的覆盖率分位数，原样应用到 test；
    - test oracle（泄漏对照）：tau 直接取 test 自身的覆盖率分位数，
      用于复现 Ablation #5 声称的「test 调优导致精度虚高」。"""
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
    """风险分数对错误的判别 AUC（越高越好）。"""
    err_t = (1 - correct_test).astype(int)
    if len(np.unique(err_t)) < 2: return np.nan
    return float(roc_auc_score(err_t, risk_test))

def exp_D_all_methods(C, pred, y_np, data, test_mask, val_mask, t_edge=0.0, t_agg=0.0):
    """所有方法（conflict / msp / entropy / mc_dropout / neighborhood）走同一协议。

    统一用「预算协议」：按风险分排序、固定保留 cov 比例的节点。
    同时给出 val 定阈值（无泄漏）与 test 定阈值（oracle 泄漏对照）两套数。
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
    # 邻域一致性基线（零成本，只用预测标签）
    nc = neighborhood_inconsistency(pred["y_pred"], pred["src"], pred["dst"], pred["deg"])
    methods["neighborhood"] = (nc[val_mask], nc[test_mask])

    # MC-Dropout（同时记录真实耗时，Scalability 一节引用这个数）
    mc_risk, t_mc = mc_dropout_risk(pred["model"], data)
    methods["mc_dropout"] = (mc_risk[val_mask], mc_risk[test_mask])

    for name, (r_v, r_t) in methods.items():
        results[name] = abstention_eval(r_v, correct_v, r_t, correct_t)
        aucs[name] = method_auc(r_t, correct_t)

    # 融合审计分（数据集级选 λ）：这里只产出部件。
    # λ 在 run_one_dataset 里用 5 个 seed 的验证集 AUC 平均曲线选择——
    # 比 per-seed 各选各的稳定得多（小图 val 噪声会被平均掉），审计阶段仍 label-free。
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

    # 冲突分全流程耗时：边特征/边分类器阶段(t_edge) + 节点级聚合阶段(t_agg)
    t_conflict = float(t_edge) + float(t_agg)
    return results, aucs, t_mc, t_conflict

def risk_coverage_curve(risk_test, correct_test, n_pts=10):
    """按风险升序保留前 x 比例节点的覆盖-精度曲线（画图用）。"""
    order = np.argsort(risk_test)
    cum_correct = np.cumsum(correct_test[order])
    coverages = np.linspace(0.1, 1.0, n_pts)
    accs = []
    for cov in coverages:
        n_keep = max(1, int(cov * len(risk_test)))
        accs.append(cum_correct[n_keep - 1] / n_keep)
    return coverages, np.array(accs)

# ============================================================
# 绘图（1x4 诊断图，供 make_2x2.py 切分）
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
        # 标出 val 调阈值协议在 80% 覆盖率下的实际工作点
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
# 曲线聚合工具
# ============================================================
def average_curves_on_targets(curve_lists, key):
    """多条曲线按相同目标点取均值（缺失点忽略）。"""
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
    """各 seed 的 p_n 目标不同，插值到公共网格后取均值。"""
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
# 单数据集单 seed
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
        # 【零选择部署模式】面向「模型训好、来了新图、没有 val/test 标签」的场景。
        # 与前几版的根本区别：没有候选集、没有 val 选择、没有守卫阈值。
        #   1) 基冲突固定为 soft：c_uv = I(same)(1−p_uv) + I(diff)p_uv
        #      它是硬指标的后验期望，只去掉伯努利抽样噪声，不引入新信息；
        #   2) 聚合固定为 deg：C(v)=Σ_u c_uv/d_v，值域 [0,1]，含义是"邻居冲突比例"，
        #      量纲与论文 Eq.(expect_cv) 一致，边界 h*=1/C 可直接照搬；
        #   3) 组合固定为 C̃(v)=C(v)·(1−M(v))，M 为邻居边的平均冲突（本底）。
        #      β=1 写死，不做选择：M 小时式子≈C（同配图自动退化为平滑），
        #      M 大时自动压掉普遍存在的冲突、只留超出本底的部分（异配图自动转对比）。
        src, dst, deg = pred["src"], pred["dst"], pred["deg"]
        same = (pred["y_pred"][src] == pred["y_pred"][dst]).astype(np.float32)
        p = np.asarray(pred["p_uv"], np.float32)
        c_e = (same * (1.0 - p) + (1.0 - same) * p).astype(np.float32)
        t0 = time.perf_counter()
        if CONFLICT_MODE == "curate2":
            # 变体：聚合改为 sym（邻居证据按 1/√d_u 打折），组合改为算术平滑。
            # 公式仍然唯一，只是两个算子的选择不同；不读 val、不读标签。
            # 消融开关（默认即主方法）：
            #   CF_AGG=sym|mean      节点级聚合算子，mean 退回度归一化
            #   CF_COMBINE=avg|none  none 表示丢掉邻居本底项，只用 C_self
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

        # ---- 现场诊断量：全部只用模型输出与图结构，审计期不读任何标签 ----
        # p_e 按论文 §III 的定义取「预测器标记边一致的概率」，不是准确率。
        p_e_obs = float(pred["e_pred"].mean())
        cbar_chosen = float(C_self.mean())
        # 恒等式（逐边精确，无需 ê 与同类指示独立）：
        #   C̄ = q + p_e − 2·E[I·ê]  ⇒  q = C̄ − p_e + 2·s
        s_obs = float((same * pred["e_pred"]).mean())
        q_hat = cbar_chosen - p_e_obs + 2.0 * s_obs
        # 边界判断：q̂ 与 1/C 比较决定「高冲突 = 错分」还是「高冲突 = 局部稀有」。
        # 仅当 p_e > 1/2 时该判据有效（论文 §III 已声明此前提）。
        mech_chosen = "trust" if (p_e_obs > 0.5 and q_hat > 1.0 / C_num) else "unreliable"
    elif CONFLICT_MODE == "nbr3" and pred.get("p_uv") is not None:
        # 【邻居冲突的第三种用法：机制由冲突场自身的平均水平决定】
        # 只读冲突值，不用 h、不用标签、不做逐数据集调参：
        #   C̄ ≤ τ → 冲突稀疏，邻居冲突主要是噪声 → 平滑 C̃=(C+M)/2
        #   C̄ > τ → 冲突遍地，邻居冲突是"本底"     → 对比 C̃=C·(1−M)
        # 唯一用 val 决定的是基冲突值取 hard 还是 soft（与原协议一致），
        # 外加 0.005 的守卫：gate 后的分数若不如 {hard,soft} 的 C，就退回 C。
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
            cbar = float(C_self.mean())          # 平均冲突水平，审计期可直接观测
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
                # 机制由 C̄ 固定，不做逐数据集挑机制；val 只做两件事：
                # (1) 基冲突值 hard/soft 二选一（与原协议一致）；
                # (2) 邻居项在 val 上确有增益（≥NBR_MARGIN）才启用，否则退回 C。
                # 第 (2) 条是「开关」不是「选择」：全数据集共用同一阈值，
                # 不引入任何额外信息（不看 h、不看标签、不看图规模）。
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
        # 【邻居冲突的第二种用法：换机制而不是混数值】
        # 对每种基冲突值 c_e，先算一次邻居本底 M(v)（=边的邻居平均冲突的节点级聚合），
        # 然后并行给出三类判决：
        #   C            —— 原方法（自己边上的冲突）
        #   lin+0.5      —— 平滑：把邻居冲突当降噪用的第二观测
        #   prod{b}      —— 乘性对比：C·(1−b·M)，把 M 当"本底冲突"，
        #                   只有"高出本底"的部分才算可疑（异配图上冲突遍地，这条最关键）
        #   max          —— 冲突稀疏时的最坏邻边
        # 全部由 (C_self, M) 两个向量 O(N) 组合得到，边只走 2 次 bincount，扫描不加成本。
        # 选优仍只看 val AUC，且提升不足 NBR_MARGIN 就退回 {hard,soft} 的 C。
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
        # 【nbr2 + 只改节点级聚合算子：不用模型中间结果、不用原始特征、不碰分类置信度】
        # 候选 = {hard, soft} × {deg, sym} × {C, lin, prod×3, max}
        #   deg: C(v) = Σ_u c_uv / d_v                                   （原方法，算术平均）
        #   sym: C(v) = Σ_u c_uv / (d_v · √d_u)                          （邻居证据按 √d_u 打折）
        # sym 的作用是压住 hub 邻居：高度数邻居与谁冲突都不稀奇，它对每个邻居的证据强度
        # 应按 1/√d_u 削弱；自己这一侧仍除以 d_v，保证量纲是"平均冲突"而不是总和。
        # tolokers/squirrel/wisconsin/cornell/texas 这类低度、度分布极偏的图上 sym 有效，
        # 其余图在 val 上会选回 deg，结果不变。
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
        # 【nbr2 + 冲突侧两处改动，仍不引入任何分类置信度（不碰 MSP）】
        # (1) 聚合算子：deg（除以度）与 sym（D^{-1/2}AD^{-1/2}）在 val 上二选一。
        #     同配高维图上"除以度"更稳，低度/异配图上"对称"更稳，谁也压不过谁。
        # (2) 基冲突值加上「模型中间结果」：把边判定概率换成 (1+cos)/2，
        #     cos 取 GCN 第一层之后的隐藏表征 z —— 不需要拟合，仍是"两端判定是否一致"。
        # 守卫仍以原方法（deg 聚合的 C）为参照，提升不足 NBR_MARGIN 就退回。
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
        # 【nbr2 的两处冲突侧改动，仍然不用任何分类置信度（不碰 MSP）】
        # (1) 聚合算子：对称归一化 D^{-1/2}AD^{-1/2}（C 与邻居本底 M 用同一算子）
        # (2) 基冲突值扩到「模型中间结果」与「原始特征」：
        #     把边判定概率换成 (1+cos)/2 —— 隐藏层 z 的余弦、输入特征 x 的余弦。
        #     这两个量不需要拟合，仍是"冲突"族（两端判定是否一致），只是换度量空间。
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
            # 高维大图（Squirrel/Pubmed）跳过原始空间：O(|E|·d) 会破坏复杂度声明
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
        # 【只换聚合算子：对称归一化代替除以度】
        # C(v) = Σ_{u∈N(v)} c_uv / sqrt(d_u d_v)，即 GCN 自己的 D^{-1/2}AD^{-1/2}。
        # 不引入新的选择维度：base 仍按 val AUC 在 hard/soft 之间二选一（与原协议一致），
        # 没有逐数据集挑机制，因此自由度与原方法相同。
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
        # 候选集 = {hard, soft} × {α=0} ∪ {α∈NBR_ALPHAS}，外加 soft_max（α=0）。
        # 因为 C(α)=(1−α)C_self+α·C_nbr 对 α 线性，每种基冲突值只走 2 次 bincount
        # 拿到 C_self / C_nbr，整族 α 是 O(N) 混合，扫参数不重复走边。
        # 选择协议与阈值调优一致：只看 val AUC，1 次选择、不碰 test；
        # 若最优候选相对 {hard, soft} 的提升不足 NBR_MARGIN，则退回，
        # 保证「在现有效果的前提下」不因选择噪声变差。
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

    # 【① 真实 AUC】test 上 C 对 node error 的 AUC
    auc_natural = method_auc(C_metric[test_mask], (pred["y_pred"][test_mask] == y_np[test_mask]).astype(float))

    # 【②】abstention：全部方法走同一协议
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
# 单数据集 5 seed 汇总
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

    # ---- 融合审计分：数据集级 λ（5 个 seed 的 val AUC 曲线平均后选择，小图更稳）----
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
        # 无害守卫：最优 λ 的平均 val AUC 没有明显超过纯置信项就退回 λ=1
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

    # 覆盖-精度曲线（取 seed 0 的排序，代表图；工作点用均值）
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
        # 边预测器在多少个 seed 上真正拟合成功（<1 表示该数据集上会退化，p_e 留 NaN）
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
                        help="sup: 训练边标签拟合的 LR（主配置）; unsup: 纯余弦阈值（Ablation #1）")
    parser.add_argument("--edge-space", type=str, default="auto", choices=["auto", "emb", "raw"],
                        help="auto: d>64 用 64 维 embedding，否则用原始特征（默认）; "
                             "emb: 强制 embedding; raw: 强制原始特征（旧实现 O(|E|·d)）")
    parser.add_argument("--edge-feat", type=str, default="full", choices=["cos", "cos_deg", "full", "pred"],
                        help="边分类器输入特征: cos / cos+度数差 / 再加二值重叠度")
    args = parser.parse_args()

    ds_names = [d.strip() for d in args.datasets.split(",")]
    os.makedirs(args.outdir, exist_ok=True)
    summary_rows, detailed = [], {}

    print(f"\n{'='*80}\n实验：{len(ds_names)} 数据集 x {len(SEEDS)} seeds | "
          f"edge_mode={args.edge_mode} edge_space={args.edge_space} "
          f"edge_feat={args.edge_feat} | MC-Dropout {N_DROPOUT_PASSES} 次 | "
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
            print(f"[{time.strftime('%H:%M:%S')}] {name} 完成，耗时 {time.time()-t0:.1f}s\n")
        except Exception as e:
            import traceback
            print(f"\n[错误] {name} 运行失败: {e}")
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

    print("\n" + "="*120 + "\n v2 汇总表（val 调阈值协议 + 真实 AUC + test 边 p_e）\n" + "="*120)
    show_cols = ["dataset", "h", "p_e_test", "skill_gain", "AUC", "auc_msp", "E_C",
                 "base_acc_pm", "acc80_pm", "acc_at_80_msp", "acc_at_80_nc",
                 "acc_at_80_mc", "t_conflict", "t_mc_dropout", "t_fit"]
    try: print(df_out[show_cols].to_markdown(index=False, tablefmt="grid"))
    except: print(df_out[show_cols].to_string(index=False))

    print("\n" + "="*80 + "\n 雷达图坐标 (E[C], base_acc) mean±std：\n" + "="*80)
    for r in summary_rows:
        print(f"({r['E_C']:.3f}, {r['base_acc']:.3f}) pm ({r['E_C']:.3f},{r['base_acc_std']:.3f}) % {r['dataset']}")

    with open(os.path.join(args.outdir, "global_summary_results_v2.json"), "w") as f:
        json.dump(detailed, f, indent=2, default=float)
    df.to_csv(os.path.join(args.outdir, "global_summary_table_v2.csv"), index=False)
    print(f"\n[done] JSON -> {args.outdir}/global_summary_results_v2.json")
    print(f"[done] CSV  -> {args.outdir}/global_summary_table_v2.csv")
