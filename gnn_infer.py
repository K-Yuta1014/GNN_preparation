# gnn_infer.py (Windows/CPU inference; PyG + RDKit + PyTorch)

import os
from typing import List, Dict, Any, Tuple, Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from rdkit import Chem

import torch_geometric
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing, global_add_pool
from torch_geometric.loader import DataLoader

import numpy as np

# ----------------------------
# Discrete feature constants
# ----------------------------
MAX_ATOMIC_NUM = 100
MAX_DEGREE = 5

BOND_TYPES = {
    Chem.rdchem.BondType.SINGLE: 0,
    Chem.rdchem.BondType.DOUBLE: 1,
    Chem.rdchem.BondType.TRIPLE: 2,
    Chem.rdchem.BondType.AROMATIC: 3,
}
NUM_BOND_TYPES = 4

MIN_FC, MAX_FC = -2, 2
FC_OFFSET = -MIN_FC
NUM_FC = (MAX_FC - MIN_FC + 1)

HYB_MAP = {
    Chem.rdchem.HybridizationType.SP: 0,
    Chem.rdchem.HybridizationType.SP2: 1,
    Chem.rdchem.HybridizationType.SP3: 2,
    Chem.rdchem.HybridizationType.SP3D: 3,
    Chem.rdchem.HybridizationType.SP3D2: 4,
}
HYB_UNKNOWN = 5
NUM_HYB = 6


def atom_features(atom: Chem.rdchem.Atom) -> torch.Tensor:
    atomic_num = min(atom.GetAtomicNum(), MAX_ATOMIC_NUM)
    degree = min(atom.GetDegree(), MAX_DEGREE)
    aromatic = int(atom.GetIsAromatic())

    formal_charge = atom.GetFormalCharge()
    formal_charge = max(MIN_FC, min(formal_charge, MAX_FC)) + FC_OFFSET

    hyb_idx = HYB_MAP.get(atom.GetHybridization(), HYB_UNKNOWN)

    return torch.tensor([atomic_num, degree, aromatic, formal_charge, hyb_idx], dtype=torch.long)


def bond_features(bond: Chem.rdchem.Bond) -> torch.Tensor:
    bond_type = BOND_TYPES.get(bond.GetBondType(), 0)
    conjugated = int(bond.GetIsConjugated())
    in_ring = int(bond.IsInRing())
    return torch.tensor([bond_type, conjugated, in_ring], dtype=torch.long)


def canonicalize_smiles(smi: str) -> Optional[str]:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def serializable_to_scalers(d: Dict[str, Dict[str, float]]) -> Dict[str, Tuple[float, float]]:
    return {k: (float(v["median"]), float(v["iqr"])) for k, v in d.items()}


def inverse_robust_scalar(x_scaled: float, med: float, iqr: float) -> float:
    return float(x_scaled * iqr + med)


def smiles_to_pyg_discrete_v2(smiles: str) -> Optional[Data]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    x = torch.stack([atom_features(a) for a in mol.GetAtoms()], dim=0)

    edge_index_list, edge_attr_list = [], []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bf = bond_features(bond)
        edge_index_list += [[i, j], [j, i]]
        edge_attr_list  += [bf, bf]

    if len(edge_index_list) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr  = torch.empty((0, 3), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        edge_attr  = torch.stack(edge_attr_list, dim=0)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.smiles = smiles
    return data


class EdgeCondLinearLayer(MessagePassing):
    def __init__(self, hidden_dim: int, num_bond_types: int):
        super().__init__(aggr="add")
        self.hidden_dim = hidden_dim
        self.num_bond_types = num_bond_types

        self.W = nn.Parameter(torch.empty(num_bond_types, hidden_dim, hidden_dim))
        nn.init.xavier_uniform_(self.W)

        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

    def forward(self, x, edge_index, bond_type):
        m = self.propagate(edge_index, x=x, bond_type=bond_type)
        return self.gru(m, x)

    def message(self, x_j, bond_type):
        bt = bond_type.clamp(0, self.num_bond_types - 1)
        W_bt = self.W[bt]
        return torch.bmm(W_bt, x_j.unsqueeze(-1)).squeeze(-1)


class MPNNRegressor(nn.Module):
    def __init__(self, hidden_dim: int = 128, num_layers: int = 3, num_targets: int = 1, dropout: float = 0.0):
        super().__init__()
        self.emb_atomic = nn.Embedding(MAX_ATOMIC_NUM + 1, 64)
        self.emb_degree = nn.Embedding(MAX_DEGREE + 1, 16)
        self.emb_aroma  = nn.Embedding(2, 8)
        self.emb_fc     = nn.Embedding(NUM_FC, 8)
        self.emb_hyb    = nn.Embedding(NUM_HYB, 8)

        node_in_dim = 64 + 16 + 8 + 8 + 8
        self.node_proj = nn.Linear(node_in_dim, hidden_dim)

        self.layers = nn.ModuleList([
            EdgeCondLinearLayer(hidden_dim=hidden_dim, num_bond_types=NUM_BOND_TYPES)
            for _ in range(num_layers)
        ])

        self.fp_dim = 1024
        self.node_to_fp = nn.Linear(hidden_dim, self.fp_dim)

        self.fc1 = nn.Linear(self.fp_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.out = nn.Linear(256, num_targets)

        self.dropout = float(dropout)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = data.batch

        atomic_num    = x[:, 0].clamp(0, MAX_ATOMIC_NUM)
        degree        = x[:, 1].clamp(0, MAX_DEGREE)
        aromatic      = x[:, 2].clamp(0, 1)
        formal_charge = x[:, 3].clamp(0, NUM_FC - 1)
        hybridization = x[:, 4].clamp(0, NUM_HYB - 1)

        h = torch.cat([
            self.emb_atomic(atomic_num),
            self.emb_degree(degree),
            self.emb_aroma(aromatic),
            self.emb_fc(formal_charge),
            self.emb_hyb(hybridization),
        ], dim=-1)

        h = self.node_proj(h)

        bond_type = edge_attr[:, 0].clamp(0, NUM_BOND_TYPES - 1)
        for layer in self.layers:
            h = layer(h, edge_index, bond_type)

        g = global_add_pool(self.node_to_fp(h), batch)

        z = F.relu(self.bn1(self.fc1(g)))
        if self.dropout > 0:
            z = F.dropout(z, p=self.dropout, training=self.training)
        z = F.relu(self.bn2(self.fc2(z)))
        if self.dropout > 0:
            z = F.dropout(z, p=self.dropout, training=self.training)

        return self.out(z)


PREPROCESS_REGISTRY: Dict[str, Callable[[str], Any]] = {
    "discrete_atom_bond_v2": smiles_to_pyg_discrete_v2,
}


def build_mpnn_edgecond_gru(model_config: Dict[str, Any]) -> nn.Module:
    return MPNNRegressor(
        hidden_dim=int(model_config["hidden_dim"]),
        num_layers=int(model_config["num_layers"]),
        num_targets=int(model_config["num_targets"]),
        dropout=float(model_config.get("dropout", 0.0)),
    )


MODEL_REGISTRY: Dict[str, Callable[[Dict[str, Any]], nn.Module]] = {
    "mpnn_edgecond_gru": build_mpnn_edgecond_gru,
}

class MahalanobisAD:
    """
    GNN fingerprint g を入力に、Mahalanobis距離でAD(in/out)を返す。
    """
    def __init__(self, ad_dir: str):
        self.ad_dir = ad_dir
        self.mu = np.load(os.path.join(ad_dir, "md_mu.npy"))                      # (1024,)
        self.inv_cov = np.load(os.path.join(ad_dir, "md_inv_cov.npy"))            # (1024,1024)
        self.thr = float(np.load(os.path.join(ad_dir, "md_thr.npy"))[0])          # scalar
        self.median = np.load(os.path.join(ad_dir, "md_median.npy"))
        self.iqr    = np.load(os.path.join(ad_dir, "md_iqr.npy"))


    def score(self, g_1d: np.ndarray) -> float:
        """
        g_1d: (1024,)
        returns: Mahalanobis distance (float)
        """
        G = np.asarray(g_1d, dtype=np.float64).reshape(1, -1)      # (1,1024)
        Gs = (G - self.median) / self.iqr                          # standardized
        D = Gs - self.mu.reshape(1, -1)
        md = float(np.einsum("bi,ij,bj->b", D, self.inv_cov, D)[0])
        return md

    def label(self, g_1d: np.ndarray) -> str:
        md = self.score(g_1d)
        return "out" if md > self.thr else "in"

class GenericGNNPredictor:
    def __init__(self,
                 ckpt_path: str,
                 device: Optional[torch.device] = None,
                 ad_dir: Optional[str] = None
                 ):
        self.ckpt_path = ckpt_path
        self.device = device if device is not None else torch.device("cpu")

        try:
            payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except TypeError:
            # older torch without weights_only argument
            payload = torch.load(ckpt_path, map_location="cpu")

        self.dataset_name = payload.get("dataset_name", "unknown")
        self.model_config = payload["model_config"]
        self.preprocess_config = payload["preprocess_config"]
        self.scalers = serializable_to_scalers(payload["scalers"])

        self.target_cols = list(self.model_config["target_cols"])
        self.num_targets = int(self.model_config["num_targets"])

        model_name = self.model_config.get("model_name")
        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model_name='{model_name}'. Available: {list(MODEL_REGISTRY.keys())}")
        self.model = MODEL_REGISTRY[model_name](self.model_config).to(self.device)
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()

        preprocess_name = self.preprocess_config.get("preprocess_name")
        if preprocess_name not in PREPROCESS_REGISTRY:
            raise ValueError(f"Unknown preprocess_name='{preprocess_name}'. Available: {list(PREPROCESS_REGISTRY.keys())}")
        self.smiles_to_data = PREPROCESS_REGISTRY[preprocess_name]

        self.do_canonicalize = bool(self.preprocess_config.get("smiles_canonicalize", True))

        self.ad = MahalanobisAD(ad_dir) if ad_dir else None

    @torch.no_grad()
    def predict_smiles(self, smiles: str) -> Dict[str, Any]:
        smi_in = smiles
        smi = smiles

        if self.do_canonicalize:
            smi2 = canonicalize_smiles(smi)
            if smi2 is None:
                return {"ok": False, "smiles_in": smi_in, "error": "Invalid SMILES"}
            smi = smi2

        data = self.smiles_to_data(smi)
        if data is None:
            return {"ok": False, "smiles_in": smi_in, "smiles_used": smi, "error": "Failed to build graph"}

        batch = next(iter(DataLoader([data], batch_size=1, shuffle=False))).to(self.device)

        pred_scaled = self.model(batch).detach().cpu().view(-1)
        if pred_scaled.numel() != self.num_targets:
            return {"ok": False, "smiles_in": smi_in, "smiles_used": smi,
                    "error": f"Pred dim mismatch: got {pred_scaled.numel()} expected {self.num_targets}"}

        pred_scaled_dict = {t: float(pred_scaled[i].item()) for i, t in enumerate(self.target_cols)}
        pred_raw_dict = {}
        for i, t in enumerate(self.target_cols):
            med, iqr = self.scalers[t]
            pred_raw_dict[t] = inverse_robust_scalar(float(pred_scaled[i].item()), med, iqr)

        return {
            "ok": True,
            "dataset": self.dataset_name,
            "smiles_in": smi_in,
            "smiles_used": smi,
            "pred_scaled": pred_scaled_dict,
            "pred_raw": pred_raw_dict,
        }
    
    @torch.no_grad()
    def embed_smiles(self, smiles: str) -> Dict[str, Any]:
        """
        fingerprint g（1024次元）を返す。AD用。
        """
        smi_in = smiles
        smi = smiles

        if self.do_canonicalize:
            smi2 = canonicalize_smiles(smi)
            if smi2 is None:
                return {"ok": False, "smiles_in": smi_in, "error": "Invalid SMILES"}
            smi = smi2

        data = self.smiles_to_data(smi)
        if data is None:
            return {"ok": False, "smiles_in": smi_in, "smiles_used": smi, "error": "Failed to build graph"}

        batch = next(iter(DataLoader([data], batch_size=1, shuffle=False))).to(self.device)

        m = self.model
        m.eval()

        x = batch.x
        edge_index = batch.edge_index
        edge_attr  = batch.edge_attr
        batch_vec  = batch.batch

        atomic_num    = x[:, 0].clamp(0, MAX_ATOMIC_NUM)
        degree        = x[:, 1].clamp(0, MAX_DEGREE)
        aromatic      = x[:, 2].clamp(0, 1)
        formal_charge = x[:, 3].clamp(0, NUM_FC - 1)
        hybridization = x[:, 4].clamp(0, NUM_HYB - 1)

        h = torch.cat([
            m.emb_atomic(atomic_num),
            m.emb_degree(degree),
            m.emb_aroma(aromatic),
            m.emb_fc(formal_charge),
            m.emb_hyb(hybridization),
        ], dim=-1)

        h = m.node_proj(h)

        bond_type = edge_attr[:, 0].clamp(0, NUM_BOND_TYPES - 1)
        for layer in m.layers:
            h = layer(h, edge_index, bond_type)

        g = global_add_pool(m.node_to_fp(h), batch_vec)   # [1, 1024]
        g = g.detach().cpu().view(-1).numpy()

        return {"ok": True, "smiles_in": smi_in, "smiles_used": smi, "g": g}

    @torch.no_grad()
    def predict_smiles_with_ad(self, smiles: str) -> Dict[str, Any]:
        """
        予測 + AD判定（ad_dir がある場合）。
        戻り値に ad_label / ad_score を追加する。
        """
        res = self.predict_smiles(smiles)

        # 予測自体が失敗したらそのまま返す
        if not res.get("ok", False):
            return res

        # ADが無い場合は予測のみ
        if self.ad is None:
            res["ad"] = {"ok": False, "reason": "ad_not_loaded"}
            return res

        emb = self.embed_smiles(res.get("smiles_used", smiles))
        if not emb.get("ok", False):
            res["ad"] = {"ok": False, "reason": emb.get("error", "embed_failed")}
            return res

        g = emb["g"]
        md = self.ad.score(g)
        lab = "out" if md > self.ad.thr else "in"

        res["ad"] = {
            "ok": True,
            "method": "mahalanobis",
            "score": float(md),
            "threshold": float(self.ad.thr),
            "label": lab,
        }
        return res
    
def load_predictor_generic(
    dataset_name: str,
    save_root: str = "models",
    ckpt_name: str = "checkpoint.pt",
    device: Optional[torch.device] = None,
    ad_dir: Optional[str] = None,
) -> GenericGNNPredictor:
    ckpt_path = os.path.join(save_root, dataset_name, ckpt_name)
    return GenericGNNPredictor(ckpt_path=ckpt_path, device=device, ad_dir=ad_dir)
