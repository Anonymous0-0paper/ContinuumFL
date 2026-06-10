"""
AP-CFL: Clustered Federated Learning Through Dynamic Clustering and Adaptive Participation.

Paper: "AP-CFL: Clustered Federated Learning Through Dynamic Clustering and
Adaptive Participation in Heterogeneous IoT"
IEEE Internet of Things Journal, Vol. 12, No. 10, May 2025.

Faithful baseline implementation — do not modify algorithm logic.
"""

import copy
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import AffinityPropagation
from .metrics_utils import compute_prf as _compute_prf, find_convergence as _find_convergence


# ---------------------------------------------------------------------------
# Helpers: encoder / classifier parameter split
# ---------------------------------------------------------------------------

_CLASSIFIER_KEYWORDS = ("fc", "linear", "classifier", "head", "dense", "output")


def _is_classifier_param(name: str) -> bool:
    n = name.lower()
    return any(kw in n for kw in _CLASSIFIER_KEYWORDS)


def split_state_dict(state_dict: Dict[str, torch.Tensor]) -> Tuple[Dict, Dict]:
    """Split a full state_dict into (encoder_sd, classifier_sd)."""
    encoder_sd, classifier_sd = {}, {}
    for k, v in state_dict.items():
        if _is_classifier_param(k):
            classifier_sd[k] = v
        else:
            encoder_sd[k] = v
    return encoder_sd, classifier_sd


def merge_state_dict(encoder_sd: Dict, classifier_sd: Dict) -> Dict:
    merged = {}
    merged.update(encoder_sd)
    merged.update(classifier_sd)
    return merged


def sd_to_vector(sd: Dict[str, torch.Tensor]) -> np.ndarray:
    return np.concatenate([v.detach().cpu().float().numpy().ravel() for v in sd.values()])


# ---------------------------------------------------------------------------
# APCFL class
# ---------------------------------------------------------------------------

class APCFL:
    """
    AP-CFL: server-side coordinator implementing the full algorithm.

    Exposes: train_round(), local_update(), cluster_clients(), aggregate(), evaluate().
    """

    def __init__(self, config, devices: Dict[str, Any], global_model: nn.Module, dataset: Any):
        self.config = config
        self.devices = devices
        self.dataset = dataset

        # Hyperparameters (paper defaults, overridable via config.apcfl_*)
        self.num_rounds: int = getattr(config, "apcfl_num_rounds", config.num_rounds)
        self.local_epochs: int = getattr(config, "apcfl_local_epochs", 1)
        self.batch_size: int = getattr(config, "apcfl_batch_size", 32)
        self.lr: float = getattr(config, "apcfl_lr", getattr(config, "learning_rate", 0.001))
        self.lam: float = getattr(config, "apcfl_lambda", 0.05)
        self.sampling_rate: float = getattr(config, "apcfl_sampling_rate", 0.7)

        # Compute device
        self._dev = "cpu"
        if getattr(config, "device", "cpu") .startswith("cuda") and torch.cuda.is_available():
            self._dev = "cuda"

        # --- Server state ---
        # Global encoder state dict (shared across all clients)
        full_sd = copy.deepcopy(global_model.state_dict())
        enc_sd, cls_sd = split_state_dict(full_sd)
        self.global_encoder_sd: Dict = enc_sd

        # Cluster classifiers: list of classifier state_dicts.
        # Initialise with a single cluster.
        self.cluster_classifiers: List[Dict] = [copy.deepcopy(cls_sd)]

        # Accumulating client set S^t: {client_id: {encoder_sd, classifier_sd, dataset_size, last_round}}
        self.client_pool: Dict[str, Dict] = {}

        # Reference model (for split detection)
        self._template_model = global_model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        """Full training loop."""
        accuracies: List[float] = []
        losses: List[float] = []
        precisions: List[float] = []
        recalls: List[float] = []
        f1s: List[float] = []
        round_times: List[float] = []
        cluster_counts: List[int] = []
        t_start = time.time()

        try:
            for t in range(1, self.num_rounds + 1):
                t_round = time.time()
                acc, loss, K = self.train_round(t)
                # train_round already called evaluate() — fetch full metrics
                _, _, prec, rec, f1 = self.evaluate()
                accuracies.append(acc); losses.append(loss)
                precisions.append(prec); recalls.append(rec); f1s.append(f1)
                round_times.append(time.time() - t_round)
                cluster_counts.append(K)
                print(
                    f"AP-CFL Round {t}/{self.num_rounds}: "
                    f"Acc={acc:.4f}  Loss={loss:.4f}  F1={f1:.4f}  Clusters={K}  "
                    f"Time={round_times[-1]:.2f}s"
                )
        except KeyboardInterrupt:
            print("AP-CFL training interrupted.")

        return {
            "method":         "APCFL",
            "final_accuracy": accuracies[-1] if accuracies else 0.0,
            "final_loss":     losses[-1]     if losses     else float("inf"),
            "accuracies":     accuracies,
            "losses":         losses,
            "precisions":     precisions,
            "recalls":        recalls,
            "f1s":            f1s,
            "round_times":    round_times,
            "cluster_counts": cluster_counts,
            "total_time":     time.time() - t_start,
            "convergence_round": _find_convergence(accuracies),
        }

    def train_round(self, t: int) -> Tuple[float, float, int]:
        """Execute one federated round; returns (accuracy, loss, num_clusters)."""
        # 1. Sample clients for this round
        sampled_ids = self._sample_clients()
        if not sampled_ids:
            acc, loss, prec, rec, f1 = self.evaluate()
            return acc, loss, len(self.cluster_classifiers)

        # 2. Update accumulating set S^t and last-seen round
        for cid in sampled_ids:
            if cid not in self.client_pool:
                self.client_pool[cid] = {
                    "encoder_sd": copy.deepcopy(self.global_encoder_sd),
                    "classifier_sd": copy.deepcopy(self.cluster_classifiers[0]),
                    "dataset_size": self.devices[cid].dataset_size or 1,
                    "last_round": t,
                }
            else:
                self.client_pool[cid]["last_round"] = t

        # 3. Local updates for sampled clients
        for cid in sampled_ids:
            result = self.local_update(cid, t)
            if result is not None:
                self.client_pool[cid]["encoder_sd"] = result["encoder_sd"]
                self.client_pool[cid]["classifier_sd"] = result["classifier_sd"]

        # 4. Aggregate
        self.aggregate(t)

        # 5. Evaluate
        acc, loss, _p, _r, _f = self.evaluate()
        return acc, loss, len(self.cluster_classifiers)

    def local_update(self, client_id: str, t: int) -> Optional[Dict]:
        """
        Client-side: select best cluster, then train with encoder regularisation.
        Returns dict with updated encoder_sd and classifier_sd, or None on failure.
        """
        device = self.devices[client_id]
        dataloader = device.local_dataloader
        if dataloader is None or len(dataloader) == 0:
            return None

        # --- Cluster selection: k* = argmin CE over all cluster classifiers ---
        k_star = self.cluster_clients(client_id)

        # Build a full state dict for local model
        enc_sd = copy.deepcopy(self.global_encoder_sd)
        cls_sd = copy.deepcopy(self.cluster_classifiers[k_star])
        local_sd = merge_state_dict(enc_sd, cls_sd)

        local_model = copy.deepcopy(self._template_model)
        local_model.load_state_dict(local_sd)

        # Reference encoder for regularisation (frozen copy on device)
        ref_enc_params = {
            k: v.to(self._dev)
            for k, v in self.global_encoder_sd.items()
        }

        if self._dev.startswith("cuda"):
            local_model = local_model.to(self._dev)

        local_model.train()
        optimizer = torch.optim.SGD(local_model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        try:
            for _ in range(self.local_epochs):
                for data, target in dataloader:
                    data = data.to(self._dev)
                    target = target.to(self._dev)

                    optimizer.zero_grad()
                    output = local_model(data)
                    if isinstance(output, tuple):
                        output = output[0]

                    ce_loss = criterion(output, target)

                    # Encoder regularisation: (λ/2) ||θ_en - θ^t_en||²
                    reg = torch.tensor(0.0, device=self._dev)
                    for name, param in local_model.named_parameters():
                        if not _is_classifier_param(name) and name in ref_enc_params:
                            reg = reg + ((param - ref_enc_params[name]) ** 2).sum()

                    loss = ce_loss + (self.lam / 2.0) * reg
                    loss.backward()
                    optimizer.step()
        except Exception as e:
            print(f"  AP-CFL local_update error for {client_id}: {e}")
            return None

        if self._dev.startswith("cuda"):
            local_model = local_model.cpu()

        updated_sd = local_model.state_dict()
        new_enc_sd, new_cls_sd = split_state_dict(updated_sd)
        return {"encoder_sd": new_enc_sd, "classifier_sd": new_cls_sd}

    def cluster_clients(self, client_id: str) -> int:
        """
        Client-side cluster selection: argmin_{k} CE(encoder + classifier_k, D_i).
        Returns cluster index k*.
        """
        if len(self.cluster_classifiers) == 1:
            return 0

        device = self.devices[client_id]
        dataloader = device.local_dataloader
        if dataloader is None or len(dataloader) == 0:
            return 0

        best_k, best_loss = 0, float("inf")
        criterion = nn.CrossEntropyLoss()

        for k, cls_sd in enumerate(self.cluster_classifiers):
            model = copy.deepcopy(self._template_model)
            model.load_state_dict(merge_state_dict(self.global_encoder_sd, cls_sd))
            model = model.to(self._dev)
            model.eval()

            total_loss = 0.0
            n_batches = 0
            with torch.no_grad():
                for data, target in dataloader:
                    data, target = data.to(self._dev), target.to(self._dev)
                    out = model(data)
                    if isinstance(out, tuple):
                        out = out[0]
                    total_loss += criterion(out, target).item()
                    n_batches += 1
                    if n_batches >= 3:  # limit CE eval to a few batches for speed
                        break

            avg_loss = total_loss / max(n_batches, 1)
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_k = k

        return best_k

    def aggregate(self, t: int):
        """
        Server-side:
        1. Compute TDI weights for all clients in S^t
        2. Aggregate global encoder (TDI over all S^t)
        3. Run AP clustering on classifier similarities (MADC)
        4. Aggregate cluster classifiers (TDI within each cluster)
        """
        pool = self.client_pool
        if not pool:
            return

        client_ids = list(pool.keys())

        # --- TDI weights ---
        tdi = {
            cid: pool[cid]["dataset_size"] * np.exp(-(t - pool[cid]["last_round"]))
            for cid in client_ids
        }
        total_tdi = sum(tdi.values()) or 1.0

        # --- Global encoder aggregation (all S^t) ---
        enc_keys = list(self.global_encoder_sd.keys())
        new_enc = {k: torch.zeros_like(self.global_encoder_sd[k], dtype=torch.float32) for k in enc_keys}
        for cid in client_ids:
            w = tdi[cid] / total_tdi
            for k in enc_keys:
                if k in pool[cid]["encoder_sd"]:
                    new_enc[k] += w * pool[cid]["encoder_sd"][k].float()
        self.global_encoder_sd = new_enc

        # --- AP clustering on classifier similarities ---
        cluster_assignments = self._run_ap_clustering(client_ids, pool)

        # --- Cluster classifier aggregation ---
        clusters: Dict[int, List[str]] = defaultdict(list)
        for cid, clabel in zip(client_ids, cluster_assignments):
            clusters[clabel].append(cid)

        new_classifiers = []
        cls_keys = list(self.cluster_classifiers[0].keys()) if self.cluster_classifiers else []

        for clabel in sorted(clusters.keys()):
            members = clusters[clabel]
            tdi_sum = sum(tdi[cid] for cid in members) or 1.0
            new_cls = {k: torch.zeros_like(pool[members[0]]["classifier_sd"][k], dtype=torch.float32)
                       for k in cls_keys if k in pool[members[0]]["classifier_sd"]}
            for cid in members:
                w = tdi[cid] / tdi_sum
                for k in new_cls:
                    if k in pool[cid]["classifier_sd"]:
                        new_cls[k] += w * pool[cid]["classifier_sd"][k].float()
            new_classifiers.append(new_cls)

        if new_classifiers:
            self.cluster_classifiers = new_classifiers

    def evaluate(self) -> Tuple[float, float, float, float, float]:
        """Evaluate using per-sample best-cluster selection (argmin loss over all cluster classifiers).
        Returns (accuracy, loss, precision, recall, f1)."""
        if not self.cluster_classifiers:
            return 0.0, float("inf"), 0.0, 0.0, 0.0

        # Build one model per cluster classifier
        models = []
        for cls_sd in self.cluster_classifiers:
            m = copy.deepcopy(self._template_model)
            m.load_state_dict(merge_state_dict(self.global_encoder_sd, cls_sd))
            m = m.to(self._dev)
            m.eval()
            models.append(m)

        criterion_none = nn.CrossEntropyLoss(reduction="none")
        all_preds: List[torch.Tensor] = []
        all_targets: List[torch.Tensor] = []
        total_loss_sum = 0.0
        total_samples = 0
        num_classes = 2

        try:
            loader = self.dataset.get_global_dataloader(batch_size=64, is_train=False)
            with torch.no_grad():
                for data, target in loader:
                    data, target = data.to(self._dev), target.to(self._dev)
                    per_cluster_loss = []
                    outs = []
                    for m in models:
                        out = m(data)
                        if isinstance(out, tuple):
                            out = out[0]
                        num_classes = out.shape[1]
                        per_cluster_loss.append(criterion_none(out, target))
                        outs.append(out)
                    # pick best cluster per sample
                    stacked_loss = torch.stack(per_cluster_loss, dim=0)  # [K, B]
                    best_k = stacked_loss.argmin(dim=0)                  # [B]
                    stacked_out = torch.stack(outs, dim=0)               # [K, B, C]
                    idx = best_k.view(1, -1, 1).expand(1, -1, num_classes)
                    best_out = stacked_out.gather(0, idx).squeeze(0)     # [B, C]
                    pred = best_out.argmax(1)
                    all_preds.append(pred.cpu())
                    all_targets.append(target.cpu())
                    total_loss_sum += stacked_loss.gather(0, best_k.unsqueeze(0)).sum().item()
                    total_samples += target.size(0)

            preds_t = torch.cat(all_preds)
            targets_t = torch.cat(all_targets)
            acc = preds_t.eq(targets_t).float().mean().item()
            avg_loss = total_loss_sum / max(total_samples, 1)
            precision, recall, f1 = _compute_prf(preds_t, targets_t, num_classes)
        except Exception as e:
            print(f"  AP-CFL evaluate error: {e}")
            acc, avg_loss, precision, recall, f1 = 0.0, float("inf"), 0.0, 0.0, 0.0

        return acc, avg_loss, precision, recall, f1

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sample_clients(self) -> List[str]:
        available = [
            cid for cid, dev in self.devices.items()
            if dev.is_active and dev.local_dataset
        ]
        n = max(1, int(self.sampling_rate * len(available)))
        return list(np.random.choice(available, size=n, replace=False))

    def _run_ap_clustering(self, client_ids: List[str], pool: Dict) -> List[int]:
        """
        Run Affinity Propagation on MADC similarity matrix computed over classifier params.
        Returns cluster label per client (same order as client_ids).
        """
        n = len(client_ids)
        if n == 1:
            return [0]

        # Cosine similarities between classifier vectors
        cls_vecs = np.vstack([sd_to_vector(pool[cid]["classifier_sd"]) for cid in client_ids])

        # Normalise rows for cosine similarity
        norms = np.linalg.norm(cls_vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        cls_vecs_norm = cls_vecs / norms

        C = cls_vecs_norm @ cls_vecs_norm.T  # (n, n) cosine similarity

        # MADC: SIM(i,j) = -(1/(n-2)) * sum_{z != i,j} |C(i,z) - C(j,z)|
        SIM = np.zeros((n, n))
        if n > 2:
            denom = n - 2
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    others = [z for z in range(n) if z != i and z != j]
                    SIM[i, j] = -(1.0 / denom) * np.sum(np.abs(C[i, others] - C[j, others]))
        else:
            # n == 2: use plain negative L1 distance as fallback
            SIM[0, 1] = SIM[1, 0] = -np.abs(C[0, 1] - C[1, 0])

        # Diagonal (preference) = median of off-diagonal similarities
        pref = np.median(SIM[~np.eye(n, dtype=bool)]) if n > 1 else 0.0
        np.fill_diagonal(SIM, pref)

        try:
            ap = AffinityPropagation(affinity="precomputed", preference=pref,
                                     max_iter=200, convergence_iter=15, random_state=42)
            labels = ap.fit_predict(SIM)
        except Exception as e:
            print(f"  AP-CFL: AffinityPropagation failed ({e}), falling back to single cluster.")
            labels = np.zeros(n, dtype=int)

        return labels.tolist()
