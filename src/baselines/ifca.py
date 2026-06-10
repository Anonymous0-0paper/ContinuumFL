"""
IFCA: Iterative Federated Clustering Algorithm.

Paper: "An Efficient Framework for Clustered Federated Learning"
Venue: NeurIPS 2020
Authors: Avishek Ghosh, Jichan Chung, Dong Yin, Kannan Ramchandran (UC Berkeley / DeepMind)
arXiv: https://arxiv.org/abs/2006.04088

Faithful baseline — do not modify algorithm logic.
"""

import copy
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from .metrics_utils import compute_prf as _compute_prf, find_convergence as _find_convergence


class IFCA:
    """
    Iterative Federated Clustering Algorithm (NeurIPS 2020).

    Maintains k cluster models.  Each round:
      1. Participating clients evaluate all k models on local data → pick argmin loss.
      2. Clients run local SGD on the assigned cluster model.
      3. Server averages locally updated models within each cluster.

    Exposes: train_round(), local_update(), estimate_cluster(), aggregate(), evaluate().
    """

    def __init__(self, config, devices: Dict[str, Any], global_model: nn.Module, dataset: Any):
        self.config  = config
        self.devices = devices
        self.dataset = dataset

        # Hyper-parameters (paper defaults, overridable via config.ifca_*)
        self.k: int              = getattr(config, "ifca_k",               4)
        self.num_rounds: int     = getattr(config, "ifca_num_rounds",      config.num_rounds)
        self.local_steps: int    = getattr(config, "ifca_local_steps",     5)
        self.lr: float           = getattr(config, "ifca_lr",              getattr(config, "learning_rate", 0.01))
        self.lr_decay: float     = getattr(config, "ifca_lr_decay",        1.0)   # per-round multiplier
        self.batch_size: int     = getattr(config, "ifca_batch_size",      32)
        self.sampling_rate: float = getattr(config, "ifca_sampling_rate",  0.7)
        self.variant: str        = getattr(config, "ifca_variant",         "model")   # "model" | "gradient"
        self.weight_sharing: bool = getattr(config, "ifca_weight_sharing", False)

        self._dev = "cpu"
        if getattr(config, "device", "cpu") == "cuda" and torch.cuda.is_available():
            self._dev = "cuda"

        self._template_model = global_model

        # Initialise k cluster models from the global model (warm start).
        # Random-noise init causes cluster assignments to be meaningless early on
        # and wastes many rounds escaping the noise basin.
        self.cluster_models: List[Dict[str, torch.Tensor]] = [
            copy.deepcopy(global_model.state_dict()) for _ in range(self.k)
        ]

        self._current_lr = self.lr

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        """Full training loop."""
        accuracies: List[float] = []
        losses:     List[float] = []
        precisions: List[float] = []
        recalls:    List[float] = []
        f1s:        List[float] = []
        round_times: List[float] = []
        cluster_counts: List[List[int]] = []
        t_start = time.time()

        try:
            for t in range(1, self.num_rounds + 1):
                t_round = time.time()
                acc, loss, counts = self.train_round(t)
                _, _, prec, rec, f1 = self.evaluate()
                accuracies.append(acc); losses.append(loss)
                precisions.append(prec); recalls.append(rec); f1s.append(f1)
                round_times.append(time.time() - t_round)
                cluster_counts.append(counts)
                self._current_lr *= self.lr_decay
                print(
                    f"IFCA Round {t}/{self.num_rounds}: "
                    f"Acc={acc:.4f}  Loss={loss:.4f}  F1={f1:.4f}  "
                    f"Clusters={counts}  Time={round_times[-1]:.2f}s"
                )
        except KeyboardInterrupt:
            print("IFCA training interrupted.")

        final_acc, final_loss, final_prec, final_rec, final_f1 = self.evaluate()

        return {
            "method":         "IFCA",
            "final_accuracy": final_acc,
            "final_loss":     final_loss,
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

    def train_round(self, t: int) -> Tuple[float, float, List[int]]:
        """
        One federated round.
        Returns (accuracy, loss, list_of_client_counts_per_cluster).
        """
        sampled_ids = self._sample_clients()
        if not sampled_ids:
            acc, loss, _p, _r, _f = self.evaluate()
            return acc, loss, [0] * self.k

        assignments: Dict[str, int] = {}
        updates:     Dict[str, Any] = {}

        for cid in sampled_ids:
            device = self.devices[cid]
            if not device.is_active or not device.local_dataset:
                continue

            k_star = self.estimate_cluster(cid)
            assignments[cid] = k_star

            result = self.local_update(cid, k_star)
            if result is not None:
                updates[cid] = result

        self.aggregate(assignments, updates)

        acc, loss, _p, _r, _f = self.evaluate()
        counts = [
            sum(1 for cid in assignments if assignments[cid] == j)
            for j in range(self.k)
        ]
        return acc, loss, counts

    def estimate_cluster(self, client_id: str) -> int:
        """
        Client evaluates all k cluster models on local data.
        Returns argmin_{j} loss(cluster_model_j, D_i).  (Loss-based, not cosine.)
        """
        device = self.devices[client_id]
        loader = device.local_dataloader
        if loader is None or len(loader) == 0:
            return 0

        criterion = nn.CrossEntropyLoss()
        best_k, best_loss = 0, float("inf")

        for k_idx, sd in enumerate(self.cluster_models):
            model = copy.deepcopy(self._template_model)
            model.load_state_dict(sd)
            model = model.to(self._dev)
            model.eval()

            total_loss = 0.0
            n_batches  = 0
            with torch.no_grad():
                for data, target in loader:
                    data, target = data.to(self._dev), target.to(self._dev)
                    out = model(data)
                    if isinstance(out, tuple):
                        out = out[0]
                    total_loss += criterion(out, target).item()
                    n_batches  += 1
                    if n_batches >= 3:   # limit eval to a few batches for speed
                        break

            avg = total_loss / max(n_batches, 1)
            if avg < best_loss:
                best_loss = avg
                best_k    = k_idx

        return best_k

    def local_update(self, client_id: str, k_star: int) -> Optional[Dict]:
        """
        Option II (model averaging): run local_steps SGD steps from cluster model k*.
        Option I  (gradient): return gradient update instead of updated model.
        Returns a dict with 'type' key.
        """
        device_obj = self.devices[client_id]
        loader = device_obj.local_dataloader
        if loader is None or len(loader) == 0:
            return None

        model = copy.deepcopy(self._template_model)
        model.load_state_dict(self.cluster_models[k_star])
        model = model.to(self._dev)

        # Weight sharing: freeze shared layers, optimise only final layer
        if self.weight_sharing:
            # Identify last FC-style layer by checking param names that contain "classifier" / "fc" / "linear"
            classifier_params = [
                p for name, p in model.named_parameters()
                if any(kw in name.lower() for kw in ("fc", "linear", "classifier", "head"))
            ]
            param_groups = classifier_params if classifier_params else list(model.parameters())
        else:
            param_groups = list(model.parameters())

        optimizer = torch.optim.SGD(param_groups, lr=self._current_lr)
        criterion = nn.CrossEntropyLoss()

        if self.variant == "gradient":
            # Option I: compute one gradient and return it
            model.train()
            try:
                data, target = next(iter(loader))
                data, target = data.to(self._dev), target.to(self._dev)
                optimizer.zero_grad()
                out = model(data)
                if isinstance(out, tuple):
                    out = out[0]
                loss = criterion(out, target)
                loss.backward()
                grad = {n: p.grad.clone().cpu() for n, p in model.named_parameters() if p.grad is not None}
                return {"type": "gradient", "gradient": grad}
            except Exception as e:
                print(f"  IFCA gradient update error for {client_id}: {e}")
                return None
        else:
            # Option II: local_steps SGD steps (model averaging)
            model.train()
            try:
                loader_iter = iter(loader)
                for _ in range(self.local_steps):
                    try:
                        data, target = next(loader_iter)
                    except StopIteration:
                        loader_iter = iter(loader)
                        data, target = next(loader_iter)
                    data, target = data.to(self._dev), target.to(self._dev)
                    optimizer.zero_grad()
                    out = model(data)
                    if isinstance(out, tuple):
                        out = out[0]
                    loss = criterion(out, target)
                    loss.backward()
                    optimizer.step()
            except Exception as e:
                print(f"  IFCA local_update error for {client_id}: {e}")
                return None

            if self._dev == "cuda":
                model = model.cpu()
            return {"type": "model", "state_dict": model.state_dict()}

    def aggregate(self, assignments: Dict[str, int], updates: Dict[str, Any]):
        """
        Server: model averaging within each cluster (Option II) or
        gradient-based update (Option I).
        Empty clusters: keep previous round's model unchanged.
        """
        if self.variant == "gradient":
            self._aggregate_gradients(assignments, updates)
        else:
            self._aggregate_models(assignments, updates)

    def _aggregate_models(self, assignments: Dict[str, int], updates: Dict[str, Any]):
        """Weighted average of locally updated models within each cluster."""
        cluster_sds:     Dict[int, List[Dict]] = defaultdict(list)
        cluster_weights: Dict[int, List[float]] = defaultdict(list)

        for cid, result in updates.items():
            if result["type"] != "model":
                continue
            k_j = assignments.get(cid, 0)
            cluster_sds[k_j].append(result["state_dict"])
            w = getattr(self.devices[cid], "dataset_size", 1) or 1
            cluster_weights[k_j].append(float(w))

        for j in range(self.k):
            sds = cluster_sds.get(j, [])
            if not sds:
                continue  # keep previous model unchanged
            ws = cluster_weights[j]
            total_w = sum(ws) or 1.0
            new_sd = {k: torch.zeros_like(v, dtype=torch.float32)
                      for k, v in sds[0].items()}
            for sd, w in zip(sds, ws):
                ww = w / total_w
                for k, v in sd.items():
                    if k in new_sd:
                        new_sd[k] += ww * v.float()
            self.cluster_models[j] = new_sd

    def _aggregate_gradients(self, assignments: Dict[str, int], updates: Dict[str, Any]):
        """Option I: gradient-based cluster model update."""
        cluster_grads:   Dict[int, List[Dict]] = defaultdict(list)
        cluster_weights: Dict[int, List[float]] = defaultdict(list)

        for cid, result in updates.items():
            if result["type"] != "gradient":
                continue
            k_j = assignments.get(cid, 0)
            cluster_grads[k_j].append(result["gradient"])
            w = getattr(self.devices[cid], "dataset_size", 1) or 1
            cluster_weights[k_j].append(float(w))

        m_total = max(len(assignments), 1)

        for j in range(self.k):
            grads = cluster_grads.get(j, [])
            if not grads:
                continue
            ws = cluster_weights[j]
            total_w = sum(ws) or 1.0
            new_sd = copy.deepcopy(self.cluster_models[j])
            for name in new_sd:
                agg_g = torch.zeros_like(new_sd[name], dtype=torch.float32)
                for grad, w in zip(grads, ws):
                    if name in grad:
                        agg_g += (w / total_w) * grad[name].float()
                new_sd[name] = new_sd[name].float() - (self._current_lr / m_total) * agg_g
            self.cluster_models[j] = new_sd

    def evaluate(self) -> Tuple[float, float, float, float, float]:
        """Evaluate by assigning each sample to its best cluster (argmin loss).
        Returns (accuracy, loss, precision, recall, f1)."""
        criterion = nn.CrossEntropyLoss(reduction="none")

        models = []
        for sd in self.cluster_models:
            m = copy.deepcopy(self._template_model)
            m.load_state_dict(sd)
            m = m.to(self._dev)
            m.eval()
            models.append(m)

        all_preds: List[torch.Tensor] = []
        all_targets_l: List[torch.Tensor] = []
        total_loss_sum = 0.0
        total_samples = 0
        num_classes = 2

        try:
            loader = self.dataset.get_global_dataloader(batch_size=64, is_train=False)
            with torch.no_grad():
                for data, target in loader:
                    data, target = data.to(self._dev), target.to(self._dev)
                    # per-sample loss for each cluster model: shape [k, B]
                    per_cluster_loss = []
                    outs = []
                    for m in models:
                        out = m(data)
                        if isinstance(out, tuple):
                            out = out[0]
                        num_classes = out.shape[1]
                        per_cluster_loss.append(criterion(out, target))
                        outs.append(out)
                    # best cluster per sample: [B]
                    stacked_loss = torch.stack(per_cluster_loss, dim=0)  # [k, B]
                    best_k = stacked_loss.argmin(dim=0)                  # [B]
                    stacked_out = torch.stack(outs, dim=0)               # [k, B, C]
                    # gather predictions from the best cluster for each sample
                    idx = best_k.view(1, -1, 1).expand(1, -1, num_classes)
                    best_out = stacked_out.gather(0, idx).squeeze(0)     # [B, C]
                    pred = best_out.argmax(1)
                    all_preds.append(pred.cpu())
                    all_targets_l.append(target.cpu())
                    total_loss_sum += stacked_loss.gather(0, best_k.unsqueeze(0)).sum().item()
                    total_samples  += target.size(0)

            preds_t   = torch.cat(all_preds)
            targets_t = torch.cat(all_targets_l)
            acc = preds_t.eq(targets_t).float().mean().item()
            avg_loss = total_loss_sum / max(total_samples, 1)
            p, r, f = _compute_prf(preds_t, targets_t, num_classes)
            return acc, avg_loss, p, r, f
        except Exception as e:
            print(f"  IFCA evaluate error: {e}")
            return 0.0, 0.0, 0.0, 0.0, 0.0

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
