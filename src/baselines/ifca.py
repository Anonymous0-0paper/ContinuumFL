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
        self.lr: float           = getattr(config, "ifca_lr",              0.01)
        self.lr_decay: float     = getattr(config, "ifca_lr_decay",        1.0)   # per-round multiplier
        self.batch_size: int     = getattr(config, "ifca_batch_size",      32)
        self.sampling_rate: float = getattr(config, "ifca_sampling_rate",  0.7)
        self.variant: str        = getattr(config, "ifca_variant",         "model")   # "model" | "gradient"
        self.weight_sharing: bool = getattr(config, "ifca_weight_sharing", False)

        self._dev = "cpu"
        if getattr(config, "device", "cpu") == "cuda" and torch.cuda.is_available():
            self._dev = "cuda"

        self._template_model = global_model

        # Initialise k cluster models with independent random seeds
        self.cluster_models: List[Dict[str, torch.Tensor]] = []
        for i in range(self.k):
            m = copy.deepcopy(global_model)
            for p in m.parameters():
                nn.init.normal_(p, mean=0.0, std=0.02)
            self.cluster_models.append(m.state_dict())

        self._current_lr = self.lr

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        """Full training loop."""
        accuracies: List[float] = []
        losses:     List[float] = []
        cluster_counts: List[List[int]] = []   # clients per cluster per round
        t_start = time.time()

        try:
            for t in range(1, self.num_rounds + 1):
                t_round = time.time()
                acc, loss, counts = self.train_round(t)
                accuracies.append(acc)
                losses.append(loss)
                cluster_counts.append(counts)

                # LR decay
                self._current_lr *= self.lr_decay

                print(
                    f"IFCA Round {t}/{self.num_rounds}: "
                    f"Acc={acc:.4f}  Loss={loss:.4f}  "
                    f"Clusters={counts}  Time={time.time()-t_round:.2f}s"
                )
        except KeyboardInterrupt:
            print("IFCA training interrupted.")

        # Evaluate final model (cluster 0 as representative)
        final_acc, final_loss = self.evaluate()

        return {
            "method":        "IFCA",
            "final_accuracy": final_acc,
            "final_loss":     final_loss,
            "accuracies":     accuracies,
            "losses":         losses,
            "total_time":     time.time() - t_start,
        }

    def train_round(self, t: int) -> Tuple[float, float, List[int]]:
        """
        One federated round.
        Returns (accuracy, loss, list_of_client_counts_per_cluster).
        """
        sampled_ids = self._sample_clients()
        if not sampled_ids:
            acc, loss = self.evaluate()
            return acc, loss, [0] * self.k

        # Per-client results: (cluster_assignment, updated_state_dict_or_gradient)
        assignments: Dict[str, int]  = {}
        updates:     Dict[str, Any]  = {}

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

        acc, loss = self.evaluate()
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

    def evaluate(self) -> Tuple[float, float]:
        """Evaluate by averaging accuracy across all k cluster models."""
        total_acc, total_loss = 0.0, 0.0
        criterion = nn.CrossEntropyLoss()

        for sd in self.cluster_models:
            model = copy.deepcopy(self._template_model)
            model.load_state_dict(sd)
            model = model.to(self._dev)
            model.eval()

            try:
                loader = self.dataset.get_global_dataloader(batch_size=64, is_train=False)
                correct, total, loss_sum = 0, 0, 0.0
                with torch.no_grad():
                    for data, target in loader:
                        data, target = data.to(self._dev), target.to(self._dev)
                        out = model(data)
                        if isinstance(out, tuple):
                            out = out[0]
                        loss_sum += criterion(out, target).item()
                        correct  += out.argmax(1).eq(target).sum().item()
                        total    += target.size(0)
                total_acc  += correct / max(total, 1)
                total_loss += loss_sum / max(len(loader), 1)
            except Exception as e:
                print(f"  IFCA evaluate error: {e}")

        n = len(self.cluster_models)
        return total_acc / max(n, 1), total_loss / max(n, 1)

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
