"""
GeoFL: Geo-Distributed Hierarchical Federated Learning with Importance-Aware Aggregation.

Paper: "GeoFL: A Framework for Efficient Geo-Distributed Cross-Device Federated Learning"
IEEE INFOCOM 2025.

Faithful baseline implementation — do not modify algorithm logic.

Three-tier architecture: clients → aggregators (zones) → central server.
"""

import copy
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _model_to_vector(state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Flatten all parameters into a single float32 vector."""
    return torch.cat([v.detach().cpu().float().reshape(-1) for v in state_dict.values()])


def _vector_to_state_dict(vec: torch.Tensor,
                          template: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    sd = {}
    ptr = 0
    for k, v in template.items():
        n = v.numel()
        sd[k] = vec[ptr:ptr + n].reshape(v.shape).to(v.dtype)
        ptr += n
    return sd


def quantize_f16(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Float32 → Float16 quantization (simulates WAN compression)."""
    return {k: v.half() for k, v in state_dict.items()}


def dequantize_f16(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Float16 → Float32 dequantization."""
    return {k: v.float() for k, v in state_dict.items()}


def compute_importance(w_agg: Dict[str, torch.Tensor],
                       w_global: Dict[str, torch.Tensor]) -> float:
    """
    Importance score S = 0.5 * L + 0.5 * (1 - Cos_norm)

    L        = ||w_agg - w_global||₂ / ||w_global||₂
    Cos_norm = (cosine_similarity(w_agg, w_global) + 1) / 2
    """
    v_agg = _model_to_vector(w_agg)
    v_glob = _model_to_vector(w_global)

    norm_glob = v_glob.norm()
    L = (v_agg - v_glob).norm() / (norm_glob + 1e-10)

    cos = F.cosine_similarity(v_agg.unsqueeze(0), v_glob.unsqueeze(0)).item()
    cos_norm = (cos + 1.0) / 2.0  # rescale [-1,1] → [0,1]

    S = 0.5 * L.item() + 0.5 * (1.0 - cos_norm)
    return float(S)


def adaptive_threshold(S0: float, S_min: float, alpha: float,
                       B_i: float, r_i: int, R0: int) -> float:
    """
    S_i = max(S_min,  S0 × (α - (1-α)×B_i)^(r_i / R0))
    """
    base = alpha - (1.0 - alpha) * B_i
    base = max(base, 1e-6)          # avoid negative base
    return max(S_min, S0 * (base ** (r_i / max(R0, 1))))


def fedavg_aggregate(state_dicts: List[Dict[str, torch.Tensor]],
                     weights: List[float]) -> Dict[str, torch.Tensor]:
    total = sum(weights) or 1.0
    norm_w = [w / total for w in weights]
    agg = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in state_dicts[0].items()}
    for sd, w in zip(state_dicts, norm_w):
        for k in agg:
            agg[k] += w * sd[k].float()
    return agg


# ---------------------------------------------------------------------------
# GeoFL class
# ---------------------------------------------------------------------------

class GeoFL:
    """
    GeoFL: three-tier geo-distributed FL with importance-aware aggregation.

    Exposes: train_round(), local_update(), compute_importance(),
             aggregator_step(), server_aggregate(), evaluate().
    """

    def __init__(self, config, devices: Dict[str, Any],
                 global_model: nn.Module, dataset: Any,
                 zones: Optional[Dict[str, List[str]]] = None):
        self.config = config
        self.devices = devices
        self.dataset = dataset

        # Hyperparameters (paper defaults, overridable via config.geofl_*)
        self.num_rounds: int = getattr(config, "geofl_num_rounds", config.num_rounds)
        self.local_epochs: int = getattr(config, "geofl_local_epochs", 1)
        self.local_steps: int = getattr(config, "geofl_local_steps", 5)
        self.batch_size: int = getattr(config, "geofl_batch_size", 16)
        self.lr: float = getattr(config, "geofl_lr", getattr(config, "learning_rate", 0.001))
        self.S0: float = getattr(config, "geofl_S0", 0.01)
        self.S_min: float = getattr(config, "geofl_S_min", 0.001)
        self.alpha: float = getattr(config, "geofl_alpha", 0.95)
        self.R0: int = getattr(config, "geofl_R0", 5)
        self.beta: float = getattr(config, "geofl_beta", 0.2)
        self.client_sampling_rate: float = getattr(config, "geofl_client_sampling_rate", 0.7)

        # Compute device
        self._dev = "cpu"
        if getattr(config, "device", "cpu") == "cuda" and torch.cuda.is_available():
            self._dev = "cuda"

        # Build zone → device mapping
        if zones is not None:
            self.zones = zones
        else:
            self.zones = self._infer_zones()

        self.aggregator_ids = sorted(self.zones.keys())
        n_agg = len(self.aggregator_ids)

        # Simulate heterogeneous WAN bandwidths ∈ [32, 171] Mbit/s (paper range), normalised to [0,1]
        rng = np.random.RandomState(42)
        raw_bw = rng.uniform(32, 171, size=n_agg)
        self.bandwidth: Dict[str, float] = {
            agg_id: float((raw_bw[i] - 32) / (171 - 32))
            for i, agg_id in enumerate(self.aggregator_ids)
        }

        # Per-aggregator state
        self.agg_model: Dict[str, Dict] = {
            agg_id: copy.deepcopy(global_model.state_dict())
            for agg_id in self.aggregator_ids
        }
        self.agg_r: Dict[str, int] = {agg_id: 0 for agg_id in self.aggregator_ids}

        # Central server state
        self.global_sd: Dict = copy.deepcopy(global_model.state_dict())
        # G_a = rounds since last upload for each aggregator
        self.G: Dict[str, int] = {agg_id: 0 for agg_id in self.aggregator_ids}
        # Last model received from each aggregator
        self.last_uploaded: Dict[str, Optional[Dict]] = {
            agg_id: copy.deepcopy(global_model.state_dict())
            for agg_id in self.aggregator_ids
        }
        # Which aggregators have uploaded at least once
        self.ever_uploaded: set = set()

        self._template_model = global_model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        """Full training loop."""
        accuracies, losses = [], []
        upload_counts = []
        t_start = time.time()

        try:
            for t in range(1, self.num_rounds + 1):
                t_round = time.time()
                acc, loss, uploads = self.train_round(t)
                accuracies.append(acc)
                losses.append(loss)
                upload_counts.append(uploads)
                print(
                    f"GeoFL Round {t}/{self.num_rounds}: "
                    f"Acc={acc:.4f}  Loss={loss:.4f}  "
                    f"Uploads={uploads}/{len(self.aggregator_ids)}  "
                    f"Time={time.time()-t_round:.2f}s"
                )
        except KeyboardInterrupt:
            print("GeoFL training interrupted.")

        return {
            "method": "GeoFL",
            "final_accuracy": accuracies[-1] if accuracies else 0.0,
            "final_loss": losses[-1] if losses else float("inf"),
            "accuracies": accuracies,
            "losses": losses,
            "upload_counts": upload_counts,
            "total_time": time.time() - t_start,
        }

    def train_round(self, t: int) -> Tuple[float, float, int]:
        """Execute one global round. Returns (accuracy, loss, num_uploads)."""
        uploads_this_round = 0

        for agg_id in self.aggregator_ids:
            # Phase 1 + 2: client training + intra-zone aggregation
            w_agg = self.aggregator_step(agg_id)
            if w_agg is None:
                continue

            # Importance + upload decision
            S_current = compute_importance(w_agg, self.agg_model[agg_id])
            S_thresh = adaptive_threshold(
                self.S0, self.S_min, self.alpha,
                self.bandwidth[agg_id], self.agg_r[agg_id], self.R0
            )

            upload = (S_current >= S_thresh) or (self.agg_r[agg_id] >= self.R0)

            if upload:
                # Quantize → upload → dequantize on server
                delta_f16 = quantize_f16({
                    k: (w_agg[k].float() - self.agg_model[agg_id][k].float())
                    for k in w_agg
                })
                w_upload = dequantize_f16(delta_f16)
                # Reconstruct full model from delta
                w_full = {k: self.agg_model[agg_id][k].float() + w_upload[k]
                          for k in w_upload}

                # Server aggregation → send back new global to this aggregator
                new_w = self.server_aggregate(agg_id, w_full)
                self.agg_model[agg_id] = new_w
                self.agg_r[agg_id] = 0
                uploads_this_round += 1
            else:
                # Buffer: update local model but don't upload
                self.agg_model[agg_id] = w_agg
                self.agg_r[agg_id] += 1

        acc, loss = self.evaluate()
        return acc, loss, uploads_this_round

    def local_update(self, client_id: str, w_init: Dict[str, torch.Tensor]) -> Optional[Dict]:
        """
        Client-side: LocalSGD for `local_steps` steps (paper: 5 steps, batch=16).
        Returns updated state_dict or None on failure.
        """
        device = self.devices[client_id]
        dataloader = device.local_dataloader
        if dataloader is None or len(dataloader) == 0:
            return None

        model = copy.deepcopy(self._template_model)
        model.load_state_dict(w_init)
        model = model.to(self._dev)
        model.train()

        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr, momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        steps = 0
        try:
            done = False
            for _ in range(self.local_epochs):
                if done:
                    break
                for data, target in dataloader:
                    if steps >= self.local_steps:
                        done = True
                        break
                    data, target = data.to(self._dev), target.to(self._dev)
                    optimizer.zero_grad()
                    out = model(data)
                    if isinstance(out, tuple):
                        out = out[0]
                    criterion(out, target).backward()
                    optimizer.step()
                    steps += 1
        except Exception as e:
            print(f"  GeoFL local_update error ({client_id}): {e}")
            return None

        if self._dev == "cuda":
            model = model.cpu()
        return model.state_dict()

    def compute_importance(self, w_agg: Dict, w_global: Dict) -> float:
        """Public wrapper around module-level compute_importance."""
        return compute_importance(w_agg, w_global)

    def aggregator_step(self, agg_id: str) -> Optional[Dict]:
        """
        Phase 1+2: population-proportional client selection, local training,
        intra-zone FedAvg aggregation. Returns aggregated zone model.
        """
        zone_clients = self.zones.get(agg_id, [])
        active = [c for c in zone_clients
                  if self.devices[c].is_active and self.devices[c].local_dataset]
        if not active:
            return None

        # Population-proportional selection
        n_select = max(1, int(self.client_sampling_rate * len(active)))
        selected = list(np.random.choice(active, size=n_select, replace=False))

        w_current = self.agg_model[agg_id]
        local_sds, local_sizes = [], []

        for cid in selected:
            result = self.local_update(cid, w_current)
            if result is not None:
                local_sds.append(result)
                local_sizes.append(self.devices[cid].dataset_size or 1)

        if not local_sds:
            return None

        return fedavg_aggregate(local_sds, local_sizes)

    def server_aggregate(self, uploading_agg: str,
                         w_uploaded: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Central server staleness-weighted aggregation (Eq. 3 from paper).

        A* = all aggregators whose models have been aggregated since
             uploading_agg's last upload.
        c_i = (G_i + 1)^{-β} / Σ_{k ∈ A*} (G_k + 1)^{-β}
        M_a = Σ_{i ∈ A*} c_i * M_i
        """
        # Store incoming upload
        self.last_uploaded[uploading_agg] = copy.deepcopy(w_uploaded)
        self.ever_uploaded.add(uploading_agg)

        # A* = aggregators that have uploaded at least once
        A_star = list(self.ever_uploaded)
        if not A_star:
            return copy.deepcopy(w_uploaded)

        # Staleness weights
        raw_w = {agg_id: (self.G[agg_id] + 1) ** (-self.beta)
                 for agg_id in A_star}
        total = sum(raw_w.values()) or 1.0
        norm_w = {agg_id: raw_w[agg_id] / total for agg_id in A_star}

        # Weighted aggregation of last-uploaded models
        result = {k: torch.zeros_like(v, dtype=torch.float32)
                  for k, v in self.last_uploaded[A_star[0]].items()}
        for agg_id in A_star:
            w = norm_w[agg_id]
            for k in result:
                result[k] += w * self.last_uploaded[agg_id][k].float()

        # Update staleness counters: reset uploader, increment others
        for agg_id in A_star:
            if agg_id == uploading_agg:
                self.G[agg_id] = 0
            else:
                self.G[agg_id] += 1

        self.global_sd = copy.deepcopy(result)
        return result

    def evaluate(self) -> Tuple[float, float]:
        """Evaluate using the current global model on the test set."""
        # Use the aggregated global_sd (or first aggregator model as fallback)
        sd = self.global_sd

        model = copy.deepcopy(self._template_model)
        model.load_state_dict(sd)
        model = model.to(self._dev)
        model.eval()

        criterion = nn.CrossEntropyLoss()
        try:
            loader = self.dataset.get_global_dataloader(batch_size=64, is_train=False)
            total_loss, correct, total = 0.0, 0, 0
            with torch.no_grad():
                for data, target in loader:
                    data, target = data.to(self._dev), target.to(self._dev)
                    out = model(data)
                    if isinstance(out, tuple):
                        out = out[0]
                    total_loss += criterion(out, target).item()
                    correct += out.argmax(1).eq(target).sum().item()
                    total += target.size(0)
            acc = correct / max(total, 1)
            avg_loss = total_loss / max(len(loader), 1)
        except Exception as e:
            print(f"  GeoFL evaluate error: {e}")
            acc, avg_loss = 0.0, float("inf")

        model.train()
        return acc, avg_loss

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _infer_zones(self) -> Dict[str, List[str]]:
        """
        Build zone mapping from device zone attributes, or fall back to
        splitting devices into equal groups if zone info is unavailable.
        """
        zone_map: Dict[str, List[str]] = defaultdict(list)

        for dev_id, dev in self.devices.items():
            zone_id = getattr(dev, "zone_id", None)
            if zone_id is not None:
                zone_map[str(zone_id)].append(dev_id)

        if zone_map:
            return dict(zone_map)

        # Fallback: split into num_zones equal groups
        num_zones = getattr(self.config, "num_zones", 4)
        dev_ids = sorted(self.devices.keys())
        chunk = max(1, len(dev_ids) // num_zones)
        zones = {}
        for i in range(num_zones):
            start = i * chunk
            end = start + chunk if i < num_zones - 1 else len(dev_ids)
            if start < len(dev_ids):
                zones[f"zone_{i}"] = dev_ids[start:end]
        return zones
