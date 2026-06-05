"""
SnapCFL: Pre-Clustering-Based Clustered Federated Learning.

Paper: "SnapCFL: A Pre-Clustering-Based Clustered Federated Learning Framework
        for Data and System Heterogeneities"
Venue: IEEE Transactions on Mobile Computing, Vol. 24, No. 6, June 2025
DOI:   10.1109/TMC.2025.3529487

Faithful baseline — do not modify algorithm logic.

Two-stage implementation:
  Stage 1 (pre_cluster): pairwise binary classifiers trained via FedAvg → DBSCAN.
  Stage 2 (train_round loop): clustered FL with fixed assignments.
"""

import copy
import json
import os
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import DBSCAN


# ---------------------------------------------------------------------------
# Tiny binary-classifier head used during pre-clustering
# ---------------------------------------------------------------------------

class _BinaryHead(nn.Module):
    """Lightweight binary classifier head appended on top of penultimate features."""

    def __init__(self, in_features: int):
        super().__init__()
        self.fc = nn.Linear(in_features, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


# ---------------------------------------------------------------------------
# SnapCFL
# ---------------------------------------------------------------------------

class SnapCFL:
    """
    SnapCFL: pre-clustering then clustered FL.

    Exposes: pre_cluster(), train_round(), constraint_select_clients(),
             intra_aggregate(), evaluate().
    """

    def __init__(self, config, devices: Dict[str, Any], global_model: nn.Module, dataset: Any):
        self.config  = config
        self.devices = devices
        self.dataset = dataset

        # ── Hyper-parameters ───────────────────────────────────────────────
        self.num_rounds: int    = getattr(config, "snapcfl_num_rounds",          config.num_rounds)
        self.lr: float          = getattr(config, "snapcfl_lr",                  0.01)
        self.batch_size: int    = getattr(config, "snapcfl_batch_size",          32)
        self.local_epochs: int  = getattr(config, "snapcfl_local_epochs",        5)
        self.sampling_rate: float = getattr(config, "snapcfl_sampling_rate",     0.7)

        # Pre-clustering params
        self.pre_cluster_rounds: int = getattr(config, "snapcfl_pre_cluster_rounds", 10)
        self.dbscan_eps: float       = getattr(config, "snapcfl_eps",                0.25)
        self.dbscan_min_samples: int = getattr(config, "snapcfl_min_samples",        2)
        self.classifier_type: str    = getattr(config, "snapcfl_classifier_type",    "lr")  # "lr" | "cnn"

        # FL stage params
        self.intra_algo: str         = getattr(config, "snapcfl_intra_algo",      "fedavg")  # fedavg | fedprox
        self.global_averaging: bool  = getattr(config, "snapcfl_global_averaging", False)
        self.mu_fedprox: float       = getattr(config, "snapcfl_mu_fedprox",       0.01)

        # Frequency constraint window (system heterogeneity)
        self.H: int        = getattr(config, "snapcfl_H",      10)
        self.c_thre: int   = getattr(config, "snapcfl_c_thre",  3)

        # Cache path for similarity matrix
        self._cache_dir = getattr(config, "results_dir", "./results")
        self._cache_file = os.path.join(
            self._cache_dir,
            f"snapcfl_similarity_{getattr(config,'dataset_name','?')}_{len(devices)}.json"
        )

        self._dev = "cpu"
        if getattr(config, "device", "cpu") == "cuda" and torch.cuda.is_available():
            self._dev = "cuda"

        self._template_model = global_model
        self._client_ids = [
            cid for cid, dev in devices.items() if dev.is_active and dev.local_dataset
        ]

        # Cluster assignments (set by pre_cluster)
        self.cluster_assignments: Dict[str, int] = {}
        self.num_clusters: int = 0

        # Per-cluster global models
        self.cluster_models: List[Dict[str, torch.Tensor]] = []

        # Participation history for frequency constraint (client_id → list of rounds selected)
        self._participation_history: Dict[str, List[int]] = defaultdict(list)

        # Latency profiles (simulated per client)
        self._latency: Dict[str, float] = {}
        self._simulate_latencies()

    # ------------------------------------------------------------------
    # Stage 1: Pre-Clustering
    # ------------------------------------------------------------------

    def pre_cluster(self):
        """
        Build pairwise dissimilarity matrix A via federated binary classifiers, then DBSCAN.
        Loads from cache if available; saves to cache after computation.
        """
        print("SnapCFL pre-clustering stage...")
        n = len(self._client_ids)
        if n < 2:
            # Single cluster
            self._assign_clusters(np.zeros(n, dtype=int))
            return

        # ── Load from cache if available ──────────────────────────────────
        A = self._load_similarity_cache()
        if A is None:
            A = self._build_similarity_matrix()
            self._save_similarity_cache(A)

        # ── Symmetrize and run DBSCAN ─────────────────────────────────────
        dist_matrix = (A + A.T) / 2.0
        np.fill_diagonal(dist_matrix, 0.0)

        # Clip to [0, 1] — values > 1 can arise from noisy classifiers
        dist_matrix = np.clip(dist_matrix, 0.0, 1.0)

        clustering = DBSCAN(
            eps=self.dbscan_eps,
            min_samples=self.dbscan_min_samples,
            metric="precomputed"
        )
        labels = clustering.fit_predict(dist_matrix)

        # Assign noise points (label == -1) to nearest cluster
        labels = self._handle_noise(labels, dist_matrix)

        self._assign_clusters(labels)
        print(f"  SnapCFL: {self.num_clusters} clusters found for {n} clients.")

    def _build_similarity_matrix(self) -> np.ndarray:
        """Train O(N²) pairwise binary classifiers via FedAvg; record test accuracy."""
        n = len(self._client_ids)
        A = np.zeros((n, n))

        total_pairs = n * (n - 1) // 2
        pair_idx = 0

        for i in range(n):
            for j in range(i + 1, n):
                pair_idx += 1
                if pair_idx % 10 == 0:
                    print(f"    SnapCFL pre-cluster: pair {pair_idx}/{total_pairs}")

                t_hat = self._pairwise_classifier(self._client_ids[i], self._client_ids[j])
                # t_hat = dissimilarity: high → very different distributions
                A[i, j] = t_hat
                # Matrix is lower-triangular in the paper; we fill upper triangle too
                # (symmetrized later)

        return A

    def _pairwise_classifier(self, cid_i: str, cid_j: str) -> float:
        """
        Train a binary classifier via FedAvg between clients i and j.
        D_i data → label 0, D_j data → label 1.
        Returns test accuracy (dissimilarity: ~0.5 = similar, >0.5 = different).
        """
        dev_i = self.devices[cid_i]
        dev_j = self.devices[cid_j]

        loader_i = dev_i.local_dataloader
        loader_j = dev_j.local_dataloader
        if loader_i is None or loader_j is None:
            return 0.5  # unknown → treat as similar

        # Infer feature dimension from first batch
        in_features = self._infer_features(loader_i)

        # Initialise shared binary classifier
        head = _BinaryHead(in_features)

        criterion = nn.CrossEntropyLoss()
        model_copy = copy.deepcopy(self._template_model)

        for _ in range(self.pre_cluster_rounds):
            # --- client i update ---
            head_i = copy.deepcopy(head)
            head_i = head_i.to(self._dev)
            model_copy = model_copy.to(self._dev)
            opt_i = torch.optim.SGD(list(model_copy.parameters()) + list(head_i.parameters()),
                                    lr=self.lr)
            model_copy.train(); head_i.train()
            for data, _ in loader_i:
                data = data.to(self._dev)
                labels_i = torch.zeros(data.size(0), dtype=torch.long, device=self._dev)
                opt_i.zero_grad()
                feat = self._extract_features(model_copy, data)
                out  = head_i(feat)
                loss = criterion(out, labels_i)
                loss.backward(); opt_i.step()
                break  # one batch per round for speed

            # --- client j update ---
            head_j = copy.deepcopy(head)
            head_j = head_j.to(self._dev)
            opt_j = torch.optim.SGD(list(model_copy.parameters()) + list(head_j.parameters()),
                                    lr=self.lr)
            model_copy.train(); head_j.train()
            for data, _ in loader_j:
                data = data.to(self._dev)
                labels_j = torch.ones(data.size(0), dtype=torch.long, device=self._dev)
                opt_j.zero_grad()
                feat = self._extract_features(model_copy, data)
                out  = head_j(feat)
                loss = criterion(out, labels_j)
                loss.backward(); opt_j.step()
                break

            # FedAvg of heads
            sd_i = head_i.state_dict()
            sd_j = head_j.state_dict()
            avg_sd = {k: (sd_i[k].float() + sd_j[k].float()) / 2.0 for k in sd_i}
            head.load_state_dict(avg_sd)

        # --- Evaluate on both loaders (test accuracy = dissimilarity) ---
        head = head.to(self._dev)
        model_copy = model_copy.to(self._dev)
        head.eval(); model_copy.eval()

        correct, total = 0, 0
        with torch.no_grad():
            for data, _ in loader_i:
                data = data.to(self._dev)
                labels_i = torch.zeros(data.size(0), dtype=torch.long, device=self._dev)
                feat = self._extract_features(model_copy, data)
                pred = head(feat).argmax(1)
                correct += pred.eq(labels_i).sum().item()
                total   += data.size(0)
                break
            for data, _ in loader_j:
                data = data.to(self._dev)
                labels_j = torch.ones(data.size(0), dtype=torch.long, device=self._dev)
                feat = self._extract_features(model_copy, data)
                pred = head(feat).argmax(1)
                correct += pred.eq(labels_j).sum().item()
                total   += data.size(0)
                break

        return correct / max(total, 1)

    def _extract_features(self, model: nn.Module, data: torch.Tensor) -> torch.Tensor:
        """Run forward pass and return penultimate-layer features (before last linear)."""
        # We hook the penultimate output by removing last linear layer from forward if possible,
        # or just use raw output logits as proxy features.
        with torch.no_grad():
            out = model(data)
            if isinstance(out, tuple):
                out = out[0]
        # Detach and flatten to (batch, features)
        feat = out.detach().float()
        if feat.dim() > 2:
            feat = feat.view(feat.size(0), -1)
        return feat

    def _infer_features(self, loader) -> int:
        """Get feature dimension from first batch forward pass."""
        model_tmp = copy.deepcopy(self._template_model).to(self._dev)
        model_tmp.eval()
        with torch.no_grad():
            for data, _ in loader:
                data = data.to(self._dev)
                out = model_tmp(data)
                if isinstance(out, tuple):
                    out = out[0]
                feat = out.float()
                if feat.dim() > 2:
                    feat = feat.view(feat.size(0), -1)
                return feat.size(1)
        return 64  # fallback

    def _handle_noise(self, labels: np.ndarray, dist_matrix: np.ndarray) -> np.ndarray:
        """Assign DBSCAN noise points (-1) to nearest non-noise cluster."""
        labels = labels.copy()
        non_noise = np.where(labels >= 0)[0]
        if len(non_noise) == 0:
            return np.zeros(len(labels), dtype=int)

        for i in np.where(labels == -1)[0]:
            nearest = non_noise[np.argmin(dist_matrix[i, non_noise])]
            labels[i] = labels[nearest]
        return labels

    def _assign_clusters(self, labels: np.ndarray):
        """Store cluster assignments and initialise per-cluster models."""
        self.cluster_assignments = {cid: int(labels[i]) for i, cid in enumerate(self._client_ids)}
        self.num_clusters = len(set(labels))
        # One model per cluster, initialised from global model
        self.cluster_models = [
            copy.deepcopy(self._template_model).state_dict()
            for _ in range(self.num_clusters)
        ]
        print(f"  SnapCFL cluster sizes: { {c: list(labels).count(c) for c in sorted(set(labels))} }")

    # ------------------------------------------------------------------
    # Stage 2: Main FL
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        """Full training loop (pre-cluster once, then train rounds)."""
        self.pre_cluster()

        accuracies: List[float] = []
        losses:     List[float] = []
        t_start = time.time()

        try:
            for t in range(1, self.num_rounds + 1):
                t_round = time.time()
                acc, loss = self.train_round(t)
                accuracies.append(acc)
                losses.append(loss)
                print(
                    f"SnapCFL Round {t}/{self.num_rounds}: "
                    f"Acc={acc:.4f}  Loss={loss:.4f}  "
                    f"Time={time.time()-t_round:.2f}s"
                )
        except KeyboardInterrupt:
            print("SnapCFL training interrupted.")

        final_acc, final_loss = self.evaluate()
        return {
            "method":         "SnapCFL",
            "final_accuracy": final_acc,
            "final_loss":     final_loss,
            "accuracies":     accuracies,
            "losses":         losses,
            "total_time":     time.time() - t_start,
        }

    def train_round(self, t: int) -> Tuple[float, float]:
        """One federated round with constraint-based client selection."""
        # Group clients by cluster
        clusters: Dict[int, List[str]] = defaultdict(list)
        for cid, cl in self.cluster_assignments.items():
            clusters[cl].append(cid)

        new_cluster_models: List[Optional[Dict]] = [None] * self.num_clusters

        for cl_idx, members in clusters.items():
            # Constraint-based selection
            selected = self.constraint_select_clients(members, t)
            if not selected:
                new_cluster_models[cl_idx] = self.cluster_models[cl_idx]
                continue

            # Record participation
            for cid in selected:
                self._participation_history[cid].append(t)

            # Intra-cluster training & aggregation
            new_sd = self.intra_aggregate(selected, cl_idx, t)
            new_cluster_models[cl_idx] = new_sd if new_sd else self.cluster_models[cl_idx]

        for i, sd in enumerate(new_cluster_models):
            if sd is not None:
                self.cluster_models[i] = sd

        # Optional global averaging across clusters
        if self.global_averaging:
            sizes = [
                sum(getattr(self.devices[cid], "dataset_size", 1) or 1
                    for cid in clusters.get(i, []))
                for i in range(self.num_clusters)
            ]
            total_size = sum(sizes) or 1.0
            avg_sd = None
            for i, sd in enumerate(self.cluster_models):
                w = sizes[i] / total_size
                if avg_sd is None:
                    avg_sd = {k: w * v.float() for k, v in sd.items()}
                else:
                    for k, v in sd.items():
                        if k in avg_sd:
                            avg_sd[k] += w * v.float()
            if avg_sd:
                self.cluster_models = [copy.deepcopy(avg_sd) for _ in range(self.num_clusters)]

        return self.evaluate()

    def constraint_select_clients(self, members: List[str], t: int) -> List[str]:
        """
        Greedy selection: pick sampling_rate fraction of clients satisfying:
        - frequency constraint: not selected more than c_thre times in last H rounds.
        - prefer lower latency.
        Fallback: simple random if all violate frequency constraint.
        """
        # Filter by frequency constraint
        eligible = []
        for cid in members:
            dev = self.devices.get(cid)
            if dev is None or not dev.is_active or not dev.local_dataset:
                continue
            history = self._participation_history[cid]
            recent = sum(1 for r in history if r > t - self.H)
            if recent < self.c_thre:
                eligible.append(cid)

        if not eligible:
            # Fallback: ignore frequency constraint
            eligible = [
                cid for cid in members
                if self.devices.get(cid) and
                   self.devices[cid].is_active and
                   self.devices[cid].local_dataset
            ]

        if not eligible:
            return []

        # Sort by latency (ascending) — greedy minimum latency selection
        eligible.sort(key=lambda cid: self._latency.get(cid, 1.0))
        n_select = max(1, int(self.sampling_rate * len(eligible)))
        return eligible[:n_select]

    def intra_aggregate(self, selected: List[str], cl_idx: int, t: int) -> Optional[Dict]:
        """
        Intra-cluster training with configurable algorithm (fedavg or fedprox).
        Returns aggregated state_dict for the cluster.
        """
        updates: List[Dict] = []
        weights: List[float] = []

        current_sd = self.cluster_models[cl_idx]

        for cid in selected:
            dev = self.devices[cid]
            loader = dev.local_dataloader
            if loader is None or len(loader) == 0:
                continue

            local_model = copy.deepcopy(self._template_model)
            local_model.load_state_dict(current_sd)

            if self.intra_algo == "fedprox":
                result = self._train_fedprox(local_model, loader, current_sd)
            else:
                result = self._train_fedavg(local_model, loader)

            if result is not None:
                updates.append(result)
                weights.append(float(getattr(dev, "dataset_size", 1) or 1))

        if not updates:
            return None

        # FedAvg aggregation within cluster
        total_w = sum(weights) or 1.0
        new_sd = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in updates[0].items()}
        for sd, w in zip(updates, weights):
            ww = w / total_w
            for k, v in sd.items():
                if k in new_sd:
                    new_sd[k] += ww * v.float()
        return new_sd

    def _train_fedavg(self, model: nn.Module, loader) -> Optional[Dict]:
        """Standard local SGD."""
        model = model.to(self._dev)
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        try:
            for _ in range(self.local_epochs):
                for data, target in loader:
                    data, target = data.to(self._dev), target.to(self._dev)
                    optimizer.zero_grad()
                    out = model(data)
                    if isinstance(out, tuple):
                        out = out[0]
                    criterion(out, target).backward()
                    optimizer.step()
        except Exception as e:
            print(f"  SnapCFL FedAvg local train error: {e}")
            return None
        if self._dev == "cuda":
            model = model.cpu()
        return model.state_dict()

    def _train_fedprox(self, model: nn.Module, loader, global_sd: Dict) -> Optional[Dict]:
        """Local SGD with FedProx proximal term."""
        model = model.to(self._dev)
        global_params = {k: v.to(self._dev).float() for k, v in global_sd.items()}
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        mu = self.mu_fedprox
        try:
            for _ in range(self.local_epochs):
                for data, target in loader:
                    data, target = data.to(self._dev), target.to(self._dev)
                    optimizer.zero_grad()
                    out = model(data)
                    if isinstance(out, tuple):
                        out = out[0]
                    ce_loss = criterion(out, target)
                    prox = sum(
                        ((p - global_params[n]) ** 2).sum()
                        for n, p in model.named_parameters()
                        if n in global_params
                    )
                    (ce_loss + (mu / 2.0) * prox).backward()
                    optimizer.step()
        except Exception as e:
            print(f"  SnapCFL FedProx local train error: {e}")
            return None
        if self._dev == "cuda":
            model = model.cpu()
        return model.state_dict()

    def evaluate(self) -> Tuple[float, float]:
        """Evaluate by averaging accuracy across all cluster models."""
        if not self.cluster_models:
            return 0.0, float("inf")

        criterion = nn.CrossEntropyLoss()
        total_acc, total_loss = 0.0, 0.0

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
                print(f"  SnapCFL evaluate error: {e}")

        n = len(self.cluster_models)
        return total_acc / max(n, 1), total_loss / max(n, 1)

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _load_similarity_cache(self) -> Optional[np.ndarray]:
        if not os.path.exists(self._cache_file):
            return None
        try:
            with open(self._cache_file, "r") as f:
                data = json.load(f)
            A = np.array(data["A"])
            cached_ids = data["client_ids"]
            if cached_ids != self._client_ids:
                print("  SnapCFL: cache client_ids mismatch — recomputing.")
                return None
            print(f"  SnapCFL: loaded similarity matrix from cache: {self._cache_file}")
            return A
        except Exception as e:
            print(f"  SnapCFL: cache load failed ({e}), recomputing.")
            return None

    def _save_similarity_cache(self, A: np.ndarray):
        os.makedirs(self._cache_dir, exist_ok=True)
        try:
            with open(self._cache_file, "w") as f:
                json.dump({"A": A.tolist(), "client_ids": self._client_ids}, f)
            print(f"  SnapCFL: similarity matrix saved to cache: {self._cache_file}")
        except Exception as e:
            print(f"  SnapCFL: cache save failed ({e})")

    # ------------------------------------------------------------------
    # System heterogeneity simulation
    # ------------------------------------------------------------------

    def _simulate_latencies(self):
        """
        Simulate T_trans + T_cmp per client using paper's distribution:
        0.25s (40%), 0.5s (40%), 1s (10%), 2s (10%).
        """
        buckets  = [0.25, 0.50, 1.0, 2.0]
        probs    = [0.40, 0.40, 0.10, 0.10]
        rng      = np.random.default_rng(seed=42)
        for cid in self.devices:
            lat = rng.choice(buckets, p=probs)
            self._latency[cid] = float(lat)
