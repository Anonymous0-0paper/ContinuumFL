"""
Baseline federated learning methods for comparison with ContinuumFL.
Implements FedAvg, FedProx, HierFL, and ClusterFL.
"""

import csv
import os
import re
import torch
import torch.nn as nn
import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from sklearn.cluster import KMeans
import copy
import torch.nn.functional as _F
from .apcfl import APCFL
from .geofl import GeoFL
from .ifca import IFCA
from .snapcfl import SnapCFL

class BaselineFLMethods:
    """
    Implements baseline federated learning methods for comparison.
    """
    
    def __init__(self, config):
        self.config = config
        self.baseline_results = {}

    # ------------------------------------------------------------------
    # CSV persistence helpers
    # ------------------------------------------------------------------

    def _results_dir(self) -> str:
        """Return (and create) the run-level results directory."""
        base = getattr(self.config, "results_dir", "./results")
        dataset  = getattr(self.config, "dataset_name",   "unknown")
        intra    = getattr(self.config, "intra_zone_alpha", "?")
        inter    = getattr(self.config, "inter_zone_alpha", "?")
        comp     = getattr(self.config, "compression_rate", 0)
        devices  = getattr(self.config, "num_devices",      "?")
        zones    = getattr(self.config, "num_zones",        "?")
        comp_pct = int(round(float(comp) * 100))
        run_name = (f"{dataset}__intra{intra}__inter{inter}"
                    f"__comp{comp_pct}pct__dev{devices}__zones{zones}")
        path = os.path.join(base, run_name)
        os.makedirs(path, exist_ok=True)
        return path

    @staticmethod
    def _slug(hparams: Dict[str, Any]) -> str:
        """Turn a hyperparameter dict into a short filesystem-safe string."""
        parts = []
        for k, v in sorted(hparams.items()):
            # shorten key: remove common prefixes, keep last word
            short_k = re.sub(r'^[a-z]+_', '', k)
            parts.append(f"{short_k}{v}")
        slug = "_".join(parts)
        # replace characters that are unsafe in filenames
        return re.sub(r'[^A-Za-z0-9._-]', '', slug)[:120]

    def save_baseline_results(self, result: Dict[str, Any], hparams: Dict[str, Any]):
        """
        Persist baseline results to CSV files inside the run results directory.

        Creates two files:
          {results_dir}/baselines/{METHOD}_{hparam-slug}/metrics.csv   — one row per round
          {results_dir}/baselines/{METHOD}_{hparam-slug}/summary.csv   — single summary row
        """
        method   = result.get("method", "unknown")
        slug     = self._slug(hparams)
        out_dir  = os.path.join(self._results_dir(), "baselines", f"{method}_{slug}")
        os.makedirs(out_dir, exist_ok=True)

        accuracies    = result.get("accuracies",    [])
        losses        = result.get("losses",        [])
        extra_keys    = [k for k in result
                         if k not in ("method", "accuracies", "losses",
                                      "final_accuracy", "final_loss",
                                      "total_time", "convergence_round")]

        # ── per-round metrics.csv ─────────────────────────────────────────
        metrics_path = os.path.join(out_dir, "metrics.csv")
        metrics_fields = ["round", "accuracy", "loss"] + extra_keys
        with open(metrics_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=metrics_fields, extrasaction="ignore")
            writer.writeheader()
            for r, (acc, loss) in enumerate(zip(accuracies, losses), start=1):
                row: Dict[str, Any] = {"round": r, "accuracy": acc, "loss": loss}
                # per-round lists stored in extra fields (e.g. cluster_counts, upload_counts)
                for k in extra_keys:
                    v = result.get(k)
                    if isinstance(v, list) and len(v) == len(accuracies):
                        row[k] = v[r - 1]
                    else:
                        row[k] = ""
                writer.writerow(row)

        # ── summary.csv ──────────────────────────────────────────────────
        summary_path = os.path.join(out_dir, "summary.csv")
        # Build ordered column list: shared config cols, then hparams, then results
        shared_fields = [
            "method", "dataset", "num_rounds", "num_devices", "num_zones",
            "intra_zone_alpha", "inter_zone_alpha", "compression_rate",
        ]
        hparam_fields = [f"hp_{k}" for k in sorted(hparams.keys())]
        result_fields = [
            "final_accuracy", "final_loss", "best_accuracy", "best_accuracy_round",
            "total_time", "convergence_round",
        ]
        all_fields = shared_fields + hparam_fields + result_fields

        best_acc   = max(accuracies) if accuracies else 0.0
        best_round = (accuracies.index(best_acc) + 1) if accuracies else 0

        row = {
            "method":           method,
            "dataset":          getattr(self.config, "dataset_name",    ""),
            "num_rounds":       getattr(self.config, "num_rounds",       ""),
            "num_devices":      getattr(self.config, "num_devices",      ""),
            "num_zones":        getattr(self.config, "num_zones",        ""),
            "intra_zone_alpha": getattr(self.config, "intra_zone_alpha", ""),
            "inter_zone_alpha": getattr(self.config, "inter_zone_alpha", ""),
            "compression_rate": getattr(self.config, "compression_rate", ""),
            "final_accuracy":   result.get("final_accuracy", ""),
            "final_loss":       result.get("final_loss",     ""),
            "best_accuracy":    best_acc,
            "best_accuracy_round": best_round,
            "total_time":       result.get("total_time",       ""),
            "convergence_round": result.get("convergence_round", ""),
        }
        for k, v in hparams.items():
            row[f"hp_{k}"] = v

        write_header = not os.path.exists(summary_path)
        with open(summary_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_fields, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        print(f"  [{method}] metrics → {metrics_path}")
        print(f"  [{method}] summary → {summary_path}")

    def run_method(self, method_name: str, devices: Dict[str, Any], 
                  global_model: nn.Module, dataset: Any) -> Dict[str, Any]:
        """Run a specific baseline method"""
        
        if method_name.lower() == 'fedavg':
            return self._run_fedavg(devices, global_model, dataset)
        elif method_name.lower() == 'fedprox':
            return self._run_fedprox(devices, global_model, dataset)
        elif method_name.lower() == 'hierfl':
            return self._run_hierfl(devices, global_model, dataset)
        elif method_name.lower() == 'clusterfl':
            return self._run_clusterfl(devices, global_model, dataset)
        elif method_name.lower() in ('apcfl', 'ap-cfl', 'ap_cfl'):
            return self._run_apcfl(devices, global_model, dataset)
        elif method_name.lower() in ('geofl', 'geo-fl', 'geo_fl'):
            return self._run_geofl(devices, global_model, dataset)
        elif method_name.lower() in ('ifca',):
            return self._run_ifca(devices, global_model, dataset)
        elif method_name.lower() in ('snapcfl', 'snap-cfl', 'snap_cfl'):
            return self._run_snapcfl(devices, global_model, dataset)
        else:
            raise ValueError(f"Unknown baseline method: {method_name}")
    
    def _run_fedavg(self, devices: Dict[str, Any], global_model: nn.Module, 
                   dataset: Any) -> Dict[str, Any]:
        """
        Run FedAvg baseline.
        
        Standard federated averaging without spatial awareness.
        """
        print("Running FedAvg baseline...")
        
        # Initialize tracking
        accuracies = []
        losses = []
        start_time = time.time()
        
        # Training loop
        try:
            for round_num in range(self.config.num_rounds):
                round_time = time.time()

                # Sample participating devices
                participating_devices = self._sample_devices(devices, 0.7)

                # Collect device updates
                device_updates = []
                device_weights = []

                for device_id in participating_devices:
                    device = devices[device_id]
                    if not device.is_active or not device.local_dataset:
                        continue

                    # Local training
                    local_model = copy.deepcopy(global_model)
                    local_model.load_state_dict(global_model.state_dict())

                    # Train locally
                    training_result = self._train_local_model(
                        local_model, device.local_dataloader,
                        self.config.local_epochs, self.config.learning_rate
                    )

                    if training_result["success"]:
                        device_updates.append(training_result["model_weights"])
                        device_weights.append(device.dataset_size)

                # FedAvg aggregation
                if device_updates:
                    global_weights = self._fedavg_aggregate(device_updates, device_weights)
                    global_model.load_state_dict(global_weights)

                # Evaluate
                accuracy, loss = self._evaluate_model(global_model, dataset)
                accuracies.append(accuracy)
                losses.append(loss)
                print(f"FedAvg Round {round_num + 1}: Accuracy={accuracy:.4f}, Loss={loss:.4f}, Time={time.time()-round_time:.2f}")
        except KeyboardInterrupt:
            print(f"Training (FedProx) interrupted by user")
        total_time = time.time() - start_time
        
        result = {
            "method": "FedAvg",
            "final_accuracy": accuracies[-1] if accuracies else 0.0,
            "final_loss": losses[-1] if losses else float('inf'),
            "accuracies": accuracies,
            "losses": losses,
            "total_time": total_time,
            "convergence_round": self._find_convergence(accuracies)
        }
        self.save_baseline_results(result, {
            "num_rounds":    self.config.num_rounds,
            "local_epochs":  self.config.local_epochs,
            "learning_rate": self.config.learning_rate,
            "batch_size":    self.config.batch_size,
            "sampling_rate": 0.7,
        })
        return result
    
    def _run_fedprox(self, devices: Dict[str, Any], global_model: nn.Module, 
                    dataset: Any) -> Dict[str, Any]:
        """
        Run FedProx baseline.
        
        Federated learning with proximal term for heterogeneous devices.
        """
        print("Running FedProx baseline...")
        
        mu = 0.01  # Proximal term coefficient
        accuracies = []
        losses = []
        start_time = time.time()

        try:
            for round_num in range(self.config.num_rounds):
                round_time = time.time()

                participating_devices = self._sample_devices(devices, 0.7)
                device_updates = []
                device_weights = []

                for device_id in participating_devices:
                    device = devices[device_id]
                    if not device.is_active or not device.local_dataset:
                        continue

                    # Local training with proximal term
                    local_model = copy.deepcopy(global_model)
                    global_weights = {name: param.clone() for name, param in global_model.named_parameters()}

                    training_result = self._train_local_model_fedprox(
                        local_model, device.local_dataloader, global_weights,
                        self.config.local_epochs, self.config.learning_rate, mu
                    )

                    if training_result["success"]:
                        device_updates.append(training_result["model_weights"])
                        device_weights.append(device.dataset_size)

                # Standard aggregation
                if device_updates:
                    global_weights = self._fedavg_aggregate(device_updates, device_weights)
                    global_model.load_state_dict(global_weights)

                # Evaluate
                accuracy, loss = self._evaluate_model(global_model, dataset)
                accuracies.append(accuracy)
                losses.append(loss)

                print(f"FedProx Round {round_num + 1}: Accuracy={accuracy:.4f}, Loss={loss:.4f}, Time={time.time()-round_time:.2f}")
        except KeyboardInterrupt:
            print(f"Training (FedProx) interrupted by user")
        total_time = time.time() - start_time
        
        result = {
            "method": "FedProx",
            "final_accuracy": accuracies[-1] if accuracies else 0.0,
            "final_loss": losses[-1] if losses else float('inf'),
            "accuracies": accuracies,
            "losses": losses,
            "total_time": total_time,
            "convergence_round": self._find_convergence(accuracies)
        }
        self.save_baseline_results(result, {
            "num_rounds":    self.config.num_rounds,
            "local_epochs":  self.config.local_epochs,
            "learning_rate": self.config.learning_rate,
            "batch_size":    self.config.batch_size,
            "mu":            0.01,
            "sampling_rate": 0.7,
        })
        return result
    
    def _run_hierfl(self, devices: Dict[str, Any], global_model: nn.Module, 
                   dataset: Any) -> Dict[str, Any]:
        """
        Run HierFL baseline.
        
        Static hierarchical aggregation based on network topology.
        """
        print("Running HierFL baseline...")
        
        # Create static clusters (zones) based on device IDs
        num_clusters = min(self.config.num_zones, len(devices) // 3)
        device_list = list(devices.keys())
        cluster_size = len(device_list) // num_clusters
        
        clusters = []
        for i in range(num_clusters):
            start_idx = i * cluster_size
            end_idx = start_idx + cluster_size if i < num_clusters - 1 else len(device_list)
            clusters.append(device_list[start_idx:end_idx])
        
        accuracies = []
        losses = []
        start_time = time.time()
        try:
            for round_num in range(self.config.num_rounds):
                round_time = time.time()

                # Two-level aggregation
                cluster_models = []
                cluster_weights = []

                for cluster in clusters:
                    # Intra-cluster aggregation — 70% participation per cluster
                    cluster_device_updates = []
                    cluster_device_weights = []

                    cluster_devices = {d: devices[d] for d in cluster if d in devices}
                    selected = self._sample_devices(cluster_devices, 0.7)

                    for device_id in selected:
                        device = devices[device_id]
                        local_model = copy.deepcopy(global_model)
                        training_result = self._train_local_model(
                            local_model, device.local_dataloader,
                            self.config.local_epochs, self.config.learning_rate
                        )

                        if training_result["success"]:
                            cluster_device_updates.append(training_result["model_weights"])
                            cluster_device_weights.append(device.dataset_size)

                    # Aggregate within cluster
                    if cluster_device_updates:
                        cluster_model = self._fedavg_aggregate(cluster_device_updates, cluster_device_weights)
                        cluster_models.append(cluster_model)
                        cluster_weights.append(sum(cluster_device_weights))

                # Inter-cluster aggregation
                if cluster_models:
                    global_weights = self._fedavg_aggregate(cluster_models, cluster_weights)
                    global_model.load_state_dict(global_weights)

                # Evaluate
                accuracy, loss = self._evaluate_model(global_model, dataset)
                accuracies.append(accuracy)
                losses.append(loss)

                print(f"HierFL Round {round_num + 1}: Accuracy={accuracy:.4f}, Loss={loss:.4f}, Time={time.time()-round_time:.2f}")
        except KeyboardInterrupt:
            print(f"Training (HierFL) interrupted by user")
        total_time = time.time() - start_time
        
        result = {
            "method": "HierFL",
            "final_accuracy": accuracies[-1] if accuracies else 0.0,
            "final_loss": losses[-1] if losses else float('inf'),
            "accuracies": accuracies,
            "losses": losses,
            "total_time": total_time,
            "convergence_round": self._find_convergence(accuracies),
            "num_clusters": num_clusters,
        }
        self.save_baseline_results(result, {
            "num_rounds":    self.config.num_rounds,
            "local_epochs":  self.config.local_epochs,
            "learning_rate": self.config.learning_rate,
            "batch_size":    self.config.batch_size,
            "num_clusters":  num_clusters,
            "sampling_rate": 0.7,
        })
        return result

    def _run_clusterfl(self, devices, global_model, dataset):
        """
        Simple ClusterFL baseline:
        - participation rate (default 0.7)
        - softmax outputs → KL divergence → similarity matrix F
        - proximal updates toward weighted neighbors
        - FedAvg aggregation
        """

        participation_rate = 0.7
        rho = 0.01
        prox_step = 1.0
        temperature = 1.0
        recluster_every = 1

        start_time = time.time()
        accuracies, losses = [], []

        # template for converting vectors ↔ state_dict
        template_state_dict = global_model.state_dict()

        W_dim = None
        last_F = None

        try:
            for round_num in range(self.config.num_rounds):
                round_time = time.time()

                # --- Sample participants (70% via shared helper) ---
                participating = self._sample_devices(devices, participation_rate)
                local_state_dicts = {}
                device_weights = []

                # --- Local training ---
                for cid in participating:
                    device = devices[cid]
                    if not device.is_active or not device.local_dataset:
                        continue

                    local_model = copy.deepcopy(global_model)
                    res = self._train_local_model(local_model, device.local_dataloader,
                                            self.config.local_epochs, self.config.learning_rate)
                    if not res.get("success", False):
                        continue

                    local_state_dicts[cid] = res["model_weights"]
                    device_weights.append(device.dataset_size)

                if not local_state_dicts:
                    acc, loss = self._evaluate_model(global_model, dataset)
                    accuracies.append(acc)
                    losses.append(loss)
                    continue

                client_ids = list(local_state_dicts.keys())
                state_dicts = [local_state_dicts[cid] for cid in client_ids]

                # --- Convert to vectors ---
                W = np.vstack([self.state_dict_to_vector(sd) for sd in state_dicts])
                if W_dim is None:
                    W_dim = W.shape[1]

                # --- Compute similarity F ---
                if (round_num % recluster_every == 0) or last_F is None:
                    # compute average softmax outputs
                    outputs = []
                    for sd in state_dicts:
                        m = copy.deepcopy(global_model)
                        m.load_state_dict(sd)
                        logits = self.predict_logits(m, dataset)
                        outputs.append(torch.softmax(torch.tensor(logits) / temperature, dim=1).numpy())
                    m = len(outputs)
                    F_mat = np.zeros((m, m))
                    for i in range(m):
                        for j in range(m):
                            p, q = outputs[i], outputs[j]
                            F_mat[i, j] = np.mean(np.sum(p * np.log((p + 1e-10) / (q + 1e-10)), axis=1))
                    # similarity = 1 - normalized KL
                    F_mat = 1 - (F_mat / (F_mat.max() + 1e-10))
                    F_mat = np.clip(F_mat, 0, 1)
                    # normalize columns
                    F_mat = F_mat / (F_mat.sum(axis=0, keepdims=True) + 1e-10)
                    last_F = F_mat
                else:
                    F_mat = last_F

                # --- Proximal update toward neighbors ---
                W_new = W - prox_step * rho * (W - F_mat.T @ W)

                # --- Convert back & aggregate (FedAvg) ---
                updated_state_dicts = [self.vector_to_state_dict(vec, template_state_dict) for vec in W_new]
                global_weights = self._fedavg_aggregate(updated_state_dicts, device_weights)
                global_model.load_state_dict(global_weights)

                # --- Eval ---
                acc, loss = self._evaluate_model(global_model, dataset)
                accuracies.append(acc)
                losses.append(loss)
                print(f"ClusterFL Round {round_num + 1}: Accuracy={acc:.4f}, Loss={loss:.4f}, Time={time.time()-round_time:.2f}")
        except KeyboardInterrupt:
            print("Training (ClusterFL) interrupted by user")
        result = {
            "method": "ClusterFL",
            "final_accuracy": accuracies[-1] if accuracies else 0.0,
            "final_loss": losses[-1] if losses else float('inf'),
            "accuracies": accuracies,
            "losses": losses,
            "total_time": time.time() - start_time,
        }
        self.save_baseline_results(result, {
            "num_rounds":      self.config.num_rounds,
            "local_epochs":    self.config.local_epochs,
            "learning_rate":   self.config.learning_rate,
            "batch_size":      self.config.batch_size,
            "rho":             0.01,
            "temperature":     1.0,
            "sampling_rate":   participation_rate,
        })
        return result

    def _run_apcfl(self, devices: Dict[str, Any], global_model: nn.Module,
                   dataset: Any) -> Dict[str, Any]:
        """AP-CFL: Affinity-Propagation Clustered FL with TDI-weighted aggregation."""
        print("Running AP-CFL baseline...")
        apcfl = APCFL(self.config, devices, global_model, dataset)
        result = apcfl.run()
        self.save_baseline_results(result, {
            "num_rounds":      getattr(self.config, "apcfl_num_rounds",      self.config.num_rounds),
            "local_epochs":    getattr(self.config, "apcfl_local_epochs",    1),
            "learning_rate":   getattr(self.config, "apcfl_lr",              0.0001),
            "lambda":          getattr(self.config, "apcfl_lambda",          0.05),
            "sampling_rate":   getattr(self.config, "apcfl_sampling_rate",   0.7),
            "batch_size":      getattr(self.config, "apcfl_batch_size",      32),
        })
        return result

    def _run_geofl(self, devices: Dict[str, Any], global_model: nn.Module,
                   dataset: Any) -> Dict[str, Any]:
        """GeoFL: geo-distributed hierarchical FL with importance-aware aggregation."""
        print("Running GeoFL baseline...")
        geofl = GeoFL(self.config, devices, global_model, dataset)
        result = geofl.run()
        self.save_baseline_results(result, {
            "num_rounds":      getattr(self.config, "geofl_num_rounds",             self.config.num_rounds),
            "local_steps":     getattr(self.config, "geofl_local_steps",            5),
            "learning_rate":   getattr(self.config, "geofl_lr",                     0.001),
            "S0":              getattr(self.config, "geofl_S0",                     0.01),
            "S_min":           getattr(self.config, "geofl_S_min",                  0.001),
            "alpha":           getattr(self.config, "geofl_alpha",                  0.95),
            "R0":              getattr(self.config, "geofl_R0",                     5),
            "beta":            getattr(self.config, "geofl_beta",                   0.2),
            "sampling_rate":   getattr(self.config, "geofl_client_sampling_rate",   0.7),
            "batch_size":      getattr(self.config, "geofl_batch_size",             16),
        })
        return result

    def _run_ifca(self, devices: Dict[str, Any], global_model: nn.Module,
                  dataset: Any) -> Dict[str, Any]:
        """IFCA: Iterative Federated Clustering Algorithm (NeurIPS 2020)."""
        print("Running IFCA baseline...")
        ifca = IFCA(self.config, devices, global_model, dataset)
        result = ifca.run()
        self.save_baseline_results(result, {
            "num_rounds":    getattr(self.config, "ifca_num_rounds",    self.config.num_rounds),
            "k":             getattr(self.config, "ifca_k",             4),
            "local_steps":   getattr(self.config, "ifca_local_steps",   5),
            "learning_rate": getattr(self.config, "ifca_lr",            0.01),
            "lr_decay":      getattr(self.config, "ifca_lr_decay",      1.0),
            "batch_size":    getattr(self.config, "ifca_batch_size",    32),
            "sampling_rate": getattr(self.config, "ifca_sampling_rate", 0.7),
            "variant":       getattr(self.config, "ifca_variant",       "model"),
            "weight_sharing": getattr(self.config, "ifca_weight_sharing", False),
        })
        return result

    def _run_snapcfl(self, devices: Dict[str, Any], global_model: nn.Module,
                     dataset: Any) -> Dict[str, Any]:
        """SnapCFL: Pre-Clustering-Based Clustered FL (IEEE TMC 2025)."""
        print("Running SnapCFL baseline...")
        snapcfl = SnapCFL(self.config, devices, global_model, dataset)
        result  = snapcfl.run()
        self.save_baseline_results(result, {
            "num_rounds":         getattr(self.config, "snapcfl_num_rounds",          self.config.num_rounds),
            "learning_rate":      getattr(self.config, "snapcfl_lr",                  0.01),
            "batch_size":         getattr(self.config, "snapcfl_batch_size",          32),
            "local_epochs":       getattr(self.config, "snapcfl_local_epochs",        5),
            "sampling_rate":      getattr(self.config, "snapcfl_sampling_rate",       0.7),
            "pre_cluster_rounds": getattr(self.config, "snapcfl_pre_cluster_rounds",  10),
            "eps":                getattr(self.config, "snapcfl_eps",                 0.25),
            "min_samples":        getattr(self.config, "snapcfl_min_samples",         2),
            "intra_algo":         getattr(self.config, "snapcfl_intra_algo",          "fedavg"),
            "global_averaging":   getattr(self.config, "snapcfl_global_averaging",    False),
        })
        return result

    def state_dict_to_vector(self, state_dict: dict) -> np.ndarray:
        vecs = []
        for key in state_dict:
            vecs.append(state_dict[key].detach().cpu().numpy().ravel())
        return np.concatenate(vecs)

    def vector_to_state_dict(self, vec: np.ndarray, template_state_dict: dict) -> dict:
        new_state_dict = {}
        pointer = 0
        for key, param in template_state_dict.items():
            shape = param.shape
            numel = param.numel()
            new_state_dict[key] = torch.tensor(vec[pointer:pointer+numel].reshape(shape))
            pointer += numel
        return new_state_dict

    def predict_logits(self, model, dataset):
        device = self.config.device

        model = copy.deepcopy(model)
        model = model.to(device)
        model.eval()
        logits_list = []
        test_dataloader = dataset.get_global_dataloader(batch_size=64, is_train=False)

        with torch.no_grad():
            for data, target in test_dataloader:
                if device == 'cuda':
                    data, target = data.cuda(), target.cuda()
                if self.config.dataset_name == 'shakespeare':
                    out, _ = model(data)
                else:
                    out = model(data)
                logits_list.append(out.detach().cpu().numpy())
        return np.vstack(logits_list)

    def _sample_devices(self, devices: Dict[str, Any], participation_rate: float) -> List[str]:
        """Sample devices for participation"""
        available_devices = [
            device_id for device_id, device in devices.items()
            if device.is_active and device.local_dataset
        ]
        
        num_participants = max(1, int(participation_rate * len(available_devices)))
        return np.random.choice(available_devices, size=num_participants, replace=False).tolist()
    
    def _train_local_model(self, model: nn.Module, dataloader: torch.utils.data.DataLoader,
                          epochs: int, learning_rate: float) -> Dict[str, Any]:
        """Train local model using standard SGD"""
        if dataloader is None or len(dataloader) == 0:
            return {"success": False}
        
        # Move model to appropriate device
        device = self.config.device
        if device == 'cuda' and torch.cuda.is_available():
            model = model.cuda()
        else:
            model = model.cpu()
            device = 'cpu'
        
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        use_cuda = device == 'cuda'
        scaler = torch.amp.GradScaler('cuda') if use_cuda else None

        try:
            for epoch in range(epochs):
                for data, target in dataloader:
                    if use_cuda:
                        data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)

                    optimizer.zero_grad()
                    if use_cuda:
                        with torch.amp.autocast('cuda'):
                            output = model(data)
                            if isinstance(output, tuple):
                                output = output[0]
                            loss = criterion(output, target)
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        output = model(data)
                        if isinstance(output, tuple):
                            output = output[0]
                        loss = criterion(output, target)
                        loss.backward()
                        optimizer.step()

            if use_cuda:
                model = model.cpu()

            return {
                "success": True,
                "model_weights": model.state_dict()
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _train_local_model_fedprox(self, model: nn.Module, dataloader: torch.utils.data.DataLoader,
                                  global_weights: Dict[str, torch.Tensor], epochs: int, 
                                  learning_rate: float, mu: float) -> Dict[str, Any]:
        """Train local model with FedProx proximal term"""
        if dataloader is None or len(dataloader) == 0:
            return {"success": False}
        
        use_cuda = self.config.device == 'cuda' and torch.cuda.is_available()
        if use_cuda:
            model = model.cuda()
            global_weights = {k: v.cuda() for k, v in global_weights.items()}
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        scaler = torch.amp.GradScaler('cuda') if use_cuda else None

        try:
            for epoch in range(epochs):
                for data, target in dataloader:
                    if use_cuda:
                        data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)

                    optimizer.zero_grad()
                    if use_cuda:
                        with torch.amp.autocast('cuda'):
                            output = model(data)
                            if isinstance(output, tuple):
                                output = output[0]
                            loss = criterion(output, target)
                            proximal_term = sum(
                                torch.sum((p - global_weights[n]) ** 2)
                                for n, p in model.named_parameters() if n in global_weights
                            )
                            total_loss = loss + (mu / 2) * proximal_term
                        scaler.scale(total_loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        output = model(data)
                        if isinstance(output, tuple):
                            output = output[0]
                        loss = criterion(output, target)
                        proximal_term = sum(
                            torch.sum((p - global_weights[n]) ** 2)
                            for n, p in model.named_parameters() if n in global_weights
                        )
                        total_loss = loss + (mu / 2) * proximal_term
                        total_loss.backward()
                        optimizer.step()

            if use_cuda:
                model = model.cpu()

            return {
                "success": True,
                "model_weights": model.state_dict()
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _fedavg_aggregate(self, model_updates: List[Dict[str, torch.Tensor]], 
                         weights: List[float]) -> Dict[str, torch.Tensor]:
        """Perform FedAvg aggregation"""
        if not model_updates:
            return {}
        
        # Normalize weights
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        # Initialize aggregated model
        aggregated_weights = {}
        first_model = model_updates[0]
        
        for param_name in first_model.keys():
            aggregated_weights[param_name] = torch.zeros_like(first_model[param_name], dtype=torch.float32)
        
        # Weighted aggregation
        for model_update, weight in zip(model_updates, normalized_weights):
            for param_name, param_tensor in model_update.items():
                if param_name in aggregated_weights:
                    aggregated_weights[param_name] += weight * param_tensor
        
        return aggregated_weights
    
    def _evaluate_model(self, model: nn.Module, dataset: Any) -> Tuple[float, float]:
        """Evaluate model on test dataset"""
        # Move model to appropriate device
        device = self.config.device
        if device == 'cuda' and torch.cuda.is_available():
            model = model.cuda()
        else:
            model = model.cpu()
            device = 'cpu'
        
        model.eval()
        
        try:
            test_dataloader = dataset.get_global_dataloader(batch_size=64, is_train=False)
            
            total_loss = 0.0
            correct = 0
            total = 0
            criterion = nn.CrossEntropyLoss()
            
            # Move criterion to same device
            if device == 'cuda':
                criterion = criterion.cuda()
            
            with torch.no_grad():
                for data, target in test_dataloader:
                    # Move data to device
                    if device == 'cuda':
                        data, target = data.cuda(), target.cuda()
                    
                    output = model(data)
                    
                    if isinstance(output, tuple):
                        output = output[0]
                    
                    loss = criterion(output, target)
                    total_loss += loss.detach().item()
                    
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
                    total += target.size(0)
            
            accuracy = correct / max(total, 1)
            avg_loss = total_loss / max(len(test_dataloader), 1)
            
        except Exception as e:
            print(f"Evaluation error: {e}")
            accuracy = 0.0
            avg_loss = float('inf')
        
        model.train()
        return accuracy, avg_loss
    
    def _find_convergence(self, accuracies: List[float], window_size: int = 10) -> int:
        """Find convergence round based on accuracy stabilization"""
        if len(accuracies) < window_size:
            return -1
        
        for i in range(window_size, len(accuracies)):
            recent_acc = accuracies[i-window_size:i]
            acc_variance = np.var(recent_acc)
            
            if acc_variance < 0.001:  # Convergence threshold
                return i - window_size
        
        return -1  # Not converged
    
    def compare_methods(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Compare results from different baseline methods"""
        comparison = {
            "method_comparison": {},
            "best_method": "",
            "best_accuracy": 0.0,
            "convergence_comparison": {},
            "efficiency_comparison": {}
        }
        
        best_accuracy = 0.0
        best_method = ""
        
        for method_name, result in results.items():
            final_acc = result.get("final_accuracy", 0.0)
            final_loss = result.get("final_loss", float('inf'))
            total_time = result.get("total_time", 0.0)
            conv_round = result.get("convergence_round", -1)
            
            comparison["method_comparison"][method_name] = {
                "final_accuracy": final_acc,
                "final_loss": final_loss,
                "total_time": total_time,
                "convergence_round": conv_round
            }
            
            if final_acc > best_accuracy:
                best_accuracy = final_acc
                best_method = method_name
            
            # Efficiency metrics
            if conv_round > 0:
                comparison["convergence_comparison"][method_name] = conv_round
            
            comparison["efficiency_comparison"][method_name] = {
                "time_per_round": total_time / self.config.num_rounds,
                "accuracy_per_time": final_acc / max(total_time, 1e-6)
            }
        
        comparison["best_method"] = best_method
        comparison["best_accuracy"] = best_accuracy
        
        return comparison