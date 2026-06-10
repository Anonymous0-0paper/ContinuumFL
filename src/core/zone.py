"""
Zone module for ContinuumFL framework.
Implements spatial zones that aggregate edge devices with similar characteristics.
"""
import logging
import os
import random

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Set
from collections import defaultdict, deque
import time

from torch import Tensor

from .device import EdgeDevice

logger = logging.getLogger("ContinuumFL.Zone")


# ─────────────────────────────────────────────────────────────────────────────
# Thread worker
# ─────────────────────────────────────────────────────────────────────────────

def run_training(args: dict[str, Any]):
    device = args["device"]
    global_model = args["model"]
    epochs = args["epochs"]
    learning_rate = args["learning_rate"]
    comp_device = args["comp_device"]
    enable_failure = args["enable_failure"]
    device_failure_probability = args["device_failure_probability"]
    dataset_name = args.get("dataset_name", "cifar100")

    failure = False
    if enable_failure:
        # FIX: simulate_failure now correctly returns bool and does NOT repair the device
        failure = device.simulate_failure(device_failure_probability)

    # active device fails this round
    if device.is_active and failure:
        logger.debug(f"[run_training] {device.device_id} failed this round — skipping")
        return None

    # device active or failed device is repaired
    if device.is_active or device.simulate_repair():
        result = device.local_train(
            global_model,
            epochs=epochs,
            learning_rate=learning_rate,
            device=comp_device,
            dataset_name=dataset_name
        )
        return result, device.device_id

    logger.debug(f"[run_training] {device.device_id} failed and not repaired — skipping")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Zone
# ─────────────────────────────────────────────────────────────────────────────

class Zone:
    """
    Represents a spatial zone in the ContinuumFL framework.

    Based on the paper's system model, each zone z_k contains:
    - A subset of devices D_k
    - Zone aggregator (edge server)
    - Spatial correlation with neighboring zones
    - Zone-level objective F_k(w)
    """

    def __init__(self, zone_id: str, edge_server_id: str,
                 compression_rate: float, enable_compression: bool = False):
        self.zone_id = zone_id
        self.edge_server_id = edge_server_id

        self.is_active = True

        # Device management
        self.devices: Dict[str, EdgeDevice] = {}
        self.device_ids: Set[str] = set()

        # Zone-level model and aggregation
        self.zone_model: Optional[nn.Module] = None
        self.aggregated_weights: Optional[Dict[str, torch.Tensor]] = None

        # Spatial characteristics
        self.centroid: Optional[Tuple[float, float]] = None
        self.neighbors: Set[str] = set()  # Neighboring zone IDs
        self.spatial_correlations: Dict[str, float] = {}  # ρ(z_k, z_j)

        # Performance tracking
        self.participation_rates = deque(maxlen=100)
        self.aggregation_times = deque(maxlen=50)
        self.communication_costs = deque(maxlen=50)
        self.accuracy_history = deque(maxlen=100)

        # Zone metrics
        self.staleness_counter = 0  # τ_k
        self.priority_score = 0.0   # P_k
        self.contribution_score = 0.0
        self.data_consistency = 0.0
        self.total_dataset_size = 0

        # Aggregation weights
        self.intra_zone_weights: Dict[str, float] = {}  # α_i^k
        self.inter_zone_weight = 0.0  # β_k

        # Resource allocation
        self.allocated_bandwidth = 0.0  # B_k
        self.compute_capacity = 0.0
        self.memory_capacity = 0.0

        # Failure handling
        self.is_operational = True
        self.backup_aggregators: List[str] = []

        self.compression_rate = compression_rate
        self.enable_compression = enable_compression

        self.sample_device_seed = 42

        logger.debug(f"[zone] Created zone {zone_id} (server={edge_server_id})")

    def add_device(self, device: EdgeDevice):
        """Add a device to this zone"""
        self.devices[device.device_id] = device
        self.device_ids.add(device.device_id)
        device.zone_id = self.zone_id
        if device.resources.bandwidth >= self.allocated_bandwidth - self.allocated_bandwidth/10 and \
                device.resources.compute_capacity >= self.compute_capacity - self.compute_capacity/10 and \
                device.resources.memory_capacity >= self.memory_capacity:
            self.backup_aggregators.append(device.device_id)
        self._update_zone_statistics()
        logger.debug(f"[zone] {self.zone_id} added device {device.device_id} (total={len(self.devices)})")

    def remove_device(self, device_id: str) -> bool:
        """Remove a device from this zone"""
        if device_id in self.devices:
            del self.devices[device_id]
            self.device_ids.discard(device_id)
            self._update_zone_statistics()
            if device_id in self.backup_aggregators:
                self.backup_aggregators.remove(device_id)
            logger.debug(f"[zone] {self.zone_id} removed device {device_id}")
            return True
        return False

    def _update_zone_statistics(self):
        """Update zone-level statistics when devices are added/removed"""
        if not self.devices:
            return

        locations = [device.location for device in self.devices.values()]
        self.centroid = (np.mean([loc[0] for loc in locations]), np.mean([loc[1] for loc in locations]))

        self.total_dataset_size = sum(device.dataset_size for device in self.devices.values())

        self.compute_capacity = sum(device.resources.compute_capacity for device in self.devices.values())
        self.memory_capacity = sum(device.resources.memory_capacity for device in self.devices.values())

    def _sample_participating_devices(self) -> List[str]:
        """Sample devices for participation in current round"""
        available_devices = [
            device_id for device_id, device in self.devices.items()
            if device.is_active and device.local_dataset
        ]

        if available_devices:
            # ContinuumFL uses all available clients every round
            logger.debug(
                f"[zone] {self.zone_id} using all {len(available_devices)} available devices"
            )
            return available_devices

        logger.debug(f"[zone] {self.zone_id} no available devices for sampling")
        return []

    def perform_local_training(self, args) -> tuple[str, Any, dict[str, list[str] | int | float]]:
        """Perform local training on participating devices"""
        zone_start = time.time()

        participating_devices = self._sample_participating_devices()
        comp_device = args["comp_device"]
        global_model = args["model"]
        learning_rate = args["learning_rate"]
        epochs = args["epochs"]
        device_participation = args["device_participation"]
        enable_failure = args["enable_failure"]
        device_failure_probability = args["device_failure_probability"]
        device_updates = {}

        # FIX: renamed loop variable from `args` to `dev_arg` to avoid shadowing
        # the outer `args` parameter (old code silently corrupted `args` after the loop)
        device_args = [
            {"device": self.devices[device_id], "model": global_model, "epochs": epochs,
             "learning_rate": learning_rate,
             "comp_device": comp_device,
             "enable_failure": enable_failure,
             "device_failure_probability": device_failure_probability,
             "dataset_name": args.get("dataset_name", "cifar100")}
            for device_id in participating_devices
        ]

        failures = 0
        successes = 0

        def _process_result(res):
            nonlocal failures, successes
            if res is None:
                failures += 1
                return
            res_dict, device_id = res
            if res_dict["success"] and device_id in self.devices:
                device_updates[device_id] = res_dict["gradient"]
                device_participation[device_id] += 1
                self.devices[device_id].participation_history.append(1)
                successes += 1
                logger.debug(
                    f"[zone] {self.zone_id} device {device_id} update collected "
                    f"‖δw‖={res_dict.get('grad_norm', 0.0):.4f}  "
                    f"loss={res_dict.get('training_loss', 0.0):.4f}"
                )
            else:
                failures += 1
                logger.debug(
                    f"[zone] {self.zone_id} device {res_dict if isinstance(res_dict, str) else device_id} "
                    f"returned no update (success={res_dict.get('success') if isinstance(res_dict, dict) else 'N/A'})"
                )

        if comp_device.startswith('cuda'):
            # CUDA kernels serialize across threads — run sequentially to avoid overhead
            for dev_arg in device_args:   # FIX: was `for args in device_args`
                _process_result(run_training(dev_arg))
        else:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            max_workers = min(len(participating_devices), os.cpu_count())
            with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
                futures = [executor.submit(run_training, dev_arg) for dev_arg in device_args]
                for f in as_completed(futures):
                    _process_result(f.result())

        logger.debug(
            f"[zone] {self.zone_id} training: {successes} success, {failures} failure/skip "
            f"out of {len(participating_devices)} participating"
        )

        intra_start = time.time()
        aggregated_weights = self.intra_zone_aggregation(device_updates)
        intra_time = time.time() - intra_start

        comm_cost = self.estimate_communication_cost(device_updates, {self.zone_id: aggregated_weights})
        total_zone_time = time.time() - zone_start

        logger.debug(
            f"[zone] {self.zone_id} done: intra={intra_time*1000:.1f}ms "
            f"total={total_zone_time*1000:.1f}ms  comm={comm_cost:.2f}MB"
        )

        return self.zone_id, aggregated_weights, {
            "participating_devices": participating_devices,
            "num_device_updates": len(device_updates),
            "intra_time": intra_time,
            "communication_cost": comm_cost,
        }

    def estimate_communication_cost(self, device_updates: Dict[str, Dict[str, torch.Tensor]],
                                    zone_weights: Dict[str, Dict[str, torch.Tensor]]) -> float:
        """Estimate communication cost in MB for this round"""
        total_cost = 0.0
        compression_rate = self.compression_rate if self.enable_compression else 1.0

        for device_id, weights in device_updates.items():
            device_size = sum(param.numel() * 4 for param in weights.values()) / (1024 * 1024)
            total_cost += device_size * compression_rate

        for zone_id, weights in zone_weights.items():
            zone_size = sum(param.numel() * 4 for param in weights.values()) / (1024 * 1024)
            total_cost += zone_size * compression_rate

        return total_cost

    def compute_intra_zone_weights(self) -> Dict[str, float]:
        """
        Compute aggregation weights for devices within the zone.

        Implements Equation (10):
        α_i^k = (n_i * q_i * r_i) / Σ(n_j * q_j * r_j)
        """
        if not self.devices:
            return {}

        weights = {}
        total_weight = 0.0

        for device_id, device in self.devices.items():
            if device.is_active:
                device.estimate_data_quality()

                dataset_size = device.dataset_size
                quality_score = device.data_quality_score
                reliability_score = device.reliability_score

                weight = dataset_size * quality_score * reliability_score
                weights[device_id] = weight
                total_weight += weight

        if total_weight > 0:
            for device_id in weights:
                weights[device_id] /= total_weight

        self.intra_zone_weights = weights
        logger.debug(
            f"[zone] {self.zone_id} intra weights computed for {len(weights)} devices "
            f"(total_weight={total_weight:.2f})"
        )
        return weights

    def intra_zone_aggregation(self, device_updates: Dict[str, Dict[str, torch.Tensor]]) -> dict[Any, Any]:
        """
        Perform intra-zone aggregation of device updates.

        Implements Equation (9):
        w_k^(t+1) = Σ α_i^k * w_i^(t)
        """
        start_time = time.time()

        weights = self.compute_intra_zone_weights()

        if not weights or not device_updates:
            logger.debug(f"[zone] {self.zone_id} intra-agg: no updates or weights — returning empty")
            return {}

        # Initialize aggregated weights from first available update
        aggregated_weights = {}
        original_dtypes = {}
        first_device_id = next(iter(device_updates.keys()))
        first_weights = device_updates[first_device_id]

        for param_name, param_tensor in first_weights.items():
            original_dtypes[param_name] = param_tensor.dtype
            if param_tensor.dtype != torch.float32:
                aggregated_weights[param_name] = torch.zeros_like(param_tensor, dtype=torch.float32)
            else:
                aggregated_weights[param_name] = torch.zeros_like(param_tensor)

        total_weight = 0.0
        for device_id, device_weights in device_updates.items():
            if device_id in weights and weights[device_id] > 0:
                weight = weights[device_id]
                for param_name, param_tensor in device_weights.items():
                    if param_tensor.dtype != torch.float32:
                        param_tensor = param_tensor.float()
                    aggregated_weights[param_name] += weight * param_tensor
                total_weight += weight

        if total_weight > 0 and abs(total_weight - 1.0) > 1e-6:
            for param_name in aggregated_weights:
                aggregated_weights[param_name] = aggregated_weights[param_name] / total_weight

        for param_name in aggregated_weights:
            if original_dtypes[param_name] != torch.float32:
                aggregated_weights[param_name] = aggregated_weights[param_name].to(original_dtypes[param_name])

        if self.enable_compression:
            compressed_weights = self._apply_gradient_compression(aggregated_weights, self.compression_rate)
            self.aggregated_weights = compressed_weights
        else:
            self.aggregated_weights = aggregated_weights

        aggregation_time = time.time() - start_time
        self.aggregation_times.append(aggregation_time)

        # Debug: log aggregated weight norm
        if self.aggregated_weights:
            agg_norm = float(np.sqrt(sum(
                v.float().norm(2).item() ** 2
                for v in self.aggregated_weights.values()
            )))
        else:
            agg_norm = 0.0
        logger.debug(
            f"[zone] {self.zone_id} intra-agg done: {len(device_updates)} updates "
            f"total_weight={total_weight:.4f}  ‖agg_delta‖={agg_norm:.4f}  "
            f"time={aggregation_time*1000:.1f}ms"
        )

        return self.aggregated_weights

    def _apply_gradient_compression(self, weights: Dict[str, torch.Tensor],
                                  compression_rate: float = 0.1) -> Dict[str, Tensor]:
        """Apply top-k compression to aggregated weights"""
        compressed_weights = {}

        for param_name, param_tensor in weights.items():
            flat_tensor = param_tensor.flatten()
            k = max(1, int(compression_rate * len(flat_tensor)))

            _, top_k_indices = torch.topk(torch.abs(flat_tensor), k)

            sparse_tensor = torch.zeros_like(flat_tensor)
            sparse_tensor[top_k_indices] = flat_tensor[top_k_indices]

            compressed_weights[param_name] = sparse_tensor.reshape(param_tensor.shape)

        return compressed_weights

    def compute_zone_contribution_score(self) -> float:
        """
        Compute zone contribution score for inter-zone aggregation.

        Implements Equation (14):
        Score_k = |D_k| * n̄_k * exp(-Var({∇F_i})) * (1 - L_k^val)

        FIX: replaced `np.random.random()` placeholder for validation_accuracy with a
        deterministic value derived from device gradient histories.  Random noise caused
        zone weights to change wildly each round, destabilizing the aggregation and
        freezing global accuracy for the first 5 rounds.
        """
        if not self.devices:
            return 0.0

        num_devices = len([d for d in self.devices.values() if d.is_active])
        avg_dataset_size = self.total_dataset_size / max(num_devices, 1)
        data_contribution = num_devices * avg_dataset_size

        # Gradient consistency: use stored norm summaries (no full tensors needed)
        gradient_norms = []
        for device in self.devices.values():
            if device.is_active and device.gradient_history:
                gradient_norms.append(device.gradient_history[-1]["total_norm"])

        if gradient_norms:
            gradient_variance = float(np.var(gradient_norms))
            consistency_score = float(np.exp(-0.0001 * gradient_variance))
        else:
            # No history yet — use a neutral consistency score (first round)
            consistency_score = 1.0

        # FIX: use a deterministic proxy for validation accuracy instead of random noise.
        # We use the mean reliability score of active devices (∈ [0,1]), which is
        # based on participation history and updates every round without randomness.
        active_devices = [d for d in self.devices.values() if d.is_active]
        if active_devices:
            validation_accuracy = float(np.mean([d.reliability_score for d in active_devices]))
        else:
            validation_accuracy = 0.5

        score = data_contribution * consistency_score * validation_accuracy
        self.contribution_score = score

        logger.debug(
            f"[zone] {self.zone_id} contribution: n_dev={num_devices} "
            f"avg_data={avg_dataset_size:.0f} consistency={consistency_score:.4f} "
            f"val_proxy={validation_accuracy:.4f} score={score:.2f}"
        )

        return score

    def compute_spatial_correlation(self, other_zone: 'Zone') -> float:
        """
        Compute spatial correlation with another zone.

        Implements Equation (1):
        ρ(z_k, z_j) = exp(-Σ dist(d_i, d_j) / (|D_k| * |D_j| * σ))
        """
        if not self.devices or not other_zone.devices:
            return 0.0

        total_distance = 0.0
        count = 0

        for device_i in self.devices.values():
            for device_j in other_zone.devices.values():
                distance = device_i.compute_spatial_distance(device_j)
                total_distance += distance
                count += 1

        if count == 0:
            return 0.0

        avg_distance = total_distance / count
        sigma = 10.0

        correlation = np.exp(-avg_distance / sigma)
        return correlation

    def update_spatial_correlations(self, zones: Dict[str, 'Zone']):
        """Update spatial correlations with all other zones"""
        for other_zone_id, other_zone in zones.items():
            if other_zone_id != self.zone_id:
                correlation = self.compute_spatial_correlation(other_zone)
                self.spatial_correlations[other_zone_id] = correlation

    def simulate_failure(self, failure_probability):
        if self.edge_server_id == 'cloud_coordinator':
            return False

        rng = random.random()

        if rng < failure_probability:
            self.is_active = False
            logger.debug(f"[zone] {self.zone_id} zone FAILED (p={failure_probability:.3f})")
            return True

        return False

    def identify_neighbor_zones(self, zones: Dict[str, 'Zone'],
                              correlation_threshold: float = 0.5):
        """Identify neighboring zones based on spatial correlation"""
        self.neighbors.clear()

        for zone_id, correlation in self.spatial_correlations.items():
            if correlation > correlation_threshold:
                self.neighbors.add(zone_id)

        logger.debug(f"[zone] {self.zone_id} neighbors (threshold={correlation_threshold}): {self.neighbors}")

    def compute_priority_score(self) -> float:
        """
        Compute priority score for resource allocation.

        Implements Equation (20):
        P_k = β_k * exp(τ_k) * |D_k^ready| / |D_k|
        """
        if not self.devices:
            return 0.0

        active_devices = [d for d in self.devices.values() if d.is_active]
        total_devices = len(self.devices)
        ready_ratio = len(active_devices) / max(total_devices, 1)

        staleness_factor = np.exp(self.staleness_counter)

        priority = self.inter_zone_weight * staleness_factor * ready_ratio
        self.priority_score = priority

        return priority

    def allocate_bandwidth(self, total_bandwidth: float,
                          zone_priorities: Dict[str, float]) -> float:
        """
        Allocate bandwidth based on zone priority.

        Implements Equation (19):
        B_k = B_total * (β_k * (1 + τ_k/τ_max)) / Σ(...)
        """
        total_priority = sum(zone_priorities.values())
        if total_priority > 0:
            self.allocated_bandwidth = total_bandwidth * (zone_priorities.get(self.zone_id, 0) / total_priority)
        else:
            self.allocated_bandwidth = 0.0

        return self.allocated_bandwidth

    def handle_device_failure(self, failed_device_id: str):
        """Handle failure of a device in the zone"""
        if failed_device_id in self.devices:
            self.devices[failed_device_id].is_active = False
            self._update_zone_statistics()
            logger.debug(f"[zone] {self.zone_id} device {failed_device_id} marked inactive")

    def promote_backup_aggregator(self) -> Optional[str]:
        """Promote a backup device to serve as zone aggregator"""
        if not self.devices:
            return None

        best_device = max(
            [d for d in self.devices.values() if d.is_active],
            key=lambda d: d.resources.compute_capacity,
            default=None
        )

        if best_device:
            return best_device.device_id

        return None

    def get_zone_info(self) -> Dict[str, Any]:
        """Get comprehensive zone information"""
        active_devices = [d for d in self.devices.values() if d.is_active]

        return {
            "zone_id": self.zone_id,
            "edge_server_id": self.edge_server_id,
            "centroid": self.centroid,
            "num_devices": len(self.devices),
            "active_devices": len(active_devices),
            "total_dataset_size": self.total_dataset_size,
            "metrics": {
                "contribution_score": self.contribution_score,
                "priority_score": self.priority_score,
                "staleness": self.staleness_counter,
                "inter_zone_weight": self.inter_zone_weight,
                "allocated_bandwidth": self.allocated_bandwidth
            },
            "resources": {
                "compute_capacity": self.compute_capacity,
                "memory_capacity": self.memory_capacity
            },
            "correlations": dict(self.spatial_correlations),
            "neighbors": list(self.neighbors),
            "status": {
                "is_operational": self.is_operational,
                "last_aggregation_time": self.aggregation_times[-1] if self.aggregation_times else 0
            }
        }

    def __len__(self):
        return len(self.devices)

    def __contains__(self, device_id: str):
        return device_id in self.devices

    def __repr__(self):
        return f"Zone(id={self.zone_id}, devices={len(self.devices)}, centroid={self.centroid})"
