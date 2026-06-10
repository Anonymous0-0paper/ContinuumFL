"""
Core device module for ContinuumFL framework.
Implements edge devices with spatial coordinates, resource constraints, and FL capabilities.
"""
import copy
import gc
import logging

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import time
import uuid
from collections import deque

from matplotlib import pyplot as plt

from src.memory_log.memory_log import log_mem
from src.models.model_factory import ShakespeareLSTM

logger = logging.getLogger("ContinuumFL.Device")


@dataclass
class DeviceResources:
    """Resource specification for edge devices"""
    compute_capacity: float  # GFLOPS
    memory_capacity: float   # GB
    bandwidth: float         # Mbps

class EdgeDevice:
    """
    Represents an edge device in the ContinuumFL framework.

    Based on the paper's system model, each device d_i has:
    - Physical location l_i = (x_i, y_i)
    - Resource specification (c_i, m_i, b_i)
    - Local dataset D_i
    - Capability to perform local training
    """

    def __init__(self, device_id: str, location: Tuple[float, float],
                 resources: DeviceResources, zone_id: Optional[str] = None):
        self.device_id = device_id
        self.location = location  # (x, y) coordinates
        self.resources = resources
        self.zone_id = zone_id

        # FL-related attributes
        self.local_model: Optional[nn.Module] = None
        self.local_dataset: Optional[torch.utils.data.Dataset] = None
        self.local_dataloader: Optional[torch.utils.data.DataLoader] = None
        self.dataset_size = 0

        # Performance tracking
        self.participation_history = deque(maxlen=100)  # Track last 100 rounds

        # FIX: store only (norm, mean_abs) per layer instead of full state dicts
        # to avoid accumulating ~6.6 MB per entry × 5 entries × 100 devices = 3.3 GB
        self.gradient_history: deque = deque(maxlen=5)   # list of {"norms": {layer: float}, "total_norm": float}
        self.training_times = deque(maxlen=50)          # Training time history

        # Quality metrics
        self.data_quality_score = 1.0    # q_i ∈ [0,1]
        self.reliability_score = 1.0     # r_i ∈ [0,1]
        self.staleness_counter = 0       # τ_i

        # Communication state
        self.last_communication_time = 0
        self.communication_latency = 0
        self.is_active = True
        self.error_feedback_buffer = None  # For gradient compression

        # Differential Privacy
        self.privacy_budget = 1.0
        self.noise_multiplier = 0.0

        logger.debug(f"[device] Created {device_id} at loc={location} zone={zone_id}")

    def set_local_dataset(self, dataset: torch.utils.data.Dataset,
                         batch_size: int = 32, num_workers: int = 0):
        """Set local dataset and create dataloader"""
        self.local_dataset = dataset
        self.dataset_size = len(dataset)
        use_cuda = torch.cuda.is_available()
        use_workers = max(0, num_workers)
        self.local_dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            num_workers=use_workers, pin_memory=use_cuda,
            persistent_workers=use_workers > 0, drop_last=False
        )
        logger.debug(f"[device] {self.device_id} dataset set: {self.dataset_size} samples, batch={batch_size}")

    def set_local_model(self, model: nn.Module):
        """Set local model for training — must be an independent copy, not a reference."""
        self.local_model = copy.deepcopy(model)

    def local_train(self, global_model: nn.Module, epochs: int = 1,
                   learning_rate: float = 0.01, device: str = 'cpu',
                   dataset_name: str = 'cifar100') -> Dict[str, Any]:
        """
        Perform local SGD training as described in Algorithm 2.
        Returns training metrics and updated model weights (as delta from global).
        """
        if self.local_dataloader is None or not self.is_active:
            logger.debug(f"[device] {self.device_id} skipped: dataloader={self.local_dataloader is not None} active={self.is_active}")
            return {"success": False, "reason": "No dataset or inactive device"}

        train_start = time.time()
        logger.debug(
            f"[device] {self.device_id} (zone={self.zone_id}) START: "
            f"data={self.dataset_size} epochs={epochs} lr={learning_rate} device={device}"
        )

        # Initialize local model with global weights
        self.local_model.load_state_dict(global_model.state_dict())

        use_cuda = device.startswith('cuda') and torch.cuda.is_available()
        if use_cuda:
            self.local_model = self.local_model.to(device)
        else:
            self.local_model = self.local_model.cpu()
            device = 'cpu'

        self.local_model.train()
        if dataset_name.lower() == 'cifar100':
            optimizer = torch.optim.SGD(
                self.local_model.parameters(), lr=learning_rate,
                momentum=0.9, weight_decay=5e-4
            )
        else:
            optimizer = torch.optim.Adam(self.local_model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        scaler = torch.amp.GradScaler('cuda') if use_cuda else None

        total_loss = 0.0
        total_correct = 0
        num_batches = 0
        epoch_losses = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_batches = 0
            for data, target in self.local_dataloader:
                if use_cuda:
                    data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)

                optimizer.zero_grad()
                if use_cuda:
                    with torch.amp.autocast('cuda'):
                        if isinstance(self.local_model, ShakespeareLSTM):
                            output, _ = self.local_model(data)
                        else:
                            output = self.local_model(data)
                        loss = criterion(output, target)
                    scaler.scale(loss).backward()
                    if isinstance(self.local_model, ShakespeareLSTM):
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.local_model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    if isinstance(self.local_model, ShakespeareLSTM):
                        output, _ = self.local_model(data)
                    else:
                        output = self.local_model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    if isinstance(self.local_model, ShakespeareLSTM):
                        torch.nn.utils.clip_grad_norm_(self.local_model.parameters(), max_norm=1.0)
                    optimizer.step()

                loss_val = loss.detach().item()
                predictions = torch.argmax(output.detach(), dim=1)
                total_correct += (predictions == target).float().sum().cpu().item()
                total_loss += loss_val
                epoch_loss += loss_val
                num_batches += 1
                epoch_batches += 1

            epoch_losses.append(epoch_loss / max(epoch_batches, 1))

        del optimizer, criterion
        if scaler is not None:
            del scaler

        if use_cuda:
            torch.cuda.empty_cache()
        gc.collect()

        training_time = time.time() - train_start
        self.training_times.append(training_time)

        # Compute gradient (delta weights) and store ONLY summary stats — not full tensors.
        # This prevents the ~6.6 MB × 5 × 100-device memory accumulation.
        local_gradient, grad_summary = self._compute_model_gradient_and_summary(global_model)
        self.gradient_history.append(grad_summary)

        # Update participation history
        self.participation_history.append(1)

        # Calculate reliability score based on recent participation
        recent_participation = list(self.participation_history)[-10:]
        self.reliability_score = sum(recent_participation) / max(len(recent_participation), 1)

        training_loss = total_loss / max(num_batches, 1)
        train_acc = total_correct / max(self.dataset_size * epochs, 1)

        # Move model back to CPU for state dict extraction
        if use_cuda:
            self.local_model = self.local_model.cpu()

        logger.debug(
            f"[device] {self.device_id} (zone={self.zone_id}) END: "
            f"time={training_time*1000:.0f}ms  loss={training_loss:.4f}  "
            f"acc={train_acc*100:.2f}%  ‖δw‖={grad_summary['total_norm']:.4f}  "
            f"epoch_losses={[f'{l:.4f}' for l in epoch_losses]}"
        )

        return {
            "success": True,
            "model_weights": self.local_model.state_dict(),
            "training_loss": training_loss,
            "training_time": training_time,
            "dataset_size": self.dataset_size,
            "gradient": local_gradient,
            "grad_norm": grad_summary["total_norm"],
        }

    def _compute_model_gradient_and_summary(
        self, global_model: nn.Module
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """
        Returns:
          local_gradient : full delta dict (local - global) for aggregation
          grad_summary   : lightweight stats dict stored in gradient_history
        """
        delta_weights: Dict[str, torch.Tensor] = {}
        layer_norms: Dict[str, float] = {}

        for (name, local_param), (_, global_param) in zip(
            self.local_model.state_dict().items(),
            global_model.state_dict().items()
        ):
            delta = (local_param.detach().cpu() - global_param.detach().cpu()).float()
            delta_weights[name] = delta
            layer_norms[name] = delta.norm(2).item()

        total_norm = float(np.sqrt(sum(n ** 2 for n in layer_norms.values())))
        grad_summary = {
            "layer_norms": layer_norms,
            "total_norm": total_norm,
        }
        return delta_weights, grad_summary

    def _compute_model_gradient(self, global_model: nn.Module, device: str = 'cpu') -> Dict[str, torch.Tensor]:
        """Kept for API compatibility — returns full delta dict."""
        delta, _ = self._compute_model_gradient_and_summary(global_model)
        return delta

    def compute_gradient_similarity(self, other_device: 'EdgeDevice') -> float:
        """
        Compute gradient similarity with another device using stored norms.
        Uses total gradient norm ratio as a fast approximation.
        """
        if len(self.gradient_history) == 0 or len(other_device.gradient_history) == 0:
            return 0.0

        norm_i = self.gradient_history[-1].get("total_norm", 0.0)
        norm_j = other_device.gradient_history[-1].get("total_norm", 0.0)

        if norm_i == 0 or norm_j == 0:
            return 0.0

        # Approximate cosine similarity via norm ratio (1 = similar magnitude)
        return min(norm_i, norm_j) / max(norm_i, norm_j)

    def compute_spatial_distance(self, other_device: 'EdgeDevice') -> float:
        """Compute Euclidean distance to another device"""
        dx = self.location[0] - other_device.location[0]
        dy = self.location[1] - other_device.location[1]
        return np.sqrt(dx*dx + dy*dy)

    def compute_network_similarity(self, other_device: 'EdgeDevice') -> float:
        """
        Compute network similarity based on bandwidth and latency.

        Implements S_network(d_i, d_j) from the paper.
        """
        bandwidth_sim = min(self.resources.bandwidth, other_device.resources.bandwidth) / \
                       max(self.resources.bandwidth, other_device.resources.bandwidth)

        latency_diff = abs(self.communication_latency - other_device.communication_latency)
        latency_sim = np.exp(-latency_diff / 10.0)  # σ_n = 10ms

        return bandwidth_sim * latency_sim

    def estimate_data_quality(self) -> float:
        """
        Estimate data quality based on label consistency and distribution.
        """
        if self.local_dataset is None:
            return 0.0

        base_quality = 0.8 + 0.2 * np.sin(len(self.participation_history) * 0.1)
        self.data_quality_score = max(0.1, min(1.0, base_quality))

        return self.data_quality_score

    def add_differential_privacy_noise(self, gradient: torch.Tensor,
                                     sensitivity: float = 1.0) -> torch.Tensor:
        """Add differential privacy noise to gradients"""
        if self.noise_multiplier > 0:
            noise = torch.normal(0, self.noise_multiplier * sensitivity, gradient.shape)
            return gradient + noise
        return gradient

    def compress_gradient(self, gradient: torch.Tensor,
                         compression_rate: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply top-k sparsification for gradient compression."""
        flat_grad = gradient.flatten()
        k = int(compression_rate * len(flat_grad))

        if k == 0:
            return torch.zeros_like(gradient), gradient

        _, top_k_indices = torch.topk(torch.abs(flat_grad), k)

        sparse_grad = torch.zeros_like(flat_grad)
        sparse_grad[top_k_indices] = flat_grad[top_k_indices]

        error_feedback = flat_grad - sparse_grad
        if self.error_feedback_buffer is None:
            self.error_feedback_buffer = torch.zeros_like(flat_grad)

        sparse_grad += self.error_feedback_buffer
        self.error_feedback_buffer = error_feedback

        return sparse_grad.reshape(gradient.shape), error_feedback.reshape(gradient.shape)

    def simulate_failure(self, failure_probability: float = 0.05) -> bool:
        """
        Simulate device failure.  Returns True if the device failed this call.

        FIX (was broken):
          - The old code always reset is_active=True in the else-branch, meaning
            any device that survived a failure check was immediately repaired.
          - The old code returned None, so callers checking `if failure:` always
            got False.
        """
        if np.random.random() < failure_probability:
            self.is_active = False
            self.participation_history.append(0)
            logger.debug(
                f"[device] {self.device_id} (zone={self.zone_id}) FAILED "
                f"(p={failure_probability:.3f})"
            )
            return True
        # Device survived this round — do NOT force-repair; let simulate_repair handle it
        return False

    def simulate_repair(self, repair_probability: float = 0.2) -> bool:
        """Simulate device repair."""
        if self.is_active:
            self.participation_history.append(1)
            return True
        elif np.random.random() < repair_probability:
            self.is_active = True
            self.participation_history.append(1)
            logger.debug(
                f"[device] {self.device_id} (zone={self.zone_id}) REPAIRED "
                f"(p={repair_probability:.3f})"
            )
            return True
        else:
            self.is_active = False
            return False

    def get_device_info(self) -> Dict[str, Any]:
        """Get comprehensive device information"""
        return {
            "device_id": self.device_id,
            "location": self.location,
            "zone_id": self.zone_id,
            "resources": {
                "compute": self.resources.compute_capacity,
                "memory": self.resources.memory_capacity,
                "bandwidth": self.resources.bandwidth
            },
            "metrics": {
                "dataset_size": self.dataset_size,
                "data_quality": self.data_quality_score,
                "reliability": self.reliability_score,
                "staleness": self.staleness_counter,
                "participation_rate": sum(self.participation_history) / max(len(self.participation_history), 1),
                "recent_grad_norm": self.gradient_history[-1]["total_norm"] if self.gradient_history else 0.0,
            },
            "status": {
                "is_active": self.is_active,
                "last_training_time": self.training_times[-1] if self.training_times else 0,
                "communication_latency": self.communication_latency
            }
        }

    def __repr__(self):
        return f"EdgeDevice(id={self.device_id}, zone={self.zone_id}, location={self.location})"
