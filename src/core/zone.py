"""
Zone module for ContinuumFL framework.
Implements spatial zones that aggregate edge devices with similar characteristics.
"""
import os

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Set
from collections import defaultdict, deque
import time

from torch import Tensor

from .device import EdgeDevice

# ThreadFunc
def run_training(args: dict[str, Any]):
    device = args["device"]
    global_model = args["model"]
    epochs = args["epochs"]
    learning_rate = args["learning_rate"]
    comp_device = args["comp_device"]

    # active device fails
    if device.is_active and device.simulate_failure():
        return None

    # device active or failed device is repaired
    if device.is_active or device.simulate_repair():
        return device.local_train(
            global_model,
            epochs=epochs,
            learning_rate=learning_rate,
            device=comp_device), device.device_id

    # device failed and was not repaired
    return None

class Zone:
    """
    Represents a spatial zone in the ContinuumFL framework.
    
    Based on the paper's system model, each zone z_k contains:
    - A subset of devices D_k
    - Zone aggregator (edge server)
    - Spatial correlation with neighboring zones
    - Zone-level objective F_k(w)
    """
    
    def __init__(self, zone_id: str, edge_server_id: str, compression_rate: float):
        self.zone_id = zone_id
        self.edge_server_id = edge_server_id
        
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
        
        # For memory pooling
        self._tensor_cache = {}
    
    def add_device(self, device: EdgeDevice):
        """Add a device to this zone"""
        self.devices[device.device_id] = device
        self.device_ids.add(device.device_id)
        device.zone_id = self.zone_id
        
        # Update zone statistics
        self._update_zone_statistics()
    
    def remove_device(self, device_id: str) -> bool:
        """Remove a device from this zone"""
        if device_id in self.devices:
            del self.devices[device_id]
            self.device_ids.discard(device_id)
            self._update_zone_statistics()
            return True
        return False
    
    def _update_zone_statistics(self):
        """Update zone-level statistics when devices are added/removed"""
        if not self.devices:
            return
        
        # Update centroid
        locations = [device.location for device in self.devices.values()]
        self.centroid = (np.mean([loc[0] for loc in locations]), np.mean([loc[1] for loc in locations]))
        
        # Update total dataset size
        self.total_dataset_size = sum(device.dataset_size for device in self.devices.values())
        
        # Update resource capacity
        self.compute_capacity = sum(device.resources.compute_capacity for device in self.devices.values())
        self.memory_capacity = sum(device.resources.memory_capacity for device in self.devices.values())

    def _sample_participating_devices(self) -> List[str]:
        """Sample devices for participation in current round"""
        # Simple participation strategy: random sampling with availability check
        available_devices = [
            device_id for device_id, device in self.devices.items()
            if device.is_active and device.local_dataset
        ]

        if available_devices:
            # Sample fraction of available devices
            participation_rate = 0.7  # 70% participation rate
            num_participants = max(1, int(participation_rate * len(available_devices)))

            participating = np.random.choice(
                available_devices, size=num_participants, replace=False
            ).tolist()

            return participating

        return []

    def perform_local_training(self, args) -> tuple[str, dict[str, Tensor], dict[str, list[str] | int]]:
        """Perform local training on participating devices"""

        participating_devices = self._sample_participating_devices()
        comp_device = args["comp_device"]
        global_model = args["model"]
        learning_rate = args["learning_rate"]
        epochs = args["epochs"]
        device_participation = args["device_participation"]
        device_updates = {}

        from concurrent.futures import ThreadPoolExecutor, as_completed
        if comp_device == 'cuda':
            max_workers = len(participating_devices)
        else:
            max_workers = min(len(participating_devices), os.cpu_count())

        device_args = [
            {"device": self.devices[device_id], "model": global_model, "epochs": epochs,
             "learning_rate": learning_rate,
             "comp_device": comp_device} for device_id in participating_devices]
        with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
            futures = [executor.submit(run_training, args) for args in device_args]

            for f in as_completed(futures):
                res = f.result()
                if res is None:
                    continue

                res_dict, device_id = res
                if res_dict["success"]:
                    device_updates[device_id] = res_dict["gradient"]
                    device_participation[device_id] += 1

                    # Update device reliability based on participation
                    self.devices[device_id].participation_history.append(1)
        intra_start = time.time()
        aggregated_weights = self.intra_zone_aggregation(device_updates)
        intra_time = time.time() - intra_start

        comm_cost = self.estimate_communication_cost(device_updates, {self.zone_id: aggregated_weights})
        return self.zone_id, aggregated_weights, {"participating_devices": participating_devices,
                                                  "num_device_updates": len(device_updates),
                                                  "intra_time": intra_time,
                                                  "communication_cost": comm_cost}

    def estimate_communication_cost(self, device_updates: Dict[str, Dict[str, torch.Tensor]],
                                    zone_weights: Dict[str, Dict[str, torch.Tensor]]) -> float:
        """Estimate communication cost in MB for this round"""
        total_cost = 0.0

        # Device to zone communication (uplink)
        for device_id, weights in device_updates.items():
            device_size = sum(param.numel() * 4 for param in weights.values()) / (1024 * 1024)  # 4 bytes per float32
            compressed_size = device_size * self.compression_rate  # Apply compression
            total_cost += compressed_size

        # Zone to cloud communication (inter-zone)
        for zone_id, weights in zone_weights.items():
            zone_size = sum(param.numel() * 4 for param in weights.values()) / (1024 * 1024)
            # Apply delta encoding (assume 50% reduction)
            delta_encoded_size = zone_size * 0.5
            total_cost += delta_encoded_size * 2  # Bidirectional

        return total_cost

    def compute_intra_zone_weights(self) -> Dict[str, float]:
        """
        Compute aggregation weights for devices within the zone.
        
        Implements Equation (10) from the paper:
        α_i^k = (n_i * q_i * r_i) / Σ(n_j * q_j * r_j)
        """
        if not self.devices:
            return {}
        
        weights = {}
        total_weight = 0.0
        
        for device_id, device in self.devices.items():
            if device.is_active:
                # Update device quality metrics
                device.estimate_data_quality()
                
                # Compute weight components
                dataset_size = device.dataset_size
                quality_score = device.data_quality_score
                reliability_score = device.reliability_score
                
                weight = dataset_size * quality_score * reliability_score
                weights[device_id] = weight
                total_weight += weight
        
        # Normalize weights
        if total_weight > 0:
            for device_id in weights:
                weights[device_id] /= total_weight
        
        self.intra_zone_weights = weights
        return weights
    
    def intra_zone_aggregation(self, device_updates: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Perform intra-zone aggregation of device updates.
        
        Implements Equation (9):
        w_k^(t+1) = Σ α_i^k * w_i^(t)
        """
        start_time = time.time()
        
        # Compute aggregation weights
        weights = self.compute_intra_zone_weights()
        
        if not weights or not device_updates:
            return {}
        
        # Initialize aggregated weights with memory pooling
        aggregated_weights = {}
        original_dtypes = {}  # Track original data types
        first_device_id = next(iter(device_updates.keys()))
        first_weights = device_updates[first_device_id]
        
        for param_name, param_tensor in first_weights.items():
            original_dtypes[param_name] = param_tensor.dtype
            # Ensure aggregated weights are float type for aggregation operations
            if param_tensor.dtype != torch.float32:
                # Use tensor cache to reduce allocations
                cache_key = f"aggregated_{param_name}_{param_tensor.shape}"
                if cache_key in self._tensor_cache:
                    aggregated_weights[param_name] = self._tensor_cache[cache_key]
                    aggregated_weights[param_name].zero_()
                else:
                    aggregated_tensor = torch.zeros_like(param_tensor, dtype=torch.float32)
                    self._tensor_cache[cache_key] = aggregated_tensor
                    aggregated_weights[param_name] = aggregated_tensor
            else:
                cache_key = f"aggregated_{param_name}_{param_tensor.shape}"
                if cache_key in self._tensor_cache:
                    aggregated_weights[param_name] = self._tensor_cache[cache_key]
                    aggregated_weights[param_name].zero_()
                else:
                    aggregated_tensor = torch.zeros_like(param_tensor)
                    self._tensor_cache[cache_key] = aggregated_tensor
                    aggregated_weights[param_name] = aggregated_tensor
        
        # Weighted aggregation with in-place operations
        total_weight = 0.0
        for device_id, device_weights in device_updates.items():
            if device_id in weights and weights[device_id] > 0:
                weight = weights[device_id]
                for param_name, param_tensor in device_weights.items():
                    # Ensure tensor is float for aggregation operations
                    if param_tensor.dtype != torch.float32:
                        param_tensor = param_tensor.float()
                    # Use in-place operations to reduce memory fragmentation
                    aggregated_weights[param_name].add_(param_tensor, alpha=weight)
                total_weight += weight
        
        # Normalize if needed using in-place operations
        if total_weight > 0 and abs(total_weight - 1.0) > 1e-6:
            for param_name in aggregated_weights:
                aggregated_weights[param_name].div_(total_weight)
        
        # Convert back to original data types with in-place operations where possible
        for param_name in aggregated_weights:
            if original_dtypes[param_name] != torch.float32:
                # For type conversion, we need to create a new tensor
                converted_tensor = aggregated_weights[param_name].to(original_dtypes[param_name])
                aggregated_weights[param_name] = converted_tensor
        
        # Apply gradient compression (temporarily disabled)
        # compressed_weights = self._apply_gradient_compression(aggregated_weights)
        
        self.aggregated_weights = aggregated_weights # TODO use compressed weights
        
        # Track aggregation time
        aggregation_time = time.time() - start_time
        self.aggregation_times.append(aggregation_time)
        
        return aggregated_weights # TODO return compressed weights
    
    def _apply_gradient_compression(self, weights: Dict[str, torch.Tensor], 
                                  compression_rate: float = 0.1) -> Dict[str, torch.Tensor]:
        """Apply top-k compression to aggregated weights"""
        compressed_weights = {}
        
        for param_name, param_tensor in weights.items():
            flat_tensor = param_tensor.flatten()
            k = max(1, int(compression_rate * len(flat_tensor)))
            
            # Get top-k elements by magnitude
            _, top_k_indices = torch.topk(torch.abs(flat_tensor), k)
            
            # Create sparse tensor using in-place operations where possible
            cache_key = f"sparse_{param_name}_{flat_tensor.shape}"
            if cache_key in self._tensor_cache:
                sparse_tensor = self._tensor_cache[cache_key]
                sparse_tensor.zero_()
            else:
                sparse_tensor = torch.zeros_like(flat_tensor)
                self._tensor_cache[cache_key] = sparse_tensor
            
            # Use in-place scatter operation
            sparse_tensor.scatter_(0, top_k_indices, flat_tensor[top_k_indices])
            
            compressed_weights[param_name] = sparse_tensor.reshape(param_tensor.shape)
        
        return compressed_weights

    def compute_spatial_correlation(self, other_zone: 'Zone') -> float:
        """
        Compute spatial correlation with another zone.
        
        Implements Equation (1):
        ρ(z_k, z_j) = exp(-Σ dist(d_i, d_j) / (|D_k| * |D_j| * σ))
        """
        if not self.devices or not other_zone.devices:
            return 0.0
        
        # Use GPU acceleration if available
        use_gpu = torch.cuda.is_available()
        
        if use_gpu:
            try:
                # Convert locations to GPU tensors for parallel computation
                locations_i = torch.tensor([[d.location[0], d.location[1]] for d in self.devices.values()], 
                                         dtype=torch.float32, device='cuda')
                locations_j = torch.tensor([[d.location[0], d.location[1]] for d in other_zone.devices.values()], 
                                         dtype=torch.float32, device='cuda')
                
                # Compute pairwise distances in parallel
                diff = locations_i.unsqueeze(1) - locations_j.unsqueeze(0)
                distances = torch.norm(diff, dim=2)
                total_distance = torch.sum(distances).cpu().item()
                count = distances.numel()
                
                if count == 0:
                    return 0.0
                
                avg_distance = total_distance / count
                sigma = 10.0  # Distance scaling parameter
                
                correlation = np.exp(-avg_distance / sigma)
                return correlation
            except Exception:
                # Fallback to CPU computation if GPU fails
                pass
        
        # Fallback to CPU computation
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
        sigma = 10.0  # Distance scaling parameter
        
        correlation = np.exp(-avg_distance / sigma)
        return correlation
    
    def update_spatial_correlations(self, zones: Dict[str, 'Zone']):
        """Update spatial correlations with all other zones"""
        for other_zone_id, other_zone in zones.items():
            if other_zone_id != self.zone_id:
                correlation = self.compute_spatial_correlation(other_zone)
                self.spatial_correlations[other_zone_id] = correlation
    
    def identify_neighbor_zones(self, zones: Dict[str, 'Zone'], 
                              correlation_threshold: float = 0.5):
        """Identify neighboring zones based on spatial correlation"""
        self.neighbors.clear()
        
        for zone_id, correlation in self.spatial_correlations.items():
            if correlation > correlation_threshold:
                self.neighbors.add(zone_id)
    
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
    
    def compute_zone_contribution_score(self) -> float:
        """
        Compute zone contribution score for aggregation weighting.
        
        Implements Equation (14):
        Score_k = |D_k|·n̄_k · exp(-Var({∇F_i})) · (1 - L_k^val)
        """
        if not self.devices:
            return 0.0
        
        #|D_k| - number of devices in zone
        num_devices = len(self.devices)
        
        # n̄_k - average dataset size per device
        avg_dataset_size = self.total_dataset_size / max(num_devices, 1)
        
        # exp(-Var({∇F_i})) - gradient consistency (inverse of variance)
        if len(self.devices) > 1:
            # Compute gradient variance across devices
            gradients = []
            for device in self.devices.values():
                if device.gradient_history:
                    gradients.append(device.gradient_history[-1])  # Most recent gradient
            
            if len(gradients) > 1:
                # Compute variance of gradients
                gradient_var = 0.0
                for param_name in gradients[0].keys():
                    param_gradients = [g[param_name] for g in gradients if param_name in g]
                    if param_gradients:
                        #Flatten all gradients for this parameter
                        flat_gradients = [g.flatten() for g in param_gradients]
                        # Compute variance across devices for this parameter
                        stacked = torch.stack(flat_gradients)
                        param_var = torch.var(stacked, dim=0).mean().item()
                        gradient_var += param_var
                
                gradient_consistency = np.exp(-gradient_var)
            else:
                gradient_consistency = 1.0
        else:
            gradient_consistency = 1.0
        
        # (1 - L_k^val) - validation accuracy factor (simplified as 1.0 for now)
        validation_factor = 1.0
        
        # Compute final score with smoothing to prevent extreme variations
        score = num_devices * avg_dataset_size * gradient_consistency * validation_factor
        
        # Apply smoothing to prevent extreme score variations
        smoothing_factor = 0.1
        self.contribution_score = (1 - smoothing_factor) * self.contribution_score + smoothing_factor * score if hasattr(self, 'contribution_score') else score
        
        return self.contribution_score
    
    def handle_device_failure(self, failed_device_id: str):
        """Handle failure of a device in the zone"""
        if failed_device_id in self.devices:
            self.devices[failed_device_id].is_active = False
            # Update zone statistics
            self._update_zone_statistics()
    
    def promote_backup_aggregator(self) -> Optional[str]:
        """Promote a backup device to serve as zone aggregator"""
        if not self.devices:
            return None
        
        # Find device with highest compute capacity
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
        """Return number of devices in zone"""
        return len(self.devices)
    
    def __contains__(self, device_id: str):
        """Check if device is in zone"""
        return device_id in self.devices
    
    def __repr__(self):
        return f"Zone(id={self.zone_id}, devices={len(self.devices)}, centroid={self.centroid})"