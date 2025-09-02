"""
Zone module for ContinuumFL framework.
Implements spatial zones that aggregate edge devices with similar characteristics.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Set
from collections import defaultdict, deque
import time
from .device import EdgeDevice

class Zone:
    """
    Represents a spatial zone in the ContinuumFL framework.
    
    Based on the paper's system model, each zone z_k contains:
    - A subset of devices D_k
    - Zone aggregator (edge server)
    - Spatial correlation with neighboring zones
    - Zone-level objective F_k(w)
    """
    
    def __init__(self, zone_id: str, edge_server_id: str):
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
        
        # Initialize aggregated weights
        aggregated_weights = {}
        original_dtypes = {}  # Track original data types
        first_device_id = next(iter(device_updates.keys()))
        first_weights = device_updates[first_device_id]
        
        for param_name, param_tensor in first_weights.items():
            original_dtypes[param_name] = param_tensor.dtype
            # Ensure aggregated weights are float type for aggregation operations
            if param_tensor.dtype != torch.float32:
                aggregated_weights[param_name] = torch.zeros_like(param_tensor, dtype=torch.float32)
            else:
                aggregated_weights[param_name] = torch.zeros_like(param_tensor)
        
        # Weighted aggregation
        total_weight = 0.0
        for device_id, device_weights in device_updates.items():
            if device_id in weights and weights[device_id] > 0:
                weight = weights[device_id]
                for param_name, param_tensor in device_weights.items():
                    # Ensure tensor is float for aggregation operations
                    if param_tensor.dtype != torch.float32:
                        param_tensor = param_tensor.float()
                    aggregated_weights[param_name] += weight * param_tensor
                total_weight += weight
        
        # Normalize if needed (should already be normalized from weight computation)
        if total_weight > 0 and abs(total_weight - 1.0) > 1e-6:
            for param_name in aggregated_weights:
                aggregated_weights[param_name] = aggregated_weights[param_name] / total_weight
        
        # Convert back to original data types
        for param_name in aggregated_weights:
            if original_dtypes[param_name] != torch.float32:
                aggregated_weights[param_name] = aggregated_weights[param_name].to(original_dtypes[param_name])
        
        # Apply gradient compression (temporarily disabled) TODO Fix Compression and replace Weights by Delta-Weights
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
            
            # Create sparse tensor
            sparse_tensor = torch.zeros_like(flat_tensor)
            sparse_tensor[top_k_indices] = flat_tensor[top_k_indices]
            
            compressed_weights[param_name] = sparse_tensor.reshape(param_tensor.shape)
        
        return compressed_weights
    
    def compute_zone_contribution_score(self) -> float:
        """
        Compute zone contribution score for inter-zone aggregation.
        
        Implements Equation (14):
        Score_k = |D_k| * n̄_k * exp(-Var({∇F_i})) * (1 - L_k^val)
        """
        if not self.devices:
            return 0.0
        
        # Data contribution component
        num_devices = len([d for d in self.devices.values() if d.is_active])
        avg_dataset_size = self.total_dataset_size / max(num_devices, 1)
        data_contribution = num_devices * avg_dataset_size
        print(f"({self.zone_id}) Num Devices {num_devices}, avg dataset size {avg_dataset_size:.2f}")
        # Gradient consistency component
        gradient_variances = []
        for device in self.devices.values():
            if device.is_active and device.gradient_history:
                grad_dict = device.gradient_history[-1]
                grad_vector = torch.cat([v.flatten() for v in grad_dict.values()])
                grad_norm = torch.norm(grad_vector).item()
                gradient_variances.append(grad_norm)
        
        if gradient_variances:
            gradient_variance = np.var(gradient_variances)
            consistency_score = np.exp(-gradient_variance)
        else:
            consistency_score = 1.0
        
        # Validation accuracy component (placeholder)
        # In practice, this would be computed on a validation set
        validation_accuracy = 0.8 + 0.1 * np.random.random()  # Placeholder
        accuracy_component = validation_accuracy
        
        score = data_contribution * consistency_score * accuracy_component
        self.contribution_score = score
        
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