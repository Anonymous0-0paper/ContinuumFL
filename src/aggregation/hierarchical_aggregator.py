"""
Hierarchical Aggregation module for ContinuumFL framework.
Implements the two-tier aggregation protocol from Section 4.2 of the paper.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, deque
import time
import copy
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading

from torch import Tensor

from ..core.zone import Zone
from ..core.device import EdgeDevice

class HierarchicalAggregator:
    """
    Implements the hierarchical aggregation protocol from Algorithm 2.
    
    Performs two-level aggregation:
    1. Intra-zone aggregation within each zone
    2. Inter-zone aggregation across zones with spatial awareness
    """
    
    def __init__(self, config):
        self.config = config
        
        # Aggregation parameters
        self.spatial_regularization = config.spatial_regularization  # λ
        self.momentum_eta = config.momentum_eta                      # η for correlation update
        self.staleness_penalty = config.staleness_penalty           # μ
        self.max_staleness = config.max_staleness                   # τ_max
        self.fairness_strength = config.fairness_strength           # α_fair
        
        # Communication optimization
        self.compression_rate = config.compression_rate              # κ
        
        # Global model state
        self.global_model: Optional[nn.Module] = None
        self.global_weights: Optional[Dict[str, torch.Tensor]] = None
        
        # Zone weights and correlations
        self.zone_base_weights: Dict[str, float] = {}      # β_k^base
        self.zone_fair_weights: Dict[str, float] = {}      # β_k^fair
        self.spatial_correlations: Dict[Tuple[str, str], float] = {}  # ρ(z_k, z_j)
        
        # Performance tracking
        self.aggregation_history = deque(maxlen=100)
        self.convergence_metrics = deque(maxlen=100)
        self.communication_costs = deque(maxlen=100)
        
        # Round state
        self.current_round = 0
        self.zone_staleness: Dict[str, int] = defaultdict(int)
        
        # Async processing
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.lock = threading.Lock()
        
        # Tensor cache for memory pooling
        self._tensor_cache = {}
    
    def set_global_model(self, model: nn.Module):
        """Initialize global model"""
        self.global_model = copy.deepcopy(model)
        self.global_weights = model.state_dict()
    
    def compute_zone_contribution_scores(self, zones: Dict[str, Zone]) -> Dict[str, float]:
        """
        Compute zone contribution scores for aggregation weighting.
        
        Implements Equation (14):
        Score_k = |D_k|·n̄_k · exp(-Var({∇F_i})) · (1 - L_k^val)
        """
        scores = {}
        
        for zone_id, zone in zones.items():
            if not zone.devices or not zone.is_operational:
                scores[zone_id] = 0.0
                continue
            
            score = zone.compute_zone_contribution_score()
            scores[zone_id] = max(0.0, score)  # Ensure non-negative
        
        # Normalize scores
        total_score = sum(scores.values())
        if total_score > 0:
            for zone_id in scores:
                scores[zone_id] /= total_score
        else:
            # Equal weights if no valid scores
            uniform_weight = 1.0 / max(len(zones), 1)
            scores = {zone_id: uniform_weight for zone_id in zones}
        
        return scores
    
    def compute_zone_base_weights(self, zones: Dict[str, Zone]) -> Dict[str, float]:
        """
        Compute base aggregation weights for zones.
        
        Implements Equation (13): β_k^base = Score_k / Σ Score_j
        """
        contribution_scores = self.compute_zone_contribution_scores(zones)
        print(f"Contribution Scores: {contribution_scores}")
        self.zone_base_weights = contribution_scores.copy()
        return self.zone_base_weights
    
    def apply_fairness_adjustment(self, base_weights: Dict[str, float]) -> Dict[str, float]:
        """
        Apply fairness adjustment to zone weights.
        
        Implements Equation (15):
        β_k^fair = β_k^base · (1 + α_fair · ((1/K - β_k^base) / (1/K)))
        """
        if not base_weights:
            return {}
        
        num_zones = len(base_weights)
        uniform_weight = 1.0 / num_zones
        fair_weights = {}
        
        for zone_id, base_weight in base_weights.items():
            # Compute deviation from uniform distribution
            deviation_ratio = (uniform_weight - base_weight) / uniform_weight
            
            # Apply fairness adjustment
            adjustment = 1.0 + self.fairness_strength * deviation_ratio
            fair_weight = base_weight * adjustment
            
            fair_weights[zone_id] = max(0.01, fair_weight)  # Ensure minimum weight
        
        # Renormalize to ensure weights sum to 1
        total_weight = sum(fair_weights.values())
        if total_weight > 0:
            for zone_id in fair_weights:
                fair_weights[zone_id] /= total_weight
        
        self.zone_fair_weights = fair_weights
        return fair_weights
    
    def apply_staleness_penalty(self, fair_weights: Dict[str, float]) -> Dict[str, float]:
        """
        Apply staleness penalty to zone weights.
        
        Implements Equation (12):
        β_k^(t) = β_k^base · exp(-μ · τ_k)
        """
        adjusted_weights = {}
        
        for zone_id, fair_weight in fair_weights.items():
            staleness = self.zone_staleness.get(zone_id, 0)
            
            # Exclude zones that are too stale
            if staleness > self.max_staleness:
                adjusted_weights[zone_id] = 0.0
            else:
                staleness_factor = np.exp(-self.staleness_penalty * staleness)
                adjusted_weights[zone_id] = max(0.01, fair_weight * staleness_factor)  # Ensure minimum weight
        print(f"Adjusted Weights: {adjusted_weights}")
        # Renormalize weights to ensure minimum participation
        total_weight = sum(adjusted_weights.values())
        if total_weight > 0:
            for zone_id in adjusted_weights:
                adjusted_weights[zone_id] /= total_weight
        else:
            # If all weights are zero, assign uniform weights
            uniform_weight = 1.0 / len(adjusted_weights)
            for zone_id in adjusted_weights:
                adjusted_weights[zone_id] = uniform_weight
        
        return adjusted_weights
    
    def update_spatial_correlations(self, zones: Dict[str, Zone]):
        """
        Update spatial correlation matrix based on model similarity.
        
        Implements Equation (16):
        ρ^(t+1)(z_k, z_j) = η·ρ^(t) + (1-η)·cos(w_k^(t) - w^(t), w_j^(t) - w^(t))
        """
        if not self.global_weights:
            return
        
        # Compute model deviations from global model
        zone_deviations = {}
        for zone_id, zone in zones.items():
            if zone.aggregated_weights and zone.is_operational:
                # Compute deviation from global model
                deviation = {}
                for param_name, global_param in self.global_weights.items():
                    if param_name in zone.aggregated_weights:
                        zone_param = zone.aggregated_weights[param_name]
                        # Use tensor cache for memory efficiency
                        cache_key = f"deviation_{zone_id}_{param_name}"
                        if cache_key in self._tensor_cache:
                            dev_tensor = self._tensor_cache[cache_key]
                            torch.sub(zone_param, global_param.cpu(), out=dev_tensor)
                            deviation[param_name] = dev_tensor
                        else:
                            dev_tensor = zone_param - global_param.cpu()
                            self._tensor_cache[cache_key] = dev_tensor
                            deviation[param_name] = dev_tensor
                    else:
                        # Use tensor cache
                        cache_key = f"zero_{param_name}_{global_param.shape}"
                        if cache_key in self._tensor_cache:
                            deviation[param_name] = self._tensor_cache[cache_key]
                        else:
                            zero_tensor = torch.zeros_like(global_param)
                            self._tensor_cache[cache_key] = zero_tensor
                            deviation[param_name] = zero_tensor
                
                zone_deviations[zone_id] = deviation
        
        # Update pairwise correlations with parallel processing
        zone_ids = list(zone_deviations.keys())
        
        # Use ThreadPoolExecutor for parallel computation
        futures = []
        results = {}
        
        def compute_correlation(i, j):
            zone_i, zone_j = zone_ids[i], zone_ids[j]
            if i <= j:  # Symmetric matrix
                correlation_key = (zone_i, zone_j) if zone_i <= zone_j else (zone_j, zone_i)
                
                if i == j:
                    # Self-correlation
                    new_correlation = 1.0
                else:
                    # Compute cosine similarity between deviations
                    try:
                        # Flatten and concatenate all parameter deviations
                        dev_i_list = [zone_deviations[zone_i][name].flatten() for name in sorted(zone_deviations[zone_i].keys())]
                        dev_j_list = [zone_deviations[zone_j][name].flatten() for name in sorted(zone_deviations[zone_j].keys())]
                        
                        # Use tensor cache for concatenated tensors
                        cache_key_i = f"concat_{zone_i}"
                        cache_key_j = f"concat_{zone_j}"
                        
                        if cache_key_i in self._tensor_cache:
                            dev_i = self._tensor_cache[cache_key_i]
                        else:
                            dev_i = torch.cat(dev_i_list)
                            self._tensor_cache[cache_key_i] = dev_i
                            
                        if cache_key_j in self._tensor_cache:
                            dev_j = self._tensor_cache[cache_key_j]
                        else:
                            dev_j = torch.cat(dev_j_list)
                            self._tensor_cache[cache_key_j] = dev_j
                        
                        # Compute cosine similarity
                        cosine_sim = torch.cosine_similarity(dev_i.unsqueeze(0), dev_j.unsqueeze(0)).item()
                        new_correlation = cosine_sim
                    except Exception:
                        # Fallback to CPU computation
                        dev_i = torch.cat([param.flatten() for param in zone_deviations[zone_i].values()])
                        dev_j = torch.cat([param.flatten() for param in zone_deviations[zone_j].values()])
                        cosine_sim = torch.cosine_similarity(dev_i.unsqueeze(0), dev_j.unsqueeze(0)).item()
                        new_correlation = cosine_sim
                
                # Apply momentum update
                old_correlation = self.spatial_correlations.get(correlation_key, 0.0)
                updated_correlation = (self.momentum_eta * old_correlation + 
                                     (1 - self.momentum_eta) * new_correlation)
                
                return correlation_key, updated_correlation
            return None, None
        
        # Submit tasks to thread pool
        for i, zone_i in enumerate(zone_ids):
            for j, zone_j in enumerate(zone_ids):
                future = self.executor.submit(compute_correlation, i, j)
                futures.append((future, i, j))
        
        # Collect results
        for future, i, j in futures:
            try:
                correlation_key, updated_correlation = future.result()
                if correlation_key is not None:
                    self.spatial_correlations[correlation_key] = updated_correlation
            except Exception:
                # Handle any exceptions in parallel computation
                pass
    
    def compute_spatial_regularization_term(self, zone_weights: Dict[str, Dict[str, torch.Tensor]],
                                          zones: Dict[str, Zone]) -> torch.Tensor:
        """
        Compute spatial regularization term for inter-zone aggregation.
        
        Implements the second term in Equation (11):
        λ · Σ Σ ρ(z_k, z_j) ||w_k - w_j||²
        """
        if not zone_weights or len(zone_weights) < 2:
            return torch.tensor(0.0)
        
        # Use tensor cache for regularization tensor
        cache_key = "regularization"
        if cache_key in self._tensor_cache:
            regularization = self._tensor_cache[cache_key]
            regularization.zero_()
        else:
            regularization = torch.tensor(0.0)
            self._tensor_cache[cache_key] = regularization
        
        zone_ids = list(zone_weights.keys())
        
        # Process in parallel where possible
        for i, zone_i in enumerate(zone_ids):
            zone_i_obj = zones.get(zone_i)
            if not zone_i_obj or zone_i not in zone_weights:
                continue
            
            # Get spatial neighbors
            for zone_j in zone_i_obj.neighbors:
                if zone_j in zone_weights and zone_j in zones:
                    correlation_key = (zone_i, zone_j) if zone_i <= zone_j else (zone_j, zone_i)
                    correlation = self.spatial_correlations.get(correlation_key, 0.0)
                    
                    if correlation > 0:
                        # Compute model difference with in-place operations
                        diff_squared = torch.tensor(0.0)
                        
                        for param_name in zone_weights[zone_i]:
                            if param_name in zone_weights[zone_j]:
                                # Use tensor cache for difference computation
                                param_diff = zone_weights[zone_i][param_name] - zone_weights[zone_j][param_name]
                                diff_squared += torch.sum(param_diff ** 2)
                        
                        regularization += correlation * diff_squared
        
        result = self.spatial_regularization * regularization
        return result
    
    async def async_inter_zone_aggregation(self, global_weights, zone_weights: Dict[str, Dict[str, torch.Tensor]],
                                         zones: Dict[str, Zone]) -> Dict[str, torch.Tensor]:
        """
        Asynchronous version of inter-zone aggregation with better resource utilization.
        """
        # Run the computation in a thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor,
            self.inter_zone_aggregation,
            global_weights,
            zone_weights,
            zones
        )
        return result
    
    def inter_zone_aggregation(self, global_weights, zone_weights: Dict[str, Dict[str, torch.Tensor]],
                             zones: Dict[str, Zone]) -> Dict[str, torch.Tensor]:
        """
        Perform inter-zone aggregation with spatial awareness.
        
        Implements Phase 3 of Algorithm 2 and Equation (11).
        """
        if not zone_weights:
            return self.global_weights or {}
        
        # Compute zone aggregation weights
        base_weights = self.compute_zone_base_weights(zones)
        fair_weights = self.apply_fairness_adjustment(base_weights)
        final_weights = self.apply_staleness_penalty(fair_weights)
        # Update spatial correlations
        self.update_spatial_correlations(zones)
        
        # Initialize global aggregated weights with memory pooling
        first_zone_weights = None
        for zone_id in sorted(zone_weights.keys()):
            if zone_weights[zone_id]:
                first_zone_weights = zone_weights[zone_id]
                break
        if first_zone_weights is None:
            return self.global_weights or {}

        global_aggregated = {}
        original_dtypes = {}  # Track original data types
        
        for param_name, param_tensor in first_zone_weights.items():
            original_dtypes[param_name] = param_tensor.dtype
            # Ensure aggregated weights are float type for aggregation operations
            cache_key = f"global_agg_{param_name}_{param_tensor.shape}"
            if param_tensor.dtype != torch.float32:
                if cache_key in self._tensor_cache:
                    global_aggregated[param_name] = self._tensor_cache[cache_key]
                    global_aggregated[param_name].zero_()
                else:
                    aggregated_tensor = torch.zeros_like(param_tensor, dtype=torch.float32)
                    self._tensor_cache[cache_key] = aggregated_tensor
                    global_aggregated[param_name] = aggregated_tensor
            else:
                if cache_key in self._tensor_cache:
                    global_aggregated[param_name] = self._tensor_cache[cache_key]
                    global_aggregated[param_name].zero_()
                else:
                    aggregated_tensor = torch.zeros_like(param_tensor)
                    self._tensor_cache[cache_key] = aggregated_tensor
                    global_aggregated[param_name] = aggregated_tensor
        
        # Weighted aggregation across zones with in-place operations
        total_weight = 0.0
        for zone_id, zone_weight_dict in zone_weights.items():
            if zone_id in final_weights and final_weights[zone_id] > 0:
                weight = final_weights[zone_id]
                
                for param_name, param_tensor in zone_weight_dict.items():
                    if param_name in global_aggregated:
                        # Ensure tensor is float for aggregation operations
                        if param_tensor.dtype != torch.float32:
                            param_tensor = param_tensor.float()
                        # Use in-place operations to reduce memory fragmentation
                        global_aggregated[param_name].add_(param_tensor, alpha=weight)
                
                total_weight += weight
        
        # Normalize if needed using in-place operations
        if total_weight > 0 and abs(total_weight - 1.0) > 1e-6:
            for param_name in global_aggregated:
                global_aggregated[param_name].div_(total_weight)
        
        # Convert back to original data types with in-place operations where possible
        for param_name in global_aggregated:
            if original_dtypes[param_name] != torch.float32:
                # For type conversion, we need to create a new tensor
                converted_tensor = global_aggregated[param_name].to(original_dtypes[param_name])
                global_aggregated[param_name] = converted_tensor
        
        # Apply spatial regularization (conceptually - in practice, this influences convergence)
        # The regularization term is computed for monitoring but not directly added to weights
        reg_term = self.compute_spatial_regularization_term(zone_weights, zones)

        # Update global weights with in-place operations where possible
        with self.lock:  # Thread-safe update
            for k in self.global_weights.keys():
                if self.global_weights[k].dtype.is_floating_point:
                    # Ensure both tensors are on the same device before operation
                    aggregated_tensor = global_aggregated[k]
                    if self.global_weights[k].device != aggregated_tensor.device:
                        aggregated_tensor = aggregated_tensor.to(self.global_weights[k].device)
                    # Use in-place addition to reduce memory allocations
                    self.global_weights[k].add_(aggregated_tensor)

        return global_aggregated
    
    def intra_zone_aggregation(self, zones: Dict[str, Zone], 
                             device_updates: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Perform intra-zone aggregation for all zones.
        
        Implements Phase 2 of Algorithm 2.
        """
        zone_aggregated_weights = {}
        
        for zone_id, zone in zones.items():
            if not zone.devices or not zone.is_operational:
                continue
            
            # Collect updates from devices in this zone
            zone_device_updates = {}
            for device_id in zone.device_ids:
                if device_id in device_updates:
                    zone_device_updates[device_id] = device_updates[device_id]
            
            if zone_device_updates:
                # Perform zone-level aggregation
                aggregated = zone.intra_zone_aggregation(zone_device_updates)
                if aggregated:
                    zone_aggregated_weights[zone_id] = aggregated
        
        return zone_aggregated_weights
    
    def federated_aggregation_round(self, zones: Dict[str, Zone], 
                                  num_device_updates: int, participating_zones: List[str], total_time, intra_time, inter_time, comm_cost) -> Dict[str, Any]:
        """
        Perform complete federated aggregation round.
        
        Implements Algorithm 2: ContinuumFL Aggregation Protocol.
        """
        round_stats = {"round": self.current_round, "participating_zones": len(participating_zones),
                       "participating_devices": num_device_updates, "aggregation_time": 0.0, "communication_cost": 0.0}
        if len(participating_zones) == 0:
            return round_stats
        
        # Update zone staleness
        for zone_id in zones:
            if zone_id in participating_zones:
                self.zone_staleness[zone_id] = 0  # Reset staleness
            else:
                self.zone_staleness[zone_id] += 1  # Increment staleness

        round_stats["aggregation_time"] = total_time
        round_stats["intra_zone_time"] = intra_time
        round_stats["inter_zone_time"] = inter_time
        
        # Estimate communication cost
        round_stats["communication_cost"] = comm_cost
        self.communication_costs.append(comm_cost)
        
        # Store aggregation history
        self.aggregation_history.append({
            "round": self.current_round,
            "zone_weights": self.zone_fair_weights.copy(),
            "spatial_correlations": dict(self.spatial_correlations),
            "participating_zones": participating_zones,
            "staleness": dict(self.zone_staleness)
        })
        
        self.current_round += 1
        
        return round_stats
    
    def get_aggregation_stats(self) -> Dict[str, Any]:
        """Get comprehensive aggregation statistics"""
        if not self.aggregation_history:
            return {}
        
        recent_history = list(self.aggregation_history)[-10:]  # Last 10 rounds
        
        # Zone participation statistics
        zone_participation = defaultdict(int)
        total_rounds = len(recent_history)
        
        for round_data in recent_history:
            for zone_id, weight in round_data["zone_weights"].items():
                if weight > 0:
                    zone_participation[zone_id] += 1
        
        avg_participation = {zone_id: count / total_rounds 
                           for zone_id, count in zone_participation.items()}
        
        # Communication efficiency
        recent_comm_costs = list(self.communication_costs)[-10:]
        avg_comm_cost = np.mean(recent_comm_costs) if recent_comm_costs else 0.0
        
        # Spatial correlation statistics
        if self.spatial_correlations:
            avg_correlation = np.mean(list(self.spatial_correlations.values()))
            max_correlation = np.max(list(self.spatial_correlations.values()))
        else:
            avg_correlation = max_correlation = 0.0
        
        return {
            "total_rounds": self.current_round,
            "zone_participation_rates": dict(avg_participation),
            "average_communication_cost_mb": avg_comm_cost,
            "spatial_correlations": {
                "average": avg_correlation,
                "maximum": max_correlation,
                "total_pairs": len(self.spatial_correlations)
            },
            "staleness_distribution": dict(self.zone_staleness),
            "current_zone_weights": dict(self.zone_fair_weights)
        }
    
    def save_aggregation_state(self, filepath: str):
        """Save aggregation state for checkpointing"""
        state = {
            "current_round": self.current_round,
            "zone_base_weights": self.zone_base_weights,
            "zone_fair_weights": self.zone_fair_weights,
            "spatial_correlations": dict(self.spatial_correlations),
            "zone_staleness": dict(self.zone_staleness),
            "aggregation_history": list(self.aggregation_history),
            "communication_costs": list(self.communication_costs)
        }
        
        torch.save(state, filepath)
    
    def load_aggregation_state(self, filepath: str):
        """Load aggregation state from checkpoint"""
        state = torch.load(filepath)
        
        self.current_round = state["current_round"]
        self.zone_base_weights = state["zone_base_weights"]
        self.zone_fair_weights = state["zone_fair_weights"]
        self.spatial_correlations = state["spatial_correlations"]
        self.zone_staleness = defaultdict(int, state["zone_staleness"])
        self.aggregation_history = deque(state["aggregation_history"], maxlen=100)
        self.communication_costs = deque(state["communication_costs"], maxlen=100)