"""
Hierarchical Aggregation module for ContinuumFL framework.
Implements the two-tier aggregation protocol from Section 4.2 of the paper.
"""

import logging

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, deque
import time
import copy

from torch import Tensor

from ..core.zone import Zone
from ..core.device import EdgeDevice

logger = logging.getLogger("ContinuumFL.Aggregator")


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
        self.zone_weights_cache: Optional[Dict[str, Dict[str, torch.Tensor]]] = {}

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

        logger.debug("[aggregator] HierarchicalAggregator initialised")

    def set_global_model(self, model: nn.Module):
        """Initialize global model"""
        self.global_model = copy.deepcopy(model)
        self.global_weights = {k: v.clone() for k, v in model.state_dict().items()}
        logger.debug(f"[aggregator] Global model set: {len(self.global_weights)} parameter tensors")

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
            scores[zone_id] = max(0.0, score)

        total_score = sum(scores.values())
        if total_score > 0:
            for zone_id in scores:
                scores[zone_id] /= total_score
        else:
            uniform_weight = 1.0 / max(len(zones), 1)
            scores = {zone_id: uniform_weight for zone_id in zones}
            logger.debug(
                f"[aggregator] All zone scores are 0 — using uniform weight={uniform_weight:.4f}"
            )

        logger.debug(
            f"[aggregator] Contribution scores: "
            + "  ".join(f"{zid}:{s:.4f}" for zid, s in sorted(scores.items(), key=lambda x: -x[1])[:5])
            + (" ..." if len(scores) > 5 else "")
        )
        return scores

    def compute_zone_base_weights(self, zones: Dict[str, Zone]) -> Dict[str, float]:
        """
        Compute base aggregation weights for zones.

        Implements Equation (13): β_k^base = Score_k / Σ Score_j
        """
        contribution_scores = self.compute_zone_contribution_scores(zones)
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
            deviation_ratio = (uniform_weight - base_weight) / uniform_weight
            adjustment = 1.0 + self.fairness_strength * deviation_ratio
            fair_weight = base_weight * adjustment
            fair_weights[zone_id] = max(0.01, fair_weight)

        total_weight = sum(fair_weights.values())
        if total_weight > 0:
            for zone_id in fair_weights:
                fair_weights[zone_id] /= total_weight

        self.zone_fair_weights = fair_weights

        min_w = min(fair_weights.values())
        max_w = max(fair_weights.values())
        logger.debug(
            f"[aggregator] Fairness-adjusted weights: "
            f"min={min_w:.4f}  max={max_w:.4f}  n_zones={num_zones}"
        )
        return fair_weights

    def apply_staleness_penalty(self, fair_weights: Dict[str, float]) -> Dict[str, float]:
        """
        Apply staleness penalty to zone weights.

        Implements Equation (12):
        β_k^(t) = β_k^base · exp(-μ · τ_k)
        """
        adjusted_weights = {}
        excluded = []

        for zone_id, fair_weight in fair_weights.items():
            staleness = self.zone_staleness.get(zone_id, 0)

            if staleness > self.max_staleness:
                adjusted_weights[zone_id] = 0.0
                excluded.append(f"{zone_id}(τ={staleness})")
            else:
                staleness_factor = np.exp(-self.staleness_penalty * staleness)
                adjusted_weights[zone_id] = fair_weight * staleness_factor

        total_weight = sum(adjusted_weights.values())
        if total_weight > 0:
            for zone_id in adjusted_weights:
                adjusted_weights[zone_id] /= total_weight

        if excluded:
            logger.debug(f"[aggregator] Zones excluded due to staleness > {self.max_staleness}: {excluded}")

        logger.debug(
            f"[aggregator] Staleness distribution: "
            + "  ".join(f"{zid}:τ={s}" for zid, s in sorted(self.zone_staleness.items(), key=lambda x: -x[1])[:5])
        )
        return adjusted_weights

    def update_spatial_correlations(self, zones: Dict[str, Zone]):
        """
        Update spatial correlation matrix based on model similarity.

        Implements Equation (16):
        ρ^(t+1)(z_k, z_j) = η·ρ^(t) + (1-η)·cos(w_k^(t) - w^(t), w_j^(t) - w^(t))
        """
        if not self.global_weights:
            return

        zone_deviations = {}
        for zone_id, zone in zones.items():
            if zone.aggregated_weights and zone.is_operational:
                deviation = {}
                for param_name, global_param in self.global_weights.items():
                    if param_name in zone.aggregated_weights:
                        zone_param = zone.aggregated_weights[param_name]
                        deviation[param_name] = zone_param - global_param.cpu()
                    else:
                        deviation[param_name] = torch.zeros_like(global_param)
                zone_deviations[zone_id] = deviation

        zone_ids = list(zone_deviations.keys())
        updated_pairs = 0
        for i, zone_i in enumerate(zone_ids):
            for j, zone_j in enumerate(zone_ids):
                if i <= j:
                    correlation_key = (zone_i, zone_j) if zone_i <= zone_j else (zone_j, zone_i)

                    if i == j:
                        new_correlation = 1.0
                    else:
                        dev_i = torch.cat([param.flatten() for param in zone_deviations[zone_i].values()])
                        dev_j = torch.cat([param.flatten() for param in zone_deviations[zone_j].values()])

                        cosine_sim = torch.cosine_similarity(
                            dev_i.unsqueeze(0), dev_j.unsqueeze(0)
                        ).item()
                        new_correlation = cosine_sim

                    old_correlation = self.spatial_correlations.get(correlation_key, 0.0)
                    updated_correlation = (
                        self.momentum_eta * old_correlation
                        + (1 - self.momentum_eta) * new_correlation
                    )
                    self.spatial_correlations[correlation_key] = updated_correlation
                    updated_pairs += 1

        if self.spatial_correlations:
            all_corr = list(self.spatial_correlations.values())
            logger.debug(
                f"[aggregator] Spatial correlations updated: {updated_pairs} pairs  "
                f"mean={np.mean(all_corr):.4f}  min={np.min(all_corr):.4f}  max={np.max(all_corr):.4f}"
            )

    def compute_spatial_regularization_term(self, zone_weights: Dict[str, Dict[str, torch.Tensor]],
                                          zones: Dict[str, Zone]) -> None | dict[str, Tensor]:
        """
        Compute spatial regularization term for inter-zone aggregation.

        Implements the second term in Equation (11):
        λ · Σ Σ ρ(z_k, z_j) (w_k - w_j)
        """
        if not zone_weights or len(zone_weights) < 2:
            return None

        zone_ids = list(zone_weights.keys())
        reg_term: dict[str, Tensor] = {}
        nonzero_pairs = 0

        for i, zone_i in enumerate(zone_ids):
            zone_i_obj = zones.get(zone_i)
            if not zone_i_obj or zone_i not in zone_weights:
                continue

            for zone_j in zone_i_obj.neighbors:
                if zone_j in self.zone_weights_cache and zone_j in zones:
                    correlation_key = (zone_i, zone_j) if zone_i <= zone_j else (zone_j, zone_i)
                    correlation = self.spatial_correlations.get(correlation_key, 0.0)

                    if correlation > 0:
                        reg = {}
                        for param_name in zone_weights[zone_i]:
                            if param_name in self.zone_weights_cache[zone_j]:
                                if param_name not in reg:
                                    reg[param_name] = torch.zeros_like(
                                        self.zone_weights_cache[zone_j][param_name]
                                    )
                                reg[param_name] += (
                                    zone_weights[zone_i][param_name]
                                    - self.zone_weights_cache[zone_j][param_name]
                                )

                        for param_name in reg:
                            if param_name not in reg_term:
                                reg_term[param_name] = torch.zeros_like(reg[param_name])
                            reg_term[param_name] += correlation * reg[param_name]
                        nonzero_pairs += 1

        for param_name in reg_term:
            reg_term[param_name] *= self.spatial_regularization

        if reg_term:
            reg_norm = float(np.sqrt(sum(
                v.float().norm(2).item() ** 2 for v in reg_term.values()
            )))
            logger.debug(
                f"[aggregator] Spatial reg term: λ={self.spatial_regularization}  "
                f"pairs={nonzero_pairs}  ‖reg‖={reg_norm:.6f}"
            )

        return reg_term

    def intra_zone_aggregation(self, zones: Dict[str, Zone],
                             device_updates: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Perform intra-zone aggregation for all zones.

        Implements Phase 2 of Algorithm 2.
        """
        zone_aggregated_weights = {}

        for zone_id, zone in zones.items():
            if not zone.devices or not zone.is_operational:
                logger.debug(f"[aggregator] Zone {zone_id} skipped (no devices or not operational)")
                continue

            zone_device_updates = {}
            for device_id in zone.device_ids:
                if device_id in device_updates:
                    zone_device_updates[device_id] = device_updates[device_id]

            if zone_device_updates:
                aggregated = zone.intra_zone_aggregation(zone_device_updates)
                if aggregated:
                    zone_aggregated_weights[zone_id] = aggregated
                    logger.debug(
                        f"[aggregator] Zone {zone_id} intra-agg: "
                        f"{len(zone_device_updates)} device updates aggregated"
                    )

        logger.debug(
            f"[aggregator] Intra-zone aggregation complete: "
            f"{len(zone_aggregated_weights)}/{len(zones)} zones produced updates"
        )
        return zone_aggregated_weights

    def inter_zone_aggregation(self, zone_weights: Dict[str, Dict[str, torch.Tensor]],
                             zones: Dict[str, Zone]) -> Dict[str, torch.Tensor]:
        """
        Perform inter-zone aggregation with spatial awareness.

        Implements Phase 3 of Algorithm 2 and Equation (11).
        """
        if not zone_weights:
            logger.debug("[aggregator] inter_zone_aggregation: no zone weights — returning global weights")
            return self.global_weights or {}

        logger.debug(
            f"[aggregator] inter_zone_aggregation: {len(zone_weights)} zones  "
            f"global_weights={'set' if self.global_weights else 'NONE'}"
        )

        # Compute zone aggregation weights
        base_weights = self.compute_zone_base_weights(zones)
        fair_weights = self.apply_fairness_adjustment(base_weights)
        final_weights = self.apply_staleness_penalty(fair_weights)

        logger.debug(
            f"[aggregator] Final zone weights (top-5): "
            + "  ".join(
                f"{zid}:{w:.4f}"
                for zid, w in sorted(final_weights.items(), key=lambda x: -x[1])[:5]
            )
        )

        # Update spatial correlations
        self.update_spatial_correlations(zones)

        # Find first valid zone weights to initialise the output tensors
        first_zone_weights = None
        for zone_id in sorted(zone_weights.keys()):
            if zone_weights[zone_id]:
                first_zone_weights = zone_weights[zone_id]
                break
        if first_zone_weights is None:
            logger.debug("[aggregator] inter_zone_aggregation: all zone weights empty")
            return self.global_weights or {}

        global_aggregated = {}
        original_dtypes = {}

        for param_name, param_tensor in first_zone_weights.items():
            original_dtypes[param_name] = param_tensor.dtype
            if param_tensor.dtype != torch.float32:
                global_aggregated[param_name] = torch.zeros_like(param_tensor, dtype=torch.float32)
            else:
                global_aggregated[param_name] = torch.zeros_like(param_tensor)

        # Weighted aggregation across zones
        total_weight = 0.0
        contributing_zones = []
        for zone_id, zone_weight_dict in zone_weights.items():
            if zone_id in final_weights and final_weights[zone_id] > 0:
                self.zone_weights_cache[zone_id] = {
                    k: v.detach().cpu() for k, v in zone_weight_dict.items()
                }
                weight = final_weights[zone_id]
                for param_name, param_tensor in zone_weight_dict.items():
                    if param_name in global_aggregated:
                        if param_tensor.dtype != torch.float32:
                            param_tensor = param_tensor.float()
                        global_aggregated[param_name] += weight * param_tensor

                total_weight += weight
                contributing_zones.append(f"{zone_id}({weight:.4f})")

        logger.debug(
            f"[aggregator] Inter-zone: total_weight={total_weight:.4f}  "
            f"zones: {contributing_zones}"
        )

        # Normalize if weights don't sum to 1
        if total_weight > 0 and abs(total_weight - 1.0) > 1e-6:
            for param_name in global_aggregated:
                global_aggregated[param_name] = global_aggregated[param_name] / total_weight

        # Convert back to original dtypes
        for param_name in global_aggregated:
            if original_dtypes[param_name] != torch.float32:
                global_aggregated[param_name] = global_aggregated[param_name].to(
                    original_dtypes[param_name]
                )

        # Spatial regularization
        reg_term = self.compute_spatial_regularization_term(zone_weights, zones)

        # Update global weights: W_new = W_old + avg(δ) + reg
        # (global_aggregated holds the weighted average of delta weights,
        # so the update is: W_new = W_old + Σ β_k δ_k + λ·reg_term)
        updated_params = 0
        for k in list(self.global_weights.keys()):
            if k in global_aggregated and self.global_weights[k].dtype.is_floating_point:
                reg = (
                    reg_term[k].cpu()
                    if reg_term and k in reg_term
                    else torch.zeros_like(global_aggregated[k])
                )
                self.global_weights[k] = (
                    self.global_weights[k].cpu() + global_aggregated[k].cpu() + reg
                )
                updated_params += 1

        # Log post-update weight norms for anomaly detection
        sample_norms = {}
        for k, v in list(self.global_weights.items())[:5]:
            n = v.float().norm(2).item()
            sample_norms[k] = n
            if np.isnan(n):
                logger.warning(f"[aggregator] ⚠ NaN in global weight layer: {k}")
            elif np.isinf(n):
                logger.warning(f"[aggregator] ⚠ Inf in global weight layer: {k}")
            elif n > 1e4:
                logger.warning(f"[aggregator] ⚠ Large norm ({n:.1f}) in global weight layer: {k}")

        total_norm = float(np.sqrt(sum(
            v.float().norm(2).item() ** 2 for v in self.global_weights.values()
        )))
        logger.debug(
            f"[aggregator] Global weights updated: {updated_params} layers  "
            f"‖W‖={total_norm:.4f}  "
            f"sample_norms={{{', '.join(f'{k}:{v:.4f}' for k,v in sample_norms.items())}}}"
        )

        return self.global_weights

    def federated_aggregation_round(self, zones: Dict[str, Zone],
                                  num_device_updates: int, participating_zones: List[str],
                                  total_time, intra_time, inter_time, comm_cost) -> Dict[str, Any]:
        """
        Record stats for a complete federated aggregation round.

        Implements Algorithm 2: ContinuumFL Aggregation Protocol.
        """
        round_stats = {
            "round": self.current_round,
            "participating_zones": 0,
            "participating_devices": num_device_updates,
            "aggregation_time": 0.0,
            "communication_cost": 0.0,
        }
        self.current_round += 1

        if len(participating_zones) == 0:
            logger.debug(f"[aggregator] Round {self.current_round}: no participating zones")
            return round_stats

        # Update zone staleness
        reset_zones = []
        staled_zones = []
        for zone_id in zones:
            if zone_id in participating_zones:
                self.zone_staleness[zone_id] = 0
                reset_zones.append(zone_id)
            else:
                self.zone_staleness[zone_id] += 1
                staled_zones.append(f"{zone_id}(τ={self.zone_staleness[zone_id]})")

        if staled_zones:
            logger.debug(f"[aggregator] Stale zones this round: {staled_zones}")

        round_stats["aggregation_time"] = total_time
        round_stats["intra_zone_time"] = intra_time
        round_stats["inter_zone_time"] = inter_time
        round_stats["participating_zones"] = len(participating_zones)
        round_stats["round"] = self.current_round
        round_stats["communication_cost"] = comm_cost
        self.communication_costs.append(comm_cost)

        self.aggregation_history.append({
            "round": self.current_round,
            "zone_weights": self.zone_fair_weights.copy(),
            "spatial_correlations": dict(self.spatial_correlations),
            "participating_zones": participating_zones,
            "staleness": dict(self.zone_staleness),
        })

        logger.debug(
            f"[aggregator] Round {self.current_round} stats: "
            f"zones={len(participating_zones)}  devs={num_device_updates}  "
            f"total={total_time*1000:.0f}ms  intra={intra_time*1000:.0f}ms  "
            f"inter={inter_time*1000:.0f}ms  comm={comm_cost:.2f}MB"
        )

        return round_stats

    def get_aggregation_stats(self) -> Dict[str, Any]:
        """Get comprehensive aggregation statistics"""
        if not self.aggregation_history:
            return {}

        recent_history = list(self.aggregation_history)[-10:]

        zone_participation = defaultdict(int)
        total_rounds = len(recent_history)

        for round_data in recent_history:
            for zone_id, weight in round_data["zone_weights"].items():
                if weight > 0:
                    zone_participation[zone_id] += 1

        avg_participation = {
            zone_id: count / total_rounds
            for zone_id, count in zone_participation.items()
        }

        recent_comm_costs = list(self.communication_costs)[-10:]
        avg_comm_cost = np.mean(recent_comm_costs) if recent_comm_costs else 0.0

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
                "total_pairs": len(self.spatial_correlations),
            },
            "staleness_distribution": dict(self.zone_staleness),
            "current_zone_weights": dict(self.zone_fair_weights),
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
            "communication_costs": list(self.communication_costs),
        }
        torch.save(state, filepath)
        logger.debug(f"[aggregator] State saved to {filepath}")

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
        logger.debug(
            f"[aggregator] State loaded from {filepath}: round={self.current_round}"
        )
