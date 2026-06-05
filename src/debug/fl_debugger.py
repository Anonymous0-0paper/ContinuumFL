"""
FLDebugger — comprehensive per-round diagnostic logger for ContinuumFL.

Tracks memory, weight norms, gradient norms, aggregation weights, device
participation, timing, zone health, and staleness every round.  All output
goes through Python's logging module at DEBUG level so it can be silenced in
production without code changes.
"""

import gc
import logging
import os
import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import psutil
import torch
import torch.nn as nn

logger = logging.getLogger("ContinuumFL.Debug")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _rss_mb() -> float:
    """Current process RSS in MB (fast /proc read)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except Exception:
        pass
    return psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2


def _gpu_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 ** 2
    return 0.0


def _gpu_reserved_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.memory_reserved() / 1024 ** 2
    return 0.0


def _model_weight_norms(model: nn.Module) -> Dict[str, float]:
    """L2 norm of each named parameter."""
    return {
        name: param.data.norm(2).item()
        for name, param in model.named_parameters()
    }


def _state_dict_weight_norms(state: Dict[str, torch.Tensor]) -> Dict[str, float]:
    return {k: v.float().norm(2).item() for k, v in state.items()}


def _summarise_norms(norms: Dict[str, float]) -> Dict[str, float]:
    """Return min / mean / max / total of a norm dict."""
    vals = list(norms.values())
    if not vals:
        return {}
    return {
        "min":   float(np.min(vals)),
        "mean":  float(np.mean(vals)),
        "max":   float(np.max(vals)),
        "total": float(np.sum(vals)),
        "n_layers": len(vals),
    }


def _format_bar(value: float, total: float, width: int = 20) -> str:
    frac = min(value / max(total, 1e-9), 1.0)
    filled = int(frac * width)
    return "[" + "█" * filled + "░" * (width - filled) + f"] {value:.4f}"


# ─────────────────────────────────────────────────────────────────────────────
# Main debugger class
# ─────────────────────────────────────────────────────────────────────────────

class FLDebugger:
    """
    Attach one instance to the coordinator and call the hooks documented below
    at the corresponding points in the training loop.

    Usage
    -----
    debugger = FLDebugger(config)

    # inside run_federated_learning:
    debugger.on_round_start(round_num, global_model)
    debugger.on_zone_training_done(zone_id, zone_result, device_updates)
    debugger.on_aggregation_done(round_num, global_model, zone_weights,
                                  final_agg_weights, aggregation_stats)
    debugger.on_round_end(round_num, round_metrics, round_time)
    debugger.on_training_end(final_results)
    """

    def __init__(self, config):
        self.config = config
        self._round_start_time: float = 0.0
        self._prev_rss: float = 0.0
        self._prev_gpu: float = 0.0

        # Per-round history (ring buffer keeps last 200)
        self.memory_history:  deque = deque(maxlen=200)
        self.accuracy_history: deque = deque(maxlen=200)
        self.loss_history:     deque = deque(maxlen=200)
        self.weight_norm_history: deque = deque(maxlen=200)

        # Counters
        self.total_device_calls: int = 0
        self.total_device_failures: int = 0
        self.zone_participation_counts: Dict[str, int] = defaultdict(int)

        logger.debug("FLDebugger initialised")

    # ── round lifecycle ───────────────────────────────────────────────────────

    def on_round_start(self, round_num: int, global_model: nn.Module) -> None:
        self._round_start_time = time.time()
        rss = _rss_mb()
        gpu = _gpu_mb()
        gpu_res = _gpu_reserved_mb()
        delta_rss = rss - self._prev_rss
        delta_gpu = gpu - self._prev_gpu
        self._prev_rss = rss
        self._prev_gpu = gpu

        norms = _model_weight_norms(global_model)
        norm_summary = _summarise_norms(norms)
        self.weight_norm_history.append(norm_summary)

        sep = "=" * 72
        logger.debug(
            f"\n{sep}\n"
            f"  ROUND {round_num} START\n"
            f"{sep}\n"
            f"  Memory  : CPU={rss:.1f} MB (Δ{delta_rss:+.1f}), "
            f"GPU allocated={gpu:.1f} MB (Δ{delta_gpu:+.1f}), "
            f"GPU reserved={gpu_res:.1f} MB\n"
            f"  GC objs : {len(gc.get_objects()):,}\n"
            f"  Model norms  : min={norm_summary.get('min',0):.4f}  "
            f"mean={norm_summary.get('mean',0):.4f}  "
            f"max={norm_summary.get('max',0):.4f}  "
            f"total={norm_summary.get('total',0):.4f}  "
            f"layers={norm_summary.get('n_layers',0)}\n"
            f"{sep}"
        )

    def on_zone_training_done(
        self,
        zone_id: str,
        zone_result: Tuple[str, Any, Dict],
        device_updates: Dict[str, Any],
    ) -> None:
        _, aggregated_weights, stats = zone_result
        n_devices     = stats.get("num_device_updates", 0)
        n_participating = len(stats.get("participating_devices", []))
        intra_time    = stats.get("intra_time", 0.0)
        comm_cost     = stats.get("communication_cost", 0.0)

        self.zone_participation_counts[zone_id] += 1
        self.total_device_calls += n_participating

        # Zone aggregated weight norm (device_updates == aggregated state dict {layer: tensor})
        grad_norm_lines = []
        if aggregated_weights:
            try:
                agg_vals = [v.float() for v in aggregated_weights.values()
                            if isinstance(v, torch.Tensor) and v.layout == torch.strided]
                agg_norm = torch.cat([v.flatten() for v in agg_vals]).norm(2).item() if agg_vals else 0.0
            except Exception:
                agg_norm = 0.0
        else:
            agg_norm = 0.0

        lines = [
            f"  ├─ Zone {zone_id}: {n_devices}/{n_participating} devices updated",
            f"  │   intra_time={intra_time*1000:.1f}ms  comm_cost={comm_cost:.2f}MB  ‖agg‖={agg_norm:.4f}",
        ]
        if grad_norm_lines:
            lines.append("  │   Device gradient norms:")
            lines.extend(grad_norm_lines)

        logger.debug("\n".join(lines))

    def on_aggregation_done(
        self,
        round_num: int,
        global_model: nn.Module,
        zone_weights: Dict[str, Dict[str, torch.Tensor]],
        final_zone_weights: Dict[str, float],
        aggregation_stats: Dict[str, Any],
    ) -> None:
        norms = _model_weight_norms(global_model)
        norm_summary = _summarise_norms(norms)

        # Per-layer norm anomaly detection (NaN / Inf / explosion)
        anomalies = []
        for name, norm in norms.items():
            if np.isnan(norm):
                anomalies.append(f"    ⚠ NaN  in layer: {name}")
            elif np.isinf(norm):
                anomalies.append(f"    ⚠ Inf  in layer: {name}")
            elif norm > 1e4:
                anomalies.append(f"    ⚠ Large norm ({norm:.1f}) in layer: {name}")

        # Zone weight distribution bar chart
        weight_lines = []
        max_w = max(final_zone_weights.values(), default=1e-9)
        for zid in sorted(final_zone_weights.keys()):
            w = final_zone_weights[zid]
            bar = _format_bar(w, max_w)
            weight_lines.append(f"      {zid:12s}: {bar}")

        sep = "-" * 72
        msg_parts = [
            f"\n{sep}",
            f"  ROUND {round_num} AGGREGATION DONE",
            f"  Participating zones : {aggregation_stats.get('participating_zones', '?')}",
            f"  Participating devs  : {aggregation_stats.get('participating_devices', '?')}",
            f"  Intra-zone time     : {aggregation_stats.get('intra_zone_time', 0)*1000:.1f} ms",
            f"  Inter-zone time     : {aggregation_stats.get('inter_zone_time', 0)*1000:.1f} ms",
            f"  Comm cost           : {aggregation_stats.get('communication_cost', 0):.2f} MB",
            f"  Global weight norms : min={norm_summary.get('min',0):.4f}  "
            f"mean={norm_summary.get('mean',0):.4f}  "
            f"max={norm_summary.get('max',0):.4f}  "
            f"total={norm_summary.get('total',0):.4f}",
        ]
        if anomalies:
            msg_parts.append("  ⚠ WEIGHT ANOMALIES DETECTED:")
            msg_parts.extend(anomalies)
        if weight_lines:
            msg_parts.append("  Zone aggregation weights:")
            msg_parts.extend(weight_lines)
        msg_parts.append(sep)
        logger.debug("\n".join(msg_parts))

    def on_round_end(
        self,
        round_num: int,
        round_metrics: Dict[str, Any],
        round_time: float,
    ) -> None:
        rss = _rss_mb()
        gpu = _gpu_mb()

        accuracy = round_metrics.get("global_accuracy", 0.0)
        loss     = round_metrics.get("global_loss", float("inf"))
        self.accuracy_history.append(accuracy)
        self.loss_history.append(loss)
        self.memory_history.append({"rss": rss, "gpu": gpu, "round": round_num})

        # Accuracy trend (last 5)
        recent_acc = list(self.accuracy_history)[-5:]
        if len(recent_acc) >= 2:
            trend = "↑" if recent_acc[-1] > recent_acc[-2] else ("↓" if recent_acc[-1] < recent_acc[-2] else "→")
        else:
            trend = "–"

        # Memory growth warning
        mem_warning = ""
        if len(self.memory_history) >= 3:
            rss_vals = [e["rss"] for e in list(self.memory_history)[-3:]]
            if all(rss_vals[i] < rss_vals[i+1] for i in range(len(rss_vals)-1)):
                growth = rss_vals[-1] - rss_vals[0]
                mem_warning = f"  ⚠ CPU RSS growing: +{growth:.1f} MB over last 3 rounds"

        sep = "=" * 72
        lines = [
            f"\n{sep}",
            f"  ROUND {round_num} END  ({round_time:.2f}s)",
            f"  Accuracy : {accuracy*100:.4f}%  {trend}   Loss: {loss:.4f}",
            f"  Memory   : CPU={rss:.1f} MB  GPU={gpu:.1f} MB",
        ]
        if mem_warning:
            lines.append(mem_warning)

        # Zone accuracy summary
        zone_metrics = round_metrics.get("zone_metrics", {})
        if zone_metrics:
            dead_zones = [zid for zid, zm in zone_metrics.items() if zm.get("accuracy", 0) < 0.01]
            if dead_zones:
                lines.append(f"  ⚠ Low-accuracy zones (<1%): {dead_zones}")

        lines.append(sep)
        logger.debug("\n".join(lines))

    def on_training_end(self, final_results: Dict[str, Any]) -> None:
        sep = "=" * 72
        lines = [
            f"\n{sep}",
            "  TRAINING COMPLETE — FINAL DEBUG SUMMARY",
            f"{sep}",
            f"  Total rounds        : {final_results.get('total_rounds', '?')}",
            f"  Final accuracy      : {final_results.get('final_accuracy', 0)*100:.4f}%",
            f"  Final loss          : {final_results.get('final_loss', 0):.4f}",
            f"  Total training time : {final_results.get('total_training_time', 0):.1f}s",
            f"  Total device calls  : {self.total_device_calls}",
            f"  Total device fails  : {self.total_device_failures}",
            f"  Comm cost (total)   : {final_results.get('total_communication_cost', 0):.2f} MB",
        ]

        # Memory growth over all rounds
        if self.memory_history:
            rss_vals = [e["rss"] for e in self.memory_history]
            lines.append(
                f"  RSS: start={rss_vals[0]:.1f} MB  end={rss_vals[-1]:.1f} MB  "
                f"delta={rss_vals[-1]-rss_vals[0]:+.1f} MB"
            )

        # Zone participation
        if self.zone_participation_counts:
            lines.append("  Zone participation counts:")
            for zid in sorted(self.zone_participation_counts.keys()):
                lines.append(f"      {zid}: {self.zone_participation_counts[zid]} rounds")

        lines.append(sep)
        logger.debug("\n".join(lines))

    # ── device-level helpers (called from device.py) ──────────────────────────

    @staticmethod
    def log_device_training_start(device_id: str, zone_id: str, dataset_size: int,
                                   epochs: int, lr: float, comp_device: str) -> None:
        logger.debug(
            f"  [device] {device_id} (zone={zone_id}) START train: "
            f"data={dataset_size} epochs={epochs} lr={lr} device={comp_device}"
        )

    @staticmethod
    def log_device_training_end(device_id: str, zone_id: Optional[str],
                                 training_time: float, loss: float,
                                 grad_norm: float) -> None:
        logger.debug(
            f"  [device] {device_id} (zone={zone_id}) END train: "
            f"time={training_time*1000:.0f}ms  loss={loss:.4f}  ‖δw‖={grad_norm:.4f}"
        )

    @staticmethod
    def log_device_failure(device_id: str, zone_id: Optional[str], event: str) -> None:
        logger.debug(f"  [device] {device_id} (zone={zone_id}) FAILURE EVENT: {event}")

    @staticmethod
    def log_device_repair(device_id: str, zone_id: Optional[str]) -> None:
        logger.debug(f"  [device] {device_id} (zone={zone_id}) REPAIRED → active")

    # ── zone-level helpers ────────────────────────────────────────────────────

    @staticmethod
    def log_intra_zone_aggregation(zone_id: str, n_updates: int,
                                    weights: Dict[str, float],
                                    agg_norm: float, time_ms: float) -> None:
        w_str = "  ".join(f"{d}:{w:.3f}" for d, w in list(weights.items())[:5])
        logger.debug(
            f"  [zone] {zone_id} intra-agg: {n_updates} updates  "
            f"‖agg‖={agg_norm:.4f}  time={time_ms:.1f}ms\n"
            f"         weights (top-5): {w_str}"
        )

    @staticmethod
    def log_zone_contribution_score(zone_id: str, n_devices: int,
                                     avg_data: float, consistency: float,
                                     score: float) -> None:
        logger.debug(
            f"  [zone] {zone_id} contribution: n_dev={n_devices}  "
            f"avg_data={avg_data:.0f}  consistency={consistency:.4f}  score={score:.4f}"
        )

    # ── aggregator-level helpers ──────────────────────────────────────────────

    @staticmethod
    def log_staleness(zone_staleness: Dict[str, int]) -> None:
        stale = {zid: s for zid, s in zone_staleness.items() if s > 0}
        if stale:
            logger.debug(f"  [agg] Stale zones: { {k: v for k, v in sorted(stale.items(), key=lambda x: -x[1])} }")

    @staticmethod
    def log_zone_weights(stage: str, weights: Dict[str, float]) -> None:
        top = sorted(weights.items(), key=lambda x: -x[1])[:5]
        bottom = sorted(weights.items(), key=lambda x: x[1])[:3]
        logger.debug(
            f"  [agg] {stage} weights — "
            f"top5: { {k: f'{v:.4f}' for k,v in top} }  "
            f"bot3: { {k: f'{v:.4f}' for k,v in bottom} }"
        )

    @staticmethod
    def log_spatial_correlations(correlations: Dict, top_n: int = 5) -> None:
        if not correlations:
            return
        sorted_corr = sorted(correlations.items(), key=lambda x: -abs(x[1]))[:top_n]
        logger.debug(
            f"  [agg] Spatial correlations (top-{top_n}): "
            + "  ".join(f"{k}:{v:.3f}" for k, v in sorted_corr)
        )
