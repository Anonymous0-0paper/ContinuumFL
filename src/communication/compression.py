"""
Communication compression module for ContinuumFL framework.
Implements gradient compression, delta encoding, and caching optimizations.
"""

import torch
import numpy as np
from typing import Dict, Tuple, List, Optional, Any
import time
from collections import defaultdict, deque

class GradientCompressor:
    """
    Implements communication optimization techniques from Section 4.4:
    - Top-k sparsification
    - Delta encoding  
    - Opportunistic caching
    """
    
    def __init__(self, config):
        self.config = config
        self.compression_rate = config.compression_rate  # κ (top-k ratio)
        self.quantization_bits = config.quantization_bits
        
        # Error feedback for unbiased compression
        self.error_feedback_buffers: Dict[str, torch.Tensor] = {}
        
        # Caching system
        self.layer_cache: Dict[str, Dict[str, torch.Tensor]] = {}
        self.cache_versions: Dict[str, Dict[str, int]] = {}
        self.layer_stability_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
        self.cache_threshold = config.cache_threshold
        
        # Delta encoding state
        self.previous_models: Dict[str, Dict[str, torch.Tensor]] = {}
        
        # Communication statistics
        self.compression_stats = {
            "original_size": 0,
            "compressed_size": 0,
            "compression_ratio": 0.0,
            "cache_hits": 0,
            "cache_misses": 0
        }
    
    def compress_gradients(self, gradients: Dict[str, torch.Tensor], 
                         device_id: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """
        Apply top-k sparsification to gradients.
        
        Implements gradient compression from Section 4.4.
        """
        compressed_gradients = {}
        compression_info = {
            "compression_rate": self.compression_rate,
            "original_size": 0,
            "compressed_size": 0,
            "sparsity": 0.0
        }
        
        total_original_elements = 0
        total_compressed_elements = 0
        
        for param_name, grad_tensor in gradients.items():
            # Flatten gradient
            flat_grad = grad_tensor.flatten()
            original_size = len(flat_grad)
            total_original_elements += original_size
            
            # Determine number of elements to keep
            k = max(1, int(self.compression_rate * original_size))
            
            # Get top-k elements by magnitude
            abs_grad = torch.abs(flat_grad)
            _, top_k_indices = torch.topk(abs_grad, k)
            
            # Create sparse gradient
            sparse_grad = torch.zeros_like(flat_grad)
            sparse_grad[top_k_indices] = flat_grad[top_k_indices]
            
            # Apply error feedback
            if device_id not in self.error_feedback_buffers:
                self.error_feedback_buffers[device_id] = torch.zeros_like(flat_grad)
            
            # Add accumulated error from previous rounds
            sparse_grad += self.error_feedback_buffers[device_id]
            
            # Update error feedback buffer
            error = flat_grad - sparse_grad
            self.error_feedback_buffers[device_id] = error
            
            # Reshape back to original shape
            compressed_gradients[param_name] = sparse_grad.reshape(grad_tensor.shape)
            total_compressed_elements += k
        
        # Update compression statistics
        compression_info["original_size"] = total_original_elements
        compression_info["compressed_size"] = total_compressed_elements
        compression_info["sparsity"] = 1.0 - (total_compressed_elements / max(total_original_elements, 1))
        
        self.compression_stats["original_size"] += total_original_elements
        self.compression_stats["compressed_size"] += total_compressed_elements
        
        return compressed_gradients, compression_info
    
    def apply_delta_encoding(self, current_model: Dict[str, torch.Tensor], 
                           entity_id: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """
        Apply delta encoding for inter-zone communication.
        
        Implements delta encoding from Section 4.4.
        """
        if entity_id not in self.previous_models:
            # First transmission - send full model
            self.previous_models[entity_id] = {k: v.clone() for k, v in current_model.items()}
            return current_model, {"encoding_type": "full", "compression_ratio": 1.0}
        
        previous_model = self.previous_models[entity_id]
        delta_weights = {}
        
        total_original_size = 0
        total_delta_size = 0
        
        for param_name, current_param in current_model.items():
            if param_name in previous_model:
                # Compute delta
                delta = current_param - previous_model[param_name]
                delta_weights[param_name] = delta
                
                # Track sizes for compression ratio
                total_original_size += current_param.numel()
                total_delta_size += torch.count_nonzero(torch.abs(delta) > 1e-6).item()
            else:
                # New parameter - send full
                delta_weights[param_name] = current_param
                total_original_size += current_param.numel()
                total_delta_size += current_param.numel()
        
        # Apply quantization to deltas
        quantized_deltas = self._quantize_tensors(delta_weights)
        
        # Update previous model state
        self.previous_models[entity_id] = {k: v.clone() for k, v in current_model.items()}
        
        compression_ratio = total_delta_size / max(total_original_size, 1)
        
        return quantized_deltas, {
            "encoding_type": "delta",
            "compression_ratio": compression_ratio,
            "original_size": total_original_size,
            "delta_size": total_delta_size
        }
    
    def _quantize_tensors(self, tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply quantization to reduce communication overhead"""
        if self.quantization_bits >= 32:
            return tensors  # No quantization needed
        
        quantized = {}
        
        for name, tensor in tensors.items():
            # Simple uniform quantization
            tensor_min = tensor.min()
            tensor_max = tensor.max()
            
            if tensor_min == tensor_max:
                quantized[name] = tensor
                continue
            
            # Quantize to specified bits
            num_levels = 2 ** self.quantization_bits - 1
            scale = (tensor_max - tensor_min) / num_levels
            
            # Quantize
            quantized_tensor = torch.round((tensor - tensor_min) / scale)
            
            # Dequantize
            dequantized = quantized_tensor * scale + tensor_min
            quantized[name] = dequantized
        
        return quantized
    
    def update_layer_cache(self, model_weights: Dict[str, torch.Tensor], 
                          entity_id: str) -> Dict[str, Any]:
        """
        Update cached layers based on stability.
        
        Implements opportunistic caching from Section 4.4.
        """
        if entity_id not in self.layer_cache:
            self.layer_cache[entity_id] = {}
            self.cache_versions[entity_id] = {}
        
        cache_updates = {}
        cache_info = {
            "cache_hits": 0,
            "cache_misses": 0,
            "stable_layers": [],
            "updated_layers": []
        }
        
        for layer_name, current_weights in model_weights.items():
            # Check if layer is in cache
            if layer_name in self.layer_cache[entity_id]:
                cached_weights = self.layer_cache[entity_id][layer_name]
                
                # Compute stability metric
                stability = self._compute_layer_stability(current_weights, cached_weights)
                self.layer_stability_history[f"{entity_id}_{layer_name}"].append(stability)
                
                # Check if layer should remain cached
                recent_stability = list(self.layer_stability_history[f"{entity_id}_{layer_name}"])
                avg_stability = np.mean(recent_stability) if recent_stability else 0.0
                
                if avg_stability > self.cache_threshold:
                    # Layer is stable - cache hit
                    cache_info["cache_hits"] += 1
                    cache_info["stable_layers"].append(layer_name)
                    # Don't include in transmission
                else:
                    # Layer changed significantly - cache miss
                    cache_info["cache_misses"] += 1
                    cache_info["updated_layers"].append(layer_name)
                    cache_updates[layer_name] = current_weights
                    
                    # Update cache
                    self.layer_cache[entity_id][layer_name] = current_weights.clone()
                    self.cache_versions[entity_id][layer_name] = self.cache_versions[entity_id].get(layer_name, 0) + 1
            else:
                # New layer - cache miss
                cache_info["cache_misses"] += 1
                cache_info["updated_layers"].append(layer_name)
                cache_updates[layer_name] = current_weights
                
                # Add to cache
                self.layer_cache[entity_id][layer_name] = current_weights.clone()
                self.cache_versions[entity_id][layer_name] = 1
        
        # Update global cache statistics
        self.compression_stats["cache_hits"] += cache_info["cache_hits"]
        self.compression_stats["cache_misses"] += cache_info["cache_misses"]
        
        return cache_updates, cache_info
    
    def _compute_layer_stability(self, current: torch.Tensor, previous: torch.Tensor) -> float:
        """
        Compute stability metric for a layer.
        
        Implements stability measure from Equation (17).
        """
        if current.shape != previous.shape:
            return 0.0
        
        # Compute relative change
        diff_norm = torch.norm(current - previous)
        current_norm = torch.norm(current)
        
        if current_norm == 0:
            return 1.0 if diff_norm == 0 else 0.0
        
        relative_change = diff_norm / current_norm
        stability = 1.0 - relative_change.item()
        
        return max(0.0, min(1.0, stability))
    
    def compress_for_transmission(self, model_weights: Dict[str, torch.Tensor],
                                entity_id: str, use_caching: bool = True) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """
        Complete compression pipeline for transmission.
        
        Combines gradient compression, delta encoding, and caching.
        """
        start_time = time.time()
        
        compression_info = {
            "entity_id": entity_id,
            "compression_steps": [],
            "total_compression_ratio": 1.0,
            "processing_time": 0.0
        }
        
        # Step 1: Apply caching (if enabled)
        if use_caching:
            cached_weights, cache_info = self.update_layer_cache(model_weights, entity_id)
            compression_info["compression_steps"].append(("caching", cache_info))
            weights_to_transmit = cached_weights
        else:
            weights_to_transmit = model_weights
        
        # Step 2: Apply delta encoding
        if weights_to_transmit:
            delta_weights, delta_info = self.apply_delta_encoding(weights_to_transmit, entity_id)
            compression_info["compression_steps"].append(("delta_encoding", delta_info))
            weights_to_transmit = delta_weights
        
        # Step 3: Apply gradient compression (sparsification)
        if weights_to_transmit:
            compressed_weights, compression_detail = self.compress_gradients(weights_to_transmit, entity_id)
            compression_info["compression_steps"].append(("sparsification", compression_detail))
            final_weights = compressed_weights
        else:
            final_weights = {}
        
        # Calculate overall compression ratio
        original_size = sum(param.numel() for param in model_weights.values())
        compressed_size = sum(param.numel() for param in final_weights.values())
        
        if original_size > 0:
            compression_info["total_compression_ratio"] = compressed_size / original_size
        
        compression_info["processing_time"] = time.time() - start_time
        
        return final_weights, compression_info
    
    def decompress_from_transmission(self, compressed_weights: Dict[str, torch.Tensor],
                                   entity_id: str, reconstruction_info: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Decompress received weights to reconstruct full model.
        
        Reverses the compression pipeline.
        """
        # This would implement the reverse operations
        # For now, return as-is since we're primarily focused on compression
        return compressed_weights
    
    def get_compression_statistics(self) -> Dict[str, Any]:
        """Get comprehensive compression statistics"""
        total_original = self.compression_stats["original_size"]
        total_compressed = self.compression_stats["compressed_size"]
        
        if total_original > 0:
            overall_compression_ratio = total_compressed / total_original
        else:
            overall_compression_ratio = 1.0
        
        cache_total = self.compression_stats["cache_hits"] + self.compression_stats["cache_misses"]
        cache_hit_rate = self.compression_stats["cache_hits"] / max(cache_total, 1)
        
        return {
            "overall_compression_ratio": overall_compression_ratio,
            "total_original_elements": total_original,
            "total_compressed_elements": total_compressed,
            "cache_hit_rate": cache_hit_rate,
            "cache_hits": self.compression_stats["cache_hits"],
            "cache_misses": self.compression_stats["cache_misses"],
            "active_caches": len(self.layer_cache),
            "error_feedback_buffers": len(self.error_feedback_buffers)
        }
    
    def reset_compression_state(self):
        """Reset compression state for new experiment"""
        self.error_feedback_buffers.clear()
        self.layer_cache.clear()
        self.cache_versions.clear()
        self.layer_stability_history.clear()
        self.previous_models.clear()
        
        self.compression_stats = {
            "original_size": 0,
            "compressed_size": 0,
            "compression_ratio": 0.0,
            "cache_hits": 0,
            "cache_misses": 0
        }
    
    def estimate_bandwidth_savings(self, original_model_size_mb: float) -> Dict[str, float]:
        """Estimate bandwidth savings from compression"""
        stats = self.get_compression_statistics()
        compression_ratio = stats["overall_compression_ratio"]
        
        # Estimate savings
        compressed_size_mb = original_model_size_mb * compression_ratio
        savings_mb = original_model_size_mb - compressed_size_mb
        savings_percentage = (1.0 - compression_ratio) * 100
        
        return {
            "original_size_mb": original_model_size_mb,
            "compressed_size_mb": compressed_size_mb,
            "savings_mb": savings_mb,
            "savings_percentage": savings_percentage,
            "compression_ratio": compression_ratio
        }