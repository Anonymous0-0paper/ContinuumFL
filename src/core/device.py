"""
Core device module for ContinuumFL framework.
Implements edge devices with spatial coordinates, resource constraints, and FL capabilities.
"""
import gc

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


@dataclass
class DeviceResources:
    """Resource specification for edge devices"""
    compute_capacity: float  # GFLOPS
    memory_capacity: float   # GB
    bandwidth: float         # Mbps

class TensorPool:
    """Simple tensor memory pool to reduce allocations"""
    def __init__(self):
        self.pool = {}  # {(shape, dtype, device): [available_tensors]}
        self.used = set()  # Set of currently used tensors
        self.max_pool_size = 100  # Maximum number of tensors to keep in pool per shape
    
    def get_tensor(self, shape, dtype=torch.float32, device='cpu'):
        """Get a tensor from the pool or create a new one"""
        key = (shape, dtype, device)
        if key in self.pool and self.pool[key]:
            # Reuse existing tensor from pool
            tensor = self.pool[key].pop()
            tensor.zero_()  # Reset tensor values
            self.used.add(tensor)
            return tensor
        else:
            # Create new tensor
            tensor = torch.zeros(shape, dtype=dtype, device=device)
            self.used.add(tensor)
            return tensor
    
    def return_tensor(self, tensor):
        """Return a tensor to the pool for reuse"""
        if tensor in self.used:
            self.used.discard(tensor)
            key = (tensor.shape, tensor.dtype, tensor.device)
            if key not in self.pool:
                self.pool[key] = []
            # Only keep a limited number of tensors in pool to prevent memory bloat
            if len(self.pool[key]) < self.max_pool_size:
                self.pool[key].append(tensor)
    
    def clear(self):
        """Clear all tensors from the pool"""
        self.pool.clear()
        self.used.clear()

# Global tensor pool instance
_global_tensor_pool = TensorPool()

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
        self.gradient_history = deque(maxlen=5)        # Store recent gradients
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
        
        # Tensor pool for this device
        self.tensor_pool = _global_tensor_pool
    
    def set_local_dataset(self, dataset: torch.utils.data.Dataset, 
                         batch_size: int = 32, num_workers: int = 0):
        """Set local dataset and create dataloader"""
        self.local_dataset = dataset
        self.dataset_size = len(dataset)
        self.local_dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, 
            num_workers=num_workers, drop_last=False
        )
    
    def set_local_model(self, model: nn.Module):
        """Set local model for training"""
        self.local_model = model.clone() if hasattr(model, 'clone') else model
    
    def local_train(self, global_model: nn.Module, epochs: int = 1, 
                   learning_rate: float = 0.01, device: str = 'cpu') -> Dict[str, Any]:
        """
        Perform local SGD training as described in Algorithm 2.
        
        Returns training metrics and updated model weights.
        """
        if self.local_dataloader is None or not self.is_active:
            return {"success": False, "reason": "No dataset or inactive device"}

        start_time = time.time()
        
        # Initialize local model with global weights
        self.local_model.load_state_dict(global_model.state_dict())

        # Move model to appropriate device
        if device == 'cuda' and torch.cuda.is_available():
            self.local_model = self.local_model.cuda()
        else:
            self.local_model = self.local_model.cpu()
            device = 'cpu'  # Ensure consistency
        
        self.local_model.train()
        # Setup optimizer with the provided learning rate
        optimizer = torch.optim.Adam(self.local_model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # Move criterion to same device
        if device == 'cuda':
            criterion = criterion.cuda()
        
        total_loss = 0.0
        total_correct = 0
        num_batches = 0
        # Local training loop
        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(self.local_dataloader):
                # Move data to device
                if device == 'cuda':
                    data, target = data.cuda(), target.cuda()

                optimizer.zero_grad()
                if isinstance(self.local_model, ShakespeareLSTM):

                    output, _ = self.local_model(data)
                else:
                    output = self.local_model(data)
                predictions = torch.argmax(output, dim=1)
                correct = (predictions == target).float().sum()
                total_correct += correct.detach().cpu().numpy()
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                total_loss += loss.detach().item()
                num_batches += 1

        if device == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()

        training_time = time.time() - start_time
        self.training_times.append(training_time)
        
        # Calculate gradient for similarity computation (move to CPU for storage)
        local_gradient = self._compute_model_gradient(global_model, device)
        self.gradient_history.append(local_gradient)

        # Update participation history
        self.participation_history.append(1)  # Participated in this round
        
        # Calculate reliability score based on recent participation
        recent_participation = list(self.participation_history)[-10:]  # Last 10 rounds
        self.reliability_score = sum(recent_participation) / max(len(recent_participation), 1)
        
        # Move model back to CPU for state dict extraction
        if device == 'cuda':
            self.local_model = self.local_model.cpu()

        training_loss = total_loss / max(num_batches, 1)
        # print(f"({self.device_id}) Training finished. Loss: {training_loss:.3f}, Accuracy: {total_correct*100.0 / (self.dataset_size * epochs):.3f}%")
        return {
            "success": True,
            "model_weights": self.local_model.state_dict(),
            "training_loss": training_loss,
            "training_time": training_time,
            "dataset_size": self.dataset_size,
            "gradient": local_gradient
        }
    
    def _compute_model_gradient(self, global_model: nn.Module, device: str = 'cpu') -> Dict[str, torch.Tensor]:
        """Compute gradient difference between local and global model"""
        # Ensure both models are on CPU for comparison
        delta_weights = {}
        for (name, local_param), (_, global_param) in zip(self.local_model.state_dict().items(),
                                                          global_model.state_dict().items()):
            # Use in-place operations to reduce memory fragmentation
            local_cpu = local_param.detach().cpu()
            global_cpu = global_param.detach().cpu()
            
            # Try to reuse tensors from pool when possible
            if name in self.tensor_pool.pool:
                # Reuse existing tensor if available
                delta_tensor = self.tensor_pool.get_tensor(local_cpu.shape, local_cpu.dtype, 'cpu')
                torch.sub(local_cpu, global_cpu, out=delta_tensor)
                delta_weights[name] = delta_tensor.float()
            else:
                # Create new tensor with in-place operations where possible
                delta_tensor = self.tensor_pool.get_tensor(local_cpu.shape, local_cpu.dtype, 'cpu')
                torch.sub(local_cpu, global_cpu, out=delta_tensor)
                delta_weights[name] = delta_tensor.float()
        return delta_weights
    
    def compute_gradient_similarity(self, other_device: 'EdgeDevice') -> float:
        """
        Compute gradient similarity with another device.
        
        Implements S_data(d_i, d_j) from Equation (6) in the paper.
        """
        if len(self.gradient_history) == 0 or len(other_device.gradient_history) == 0:
            return 0.0

        def flatten_grads(grad):
            # Use tensor pool for flattened gradients
            flat_shape = sum(g.numel() for g in grad.values())
            flat_tensor = self.tensor_pool.get_tensor((flat_shape,), dtype=torch.float32, device='cpu')
            
            idx = 0
            for g in grad.values():
                flat_g = g.view(-1)
                # Use in-place copy to reduce memory fragmentation
                flat_tensor[idx:idx + flat_g.numel()].copy_(flat_g)
                idx += flat_g.numel()
            
            return flat_tensor[:idx]

        grad_i = self.gradient_history[-1]  # Most recent gradient
        grad_j = other_device.gradient_history[-1]
        
        # Use tensor pool for computations
        flat_grad_i = flatten_grads(grad_i)
        flat_grad_j = flatten_grads(grad_j)
        
        # Cosine similarity with in-place operations
        dot_product = torch.dot(flat_grad_i, flat_grad_j)
        norm_i = torch.norm(flat_grad_i)
        norm_j = torch.norm(flat_grad_j)
        
        # Return tensors to pool
        self.tensor_pool.return_tensor(flat_grad_i)
        self.tensor_pool.return_tensor(flat_grad_j)
        
        if norm_i == 0 or norm_j == 0:
            return 0.0
        
        similarity = dot_product / (norm_i * norm_j)
        return similarity.item()
    
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
        # Bandwidth compatibility
        bandwidth_sim = min(self.resources.bandwidth, other_device.resources.bandwidth) / \
                       max(self.resources.bandwidth, other_device.resources.bandwidth)
        
        # Latency similarity (assuming similar latency within zones)
        latency_diff = abs(self.communication_latency - other_device.communication_latency)
        latency_sim = np.exp(-latency_diff / 10.0)  # σ_n = 10ms
        
        return bandwidth_sim * latency_sim
    
    def estimate_data_quality(self) -> float:
        """
        Estimate data quality based on label consistency and distribution.
        
        This is a simplified implementation. In practice, this would involve
        more sophisticated data quality assessment techniques.
        """
        if self.local_dataset is None:
            return 0.0
        
        # Placeholder implementation
        # In practice, this could involve:
        # - Label noise detection
        # - Distribution similarity to global distribution
        # - Data completeness metrics
        
        # For now, randomly vary quality with some stability
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
        """
        Apply top-k sparsification for gradient compression.
        
        Implements the gradient compression technique from Section 4.4.
        """
        flat_grad = gradient.flatten()
        k = int(compression_rate * len(flat_grad))
        
        if k == 0:
            # Use tensor pool for zero tensor
            zero_tensor = self.tensor_pool.get_tensor(gradient.shape, gradient.dtype, gradient.device)
            zero_tensor.zero_()
            return zero_tensor, gradient
        
        # Get top-k indices
        _, top_k_indices = torch.topk(torch.abs(flat_grad), k)
        
        # Create sparse gradient using tensor pool
        sparse_grad = self.tensor_pool.get_tensor(flat_grad.shape, flat_grad.dtype, flat_grad.device)
        sparse_grad.zero_()
        sparse_grad.scatter_(0, top_k_indices, flat_grad[top_k_indices])
        
        # Error feedback buffer with in-place operations
        error_feedback = self.tensor_pool.get_tensor(flat_grad.shape, flat_grad.dtype, flat_grad.device)
        torch.sub(flat_grad, sparse_grad, out=error_feedback)
        
        if self.error_feedback_buffer is None:
            self.error_feedback_buffer = self.tensor_pool.get_tensor(flat_grad.shape, flat_grad.dtype, flat_grad.device)
            self.error_feedback_buffer.zero_()
        
        # Add previous error in-place
        sparse_grad.add_(self.error_feedback_buffer)
        self.error_feedback_buffer.copy_(error_feedback)
        
        # Reshape and return tensors from pool
        sparse_result = sparse_grad.reshape(gradient.shape)
        error_result = error_feedback.reshape(gradient.shape)
        
        return sparse_result, error_result

    def simulate_failure(self, failure_probability: float = 0.05):
        """Simulate device failure or unavailability"""
        if np.random.random() < failure_probability:
            self.is_active = False
            self.participation_history.append(0)
        else:
            self.is_active = True
    def simulate_repair(self, repair_probability: float = 0.2):
        """Simulate device repair to avoid complete outage"""
        if self.is_active:
            self.participation_history.append(1)
            return True
        elif np.random.random() < repair_probability:
            self.is_active = True
            self.participation_history.append(1)
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
                "participation_rate": sum(self.participation_history) / max(len(self.participation_history), 1)
            },
            "status": {
                "is_active": self.is_active,
                "last_training_time": self.training_times[-1] if self.training_times else 0,
                "communication_latency": self.communication_latency
            }
        }
    
    def __repr__(self):
        return f"EdgeDevice(id={self.device_id}, zone={self.zone_id}, location={self.location})"