"""
Core device module for ContinuumFL framework.
Implements edge devices with spatial coordinates, resource constraints, and FL capabilities.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import time
import uuid
from collections import deque

from src.models.model_factory import ShakespeareLSTM


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
        self.gradient_history = deque(maxlen=10)        # Store recent gradients
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
        
        # Setup optimizer
        optimizer = torch.optim.SGD(self.local_model.parameters(), 
                                  lr=learning_rate, momentum=0.9, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        # Move criterion to same device
        if device == 'cuda':
            criterion = criterion.cuda()
        
        total_loss = 0.0
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
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
        
        training_time = time.time() - start_time
        self.training_times.append(training_time)
        
        # Calculate gradient for similarity computation (move to CPU for storage)
        local_gradient = self._compute_model_gradient(global_model, device)
        self.gradient_history.append(local_gradient.clone())
        
        # Update participation history
        self.participation_history.append(1)  # Participated in this round
        
        # Calculate reliability score based on recent participation
        recent_participation = list(self.participation_history)[-10:]  # Last 10 rounds
        self.reliability_score = sum(recent_participation) / max(len(recent_participation), 1)
        
        # Move model back to CPU for state dict extraction
        if device == 'cuda':
            self.local_model = self.local_model.cpu()
        
        return {
            "success": True,
            "model_weights": self.local_model.state_dict(),
            "training_loss": total_loss / max(num_batches, 1),
            "training_time": training_time,
            "dataset_size": self.dataset_size,
            "gradient": local_gradient
        }
    
    def _compute_model_gradient(self, global_model: nn.Module, device: str = 'cpu') -> torch.Tensor:
        """Compute gradient difference between local and global model"""
        # Ensure both models are on CPU for comparison
        if device == 'cuda':
            local_params = torch.cat([p.cpu().flatten() for p in self.local_model.parameters()])
        else:
            local_params = torch.cat([p.flatten() for p in self.local_model.parameters()])
        
        global_params = torch.cat([p.flatten() for p in global_model.parameters()])
        return local_params - global_params
    
    def compute_gradient_similarity(self, other_device: 'EdgeDevice') -> float:
        """
        Compute gradient similarity with another device.
        
        Implements S_data(d_i, d_j) from Equation (6) in the paper.
        """
        if len(self.gradient_history) == 0 or len(other_device.gradient_history) == 0:
            return 0.0
        
        grad_i = self.gradient_history[-1]  # Most recent gradient
        grad_j = other_device.gradient_history[-1]
        
        # Cosine similarity
        dot_product = torch.dot(grad_i.flatten(), grad_j.flatten())
        norm_i = torch.norm(grad_i.flatten())
        norm_j = torch.norm(grad_j.flatten())
        
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
            return torch.zeros_like(gradient), gradient
        
        # Get top-k indices
        _, top_k_indices = torch.topk(torch.abs(flat_grad), k)
        
        # Create sparse gradient
        sparse_grad = torch.zeros_like(flat_grad)
        sparse_grad[top_k_indices] = flat_grad[top_k_indices]
        
        # Error feedback buffer
        error_feedback = flat_grad - sparse_grad
        if self.error_feedback_buffer is None:
            self.error_feedback_buffer = torch.zeros_like(flat_grad)
        
        # Add previous error
        sparse_grad += self.error_feedback_buffer
        self.error_feedback_buffer = error_feedback
        
        return sparse_grad.reshape(gradient.shape), error_feedback.reshape(gradient.shape)
    
    def simulate_failure(self, failure_probability: float = 0.05):
        """Simulate device failure or unavailability"""
        if np.random.random() < failure_probability:
            self.is_active = False
            self.participation_history.append(0)  # Did not participate
        else:
            self.is_active = True
    
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