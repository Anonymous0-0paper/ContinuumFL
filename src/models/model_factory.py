"""
Model definitions for ContinuumFL framework.
Supports ResNet for CIFAR-100, CNN for FEMNIST, and LSTM for Shakespeare.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List

class ResNet18(nn.Module):
    """ResNet-18 model for CIFAR-100"""
    
    def __init__(self, num_classes: int = 100):
        super(ResNet18, self).__init__()
        self.num_classes = num_classes
        
        # Basic ResNet-18 architecture adapted for CIFAR-100
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # ResNet blocks
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_layer(self, in_channels: int, out_channels: int, blocks: int, stride: int = 1):
        """Create a ResNet layer"""
        layers = []
        
        # First block may have stride > 1
        layers.append(BasicBlock(in_channels, out_channels, stride))
        
        # Subsequent blocks
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels, 1))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x
    
    def clone(self):
        """Create a deep copy of the model"""
        cloned_model = ResNet18(self.num_classes)
        cloned_model.load_state_dict(self.state_dict())
        return cloned_model

class BasicBlock(nn.Module):
    """Basic ResNet block"""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(BasicBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class FEMNISTNet(nn.Module):
    """CNN model for FEMNIST dataset"""
    
    def __init__(self, num_classes: int = 62):
        super(FEMNISTNet, self).__init__()
        self.num_classes = num_classes
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.dropout1 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(512, 128)
        self.dropout2 = nn.Dropout(0.5)
        
        self.fc3 = nn.Linear(128, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Convolutional layers
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        
        # Flatten for fully connected layers
        x = torch.flatten(x, 1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        
        x = self.fc3(x)
        
        return x
    
    def clone(self):
        """Create a deep copy of the model"""
        cloned_model = FEMNISTNet(self.num_classes)
        cloned_model.load_state_dict(self.state_dict())
        return cloned_model

class ShakespeareLSTM(nn.Module):
    """LSTM model for Shakespeare next-character prediction"""
    
    def __init__(self, vocab_size: int = 80, embedding_dim: int = 64, 
                 hidden_dim: int = 256, num_layers: int = 3, dropout: float = 0.2):
        super(ShakespeareLSTM, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM layers
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights"""
        # Initialize embedding
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        
        # Initialize LSTM weights
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # Set forget gate bias to 1
                n = param.size(0)
                param.data[(n//4):(n//2)].fill_(1)
        
        # Initialize linear layer
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
    
    def forward(self, x, hidden=None):
        # x shape: (batch_size, seq_length)
        batch_size, seq_length = x.size()
        
        # Embedding
        embedded = self.embedding(x)  # (batch_size, seq_length, embedding_dim)
        
        # LSTM
        if hidden is None:
            lstm_out, hidden = self.lstm(embedded)
        else:
            lstm_out, hidden = self.lstm(embedded, hidden)
        
        # Take the last time step output
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_dim)
        
        # Apply dropout and linear layer
        output = self.dropout(last_output)
        output = self.fc(output)  # (batch_size, vocab_size)
        
        return output, hidden
    
    def init_hidden(self, batch_size: int, device: torch.device):
        """Initialize hidden state"""
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        return (h0, c0)
    
    def clone(self):
        """Create a deep copy of the model"""
        cloned_model = ShakespeareLSTM(
            self.vocab_size, self.embedding_dim, 
            self.hidden_dim, self.num_layers
        )
        cloned_model.load_state_dict(self.state_dict())
        return cloned_model

class ModelFactory:
    """Factory class to create models based on configuration"""
    
    @staticmethod
    def create_model(config) -> nn.Module:
        """Create model based on dataset and configuration"""
        dataset_name = config.dataset_name.lower()
        model_config = config.get_model_config()
        
        if dataset_name == 'cifar100':
            return ResNet18(num_classes=model_config['num_classes'])
        
        elif dataset_name == 'femnist':
            return FEMNISTNet(num_classes=model_config['num_classes'])
        
        elif dataset_name == 'shakespeare':
            return ShakespeareLSTM(
                vocab_size=60,
                embedding_dim=model_config.get('embedding_dim', 64),
                hidden_dim=model_config.get('hidden_dim', 256),
                num_layers=model_config.get('num_layers', 3)
            )
        
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    @staticmethod
    def get_model_info(model: nn.Module) -> Dict[str, Any]:
        """Get model information and statistics"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Calculate model size in MB
        model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
        
        return {
            "model_type": model.__class__.__name__,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": model_size_mb,
            "layers": len(list(model.modules()))
        }
    
    @staticmethod
    def save_model(model: nn.Module, filepath: str, include_optimizer: bool = False, 
                  optimizer: Optional[torch.optim.Optimizer] = None):
        """Save model state"""
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "model_class": model.__class__.__name__,
            "model_config": ModelFactory.get_model_info(model)
        }
        
        if include_optimizer and optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        
        torch.save(checkpoint, filepath)
    
    @staticmethod
    def load_model(filepath: str, model: nn.Module, 
                  optimizer: Optional[torch.optim.Optimizer] = None) -> Dict[str, Any]:
        """Load model state"""
        checkpoint = torch.load(filepath, map_location='cpu')
        
        model.load_state_dict(checkpoint["model_state_dict"])
        
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        return checkpoint

# Additional utility functions for model operations

def model_parameters_to_vector(model: nn.Module) -> torch.Tensor:
    """Convert model parameters to a single vector"""
    return torch.cat([param.flatten() for param in model.parameters()])

def vector_to_model_parameters(vector: torch.Tensor, model: nn.Module):
    """Set model parameters from a vector"""
    pointer = 0
    for param in model.parameters():
        num_params = param.numel()
        param.data = vector[pointer:pointer + num_params].view(param.shape)
        pointer += num_params

def compute_model_norm(model: nn.Module) -> float:
    """Compute L2 norm of model parameters"""
    total_norm = 0.0
    for param in model.parameters():
        total_norm += param.data.norm(2).item() ** 2
    return total_norm ** 0.5

def compute_model_difference_norm(model1: nn.Module, model2: nn.Module) -> float:
    """Compute L2 norm of difference between two models"""
    total_norm = 0.0
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        diff = p1.data - p2.data
        total_norm += diff.norm(2).item() ** 2
    return total_norm ** 0.5

def add_noise_to_model(model: nn.Module, noise_std: float):
    """Add Gaussian noise to model parameters (for differential privacy)"""
    with torch.no_grad():
        for param in model.parameters():
            noise = torch.normal(0, noise_std, param.shape)
            param.add_(noise)

def clip_model_gradients(model: nn.Module, max_norm: float):
    """Clip model gradients to maximum norm"""
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

def compute_gradient_diversity(gradients: List[torch.Tensor]) -> float:
    """Compute diversity of gradients (variance measure)"""
    if len(gradients) < 2:
        return 0.0
    
    # Stack gradients
    stacked_grads = torch.stack(gradients)
    
    # Compute variance across gradients
    variance = torch.var(stacked_grads, dim=0)
    
    # Return mean variance
    return torch.mean(variance).item()

def estimate_model_memory_usage(model: nn.Module) -> Dict[str, float]:
    """Estimate memory usage for model training"""
    model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    
    # Rough estimates for training memory
    gradients_size = model_size  # Same size as parameters
    optimizer_size = model_size * 2  # Momentum + parameter copy
    activations_size = model_size * 0.5  # Rough estimate
    
    total_size = model_size + gradients_size + optimizer_size + activations_size
    
    return {
        "model_size_mb": model_size,
        "gradients_size_mb": gradients_size,
        "optimizer_size_mb": optimizer_size,
        "activations_size_mb": activations_size,
        "total_training_size_mb": total_size
    }