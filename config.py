# Configuration file for ContinuumFL Framework
import numpy as np
from typing import Dict, Tuple, List

class ContinuumFLConfig:
    """Configuration class for ContinuumFL framework parameters"""
    
    def __init__(self):
        # System Architecture
        self.num_devices = 100
        self.num_zones = 20
        self.num_edge_servers = 20
        self.region_size = (100.0, 100.0)  # (width, height) in km
        
        # Zone Discovery Parameters
        self.similarity_weights = {
            'spatial': 0.4,    # ω1 - spatial proximity weight
            'data': 0.4,       # ω2 - data similarity weight  
            'network': 0.2     # ω3 - network similarity weight
        }
        self.similarity_threshold = 0.6  # θ - clustering threshold
        self.min_zone_size = 3           # n_min
        self.max_zone_size = 10          # n_max
        self.distance_scaling = 10.0     # σ - distance scaling parameter
        self.stability_tradeoff = 0.3    # γ - stability vs optimality
        self.stability_threshold = 0.1   # θ_stability
        self.stability_window = 10       # W - sliding window size
        
        # Federated Learning Parameters
        self.num_rounds = 200            # T - total training rounds
        self.local_epochs = 5            # τ_local - local training epochs
        self.learning_rate = 0.01        # η - learning rate
        self.batch_size = 32
        self.momentum = 0.9
        self.weight_decay = 1e-4
        
        # Aggregation Parameters
        self.spatial_regularization = 0.1  # λ - spatial regularization
        self.momentum_eta = 0.9             # η - momentum for correlation update
        self.staleness_penalty = 0.01       # μ - staleness penalty
        self.max_staleness = 5             # τ_max - maximum staleness
        self.fairness_strength = 0.5       # α_fair - fairness enforcement
        
        # Communication Optimization
        self.compression_rate = 0.1      # κ - gradient compression rate (top-k)
        self.quantization_bits = 8       # bits for delta encoding
        self.cache_threshold = 0.95      # τ_cache - caching stability threshold
        
        # Resource Parameters
        self.total_bandwidth = 1000.0    # MB/s - total available bandwidth
        self.deadline_constraint = 60    # seconds - round deadline
        
        # Device Heterogeneity Simulation
        self.device_compute_range = (1.0, 10.0)     # GFLOPS range
        self.device_memory_range = (1.0, 8.0)       # GB range
        self.device_bandwidth_range = (10.0, 100.0) # Mbps range
        self.intra_zone_latency_range = (5, 15)     # ms range
        self.inter_zone_latency_range = (20, 100)   # ms range
        
        # Dataset Configuration
        self.dataset_name = 'cifar100'   # Options: 'cifar100', 'femnist', 'shakespeare'
        self.max_samples = 100            # Limit Dataset-Size (use -1 for full dataset)
        self.data_distribution = 'dirichlet'  # Data distribution type
        self.intra_zone_alpha = 10       # Dirichlet α for intra-zone
        self.inter_zone_alpha = 0.3      # Dirichlet α for inter-zone
        self.train_test_split = 0.8
        
        # Model Configuration
        self.model_configs = {
            'cifar100': {
                'model_type': 'resnet18',
                'num_classes': 100,
                'input_shape': (3, 32, 32)
            },
            'femnist': {
                'model_type': 'cnn',
                'num_classes': 62,
                'input_shape': (1, 28, 28)
            },
            'shakespeare': {
                'model_type': 'lstm',
                'vocab_size': 80,
                'embedding_dim': 64,
                'hidden_dim': 256,
                'num_layers': 3
            }
        }
        
        # Logging and Visualization
        self.log_level = 'INFO'
        self.save_logs = True
        self.log_dir = './logs'
        self.checkpoint_dir = './checkpoints'
        self.results_dir = './results'
        self.plot_interval = 10          # rounds between plotting
        self.save_interval = 50          # rounds between checkpoints
        
        # Experimental Settings
        self.random_seed = 42
        self.device = 'cuda'             # 'cuda' or 'cpu'
        self.num_workers = 4             # for data loading
        
        # Baseline Comparisons
        self.baselines = ['fedavg', 'fedprox', 'hierfl', 'clusterfl']
        
        # Advanced Features
        self.enable_differential_privacy = False
        self.dp_epsilon = 1.0
        self.dp_delta = 1e-5
        self.enable_secure_aggregation = False
        
    def get_model_config(self) -> Dict:
        """Get model configuration for current dataset"""
        return self.model_configs.get(self.dataset_name, self.model_configs['cifar100'])
    
    def update_config(self, **kwargs):
        """Update configuration parameters"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Warning: Unknown configuration parameter: {key}")
    
    def validate_config(self) -> bool:
        """Validate configuration parameters"""
        assert self.num_zones <= self.num_devices, "Number of zones cannot exceed number of devices"
        assert self.min_zone_size <= self.max_zone_size, "Min zone size must be <= max zone size"
        assert sum(self.similarity_weights.values()) == 1.0, "Similarity weights must sum to 1.0"
        assert 0 <= self.fairness_strength <= 1.0, "Fairness strength must be in [0, 1]"
        assert 0 < self.compression_rate <= 1.0, "Compression rate must be in (0, 1]"
        return True
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    def save_config(self, filepath: str):
        """Save configuration to file"""
        import json
        config_dict = self.to_dict()
        # Convert numpy arrays to lists for JSON serialization
        for key, value in config_dict.items():
            if isinstance(value, np.ndarray):
                config_dict[key] = value.tolist()
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load_config(cls, filepath: str):
        """Load configuration from file"""
        import json
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        config = cls()
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        return config

# Default configuration instance
config = ContinuumFLConfig()