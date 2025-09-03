"""
ContinuumFL Coordinator - Main orchestrator for the spatial-aware federated learning framework.
Implements the complete ContinuumFL protocol from the paper.
"""
from asyncio import FIRST_COMPLETED
from concurrent.futures import ProcessPoolExecutor

import torch
import torch.nn as nn
import numpy as np
import time
import os
import json
from typing import Dict, List, Optional, Any
from collections import defaultdict, deque
import logging
from concurrent.futures import ThreadPoolExecutor, wait

from .core.device import EdgeDevice, DeviceResources
from .core.zone import Zone
from .core.zone_discovery import ZoneDiscovery
from .aggregation.hierarchical_aggregator import HierarchicalAggregator
from .data.federated_dataset import FederatedDataset
from .models.model_factory import ModelFactory
from .communication.compression import GradientCompressor
from .baselines.baseline_fl import BaselineFLMethods

class ContinuumFLCoordinator:
    """
    Main coordinator for ContinuumFL framework.
    
    Implements the complete spatial-aware federated learning protocol
    from Algorithm 2 in the paper.
    """
    
    def __init__(self, config):
        self.config = config
        
        # Setup logging
        self._setup_logging()
        
        # Core components
        self.zone_discovery = ZoneDiscovery(config)
        self.aggregator = HierarchicalAggregator(config)
        self.dataset = FederatedDataset(config)
        self.compressor = GradientCompressor(config)
        
        # System state
        self.devices: Dict[str, EdgeDevice] = {}
        self.zones: Dict[str, Zone] = {}
        self.global_model: Optional[nn.Module] = None
        
        # Training state
        self.current_round = 0
        self.is_training = False
        self.training_history = deque(maxlen=1000)
        
        # Performance tracking
        self.round_times = deque(maxlen=100)
        self.accuracies = deque(maxlen=1000)
        self.losses = deque(maxlen=1000)
        self.communication_costs = deque(maxlen=1000)
        
        # Device and zone statistics
        self.device_participation = defaultdict(int)
        self.zone_performance = defaultdict(list)
        
        # Baseline comparison
        self.baseline_methods = BaselineFLMethods(config) if hasattr(config, 'baselines') else None
        
        # Set random seeds for reproducibility
        self._set_random_seeds()
        
        self.logger.info("ContinuumFL Coordinator initialized")
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_dir = self.config.log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(log_dir, 'continuumfl.log')),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger('ContinuumFL')
    
    def _set_random_seeds(self):
        """Set random seeds for reproducibility"""
        torch.manual_seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config.random_seed)
    
    def initialize_system(self):
        """Initialize the ContinuumFL system"""
        self.logger.info("Initializing ContinuumFL system...")
        
        # Check and setup device
        self._setup_compute_device()
        
        # 1. Create and prepare dataset
        self.logger.info("Preparing dataset...")
        self.dataset.download_and_prepare()
        
        # 2. Create global model
        self.logger.info("Creating global model...")
        self.global_model = ModelFactory.create_model(self.config)
        
        # Move model to appropriate device
        if self.config.device == 'cuda' and torch.cuda.is_available():
            self.global_model = self.global_model.cuda()
            self.logger.info(f"Model moved to GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.logger.info("Model will use CPU")
        
        self.aggregator.set_global_model(self.global_model)
        
        # Log model information
        model_info = ModelFactory.get_model_info(self.global_model)
        self.logger.info(f"Model: {model_info}")
        
        # 3. Create edge devices
        self.logger.info("Creating edge devices...")
        self._create_edge_devices()
        
        # 4. Initial zone discovery
        self.logger.info("Performing initial zone discovery...")
        device_list = list(self.devices.values())
        self.zones = self.zone_discovery.discover_zones(device_list)
        
        # 5. Distribute data to devices
        self.logger.info("Distributing data to devices...")
        self._distribute_data_to_devices()
        
        # 6. Setup device models
        self.logger.info("Setting up device models...")
        self._setup_device_models()
        
        self.logger.info("ContinuumFL system initialization complete")
        self.logger.info(f"System: {len(self.devices)} devices, {len(self.zones)} zones, Device: {self.config.device}")
    
    def _setup_compute_device(self):
        """Setup and verify compute device"""
        if self.config.device == 'cuda':
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                current_device = torch.cuda.current_device()
                device_name = torch.cuda.get_device_name(current_device)
                memory_total = torch.cuda.get_device_properties(current_device).total_memory / (1024**3)
                
                self.logger.info(f"CUDA device available: {device_name}")
                self.logger.info(f"GPU memory: {memory_total:.2f}GB")
                
                # Clear GPU cache
                torch.cuda.empty_cache()
            else:
                self.logger.warning("CUDA requested but not available, falling back to CPU")
                self.config.device = 'cpu'
        else:
            self.logger.info("Using CPU for computation")
    
    def _create_edge_devices(self):
        """Create edge devices with heterogeneous resources and spatial distribution"""
        region_width, region_height = self.config.region_size
        num_physical_zones = self.config.num_zones
        physical_zone_locations = []
        physical_zone_sizes = []
        for physical_zone in range(num_physical_zones):
            # Generate random location within region
            location = (
                np.random.uniform(0, region_width),
                np.random.uniform(0, region_height)
            )
            size = np.random.uniform(1, 5)

            physical_zone_locations.append(location)
            physical_zone_sizes.append(size)

        for i in range(self.config.num_devices):
            device_id = f"device_{i}"
            physical_zone_assignment = np.random.choice(num_physical_zones, 1)[0]

            # Generate random location within zone
            x = np.random.normal(physical_zone_locations[physical_zone_assignment][0], physical_zone_sizes[physical_zone_assignment])
            x = np.clip(x, 0, region_width)

            y = np.random.normal(physical_zone_locations[physical_zone_assignment][1], physical_zone_sizes[physical_zone_assignment])
            y = np.clip(y, 0, region_height)

            location = (x, y)

            # Generate heterogeneous resources
            compute_capacity = np.random.uniform(*self.config.device_compute_range)
            memory_capacity = np.random.uniform(*self.config.device_memory_range)
            bandwidth = np.random.uniform(*self.config.device_bandwidth_range)
            
            resources = DeviceResources(compute_capacity, memory_capacity, bandwidth)
            
            # Create device
            device = EdgeDevice(device_id, location, resources)
            
            # Set communication latency (will be updated based on zone assignment)
            device.communication_latency = np.random.uniform(*self.config.intra_zone_latency_range)
            
            self.devices[device_id] = device
        
        self.logger.info(f"Created {len(self.devices)} edge devices")
    
    def _distribute_data_to_devices(self):
        """Distribute data to devices based on zone assignments"""
        # Create zone-to-devices mapping
        zone_device_mapping = {}
        for zone_id, zone in self.zones.items():
            zone_device_mapping[zone_id] = list(zone.device_ids)
        
        # Distribute data
        device_datasets = self.dataset.distribute_data_to_devices(
            list(self.devices.keys()), zone_device_mapping
        )

        zone_dataset_sizes = {}
        # Assign datasets to devices
        for device_id, (train_subset, test_subset) in device_datasets.items():
            if device_id in self.devices:
                device = self.devices[device_id]
                if train_subset:
                    device.set_local_dataset(train_subset, self.config.batch_size)
                    device.estimate_data_quality()

                    if device.zone_id not in zone_dataset_sizes:
                        zone_dataset_sizes[device.zone_id] = device.dataset_size
                    else:
                        zone_dataset_sizes[device.zone_id] += device.dataset_size
        for zone_id, zone in self.zones.items():
            zone.total_dataset_size = zone_dataset_sizes[zone_id]
        # Analyze data distribution
        distribution_analysis = self.dataset.analyze_data_distribution(zones=self.zones)
        self.logger.info(f"Data distribution: {distribution_analysis}")
    
    def _setup_device_models(self):
        """Setup local models for each device"""
        for device in self.devices.values():
            if device.local_dataset:
                device.set_local_model(self.global_model)
    
    def run_federated_learning(self) -> Dict[str, Any]:
        """
        Run the complete ContinuumFL federated learning process.
        
        Implements Algorithm 2: ContinuumFL Aggregation Protocol.
        """
        self.logger.info(f"Starting federated learning for {self.config.num_rounds} rounds")
        
        self.is_training = True
        training_start_time = time.time()
        
        try:
            for round_num in range(self.config.num_rounds):
                self.current_round = round_num
                round_start_time = time.time()
                
                self.logger.info(f"\n=== Round {round_num + 1}/{self.config.num_rounds} ===")
                
                # 1. Zone discovery/update (periodic)
                if round_num % 10 == 0 and round_num > 0:  # Every 10 rounds
                    self.logger.info("Updating zone assignments...")
                    device_list = [d for d in self.devices.values() if d.is_active]
                    self.zones = self.zone_discovery.adaptive_zone_update(device_list, self.zones)

                # 2. Device sampling and local training
                participating_devices = self._sample_participating_devices()
                device_updates = self._perform_local_training(participating_devices)
                
                # 3. Hierarchical aggregation
                if device_updates:
                    global_weights, aggregation_stats = self.aggregator.federated_aggregation_round(
                        self.zones, device_updates
                    )
                    
                    # Update global model
                    if global_weights:
                        model_state = self.global_model.state_dict()
                        for k in model_state.keys():
                            if model_state[k].dtype.is_floating_point:
                                model_state[k] += global_weights[k]
                        self.global_model.load_state_dict(model_state)
                else:
                    self.logger.warning("No device updates received in this round")
                    aggregation_stats = {"participating_devices": 0, "participating_zones": 0}
                
                # 4. Evaluation
                round_metrics = self._evaluate_round(participating_devices, aggregation_stats)
                
                # 5. Broadcast updated model
                self._broadcast_global_model()
                
                # 6. Track performance
                round_time = time.time() - round_start_time
                self.round_times.append(round_time)
                self.training_history.append(round_metrics)
                
                # Log round results
                self._log_round_results(round_num, round_metrics, round_time)
                
                # Save checkpoint periodically
                if (round_num + 1) % self.config.save_interval == 0:
                    self._save_checkpoint(round_num + 1)
        
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise
        finally:
            self.is_training = False
        
        total_training_time = time.time() - training_start_time
        
        # Final evaluation and results
        final_results = self._finalize_training(total_training_time)
        
        self.logger.info("Federated learning completed")
        return final_results
    
    def _sample_participating_devices(self) -> List[str]:
        """Sample devices for participation in current round"""
        # Simple participation strategy: random sampling with availability check
        available_devices = [
            device_id for device_id, device in self.devices.items()
            if device.is_active and device.local_dataset
        ]
        
        # Sample fraction of available devices
        participation_rate = 0.7  # 70% participation rate
        num_participants = max(1, int(participation_rate * len(available_devices)))
        
        participating = np.random.choice(
            available_devices, size=num_participants, replace=False
        ).tolist()
        
        return participating

    def _perform_local_training(self, participating_devices: List[str]) -> Dict[str, Dict[str, torch.Tensor]]:
        """Perform local training on participating devices"""
        device_updates = {}

        from concurrent.futures import ThreadPoolExecutor, as_completed
        if self.config.device == 'cuda':
            max_workers = len(self.zones)
        else:
            max_workers = min(len(self.zones), os.cpu_count())
        args = {
            "comp_device": self.config.device,
            "model": self.global_model,
            "learning_rate": self.config.learning_rate,
            "epochs": self.config.epochs,
            "device_participation": self.device_participation
        }
        def _add_participating_devices(zone):
            import copy
            args_copy = copy.deepcopy(args)
            devices = set(participating_devices).intersection(zone.devices.keys())
            args_copy["participating_devices"] = devices
            return args_copy

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(zone.perform_local_training, _add_participating_devices().copy()) for zone_id, zone in self.zones.items()]
            while futures:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for f in done:
                    zone_id, aggregated_weights = f.result()
                    # Perform Inter Zone Aggregation TODO
                    futures.append(executor.submit(self.zones[zone_id].perform_local_training, args))

        self.logger.info(f"Local training completed: {len(device_updates)}/{len(participating_devices)} devices")
        return device_updates
    
    def _evaluate_round(self, participating_devices: List[str], 
                       aggregation_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate the current round performance"""
        # Global evaluation
        global_metrics = self._evaluate_global_model()
        
        # Zone-level evaluation
        zone_metrics = self._evaluate_zones()
        
        # Communication cost estimation
        comm_cost = aggregation_stats.get("communication_cost", 0.0)
        self.communication_costs.append(comm_cost)
        
        # Compile round metrics
        round_metrics = {
            "round": self.current_round,
            "global_accuracy": global_metrics.get("accuracy", 0.0),
            "global_loss": global_metrics.get("loss", float('inf')),
            "participating_devices": len(participating_devices),
            "participating_zones": aggregation_stats.get("participating_zones", 0),
            "communication_cost_mb": comm_cost,
            "zone_metrics": zone_metrics,
            "aggregation_time": aggregation_stats.get("aggregation_time", 0.0)
        }
        
        # Store metrics
        self.accuracies.append(round_metrics["global_accuracy"])
        self.losses.append(round_metrics["global_loss"])
        
        return round_metrics
    
    def _evaluate_global_model(self) -> Dict[str, float]:
        """Evaluate global model on test data"""
        if not hasattr(self.dataset, 'test_data') or self.dataset.test_data is None:
            return {"accuracy": 0.0, "loss": float('inf')}
        
        self.global_model.eval()
        test_dataloader = self.dataset.get_global_dataloader(
            batch_size=self.config.batch_size, is_train=False
        )
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        criterion = nn.CrossEntropyLoss()
        
        # Move criterion to same device as model
        if self.config.device == 'cuda' and torch.cuda.is_available():
            criterion = criterion.cuda()
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_dataloader):
                # Move data to appropriate device
                if self.config.device == 'cuda' and torch.cuda.is_available():
                    data, target = data.cuda(), target.cuda()
                
                output = self.global_model(data)
                
                # Handle different model outputs
                if isinstance(output, tuple):  # LSTM returns (output, hidden)
                    output = output[0]
                
                loss = criterion(output, target)
                total_loss += loss.item()
                
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

                if batch_idx % 20 == 0:
                    print(f"Loss: {loss}")
        accuracy = correct / max(total, 1)
        avg_loss = total_loss / max(len(test_dataloader), 1)
        
        self.global_model.train()
        
        return {"accuracy": accuracy, "loss": avg_loss}
    
    def _evaluate_zones(self) -> Dict[str, Dict[str, float]]:
        """Evaluate performance for each zone"""
        zone_metrics = {}
        
        for zone_id, zone in self.zones.items():
            if not zone.devices:
                continue
            
            # Collect zone devices' test data
            zone_test_data = []
            zone_test_labels = []
            
            for device in zone.devices.values():
                if device.local_dataset and hasattr(device, 'local_dataloader'):
                    # Get test data if available
                    test_dataloader = self.dataset.get_device_dataloader(
                        device.device_id, is_train=False
                    )
                    if test_dataloader:
                        for data, labels in test_dataloader:
                            zone_test_data.append(data)
                            zone_test_labels.append(labels)
            
            if zone_test_data:
                # Evaluate on zone data
                zone_accuracy = self._evaluate_on_data(zone_test_data, zone_test_labels)
                zone_metrics[zone_id] = {"accuracy": zone_accuracy}
            else:
                zone_metrics[zone_id] = {"accuracy": 0.0}
        
        return zone_metrics
    
    def _evaluate_on_data(self, data_list: List[torch.Tensor], 
                         labels_list: List[torch.Tensor]) -> float:
        """Evaluate model on given data"""
        if not data_list:
            return 0.0
        
        self.global_model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, labels in zip(data_list, labels_list):
                if self.config.device == 'cuda' and torch.cuda.is_available():
                    data, labels = data.cuda(), labels.cuda()
                
                output = self.global_model(data)
                
                if isinstance(output, tuple):
                    output = output[0]
                
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(labels.view_as(pred)).sum().item()
                total += labels.size(0)
        
        self.global_model.train()
        return correct / max(total, 1)
    
    def _broadcast_global_model(self):
        """Broadcast updated global model to all devices"""
        for device in self.devices.values():
            if device.is_active and device.local_model:
                device.local_model.load_state_dict(self.global_model.state_dict())
    
    def _log_round_results(self, round_num: int, round_metrics: Dict[str, Any], round_time: float):
        """Log results for current round"""
        accuracy = round_metrics["global_accuracy"]
        loss = round_metrics["global_loss"]
        participating_devices = round_metrics["participating_devices"]
        participating_zones = round_metrics["participating_zones"]
        comm_cost = round_metrics["communication_cost_mb"]
        
        self.logger.info(
            f"Round {round_num + 1} Results: "
            f"Accuracy={accuracy * 100:.4f}%, Loss={loss:.4f}, "
            f"Devices={participating_devices}, Zones={participating_zones}, "
            f"CommCost={comm_cost:.2f}MB, Time={round_time:.2f}s"
        )
        
        # Log zone-specific results
        for zone_id, zone_metrics in round_metrics["zone_metrics"].items():
            zone_acc = zone_metrics.get("accuracy", 0.0)
            self.logger.info(f"  Zone {zone_id}: Accuracy={zone_acc:.4f}")
    
    def _save_checkpoint(self, round_num: int):
        """Save training checkpoint"""
        checkpoint_dir = self.config.checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_round_{round_num}.pt")
        
        checkpoint = {
            "round": round_num,
            "global_model_state": self.global_model.state_dict(),
            "training_history": list(self.training_history),
            "device_participation": dict(self.device_participation),
            "config": self.config.to_dict()
        }
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def _finalize_training(self, total_time: float) -> Dict[str, Any]:
        """Finalize training and compile results"""
        # Final global evaluation
        final_metrics = self._evaluate_global_model()
        
        # Compile training statistics
        training_stats = {
            "total_rounds": self.current_round + 1,
            "total_training_time": total_time,
            "final_accuracy": final_metrics.get("accuracy", 0.0),
            "final_loss": final_metrics.get("loss", float('inf')),
            "average_round_time": np.mean(self.round_times) if self.round_times else 0.0,
            "total_communication_cost": sum(self.communication_costs),
            "device_participation_stats": dict(self.device_participation),
            "convergence_rounds": self._analyze_convergence()
        }
        
        # Zone discovery statistics
        discovery_stats = self.zone_discovery.get_discovery_stats()
        training_stats["zone_discovery_stats"] = discovery_stats
        
        # Aggregation statistics
        aggregation_stats = self.aggregator.get_aggregation_stats()
        training_stats["aggregation_stats"] = aggregation_stats
        
        # Save final results
        self._save_final_results(training_stats)
        
        return training_stats
    
    def _analyze_convergence(self) -> int:
        """Analyze convergence behavior"""
        if len(self.accuracies) < 10:
            return -1
        
        # Simple convergence detection: accuracy stabilizes
        recent_accuracies = list(self.accuracies)[-10:]
        accuracy_variance = np.var(recent_accuracies)
        
        # If variance is low, consider converged
        if accuracy_variance < 0.001:  # Threshold for convergence
            return len(self.accuracies) - 10
        
        return -1  # Not converged
    
    def _save_final_results(self, training_stats: Dict[str, Any]):
        """Save final training results"""
        results_dir = self.config.results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        # Save training statistics
        results_path = os.path.join(results_dir, "training_results.json")
        with open(results_path, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            json_stats = self._convert_for_json(training_stats)
            json.dump(json_stats, f, indent=2)
        
        # Save training history
        history_path = os.path.join(results_dir, "training_history.json")
        with open(history_path, 'w') as f:
            history = [self._convert_for_json(round_data) for round_data in self.training_history]
            json.dump(history, f, indent=2)
        
        # Save final model
        model_path = os.path.join(results_dir, "final_model.pt")
        torch.save(self.global_model.state_dict(), model_path)
        
        self.logger.info(f"Final results saved to {results_dir}")
    
    def _convert_for_json(self, obj):
        """Convert numpy types to JSON-serializable types"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: self._convert_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(item) for item in obj]
        else:
            return obj
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        active_devices = len([d for d in self.devices.values() if d.is_active])
        operational_zones = len([z for z in self.zones.values() if z.is_operational])
        
        return {
            "is_training": self.is_training,
            "current_round": self.current_round,
            "total_devices": len(self.devices),
            "active_devices": active_devices,
            "total_zones": len(self.zones),
            "operational_zones": operational_zones,
            "recent_accuracy": self.accuracies[-1] if self.accuracies else 0.0,
            "recent_loss": self.losses[-1] if self.losses else float('inf'),
            "total_communication_cost": sum(self.communication_costs)
        }
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training from checkpoint"""
        checkpoint = torch.load(checkpoint_path)
        
        self.global_model.load_state_dict(checkpoint["global_model_state"])
        self.current_round = checkpoint["round"]
        self.training_history = deque(checkpoint["training_history"], maxlen=1000)
        self.device_participation = defaultdict(int, checkpoint["device_participation"])
        
        # Update device models
        self._broadcast_global_model()
        
        self.logger.info(f"Checkpoint loaded from round {self.current_round}")
    
    def run_baseline_comparison(self) -> Dict[str, Any]:
        """Run comparison with baseline methods"""
        if not self.baseline_methods:
            self.logger.warning("Baseline methods not configured")
            return {}
        
        self.logger.info("Running baseline comparisons...")
        
        baseline_results = {}
        
        for baseline_name in self.config.baselines:
            self.logger.info(f"Running {baseline_name}...")
            
            # Reset system for baseline
            self._reset_for_baseline()
            
            # Run baseline method
            result = self.baseline_methods.run_method(
                baseline_name, self.devices, self.global_model, self.dataset
            )
            
            baseline_results[baseline_name] = result
            
            self.logger.info(f"{baseline_name} completed: Accuracy={result.get('final_accuracy', 0.0):.4f}")
        
        return baseline_results
    
    def _reset_for_baseline(self):
        """Reset system state for baseline comparison"""
        # Reset global model
        self.global_model = ModelFactory.create_model(self.config)
        
        # Reset device models
        for device in self.devices.values():
            if device.local_model:
                device.set_local_model(self.global_model)
        
        # Reset tracking variables
        self.current_round = 0