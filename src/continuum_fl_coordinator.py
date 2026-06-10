"""
ContinuumFL Coordinator - Main orchestrator for the spatial-aware federated learning framework.
Implements the complete ContinuumFL protocol from the paper.
"""

import csv
import torch
import torch.nn as nn
import numpy as np
import time
import os
import json
from typing import Dict, List, Optional, Any
from collections import defaultdict, deque
import logging
from concurrent.futures import ThreadPoolExecutor, Future, wait, ALL_COMPLETED, FIRST_COMPLETED

from torch import Tensor

from .core.device import EdgeDevice, DeviceResources
from .core.zone import Zone
from .core.zone_discovery import ZoneDiscovery
from .aggregation.hierarchical_aggregator import HierarchicalAggregator
from .data.federated_dataset import FederatedDataset
from .models.model_factory import ModelFactory
from .communication.compression import GradientCompressor
from .baselines.baseline_fl import BaselineFLMethods
from .memory_log.memory_log import log_mem
from .debug.fl_debugger import FLDebugger

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
        self.standalone_devices: Dict[str, EdgeDevice] = {}
        self.standalone_device_zone: Zone = None
        self.zones: Dict[str, Zone] = {}
        self.global_model: Optional[nn.Module] = None

        # Aggregation Settings
        self.async_aggregation = config.async_aggregation

        # Device Settings
        self.enable_failure = config.enable_failure
        self.zone_failure_probability = config.zone_failure_probability
        self.device_failure_probability = config.device_failure_probability

        # Training state
        self.current_round = 0
        self.is_training = False
        self.training_history = deque(maxlen=1000)

        # Per-dataset LR scheduler state
        self.current_lr = config.learning_rate
        self._lr_plateau_best_loss = float('inf')
        self._lr_plateau_counter = 0
        
        # Performance tracking
        self.round_times = deque(maxlen=1000)
        self.accuracies = deque(maxlen=1000)
        self.losses = deque(maxlen=1000)
        self.communication_costs = deque(maxlen=1000)

        # Best / last checkpoint tracking
        self.best_accuracy = 0.0
        self.best_accuracy_round = 0
        self.last_accuracy = 0.0

        # Early stopping state
        self._es_counter = 0
        self._es_best = 0.0
        
        # Device and zone statistics
        self.device_participation = defaultdict(int)
        self.zone_performance = defaultdict(list)
        
        # Baseline comparison
        self.baseline_methods = BaselineFLMethods(config) if hasattr(config, 'baselines') else None

        # Debug / diagnostics
        self.debugger = FLDebugger(config)

        # Set random seeds for reproducibility
        self._set_random_seeds()

        self.logger.info("ContinuumFL Coordinator initialized")
    
    def _setup_logging(self):
        """Setup logging configuration.

        Two handlers:
          - console + continuumfl.log : INFO and above (clean operational output)
          - continuumfl_debug.log     : DEBUG and above (full diagnostics from FLDebugger
                                        and all submodule loggers)
        """
        log_dir = self.config.log_dir
        os.makedirs(log_dir, exist_ok=True)

        fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # INFO handler — console + main log file
        info_file_handler = logging.FileHandler(os.path.join(log_dir, 'continuumfl.log'))
        info_file_handler.setLevel(logging.INFO)
        info_file_handler.setFormatter(fmt)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, self.config.log_level))
        console_handler.setFormatter(fmt)

        # DEBUG handler — separate detailed file, does not go to console
        debug_file_handler = logging.FileHandler(os.path.join(log_dir, 'continuumfl_debug.log'))
        debug_file_handler.setLevel(logging.DEBUG)
        debug_file_handler.setFormatter(fmt)

        root_logger = logging.getLogger('ContinuumFL')
        root_logger.setLevel(logging.DEBUG)  # capture everything; handlers filter
        root_logger.handlers.clear()
        root_logger.addHandler(info_file_handler)
        root_logger.addHandler(console_handler)
        root_logger.addHandler(debug_file_handler)

        self.logger = root_logger
        self.logger.info(
            f"Logging initialised: INFO → console + continuumfl.log | "
            f"DEBUG → continuumfl_debug.log"
        )
    
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
        if self.config.device.startswith('cuda') and torch.cuda.is_available():
            self.global_model = self.global_model.to(self.config.device)
            self.logger.info(f"Model moved to GPU: {torch.cuda.get_device_name(self.config.device)}")
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
        if self.config.device.startswith('cuda'):
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
            size = np.random.uniform(2, 5)

            physical_zone_locations.append(location)
            physical_zone_sizes.append(size)

        for i in range(self.config.num_devices):
            device_id = f"device_{i}"
            physical_zone_assignment = np.random.choice(num_physical_zones, 1)[0]

            # Generate random location within zone
            x = np.random.normal(physical_zone_locations[physical_zone_assignment][0],
                                 physical_zone_sizes[physical_zone_assignment])
            x = np.clip(x, 0, region_width)

            y = np.random.normal(physical_zone_locations[physical_zone_assignment][1],
                                 physical_zone_sizes[physical_zone_assignment])
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
            zone_device_mapping
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
        # distribution_analysis = self.dataset.analyze_data_distribution(zones=self.zones)

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

                dataset = self.config.dataset_name.lower()
                if dataset == 'femnist':
                    self.current_lr = 0.001
                elif dataset == 'cifar100':
                    self.current_lr = 0.01
                elif dataset == 'shakespeare':
                    self.current_lr = 0.0005
                else:
                    self.current_lr = self.config.learning_rate
                args = {
                    "comp_device": self.config.device,
                    "model": self.global_model,
                    "learning_rate": self.current_lr,
                    "epochs": self.config.local_epochs,
                    "device_participation": self.device_participation,
                    "enable_failure": self.enable_failure,
                    "device_failure_probability": self.device_failure_probability,
                    "dataset_name": self.config.dataset_name,
                }

                # 2. Start training in all zones
                executor, futures = self._start_local_training(args)
                return_when = FIRST_COMPLETED if self.async_aggregation else ALL_COMPLETED

                round_num = 0
                while futures and round_num < self.config.num_rounds:
                    round_start_time = time.time()

                    self.logger.info(f"\n=== Round {round_num + 1}/{self.config.num_rounds} ===")
                    log_mem(f"Round {round_num + 1}")
                    self.debugger.on_round_start(round_num + 1, self.global_model)

                    # 1. Zone discovery/update (periodic)
                    if round_num % 10 == 0 and round_num > 0:  # Every 10 rounds
                        self.logger.info("Updating zone assignments...")
                        device_list = [d for d in self.devices.values() if d.is_active]
                        self.zones = self.zone_discovery.adaptive_zone_update(device_list, self.zones)
                    done, _ = wait(futures, return_when=return_when)
                    waiting_time = time.time() - round_start_time
                    self.logger.info(f"Aggregating gradients of {len(done)} Zone(s)...")

                    # 3. Hierarchical aggregation
                    zone_weights = {}
                    participating_devices = []
                    total_participating_devices = []
                    start_time = time.time()
                    communication_cost = 0
                    total_num_device_updates = 0
                    intra_time = 0
                    for f in done:
                        zone_id, aggregated_weights, local_stats = f.result()

                        participating_devices = local_stats["participating_devices"]
                        total_participating_devices.extend(participating_devices)
                        num_device_updates = local_stats["num_device_updates"]
                        intra_time += local_stats["intra_time"]
                        # FIX: was added twice (lines 322 and 326 both added the same cost)
                        communication_cost += local_stats["communication_cost"]
                        total_num_device_updates += num_device_updates
                        self.logger.info(
                            f"({zone_id}) Local training completed: {num_device_updates}/{len(participating_devices)} devices")
                        zone_weights[zone_id] = aggregated_weights
                        futures.remove(f)
                        self.debugger.on_zone_training_done(
                            zone_id,
                            (zone_id, aggregated_weights, local_stats),
                            {k: v for k, v in (aggregated_weights or {}).items()},
                        )
                    intra_time /= len(done)
                    inter_start = time.time()
                    inter_zone_aggregated_weights = self.aggregator.inter_zone_aggregation(
                        zone_weights=zone_weights, zones=self.zones)
                    inter_time = time.time() - inter_start
                    total_time = time.time() - start_time
                    aggregation_stats = self.aggregator.federated_aggregation_round(zones=self.zones,
                                                                num_device_updates=total_num_device_updates,
                                                                participating_zones=list(zone_weights.keys()),
                                                                total_time=total_time,
                                                                intra_time=intra_time,
                                                                inter_time=inter_time,
                                                                comm_cost=communication_cost)
                    if inter_zone_aggregated_weights:
                        self.global_model.load_state_dict(inter_zone_aggregated_weights)
                    else:
                        self.logger.warning("No device updates received in this round")
                        aggregation_stats = {"participating_devices": 0, "participating_zones": 0}

                    self.debugger.on_aggregation_done(
                        round_num + 1,
                        self.global_model,
                        zone_weights,
                        self.aggregator.zone_fair_weights,
                        aggregation_stats,
                    )

                    # 4. Evaluation
                    round_metrics = self._evaluate_round(total_participating_devices, aggregation_stats)
                    round_metrics["waiting_time"] = waiting_time
                    # 5. Track performance
                    round_time = time.time() - round_start_time
                    round_metrics["round_time_s"] = round_time
                    self.round_times.append(round_time)
                    self.training_history.append(round_metrics)

                    # Log round results + append CSV row
                    self._log_round_results(round_num, round_metrics, round_time)
                    self._append_metrics_csv(round_metrics)
                    self.debugger.on_round_end(round_num + 1, round_metrics, round_time)

                    # Save checkpoint periodically
                    if (round_num + 1) % self.config.save_interval == 0:
                        self._save_checkpoint(round_num + 1)

                    # Early stopping
                    if getattr(self.config, 'enable_early_stopping', False):
                        acc = round_metrics["global_accuracy"]
                        min_delta = getattr(self.config, 'early_stopping_min_delta', 1e-4)
                        patience = getattr(self.config, 'early_stopping_patience', 20)
                        if acc >= self._es_best + min_delta:
                            self._es_best = acc
                            self._es_counter = 0
                        else:
                            self._es_counter += 1
                            self.logger.info(
                                f"Early stopping: no improvement for {self._es_counter}/{patience} rounds "
                                f"(best={self._es_best*100:.2f}%)"
                            )
                            if self._es_counter >= patience:
                                self.logger.info(
                                    f"Early stopping triggered at round {round_num + 1}. "
                                    f"Best accuracy: {self._es_best*100:.2f}%"
                                )
                                self._save_checkpoint(round_num + 1)
                                break

                    # Restart training for aggregated zones
                    failed_zones = {}
                    for zone_id in zone_weights.keys():
                        is_failure = self.zones[zone_id].simulate_failure(self.zone_failure_probability) if self.enable_failure else False
                        if is_failure:
                            failed_zones[zone_id] = self.zones[zone_id]
                    if failed_zones:
                        zoneless_devices = {}
                        for _, failed_zone in failed_zones.items():
                            for device_id, device in failed_zone.devices.items():
                                zoneless_devices[device_id] = device

                        standalone_devices = self.zone_discovery.handle_zone_failure(zoneless_devices=zoneless_devices, zones=self.zones, failed_zones=failed_zones)
                        if self.standalone_device_zone is None:
                            self.standalone_device_zone = Zone('standalone_devices', 'cloud_coordinator', self.config.compression_rate, self.config.enable_compression)
                            self.zones[self.standalone_device_zone.zone_id] = self.standalone_device_zone
                        for device_id, device in standalone_devices.items():
                            self.standalone_devices[device_id] = device
                            self.standalone_device_zone.add_device(device)
                    # Update LR for next round
                    round_loss = round_metrics.get("loss", float('inf'))
                    args["learning_rate"] = self._compute_round_lr(round_num + 1, round_loss)
                    self.logger.info(f"Round {round_num + 1} LR → {args['learning_rate']:.6f}")

                    if self.standalone_device_zone and self.standalone_device_zone.devices:
                        futures.append(executor.submit(self.standalone_device_zone.perform_local_training, args))
                    for zone_id, zone in self.zones.items():
                        if zone.is_active and zone_id != 'standalone_devices':
                            futures.append(executor.submit(self.zones[zone_id].perform_local_training, args))
                    self.current_round += 1
                    round_num = self.current_round

                executor.shutdown(wait=False)
        
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

        self.debugger.on_training_end(final_results)
        self.logger.info("Federated learning completed")
        return final_results

    def _start_local_training(self, args) -> tuple[
        ThreadPoolExecutor, list[Future[tuple[str, Any, Any, dict[str, list[str] | int | float]]]]]:
        """Start local training in all Zones"""

        from concurrent.futures import ThreadPoolExecutor

        if self.config.device.startswith('cuda'):
            max_workers = len(self.zones)
        else:
            max_workers = min(len(self.zones), os.cpu_count())
        executor = ThreadPoolExecutor(max_workers=max_workers)
        # Start training for all zones
        futures = [executor.submit(zone.perform_local_training, args) for zone_id, zone in self.zones.items()]

        return executor, futures
    
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
        
        round_time = time.time()  # wall-clock snapshot; actual round_time added by caller
        round_metrics = {
            "round": self.current_round,
            "global_accuracy": global_metrics.get("accuracy", 0.0),
            "global_loss": global_metrics.get("loss", float('inf')),
            "precision": global_metrics.get("precision", 0.0),
            "recall": global_metrics.get("recall", 0.0),
            "f1": global_metrics.get("f1", 0.0),
            "participating_devices": len(participating_devices),
            "participating_zones": aggregation_stats.get("participating_zones", 0),
            "communication_cost_mb": comm_cost,
            "aggregation_time": aggregation_stats.get("aggregation_time", 0.0),
            "learning_rate": self.current_lr,
            "zone_metrics": zone_metrics,
        }

        # Track best and last accuracy
        acc = round_metrics["global_accuracy"]
        self.last_accuracy = acc
        if acc > self.best_accuracy:
            self.best_accuracy = acc
            self.best_accuracy_round = self.current_round
            self._save_best_checkpoint()

        self.accuracies.append(acc)
        self.losses.append(round_metrics["global_loss"])

        return round_metrics
    
    def _evaluate_global_model(self) -> Dict[str, float]:
        """Evaluate global model on test data. Returns accuracy, loss, precision, recall, F1."""
        if not hasattr(self.dataset, 'test_data') or self.dataset.test_data is None:
            return {"accuracy": 0.0, "loss": float('inf'),
                    "precision": 0.0, "recall": 0.0, "f1": 0.0}

        self.global_model.eval()
        test_dataloader = self.dataset.get_global_dataloader(
            batch_size=self.config.batch_size, is_train=False
        )

        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []

        criterion = nn.CrossEntropyLoss()
        if self.config.device.startswith('cuda') and torch.cuda.is_available():
            criterion = criterion.to(self.config.device)

        with torch.no_grad():
            for data, target in test_dataloader:
                if self.config.device.startswith('cuda') and torch.cuda.is_available():
                    data, target = data.to(self.config.device), target.to(self.config.device)

                output = self.global_model(data)
                if isinstance(output, tuple):
                    output = output[0]

                loss = criterion(output, target)
                total_loss += loss.detach().item()

                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)

                all_preds.append(pred.cpu())
                all_targets.append(target.cpu())

        accuracy = correct / max(total, 1)
        avg_loss = total_loss / max(len(test_dataloader), 1)

        # Macro precision / recall / F1 (per-class then averaged)
        preds_t = torch.cat(all_preds)
        targets_t = torch.cat(all_targets)
        num_classes = output.shape[1]
        precision_sum = recall_sum = f1_sum = 0.0
        valid_classes = 0
        for c in range(num_classes):
            tp = ((preds_t == c) & (targets_t == c)).sum().item()
            fp = ((preds_t == c) & (targets_t != c)).sum().item()
            fn = ((preds_t != c) & (targets_t == c)).sum().item()
            p = tp / max(tp + fp, 1)
            r = tp / max(tp + fn, 1)
            f = 2 * p * r / max(p + r, 1e-8)
            if (targets_t == c).sum().item() > 0:
                precision_sum += p
                recall_sum += r
                f1_sum += f
                valid_classes += 1
        n = max(valid_classes, 1)

        self.global_model.train()
        return {
            "accuracy": accuracy,
            "loss": avg_loss,
            "precision": precision_sum / n,
            "recall": recall_sum / n,
            "f1": f1_sum / n,
        }
    
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
                if self.config.device.startswith('cuda') and torch.cuda.is_available():
                    data, labels = data.to(self.config.device), labels.to(self.config.device)
                
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
    
    def _compute_round_lr(self, round_num: int, round_loss: float) -> float:
        """Compute per-round learning rate based on dataset scheduler rules.
        Returns current_lr unchanged if enable_lr_scheduler is False."""
        if not getattr(self.config, 'enable_lr_scheduler', False):
            return self.current_lr

        import math
        dataset = self.config.dataset_name.lower()
        num_rounds = self.config.num_rounds

        if dataset == 'femnist':
            # StepLR: ×0.5 every 10 rounds, min 1e-5
            lr = 0.001 * (0.5 ** (round_num // 15))
            return max(lr, 1e-5)

        elif dataset == 'cifar100':
            # CosineAnnealingLR: 0.01 → 1e-5
            lr_max, lr_min = 0.01, 1e-5
            lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * round_num / num_rounds))
            return lr

        elif dataset == 'shakespeare':
            # ReduceLROnPlateau: factor=0.5, patience=2, min_lr=1e-5
            if round_loss < self._lr_plateau_best_loss:
                self._lr_plateau_best_loss = round_loss
                self._lr_plateau_counter = 0
            else:
                self._lr_plateau_counter += 1
                if self._lr_plateau_counter >= 2:
                    self.current_lr = max(self.current_lr * 0.5, 1e-5)
                    self._lr_plateau_counter = 0
            return self.current_lr

        return self.current_lr

    def _log_round_results(self, round_num: int, round_metrics: Dict[str, Any], round_time: float):
        """Log results for current round"""
        self.logger.info(
            f"Round {round_num + 1} | "
            f"Acc={round_metrics['global_accuracy']*100:.2f}% "
            f"(best={self.best_accuracy*100:.2f}%@R{self.best_accuracy_round+1}) | "
            f"Loss={round_metrics['global_loss']:.4f} | "
            f"P={round_metrics['precision']:.4f} R={round_metrics['recall']:.4f} "
            f"F1={round_metrics['f1']:.4f} | "
            f"Devices={round_metrics['participating_devices']} "
            f"Zones={round_metrics['participating_zones']} | "
            f"Comm={round_metrics['communication_cost_mb']:.2f}MB "
            f"LR={round_metrics['learning_rate']:.6f} "
            f"Time={round_time:.2f}s"
        )
        for zone_id, zm in round_metrics["zone_metrics"].items():
            self.logger.info(f"  Zone {zone_id}: Accuracy={zm.get('accuracy', 0.0):.4f}")

    def _results_dir(self) -> str:
        """Return the per-run results directory (mirrors main.py naming)."""
        compression_pct = int(round(self.config.compression_rate * 100))
        run_name = (
            f"{self.config.dataset_name}"
            f"__intra{self.config.intra_zone_alpha}"
            f"__inter{self.config.inter_zone_alpha}"
            f"__comp{compression_pct}pct"
            f"__dev{self.config.num_devices}"
            f"__zones{self.config.num_zones}"
        )
        path = os.path.join(self.config.results_dir, run_name)
        os.makedirs(path, exist_ok=True)
        return path

    def _append_metrics_csv(self, round_metrics: Dict[str, Any]):
        """Append one row per round to metrics.csv in the results directory."""
        csv_path = os.path.join(self._results_dir(), "metrics.csv")
        fieldnames = [
            "round", "global_accuracy", "global_loss",
            "precision", "recall", "f1",
            "participating_devices", "participating_zones",
            "communication_cost_mb", "aggregation_time",
            "round_time_s", "waiting_time", "learning_rate",
            "best_accuracy", "best_accuracy_round",
        ]
        write_header = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            row = {k: round_metrics.get(k, "") for k in fieldnames}
            row["best_accuracy"] = self.best_accuracy
            row["best_accuracy_round"] = self.best_accuracy_round + 1
            writer.writerow(row)

    def _save_best_checkpoint(self):
        """Save model checkpoint whenever a new best accuracy is reached."""
        checkpoint_dir = self.config.checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        path = os.path.join(checkpoint_dir, "best_model.pt")
        torch.save({
            "round": self.best_accuracy_round,
            "best_accuracy": self.best_accuracy,
            "global_model_state": self.global_model.state_dict(),
            "config": self.config.to_dict(),
        }, path)
        self.logger.info(
            f"New best accuracy {self.best_accuracy*100:.2f}% at round "
            f"{self.best_accuracy_round+1} — saved to {path}"
        )
    
    def _save_summary_csv(self, training_stats: Dict[str, Any]):
        """Write a one-row summary CSV for this run (appends across runs)."""
        summary_path = os.path.join(self._results_dir(), "summary.csv")
        fieldnames = [
            "dataset", "intra_zone_alpha", "inter_zone_alpha", "compression_rate",
            "num_rounds", "num_devices", "num_zones", "local_epochs", "learning_rate",
            "enable_lr_scheduler",
            "final_accuracy", "final_loss", "final_precision", "final_recall", "final_f1",
            "best_accuracy", "best_accuracy_round", "last_accuracy",
            "total_training_time", "average_round_time", "total_communication_cost",
            "convergence_rounds",
        ]
        write_header = not os.path.exists(summary_path)
        row = {
            "dataset": self.config.dataset_name,
            "intra_zone_alpha": self.config.intra_zone_alpha,
            "inter_zone_alpha": self.config.inter_zone_alpha,
            "compression_rate": self.config.compression_rate,
            "num_rounds": self.config.num_rounds,
            "num_devices": self.config.num_devices,
            "num_zones": self.config.num_zones,
            "local_epochs": self.config.local_epochs,
            "learning_rate": self.config.learning_rate,
            "enable_lr_scheduler": getattr(self.config, "enable_lr_scheduler", False),
            **{k: training_stats.get(k, "") for k in fieldnames
               if k not in ("dataset", "intra_zone_alpha", "inter_zone_alpha",
                            "compression_rate", "num_rounds", "num_devices", "num_zones",
                            "local_epochs", "learning_rate", "enable_lr_scheduler")},
        }
        with open(summary_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        self.logger.info(f"Summary CSV saved: {summary_path}")

    def _save_checkpoint(self, round_num: int):
        """Save periodic training checkpoint and always overwrite last_model.pt."""
        checkpoint_dir = self.config.checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint = {
            "round": round_num,
            "last_accuracy": self.last_accuracy,
            "best_accuracy": self.best_accuracy,
            "best_accuracy_round": self.best_accuracy_round,
            "global_model_state": self.global_model.state_dict(),
            "training_history": list(self.training_history),
            "device_participation": dict(self.device_participation),
            "config": self.config.to_dict(),
        }

        # Periodic numbered checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_round_{round_num}.pt")
        torch.save(checkpoint, checkpoint_path)

        # Always-current last checkpoint
        last_path = os.path.join(checkpoint_dir, "last_model.pt")
        torch.save(checkpoint, last_path)

        self.logger.info(
            f"Checkpoint saved: {checkpoint_path} | "
            f"last_acc={self.last_accuracy*100:.2f}% best_acc={self.best_accuracy*100:.2f}%"
        )
    
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
            "final_precision": final_metrics.get("precision", 0.0),
            "final_recall": final_metrics.get("recall", 0.0),
            "final_f1": final_metrics.get("f1", 0.0),
            "best_accuracy": self.best_accuracy,
            "best_accuracy_round": self.best_accuracy_round + 1,
            "last_accuracy": self.last_accuracy,
            "average_round_time": np.mean(self.round_times) if self.round_times else 0.0,
            "total_communication_cost": sum(self.communication_costs),
            "device_participation_stats": dict(self.device_participation),
            "convergence_rounds": self._analyze_convergence(),
        }
        self._save_summary_csv(training_stats)
        
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