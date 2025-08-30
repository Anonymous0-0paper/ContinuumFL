"""
Data loading and distribution module for ContinuumFL framework.
Handles dataset downloading, preprocessing, and non-IID distribution across zones and devices.
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from typing import Dict, List, Tuple, Optional, Any
import pickle
import json
from collections import defaultdict

from src.core.zone import Zone

try:
    import requests
except ImportError:
    print("Warning: requests not installed. Download functionality may be limited.")
    requests = None
try:
    import zipfile
except ImportError:
    print("Warning: zipfile not available")
    zipfile = None
import gzip

class FederatedDataset:
    """Base class for federated datasets with spatial non-IID distribution"""
    
    def __init__(self, config, data_dir: str = "./data"):
        self.config = config
        self.data_dir = data_dir
        self.dataset_name = config.dataset_name
        
        # Distribution parameters
        self.intra_zone_alpha = config.intra_zone_alpha  # Dirichlet α for intra-zone
        self.inter_zone_alpha = config.inter_zone_alpha  # Dirichlet α for inter-zone
        self.train_test_split = config.train_test_split
        
        # Data storage
        self.train_data = None
        self.test_data = None
        self.device_datasets: Dict[str, Dict[str, Any]] = {}
        self.zone_distributions: Dict[str, np.ndarray] = {}

        # Dataset size limitation
        self.max_samples = config.max_samples

        # Create data directory
        os.makedirs(data_dir, exist_ok=True)
    
    def download_and_prepare(self):
        """Download and prepare the dataset"""
        if self.dataset_name.lower() == 'cifar100':
            self._prepare_cifar100()
        elif self.dataset_name.lower() == 'femnist':
            self._prepare_femnist()
        elif self.dataset_name.lower() == 'shakespeare':
            self._prepare_shakespeare()
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")
    
    def _prepare_cifar100(self):
        """Prepare CIFAR-100 dataset"""
        print("Downloading CIFAR-100 dataset...")
        
        # Define transforms
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        
        # Download datasets
        full_train = torchvision.datasets.CIFAR100(
            root=self.data_dir, train=True, download=True, transform=transform_train
        )
        num_samples = min(len(full_train), self.max_samples) if self.max_samples > 0 else len(full_train)
        self.train_data = Subset(full_train, range(num_samples))
        # Attach targets so rest of code works unchanged
        self.train_data.targets = [full_train.targets[i] for i in self.train_data.indices]

        full_test = torchvision.datasets.CIFAR100(
            root=self.data_dir, train=False, download=True, transform=transform_test
        )
        num_samples = min(len(full_test), self.max_samples) if self.max_samples > 0 else len(full_test)
        self.test_data = Subset(full_test, range(num_samples))
        # Attach targets so rest of code works unchanged
        self.test_data.targets = [full_test.targets[i] for i in self.test_data.indices]
        
        print(f"CIFAR-100 loaded: {len(self.train_data)} train, {len(self.test_data)} test samples")
    
    def _prepare_femnist(self):
        """Prepare FEMNIST dataset"""
        print("Preparing FEMNIST dataset...")
        
        # Check if already processed
        femnist_path = os.path.join(self.data_dir, 'femnist')
        if os.path.exists(os.path.join(femnist_path, 'train.pkl')):
            print("Loading existing FEMNIST data...")
            with open(os.path.join(femnist_path, 'train.pkl'), 'rb') as f:
                train_dict = pickle.load(f)
            with open(os.path.join(femnist_path, 'test.pkl'), 'rb') as f:
                test_dict = pickle.load(f)
            
            self.train_data = self._dict_to_dataset(train_dict)
            self.test_data = self._dict_to_dataset(test_dict)
            return
        
        # Download and process FEMNIST
        self._download_femnist()
    
    def _download_femnist(self):
        """Download and process FEMNIST dataset"""
        femnist_url = "https://github.com/TalwalkarLab/leaf/raw/master/data/femnist/data/all_data.zip"
        femnist_path = os.path.join(self.data_dir, 'femnist')
        os.makedirs(femnist_path, exist_ok=True)
        
        # Download if not exists
        zip_path = os.path.join(femnist_path, 'all_data.zip')
        if not os.path.exists(zip_path):
            if requests:
                print("Downloading FEMNIST data...")
                response = requests.get(femnist_url)
                with open(zip_path, 'wb') as f:
                    f.write(response.content)
            else:
                print("Warning: requests not available, cannot download FEMNIST data")
                return
        
        # Extract and process
        if zipfile:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(femnist_path)
        else:
            print("Warning: zipfile not available, cannot extract FEMNIST data")
            return
        
        # Process JSON files to create unified dataset
        self._process_femnist_json(femnist_path)
    
    def _process_femnist_json(self, femnist_path: str):
        """Process FEMNIST JSON files"""
        train_files = []
        test_files = []
        
        data_dir = os.path.join(femnist_path, 'all_data')
        
        # Find train and test files
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.json'):
                    if 'train' in file:
                        train_files.append(os.path.join(root, file))
                    elif 'test' in file:
                        test_files.append(os.path.join(root, file))
        
        # Process train data
        train_data = {'images': [], 'labels': []}
        for file_path in train_files:
            with open(file_path, 'r') as f:
                data = json.load(f)
                for user in data['users']:
                    train_data['images'].extend(data['user_data'][user]['x'])
                    train_data['labels'].extend(data['user_data'][user]['y'])
        
        # Process test data
        test_data = {'images': [], 'labels': []}
        for file_path in test_files:
            with open(file_path, 'r') as f:
                data = json.load(f)
                for user in data['users']:
                    test_data['images'].extend(data['user_data'][user]['x'])
                    test_data['labels'].extend(data['user_data'][user]['y'])
        
        # Convert to tensors
        train_data['images'] = torch.tensor(train_data['images']).float().reshape(-1, 1, 28, 28) / 255.0
        train_data['labels'] = torch.tensor(train_data['labels']).long()
        test_data['images'] = torch.tensor(test_data['images']).float().reshape(-1, 1, 28, 28) / 255.0
        test_data['labels'] = torch.tensor(test_data['labels']).long()
        
        # Save processed data
        with open(os.path.join(femnist_path, 'train.pkl'), 'wb') as f:
            pickle.dump(train_data, f)
        with open(os.path.join(femnist_path, 'test.pkl'), 'wb') as f:
            pickle.dump(test_data, f)
        
        self.train_data = self._dict_to_dataset(train_data)
        self.test_data = self._dict_to_dataset(test_data)
        
        print(f"FEMNIST processed: {len(self.train_data)} train, {len(self.test_data)} test samples")
    
    def _dict_to_dataset(self, data_dict: Dict[str, torch.Tensor]) -> Dataset:
        """Convert dictionary to PyTorch dataset"""
        class DictDataset(Dataset):
            def __init__(self, images, labels):
                self.images = images
                self.labels = labels
            
            def __len__(self):
                return len(self.images)
            
            def __getitem__(self, idx):
                return self.images[idx], self.labels[idx]
        
        return DictDataset(data_dict['images'], data_dict['labels'])
    
    def _prepare_shakespeare(self):
        """Prepare Shakespeare dataset"""
        print("Preparing Shakespeare dataset...")
        
        shakespeare_path = os.path.join(self.data_dir, 'shakespeare')
        os.makedirs(shakespeare_path, exist_ok=True)
        
        # Check if already processed
        if os.path.exists(os.path.join(shakespeare_path, 'train.pkl')):
            print("Loading existing Shakespeare data...")
            with open(os.path.join(shakespeare_path, 'train.pkl'), 'rb') as f:
                self.train_data = pickle.load(f)
            with open(os.path.join(shakespeare_path, 'test.pkl'), 'rb') as f:
                self.test_data = pickle.load(f)
            return
        
        # Download and process Shakespeare data
        self._download_shakespeare()
    
    def _download_shakespeare(self):
        """Download and process Shakespeare dataset"""
        # This is a simplified version - in practice, you'd download from LEAF

        print(f"Shakespeare processed: {len(self.train_data)} train, {len(self.test_data)} test samples")
    
    def _text_dict_to_dataset(self, data_dict: Dict[str, torch.Tensor]) -> Dataset:
        """Convert text dictionary to PyTorch dataset"""
        class TextDataset(Dataset):
            def __init__(self, sequences, targets):
                self.sequences = sequences
                self.targets = targets
            
            def __len__(self):
                return len(self.sequences)
            
            def __getitem__(self, idx):
                return self.sequences[idx], self.targets[idx]
        
        return TextDataset(data_dict['sequences'], data_dict['targets'])
    
    def create_zone_distributions(self, num_zones: int, num_classes: int) -> Dict[str, np.ndarray]:
        """
        Create class distribution for each zone using Dirichlet distribution.
        
        Implements the spatial data heterogeneity model from the paper.
        """
        # Create zone-level class distributions
        zone_distributions = {}
        
        # Generate Dirichlet distributions for zones
        for zone_id in range(num_zones):
            # Use inter-zone alpha for zone-level diversity
            zone_dist = np.random.dirichlet([self.inter_zone_alpha] * num_classes)
            zone_distributions[f"zone_{zone_id}"] = zone_dist
        
        self.zone_distributions = zone_distributions
        return zone_distributions
    
    def distribute_data_to_devices(self, devices: List[str], zones: Dict[str, List[str]]) -> Dict[str, Tuple[Subset, Subset]]:
        """
        Distribute data to devices with spatial non-IID characteristics.
        
        Implements the data distribution strategy where:
        - Devices in same zone have similar data (high intra-zone similarity)
        - Devices in different zones have different data (low inter-zone similarity)
        """
        if self.train_data is None:
            raise ValueError("Dataset not prepared. Call download_and_prepare() first.")
        
        # Get number of classes
        if hasattr(self.train_data, 'classes'):
            num_classes = len(self.train_data.classes)
        elif self.dataset_name.lower() == 'cifar100':
            num_classes = 100
        elif self.dataset_name.lower() == 'femnist':
            num_classes = 62
        elif self.dataset_name.lower() == 'shakespeare':
            num_classes = len("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .,!?'-")
        else:
            num_classes = 10  # Default
        
        # Create zone distributions
        num_zones = len(zones)
        zone_distributions = self.create_zone_distributions(num_zones, num_classes)
        
        # Get labels for train and test data
        if isinstance(self.train_data.targets, torch.Tensor):
            train_labels = self.train_data.targets.numpy()
        else:
            train_labels = np.array(self.train_data.targets)
        
        if isinstance(self.test_data.targets, torch.Tensor):
            test_labels = self.test_data.targets.numpy()
        else:
            test_labels = np.array(self.test_data.targets)
        
        # Group data by class
        train_class_indices = defaultdict(list)
        test_class_indices = defaultdict(list)
        
        for idx, label in enumerate(train_labels):
            train_class_indices[label].append(idx)
        
        for idx, label in enumerate(test_labels):
            test_class_indices[label].append(idx)
        
        # Distribute data to devices
        device_datasets = {} # (dict (dev_id: (trainset, testset)))
        rng = np.random.default_rng(42)  # TODO bind random seed to config

        # ===Train set partitioning===
        device_train_indices = {}

        # Assign indices to zones
        zone_indices = {}
        zone_distribution_cache = {}
        device_distribution_cache = {}
        for clazz in train_class_indices.keys():
            # Get all indices for class and shuffle
            class_indices = train_class_indices[clazz]
            rng.shuffle(class_indices)

            # Cache Zone Distributions for test set partitioning later
            zone_distributions = dict(zip(list(zones.keys()), rng.dirichlet([self.inter_zone_alpha] * num_zones)))
            zone_distribution_cache[clazz] = zone_distributions
            # Calculate Number of examples each zone gets
            counts = {zone_id : int(zone_distributions[zone_id] * len(class_indices)) for zone_id in zones.keys()}

            # Fix rounding errors safely
            while np.array(list(counts.values())).sum() < len(class_indices): counts[max(zone_distributions, key=zone_distributions.get)] += 1
            while np.array(list(counts.values())).sum() > len(class_indices): counts[max(zone_distributions, key=zone_distributions.get)] -= 1

            # assign indices to zones
            start = 0
            for zone_id, device_list in zones.items():
                if zone_id not in zone_indices:
                    zone_indices[zone_id] = {clazz: class_indices[start:start + counts[zone_id]]}
                else:
                    zone_indices[zone_id][clazz] = class_indices[start:start + counts[zone_id]]
                start += counts[zone_id]

        # Split Zone Data for devices
        for zone_id, device_list in zones.items():
            # Get Zone data and number of devices in zone
            zn_idxs = zone_indices[zone_id]
            num_devices = len(device_list)
            # class-wise splitting
            for clazz in zn_idxs.keys():
                # Get all indices for class and shuffle
                class_indices = zn_idxs[clazz]
                rng.shuffle(class_indices)

                # Cache Zone Distributions for test set partitioning later
                device_distributions = dict(zip(device_list, rng.dirichlet([self.intra_zone_alpha] * num_devices)))
                if clazz not in device_distribution_cache:
                    device_distribution_cache[clazz] = device_distributions
                else:
                    for device_id, value in device_distributions.items():
                        if device_id not in device_distribution_cache[clazz]:
                            device_distribution_cache[clazz][device_id] = value

                # Calculate Number of examples each device gets
                counts = {device_id: int(device_distributions[device_id] * len(class_indices)) for device_id in device_list}

                # Fix rounding errors safely
                while np.array(list(counts.values())).sum() < len(class_indices):
                    counts[max(counts.keys(), key=lambda d: device_distributions[d])] += 1

                while np.array(list(counts.values())).sum() > len(class_indices):
                    counts[max(counts.keys(), key=lambda d: device_distributions[d])] -= 1

                # assign indices to devices
                start = 0
                for device_id in device_list:
                    if device_id not in device_train_indices:
                        device_train_indices[device_id] = {}
                    device_train_indices[device_id][clazz] = class_indices[start:start + counts[device_id]]

        for zone_id in zones.keys():  # use keys(), not items()
            num_samples = 0
            for clazz in zone_indices[zone_id].keys():
                num_samples += len(zone_indices[zone_id][clazz])


        # ===Test set partitioning===
        device_test_indices = {}

        for clazz in test_class_indices.keys():
            class_indices = test_class_indices[clazz]
            rng.shuffle(class_indices)

            zone_distributions = zone_distribution_cache[clazz]
            counts = {zone_id: int(zone_distributions[zone_id] * len(class_indices)) for zone_id in zones.keys()}
            while np.array(list(counts.values())).sum() < len(class_indices): counts[
                max(zone_distributions, key=zone_distributions.get)] += 1
            while np.array(list(counts.values())).sum() > len(class_indices): counts[
                max(zone_distributions, key=zone_distributions.get)] -= 1

            start = 0
            for zone_id, device_list in zones.items():
                if zone_id not in zone_indices:
                    zone_indices[zone_id] = {}
                zone_indices[zone_id][clazz] = class_indices[start:start + counts[zone_id]]
                start += counts[zone_id]
        total_samples = 0
        for zone_id, device_list in zones.items():
            zn_idxs = zone_indices[zone_id]

            for clazz in zn_idxs:
                class_indices = zn_idxs[clazz]
                rng.shuffle(class_indices)

                device_distributions = device_distribution_cache[clazz]
                counts = {device_id: int(device_distributions[device_id] * len(class_indices)) for device_id in device_list}
                while np.array(list(counts.values())).sum() < len(class_indices):
                    counts[max(counts.keys(), key=lambda d: device_distributions[d])] += 1

                while np.array(list(counts.values())).sum() > len(class_indices):
                    counts[max(counts.keys(), key=lambda d: device_distributions[d])] -= 1

                start = 0
                for device_id in device_list:
                    if device_id not in device_test_indices:
                        device_test_indices[device_id] = {}
                    device_test_indices[device_id][clazz] = class_indices[start:start + counts[device_id]]
                    start += counts[device_id]

            # Create subsets

            for device_id in device_list:
                flattened_train_indices_per_device = [x for sublist in device_train_indices[device_id].values() for x in sublist]
                flattened_test_indices_per_device = [x for sublist in device_test_indices[device_id].values() for x in
                                                     sublist]
                train_subset = Subset(self.train_data, flattened_train_indices_per_device) if flattened_train_indices_per_device else None
                test_subset = Subset(self.test_data, flattened_test_indices_per_device) if flattened_test_indices_per_device else None
                device_datasets[device_id] = (train_subset, test_subset)
                len_train_subset = len(train_subset) if train_subset else 0
                len_test_subset = len(test_subset) if test_subset else 0
                total_samples += len_train_subset + len_test_subset
                print(f"Device {device_id} ({zone_id}): {len_train_subset} train, "
                      f"{len_test_subset} test samples")
        print(f"Total Samples: {total_samples}")
        self.device_datasets = device_datasets
        return device_datasets
    
    def get_device_dataloader(self, device_id: str, batch_size: int = 32, 
                            is_train: bool = True) -> Optional[DataLoader]:
        """Get DataLoader for a specific device"""
        if device_id not in self.device_datasets:
            return None
        
        train_subset, test_subset = self.device_datasets[device_id]
        subset = train_subset if is_train else test_subset
        
        if subset is None or len(subset) == 0:
            return None
        
        return DataLoader(
            subset, 
            batch_size=batch_size, 
            shuffle=is_train, 
            num_workers=0,  # Set to 0 to avoid issues in Windows
            drop_last=False
        )
    
    def get_global_dataloader(self, batch_size: int = 32, 
                            is_train: bool = True) -> DataLoader:
        """Get DataLoader for global evaluation"""
        dataset = self.train_data if is_train else self.test_data
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=0,
            drop_last=False
        )
    
    def analyze_data_distribution(self, zones: dict[str, Zone]) -> Dict[str, Any]:
        """Analyze the data distribution across devices and zones"""
        if not self.device_datasets:
            return {}
        
        analysis = {
            "total_devices": len(self.device_datasets),
            "device_stats": {},
            "zone_stats": defaultdict(lambda: {"devices": 0, "total_samples": 0, "class_distribution": defaultdict(int)})
        }

        for zone_id, zone in zones.items():
            for device_id in zone.devices.keys():
                (train_subset, test_subset) = self.device_datasets[device_id]

                train_size = len(train_subset) if train_subset else 0
                test_size = len(test_subset) if test_subset else 0

                analysis["device_stats"][device_id] = {
                    "train_samples": train_size,
                    "test_samples": test_size,
                    "total_samples": train_size + test_size,
                    "zone": zone_id
                }

                # Update zone statistics
                analysis["zone_stats"][zone_id]["devices"] += 1
                analysis["zone_stats"][zone_id]["total_samples"] += train_size + test_size

                # Analyze class distribution for train data
                if train_subset and len(train_subset) > 0:
                    if hasattr(train_subset.dataset, 'targets'):
                        targets = train_subset.dataset.targets
                    elif hasattr(train_subset.dataset, 'labels'):
                        targets = train_subset.dataset.labels
                    else:
                        continue

                    indices = train_subset.indices
                    device_labels = [targets[i] for i in indices]

                    for label in device_labels:
                        if isinstance(label, torch.Tensor):
                            label = label.item()
                        analysis["zone_stats"][zone_id]["class_distribution"][int(label)] += 1
        return analysis

    def save_data_distribution(self, filepath: str):
        """Save device data distribution for reproducibility"""
        distribution_info = {
            "dataset_name": self.dataset_name,
            "device_datasets": {
                device_id: {
                    "train_indices": data[0].indices if data[0] else [],
                    "test_indices": data[1].indices if data[1] else []
                }
                for device_id, data in self.device_datasets.items()
            },
            "zone_distributions": self.zone_distributions,
            "config": {
                "intra_zone_alpha": self.intra_zone_alpha,
                "inter_zone_alpha": self.inter_zone_alpha,
                "train_test_split": self.train_test_split
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(distribution_info, f, indent=2)
    
    def load_data_distribution(self, filepath: str):
        """Load previously saved data distribution"""
        with open(filepath, 'r') as f:
            distribution_info = json.load(f)
        
        # Reconstruct device datasets
        device_datasets = {}
        for device_id, indices_info in distribution_info["device_datasets"].items():
            train_indices = indices_info["train_indices"]
            test_indices = indices_info["test_indices"]
            
            train_subset = Subset(self.train_data, train_indices) if train_indices else None
            test_subset = Subset(self.test_data, test_indices) if test_indices else None
            
            device_datasets[device_id] = (train_subset, test_subset)
        
        self.device_datasets = device_datasets
        self.zone_distributions = distribution_info["zone_distributions"]