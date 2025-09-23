"""
Data loading and distribution module for ContinuumFL framework.
Handles dataset downloading, preprocessing, and non-IID distribution across zones and devices.
"""

import os
import random
import datasets
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from typing import Dict, List, Tuple, Optional, Any, Union
import pickle
import json
from collections import defaultdict
from datasets import load_dataset, DownloadConfig
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

class FederatedDataset:
    """Base class for federated datasets with spatial non-IID distribution"""
    
    def __init__(self, config, data_dir: str = "./data"):
        self.config = config
        self.data_dir = data_dir
        self.dataset_name = config.dataset_name
        self.num_classes = None
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
        self.shakespeare_num_speakers = config.shakespeare_num_speakers
        # Shakespeare Specific
        self.train_indices = []
        self.test_indices = []

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
        """Prepare CIFAR-100 dataset with random subsets"""
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

        # Download full datasets
        full_train = torchvision.datasets.CIFAR100(root=self.data_dir, train=True, download=True, transform=transform_train)
        full_test = torchvision.datasets.CIFAR100(root=self.data_dir, train=False, download=True, transform=transform_test)

        # Compute number of samples
        num_train_samples = min(len(full_train), self.max_samples) if self.max_samples > 0 else len(full_train)
        num_test_samples = min(len(full_test), self.max_samples) if self.max_samples > 0 else len(full_test)

        # Random indices
        train_indices = torch.randperm(len(full_train))[:num_train_samples]
        test_indices = torch.randperm(len(full_test))[:num_test_samples]

        # Create subsets
        self.train_data = Subset(full_train, train_indices)
        self.train_data.targets = [full_train.targets[i] for i in train_indices]

        self.test_data = Subset(full_test, test_indices)
        self.test_data.targets = [full_test.targets[i] for i in test_indices]

        print(f"CIFAR-100 loaded: {len(self.train_data)} train, {len(self.test_data)} test samples")

    def _prepare_femnist(self):
        """Prepare FEMNIST dataset safely."""
        print("Preparing FEMNIST dataset...")

        femnist_path = os.path.join(self.data_dir, 'femnist')
        os.makedirs(femnist_path, exist_ok=True)

        train_file = os.path.join(femnist_path, 'train.pkl')
        test_file = os.path.join(femnist_path, 'test.pkl')

        if os.path.exists(train_file) and os.path.exists(test_file):
            print("Loading existing FEMNIST data...")
            with open(train_file, 'rb') as f:
                train_ds = pickle.load(f)
            with open(test_file, 'rb') as f:
                test_ds = pickle.load(f)
            self.process_femnist(train_ds, test_ds)
            return

        print("Downloading FEMNIST dataset...")
        download_config = DownloadConfig(cache_dir=femnist_path)
        dataset = load_dataset('flwrlabs/femnist', download_config=download_config)

        split_ds = dataset['train'].train_test_split(test_size=0.2, seed=42)
        train_ds = split_ds['train']
        test_ds = split_ds['test']

        with open(train_file, 'wb') as f:
            pickle.dump(train_ds, f)
        with open(test_file, 'wb') as f:
            pickle.dump(test_ds, f)

        self.process_femnist(train_ds, test_ds)

    def _download_femnist(self):
        """Download and process FEMNIST dataset"""
        femnist_path = os.path.join(self.data_dir, 'femnist')
        os.makedirs(femnist_path, exist_ok=True)
        
        # Download if not exists
        download_config = DownloadConfig(cache_dir=femnist_path)
        dataset = load_dataset('flwrlabs/femnist', download_config=download_config)

        split_ds = dataset['train'].train_test_split(test_size=0.2, seed=42)

        train_ds = split_ds['train']
        test_ds = split_ds['test']

        with open(os.path.join(femnist_path, 'train.pkl'), 'wb') as f:
            pickle.dump(train_ds, f)

        with open(os.path.join(femnist_path, 'test.pkl'), 'wb') as f:
            pickle.dump(test_ds, f)

        self.process_femnist(train_ds, test_ds)

    def process_femnist(self, train_ds, test_ds):
        """Process FEMNIST dataset into PyTorch-ready format."""
        # Limit samples if needed
        if self.max_samples > 0:
            train_len = len(train_ds)
            num_samples = min(self.max_samples, train_len)
            if num_samples < train_len:
                train_ds = train_ds.select(range(num_samples))
                test_ds = test_ds.select(range(min(num_samples, len(test_ds))))

        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        class FEMNISTDataset(Dataset):
            def __init__(self, hf_dataset, transform=None):
                self.dataset = hf_dataset
                self.transform = transform

                # Store data and labels for compatibility with FederatedDataset
                self.data = [item['image'] for item in hf_dataset]
                self.targets = [item['character'] for item in hf_dataset]

            def __len__(self):
                return len(self.dataset)

            def __getitem__(self, idx):
                image = self.data[idx]  # This is already a PIL image
                label = self.targets[idx]

                if self.transform:
                    image = self.transform(image)  # Apply transform directly

                return image, label

        self.train_data = FEMNISTDataset(train_ds, transform=transform)
        self.test_data = FEMNISTDataset(test_ds, transform=transform)

        print(f"FEMNIST processed: {len(self.train_data)} train, {len(self.test_data)} test samples")

    def _dict_to_dataset(self, data_dict: Dict[str, torch.Tensor]) -> Any:
        """Convert dictionary to PyTorch dataset"""

        class CustomDataset(Dataset):
            def __init__(self, data_dict: Dict[str, torch.Tensor]):
                self.data = data_dict

            def __len__(self):
                return len(self.data['image'])

            def __getitem__(self, idx_or_key):
                if isinstance(idx_or_key, str):
                    return self.data[idx_or_key]
                return {
                    'image': self.data['image'][idx_or_key],
                    'character': self.data['character'][idx_or_key]
                }

            @property
            def column_names(self):
                return self.data.keys()

            def remove_columns(self, cols: Union[str, List[str]]):
                """Return a new dataset with some columns removed"""
                if isinstance(cols, str):
                    cols = [cols]
                new_data = {k: v for k, v in self.data.items() if k not in cols}
                return CustomDataset(new_data)

            def __getitem_by_key__(self, key):
                return self.data[key]

        return CustomDataset(data_dict)

    class ShakespeareDataset(torch.utils.data.Dataset):
        def __init__(self, dataset, vocab):
            self.vocab = vocab
            self.stoi = {ch: i for i, ch in enumerate(self.vocab)}
            self.itos = {i: ch for i, ch in enumerate(self.vocab)}
            max_len = max(len(seq) for seq in dataset["x"])
            n_samples = len(dataset["x"])

            sequences_np = np.zeros((n_samples, max_len), dtype=np.int32)

            for i, seq in enumerate(dataset["x"]):
                sequences_np[i, :len(seq)] = [self.stoi[c] for c in seq]

            self.sequences = sequences_np
            self.targets = np.array([self.stoi[y] for y in dataset["y"]], dtype=np.int64)

            self.num_classes = len(self.vocab)
            self.classes = self.vocab

        def __len__(self):
            return len(self.sequences)

        def __getitem__(self, idx):
            return torch.tensor(self.sequences[idx], dtype=torch.long), self.targets[idx]

    def _prepare_shakespeare(self):
        """Prepare Shakespeare dataset"""
        print("Preparing Shakespeare dataset...")
        
        shakespeare_path = os.path.join(self.data_dir, 'shakespeare')
        os.makedirs(shakespeare_path, exist_ok=True)
        
        # Check if already processed
        if os.path.exists(os.path.join(shakespeare_path, 'train.pkl')):
            print("Loading existing Shakespeare data...")
            vocab = None
            with open(os.path.join(shakespeare_path, 'vocab.pkl'), 'rb') as f:
                vocab = pickle.load(f)
            with open(os.path.join(shakespeare_path, 'train_indices.pkl'), 'rb') as f:
                self.train_indices = pickle.load(f)
            with open(os.path.join(shakespeare_path, 'train.pkl'), 'rb') as f:
                train_dataset = pickle.load(f)
                self.train_data = self.ShakespeareDataset(train_dataset, vocab)
            with open(os.path.join(shakespeare_path, 'test_indices.pkl'), 'rb') as f:
                self.test_indices = pickle.load(f)
            with open(os.path.join(shakespeare_path, 'test.pkl'), 'rb') as f:
                test_data = pickle.load(f)
                self.test_data = self.ShakespeareDataset(test_data, vocab)
            return

        # Download and process Shakespeare dataset
        self._download_shakespeare()

    def _download_shakespeare(self):
        shakespeare_path = os.path.join(self.data_dir, 'shakespeare')
        os.makedirs(shakespeare_path, exist_ok=True)

        # Load dataset from Hugging Face
        download_config = DownloadConfig(cache_dir=shakespeare_path)
        dataset = load_dataset("flwrlabs/shakespeare", download_config=download_config)["train"]

        chars = set()
        for batch in dataset.iter(batch_size=10000):
            for text in batch['x']:
                chars.update(text)

        vocab = sorted(list(chars))
        print(f"Vocab: {len(vocab)}: {vocab}")
        max_speakers = self.shakespeare_num_speakers
        unique_speakers = dataset.unique("character_id")[:max_speakers]

        speaker_indices = defaultdict(list)
        for i, speaker in enumerate(dataset["character_id"]):
            if speaker in unique_speakers:
                speaker_indices[speaker].append(i)

        train_indices = []
        test_indices = []
        random.seed(42)
        for speaker, indices in speaker_indices.items():
            rnd = random.random()
            random.shuffle(indices)

            if rnd > 0.8:
                test_indices.extend(indices)
            else:
                train_indices.extend(indices)

        self.train_indices = train_indices
        self.test_indices = test_indices

        # Concatenate all speaker splits into one Dataset
        train_dataset = dataset.select(self.train_indices)
        test_dataset = dataset.select(self.test_indices)

        # Save raw train/test texts
        with open(os.path.join(shakespeare_path, 'vocab.pkl'), 'wb') as f:
            pickle.dump(vocab, f)
        with open(os.path.join(shakespeare_path, 'train_indices.pkl'), 'wb') as f:
            pickle.dump(train_indices, f)
        with open(os.path.join(shakespeare_path, 'train.pkl'), 'wb') as f:
            pickle.dump(train_dataset, f)
        with open(os.path.join(shakespeare_path, 'test_indices.pkl'), 'wb') as f:
            pickle.dump(test_indices, f)
        with open(os.path.join(shakespeare_path, 'test.pkl'), 'wb') as f:
            pickle.dump(test_dataset, f)

        self.train_data = self.ShakespeareDataset(train_dataset, vocab)
        self.test_data = self.ShakespeareDataset(test_dataset, vocab)

        print(f"Shakespeare processed: {len(self.train_data)} train, {len(self.test_data)} test samples")
    
    def distribute_data_to_devices(self, zones: Dict[str, List[str]]) -> Dict[str, Tuple[Subset, Subset]]:
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
            num_classes = self.train_data.num_classes
            print(f"Shakespeare has {num_classes} classes.")
        else:
            num_classes = 10  # Default
        self.num_classes = num_classes
        # Create zone distributions
        num_zones = len(zones)

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
        device_datasets = {}
        rng = np.random.default_rng(42)

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

        for zone_id in zones.keys():
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

                if self.dataset_name.lower() == 'shakespeare':
                    train_idx_map = {old_idx: new_idx for new_idx, old_idx in enumerate(self.train_indices)}
                    test_idx_map = {old_idx: new_idx for new_idx, old_idx in enumerate(self.test_indices)}

                    flattened_train_indices_per_device = [train_idx_map[idx] for idx in
                                                          flattened_train_indices_per_device if idx in train_idx_map]
                    flattened_test_indices_per_device = [test_idx_map[idx] for idx in flattened_test_indices_per_device
                                                         if idx in test_idx_map]


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
                    if isinstance(train_subset.dataset.targets, datasets.arrow_dataset.Column):
                        device_labels = torch.tensor(train_subset.dataset.targets[train_subset.indices])
                    else:
                        device_labels = torch.tensor([train_subset.dataset.targets[i] for i in train_subset.indices])
                    counts = torch.bincount(device_labels, minlength=self.num_classes)
                    for label, count in enumerate(counts.tolist()):
                        if count > 0:
                            analysis["zone_stats"][zone_id]["class_distribution"][label] += count
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