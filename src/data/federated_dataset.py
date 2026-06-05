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
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    print("Warning: librosa not installed. Speech Commands mel-spectrogram will use scipy fallback.")
    LIBROSA_AVAILABLE = False

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
        elif self.dataset_name.lower() == 'ucihar':
            self._prepare_ucihar()
        elif self.dataset_name.lower() == 'speechcommands':
            self._prepare_speechcommands()
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
            # Enforce max_samples if requested (trim loaded datasets and index lists)
            if self.max_samples > 0:
                # Trim train indices and dataset
                if len(self.train_indices) > self.max_samples:
                    keep_train = self.train_indices[:self.max_samples]
                    self.train_indices = keep_train
                    try:
                        trimmed_train = train_dataset.select(range(min(len(train_dataset), self.max_samples)))
                        self.train_data = self.ShakespeareDataset(trimmed_train, vocab)
                    except Exception:
                        # If the loaded train_dataset is not a HF dataset with select(), fall back to slicing
                        if hasattr(train_dataset, '__getitem__'):
                            trimmed_items = [train_dataset[i] for i in range(min(len(train_dataset), self.max_samples))]
                            self.train_data = self.ShakespeareDataset({'x': [it[0] for it in trimmed_items], 'y': [it[1] for it in trimmed_items]}, vocab)
                # Trim test indices and dataset
                if len(self.test_indices) > self.max_samples:
                    keep_test = self.test_indices[:self.max_samples]
                    self.test_indices = keep_test
                    try:
                        trimmed_test = test_data.select(range(min(len(test_data), self.max_samples)))
                        self.test_data = self.ShakespeareDataset(trimmed_test, vocab)
                    except Exception:
                        if hasattr(test_data, '__getitem__'):
                            trimmed_items = [test_data[i] for i in range(min(len(test_data), self.max_samples))]
                            self.test_data = self.ShakespeareDataset({'x': [it[0] for it in trimmed_items], 'y': [it[1] for it in trimmed_items]}, vocab)
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

        # Apply max_samples limit if requested (consistent with other dataset handlers)
        if self.max_samples > 0:
            num_train_samples = min(len(train_dataset), self.max_samples)
            num_test_samples = min(len(test_dataset), self.max_samples)
            if num_train_samples < len(train_dataset):
                train_dataset = train_dataset.select(range(num_train_samples))
            if num_test_samples < len(test_dataset):
                test_dataset = test_dataset.select(range(num_test_samples))

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
    
    def _prepare_ucihar(self):
        """Prepare UCI Human Activity Recognition dataset (raw inertial signals)."""
        print("Preparing UCI HAR dataset...")
        har_path = os.path.join(self.data_dir, 'ucihar')
        os.makedirs(har_path, exist_ok=True)

        cache_file = os.path.join(har_path, 'data.pkl')
        if os.path.exists(cache_file):
            print("Loading cached UCI HAR data...")
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            self._build_ucihar_datasets(data)
            return

        zip_path = os.path.join(har_path, 'UCI_HAR.zip')
        if not os.path.exists(zip_path):
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
            print(f"Downloading UCI HAR from {url} ...")
            if requests is None:
                raise RuntimeError("requests package required to download UCI HAR.")
            r = requests.get(url, stream=True)
            r.raise_for_status()
            with open(zip_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        extract_dir = os.path.join(har_path, 'extracted')
        if not os.path.exists(extract_dir):
            print("Extracting UCI HAR zip...")
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(extract_dir)

        root = os.path.join(extract_dir, 'UCI HAR Dataset')

        signal_names = [
            'body_acc_x', 'body_acc_y', 'body_acc_z',
            'body_gyro_x', 'body_gyro_y', 'body_gyro_z',
            'total_acc_x', 'total_acc_y', 'total_acc_z',
        ]

        def load_signals(split):
            arrays = []
            for name in signal_names:
                fpath = os.path.join(root, split, 'Inertial Signals', f'{name}_{split}.txt')
                arrays.append(np.loadtxt(fpath))
            # shape: (n_samples, 9, 128)
            return np.stack(arrays, axis=1).astype(np.float32)

        def load_labels(split):
            fpath = os.path.join(root, split, f'y_{split}.txt')
            return np.loadtxt(fpath, dtype=np.int64) - 1  # 0-indexed

        def load_subjects(split):
            fpath = os.path.join(root, split, f'subject_{split}.txt')
            return np.loadtxt(fpath, dtype=np.int64)

        print("Loading UCI HAR signals (this may take a moment)...")
        data = {
            'X_train': load_signals('train'),
            'y_train': load_labels('train'),
            'subjects_train': load_subjects('train'),
            'X_test': load_signals('test'),
            'y_test': load_labels('test'),
            'subjects_test': load_subjects('test'),
        }

        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)

        self._build_ucihar_datasets(data)

    def _build_ucihar_datasets(self, data):
        """Build PyTorch datasets from UCI HAR numpy arrays."""

        class UCIHARDataset(Dataset):
            def __init__(self, X, y, subjects):
                # Normalize each channel to zero-mean unit-variance
                mean = X.mean(axis=(0, 2), keepdims=True)
                std = X.std(axis=(0, 2), keepdims=True) + 1e-8
                X = (X - mean) / std
                self.X = torch.from_numpy(X)
                self.targets = y.tolist()
                self.subjects = subjects

            def __len__(self):
                return len(self.targets)

            def __getitem__(self, idx):
                return self.X[idx], self.targets[idx]

        X_tr, y_tr = data['X_train'], data['y_train']
        X_te, y_te = data['X_test'], data['y_test']

        if self.max_samples > 0:
            X_tr = X_tr[:self.max_samples]
            y_tr = y_tr[:self.max_samples]
            X_te = X_te[:self.max_samples]
            y_te = y_te[:self.max_samples]

        self.train_data = UCIHARDataset(X_tr, y_tr, data['subjects_train'][:len(y_tr)])
        self.test_data = UCIHARDataset(X_te, y_te, data['subjects_test'][:len(y_te)])
        print(f"UCI HAR loaded: {len(self.train_data)} train, {len(self.test_data)} test samples, 6 classes")

    def _prepare_speechcommands(self):
        """Prepare Google Speech Commands v2 dataset as log-mel spectrograms.

        Uses HuggingFace datasets + librosa — no torchaudio dependency.
        Spectrograms are pre-computed and cached to avoid re-processing on reload.
        """
        print("Preparing Google Speech Commands v2 dataset...")
        sc_path = os.path.join(self.data_dir, 'speechcommands')
        os.makedirs(sc_path, exist_ok=True)

        # All 35 keywords present in Speech Commands v2
        KEYWORDS = [
            'backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five',
            'follow', 'forward', 'four', 'go', 'happy', 'house', 'learn', 'left',
            'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila',
            'six', 'stop', 'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero',
        ]
        label_map = {kw: i for i, kw in enumerate(KEYWORDS)}

        def _wav_to_melspec(audio_array, sr=16000, n_mels=64, n_fft=400, hop_length=160):
            """Convert raw waveform (numpy float32, 16 kHz) to log-mel spectrogram tensor."""
            # Ensure float32 and correct sample rate
            wav = audio_array.astype(np.float32)
            if LIBROSA_AVAILABLE:
                mel = librosa.feature.melspectrogram(
                    y=wav, sr=sr, n_fft=n_fft, hop_length=hop_length,
                    n_mels=n_mels, fmin=20.0, fmax=8000.0,
                )
                log_mel = librosa.power_to_db(mel, ref=np.max, top_db=80.0)
            else:
                # Minimal numpy/scipy fallback
                from scipy.signal import spectrogram as sp_spectrogram
                from scipy.signal.windows import hann
                freqs, times, Sxx = sp_spectrogram(
                    wav, fs=sr, window=hann(n_fft), nperseg=n_fft,
                    noverlap=n_fft - hop_length, scaling='spectrum',
                )
                # Build triangular mel filter bank manually
                f_min, f_max = 20.0, 8000.0
                mel_min = 2595 * np.log10(1 + f_min / 700)
                mel_max = 2595 * np.log10(1 + f_max / 700)
                mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
                hz_points = 700 * (10 ** (mel_points / 2595) - 1)
                bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)
                fbank = np.zeros((n_mels, n_fft // 2 + 1))
                for m in range(1, n_mels + 1):
                    f_m_minus, f_m, f_m_plus = bin_points[m-1], bin_points[m], bin_points[m+1]
                    for k in range(f_m_minus, f_m):
                        fbank[m-1, k] = (k - f_m_minus) / max(f_m - f_m_minus, 1)
                    for k in range(f_m, f_m_plus):
                        fbank[m-1, k] = (f_m_plus - k) / max(f_m_plus - f_m, 1)
                mel = np.dot(fbank, Sxx)
                log_mel = 10 * np.log10(mel + 1e-10)
            # Fixed time-axis length: pad/trim to 101 frames
            target_frames = 101
            if log_mel.shape[1] < target_frames:
                log_mel = np.pad(log_mel, ((0, 0), (0, target_frames - log_mel.shape[1])))
            else:
                log_mel = log_mel[:, :target_frames]
            # Per-sample normalisation
            log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-8)
            return torch.tensor(log_mel, dtype=torch.float32).unsqueeze(0)  # (1, 64, 101)

        def _ensure_extracted(sc_path):
            """Download and extract Speech Commands v0.02 tarball if not already done."""
            import tarfile
            extract_dir = os.path.join(sc_path, 'speech_commands_v0.02')
            if os.path.isdir(extract_dir):
                return extract_dir
            tar_path = os.path.join(sc_path, 'speech_commands_v0.02.tar.gz')
            if not os.path.exists(tar_path):
                url = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
                print(f"  Downloading Speech Commands v0.02 (~2.3 GB) from {url} ...")
                if requests is None:
                    raise RuntimeError("requests package required to download Speech Commands.")
                r = requests.get(url, stream=True)
                r.raise_for_status()
                with open(tar_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=65536):
                        f.write(chunk)
            print("  Extracting Speech Commands tarball...")
            os.makedirs(extract_dir, exist_ok=True)
            with tarfile.open(tar_path, 'r:gz') as tf:
                tf.extractall(extract_dir)
            return extract_dir

        def _build_split(split_name, cache_pkl, max_s, extract_dir):
            """Build spectrogram dataset for one split (train / validation / test)."""
            if os.path.exists(cache_pkl):
                print(f"  Loading cached '{split_name}' spectrograms...")
                with open(cache_pkl, 'rb') as f:
                    return pickle.load(f)

            print(f"  Building '{split_name}' split from WAV files...")
            # Read official split lists
            val_list_path  = os.path.join(extract_dir, 'validation_list.txt')
            test_list_path = os.path.join(extract_dir, 'testing_list.txt')
            with open(val_list_path)  as fh: val_files  = set(line.strip() for line in fh)
            with open(test_list_path) as fh: test_files = set(line.strip() for line in fh)

            specs, labels, speakers = [], [], []
            for word in sorted(label_map.keys()):
                word_dir = os.path.join(extract_dir, word)
                if not os.path.isdir(word_dir):
                    continue
                for fname in sorted(os.listdir(word_dir)):
                    if not fname.endswith('.wav'):
                        continue
                    rel_path = f'{word}/{fname}'
                    # Assign to correct split
                    if split_name == 'validation' and rel_path not in val_files:
                        continue
                    if split_name == 'test' and rel_path not in test_files:
                        continue
                    if split_name == 'train' and (rel_path in val_files or rel_path in test_files):
                        continue

                    wav_path = os.path.join(extract_dir, rel_path)
                    if LIBROSA_AVAILABLE:
                        wav, sr = librosa.load(wav_path, sr=16000, mono=True)
                    else:
                        import scipy.io.wavfile as wav_io
                        sr, wav = wav_io.read(wav_path)
                        wav = wav.astype(np.float32) / 32768.0

                    target_len = 16000
                    if len(wav) < target_len:
                        wav = np.pad(wav, (0, target_len - len(wav)))
                    else:
                        wav = wav[:target_len]

                    specs.append(_wav_to_melspec(wav))
                    labels.append(label_map[word])
                    # Speaker ID is the hash part of the filename: hash_nonce.wav
                    speakers.append(fname.split('_')[0])

                    if max_s > 0 and len(specs) >= max_s:
                        break
                if max_s > 0 and len(specs) >= max_s:
                    break

            result = {'specs': specs, 'labels': labels, 'speakers': speakers}
            with open(cache_pkl, 'wb') as f:
                pickle.dump(result, f)
            return result

        class SpeechCommandsDataset(Dataset):
            def __init__(self, data_dict):
                self.specs = data_dict['specs']       # list of (1,64,101) tensors
                self.targets = data_dict['labels']    # list of ints
                self.speakers = data_dict['speakers']

            def __len__(self):
                return len(self.targets)

            def __getitem__(self, idx):
                return self.specs[idx], self.targets[idx]

        max_s = self.max_samples if self.max_samples > 0 else -1
        extract_dir = _ensure_extracted(sc_path)
        tag = "full" if max_s < 0 else max_s
        train_cache = os.path.join(sc_path, f'train_specs_{tag}.pkl')
        test_cache  = os.path.join(sc_path, f'val_specs_{tag}.pkl')

        train_data = _build_split('train',      train_cache, max_s, extract_dir)
        test_data  = _build_split('validation', test_cache,  max_s, extract_dir)

        self.train_data = SpeechCommandsDataset(train_data)
        self.test_data  = SpeechCommandsDataset(test_data)
        print(f"Speech Commands loaded: {len(self.train_data)} train, {len(self.test_data)} test, 35 classes")

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
        elif self.dataset_name.lower() == 'ucihar':
            num_classes = 6
        elif self.dataset_name.lower() == 'speechcommands':
            num_classes = 35
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

            if clazz in device_train_indices:
                zone_distributions = zone_distribution_cache[clazz]
            else:
                zone_distributions = dict(zip(list(zones.keys()), rng.dirichlet([self.inter_zone_alpha] * num_zones)))

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
            num_devices = len(device_list)

            for clazz in zn_idxs:
                class_indices = zn_idxs[clazz]
                rng.shuffle(class_indices)

                if clazz in device_train_indices:
                    device_distributions = device_distribution_cache[clazz]
                else:
                    device_distributions = dict(zip(device_list, rng.dirichlet([self.intra_zone_alpha] * num_devices)))

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