"""
Quick smoke-tests for UCI HAR and Speech Commands datasets + models.
Run with:  python -m pytest tests/test_new_datasets.py -v
Or:        python tests/test_new_datasets.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest
from config import ContinuumFLConfig
from src.models.model_factory import ModelFactory, UCIHAR_CNN_LSTM, SpeechCommandsCNN


# ---------------------------------------------------------------------------
# Model shape tests (no data download needed)
# ---------------------------------------------------------------------------

def test_ucihar_model_forward():
    model = UCIHAR_CNN_LSTM(num_classes=6)
    x = torch.randn(4, 9, 128)           # batch=4, 9 channels, 128 timesteps
    out = model(x)
    assert out.shape == (4, 6), f"Expected (4, 6), got {out.shape}"
    print(f"[PASS] UCIHAR_CNN_LSTM forward: {out.shape}")

def test_ucihar_model_clone():
    model = UCIHAR_CNN_LSTM(num_classes=6)
    cloned = model.clone()
    for p1, p2 in zip(model.parameters(), cloned.parameters()):
        assert torch.allclose(p1, p2)
    print("[PASS] UCIHAR_CNN_LSTM clone")

def test_speechcommands_model_forward():
    model = SpeechCommandsCNN(num_classes=35)
    x = torch.randn(4, 1, 64, 101)       # batch=4, 1-ch mel-spec
    out = model(x)
    assert out.shape == (4, 35), f"Expected (4, 35), got {out.shape}"
    print(f"[PASS] SpeechCommandsCNN forward: {out.shape}")

def test_speechcommands_model_clone():
    model = SpeechCommandsCNN(num_classes=35)
    cloned = model.clone()
    for p1, p2 in zip(model.parameters(), cloned.parameters()):
        assert torch.allclose(p1, p2)
    print("[PASS] SpeechCommandsCNN clone")

def test_model_factory_ucihar():
    cfg = ContinuumFLConfig()
    cfg.dataset_name = 'ucihar'
    model = ModelFactory.create_model(cfg)
    assert isinstance(model, UCIHAR_CNN_LSTM)
    info = ModelFactory.get_model_info(model)
    print(f"[PASS] ModelFactory UCI HAR: {info['total_parameters']:,} params, {info['model_size_mb']:.2f} MB")

def test_model_factory_speechcommands():
    cfg = ContinuumFLConfig()
    cfg.dataset_name = 'speechcommands'
    model = ModelFactory.create_model(cfg)
    assert isinstance(model, SpeechCommandsCNN)
    info = ModelFactory.get_model_info(model)
    print(f"[PASS] ModelFactory SpeechCommands: {info['total_parameters']:,} params, {info['model_size_mb']:.2f} MB")


# ---------------------------------------------------------------------------
# Dataset loading tests (download required; skipped if offline)
# ---------------------------------------------------------------------------

def test_ucihar_dataset_load():
    """Download UCI HAR and verify shapes/labels."""
    cfg = ContinuumFLConfig()
    cfg.dataset_name = 'ucihar'
    cfg.max_samples = 200       # keep fast

    from src.data.federated_dataset import FederatedDataset
    ds = FederatedDataset(cfg, data_dir='./data')
    ds.download_and_prepare()

    assert len(ds.train_data) <= 200
    x, y = ds.train_data[0]
    assert x.shape == (9, 128), f"Expected (9,128), got {x.shape}"
    assert 0 <= y <= 5, f"Label out of range: {y}"
    print(f"[PASS] UCI HAR dataset: {len(ds.train_data)} train, {len(ds.test_data)} test")


def test_speechcommands_dataset_load():
    """Download Speech Commands and verify shapes/labels."""

    cfg = ContinuumFLConfig()
    cfg.dataset_name = 'speechcommands'
    cfg.max_samples = 100

    from src.data.federated_dataset import FederatedDataset
    ds = FederatedDataset(cfg, data_dir='./data')
    ds.download_and_prepare()

    assert len(ds.train_data) <= 100
    x, y = ds.train_data[0]
    assert x.shape == (1, 64, 101), f"Expected (1,64,101), got {x.shape}"
    assert 0 <= y <= 34, f"Label out of range: {y}"
    print(f"[PASS] Speech Commands dataset: {len(ds.train_data)} train, {len(ds.test_data)} test")


# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=== Model tests (no download) ===")
    test_ucihar_model_forward()
    test_ucihar_model_clone()
    test_speechcommands_model_forward()
    test_speechcommands_model_clone()
    test_model_factory_ucihar()
    test_model_factory_speechcommands()

    print("\n=== Dataset tests (requires internet) ===")
    test_ucihar_dataset_load()
    test_speechcommands_dataset_load()

    print("\nAll tests passed.")
