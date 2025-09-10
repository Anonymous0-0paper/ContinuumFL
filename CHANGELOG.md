# ContinuumFL Changelog

## [Unreleased]

### Added
- Comprehensive caching mechanism for similarity computations with expiration policies
- GPU acceleration for similarity matrix computations when available
- Memory pooling for frequently allocated objects to reduce memory fragmentation
- In-place operations to minimize memory allocations
- Async/await patterns for non-blocking operations
- ProcessPoolExecutor for CPU-bound tasks to bypass Python's GIL
- Proper device consistency handling in tensor operations

### Fixed
- Cross-platform compatibility issues with memory monitoring
- Device mismatch errors in tensor operations
- Learning rate inconsistency between configuration and actual usage
- Zone counting issues in aggregation statistics
- Data distribution KeyError issues in federated datasets
- Accuracy variation issues through improved zone weight stabilization

### Improved
- Hierarchical clustering performance with GPU acceleration
- Memory efficiency through tensor pooling and in-place operations
- Training stability with hysteresis in zone reassignment
- Communication cost estimation accuracy
- Convergence behavior through better weight normalization

## [1.0.0] - 2025-09-09

### Added
- Initial release of ContinuumFL framework
- Spatial-aware federated learning implementation
- Hierarchical zone discovery and aggregation
- Support for multiple datasets (Shakespeare, CIFAR-100, FEMNIST)
- GPU acceleration capabilities
- Comprehensive logging and monitoring

### Known Issues
- High accuracy variation in early training rounds (partially addressed in unreleased version)
- Memory usage optimization opportunities (addressed in unreleased version)













# ContinuumFL: Spatial-Aware Federated Learning Framework

ContinuumFL is an implementation of the spatial-aware federated learning framework described in the paper "ContinuumFL: Spatial-Aware Federated Learning in Dynamic Edge Environments". It provides a hierarchical aggregation protocol that considers spatial relationships between edge devices to improve model convergence and communication efficiency.

## Features

- **Spatial-Aware Zone Discovery**: Dynamically clusters devices based on spatial proximity, data similarity, and network characteristics
- **Hierarchical Aggregation**: Two-tier aggregation protocol (intra-zone then inter-zone) with spatial regularization
- **Adaptive Zone Management**: Periodic zone updates based on device mobility and data distribution changes
- **Communication Optimization**: Gradient compression and delta encoding to reduce bandwidth usage
- **Resource-Aware Scheduling**: Bandwidth allocation based on zone priority and device capabilities
- **GPU Acceleration**: Parallel computation for similarity matrix calculations when CUDA is available
- **Async Processing**: Non-blocking operations for better resource utilization

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd ContinuumFL

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
# Run with Shakespeare dataset
python main.py --dataset shakespeare --num_devices 30 --num_zones 5 --num_rounds 100

# Run with CIFAR-100 dataset
python main.py --dataset cifar100 --num_devices 20 --num_zones 4 --num_rounds 50
```

### Advanced Configuration

```bash
# Run with custom configuration
python main.py --config example_config.json

# Run with specific parameters
python main.py --dataset shakespeare --max_samples 50000 --num_devices 30 --num_zones 5 --num_rounds 20 --local_epochs 5 --learning_rate 0.01 --batch_size 32
```

### Async Mode

To run in async mode, use the `--use_async` flag:

```bash
# Run in async mode for better resource utilization
python main.py --dataset shakespeare --num_devices 30 --num_zones 5 --num_rounds 100 --use_async

# Run in async mode with GPU
python main.py --dataset shakespeare --num_devices 30 --num_zones 5 --num_rounds 100 --use_async --device cuda
```

### GPU Acceleration

To leverage GPU acceleration:

```bash
# Run with GPU acceleration
python main.py --dataset shakespeare --num_devices 30 --num_zones 5 --num_rounds 100 --device cuda

# Run with CPU (default)
python main.py --dataset shakespeare --num_devices 30 --num_zones 5 --num_rounds 100 --device cpu
```

## Configuration

The framework can be configured through command-line arguments or a JSON configuration file. Key parameters include:

- `dataset`: Dataset to use (shakespeare, cifar100, femnist)
- `num_devices`: Number of edge devices to simulate
- `num_zones`: Number of spatial zones to create
- `num_rounds`: Number of federated learning rounds
- `local_epochs`: Number of local training epochs per round
- `learning_rate`: Learning rate for local training
- `batch_size`: Batch size for training
- `device`: Compute device (cuda or cpu)
- `use_async`: Enable async processing mode

## Performance Optimizations

### Memory Management
- Tensor pooling to reduce allocations
- In-place operations to minimize memory fragmentation
- Gradient compression for communication efficiency

### Computation
- GPU acceleration for similarity computations
- Caching with expiration policies for repeated calculations
- Parallel processing with ThreadPoolExecutor/ProcessPoolExecutor

### Communication
- Gradient compression with top-k sparsification
- Delta encoding for model updates
- Spatial regularization to reduce communication overhead

## Project Structure

```
ContinuumFL/
├── src/
│   ├── core/              # Core components (devices, zones)
│   ├── aggregation/       # Hierarchical aggregation protocol
│   ├── data/              # Dataset handling and distribution
│   ├── models/            # Model definitions
│   ├── communication/     # Communication optimization
│   ├── baselines/         # Baseline methods for comparison
│   └── memory_log/        # Memory monitoring
├── config.py             # Configuration management
├── main.py               # Entry point
├── continuum_fl_coordinator.py  # Main coordinator
└── requirements.txt      # Dependencies
```

## Results and Evaluation

The framework includes comprehensive logging and evaluation capabilities:

- Round-by-round accuracy and loss tracking
- Communication cost estimation
- Zone participation statistics
- Device reliability metrics
- Convergence analysis

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Based on the research paper "ContinuumFL: Spatial-Aware Federated Learning in Dynamic Edge Environments"
- Uses PyTorch for deep learning operations
- Inspired by federated learning frameworks like FedML and Flower
