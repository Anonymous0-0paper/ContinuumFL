# ContinuumFL Installation & Quick Start Guide

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Install Dependencies

```bash
# Install PyTorch (choose based on your system)
# For CPU only:
pip install torch torchvision

# For GPU (CUDA):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies:
pip install -r requirements.txt
```

Or install all dependencies at once:
```bash
pip install torch torchvision numpy pandas scikit-learn matplotlib seaborn requests tqdm pillow
```

### Step 2: Check GPU Availability (Optional but Recommended)

```bash
python test_gpu.py
```

This will:
- Check if CUDA is available
- Test GPU operations and performance
- Provide recommendations for your hardware
- Give optimal settings for your system

### Step 3: Verify Installation

```bash
python test_continuumfl.py
```

## 🚀 Quick Start

### Option 1: Auto-detect Device (Recommended)
```bash
# ContinuumFL will automatically detect and use GPU if available
python main.py --dataset cifar100 --num_devices 20 --num_zones 5 --num_rounds 10
```

### Option 2: Force GPU Usage
```bash
# Explicitly use GPU (will fallback to CPU if CUDA unavailable)
python main.py --device cuda --dataset cifar100 --num_devices 50 --num_zones 10 --num_rounds 50
```

### Option 3: CPU Only
```bash
# Force CPU usage (useful for testing or when GPU memory is limited)
python main.py --device cpu --dataset cifar100 --num_devices 20 --num_zones 5 --num_rounds 30
```

### Option 4: Full GPU-Accelerated Experiment  
```bash
# Large-scale experiment with GPU acceleration
python main.py --device cuda --dataset cifar100 --num_devices 200 --num_zones 40 --num_rounds 200 --batch_size 64 --create_visualizations
```

### Option 5: GPU with Baseline Comparison
```bash
# Compare with baselines using GPU
python main.py --device cuda --dataset cifar100 --run_baselines --create_visualizations
```

### Option 6: Use Configuration File
```bash
python main.py --config_file example_config.json
```

### Option 5: Windows Batch Script
```bash
run_experiments.bat
```

## 📊 Expected Results

After running an experiment, you should see:

1. **Console Output**: Training progress and metrics
2. **Results Directory**: Detailed results and checkpoints
3. **Visualizations**: Performance plots and analysis charts
4. **Logs**: Training logs for debugging

### Example Output Structure:
```
results/
├── experiment_results.json      # Complete results
├── training_history.json        # Round-by-round metrics  
├── final_model.pt               # Trained model
├── 01_training_curves.png       # Accuracy/loss plots
├── 02_zone_performance.png      # Zone analysis
├── 03_spatial_distribution.png  # Device layout
└── ...                          # Additional visualizations
```

## 🔧 Troubleshooting

### Common Issues:

1. **ModuleNotFoundError**: Install dependencies with pip
2. **CUDA out of memory**: 
   - Use `--device cpu` or reduce batch size with `--batch_size 16`
   - Reduce number of devices with `--num_devices 50`
   - Use gradient compression with `--compression_rate 0.05`
3. **CUDA not available**: 
   - Install CUDA-enabled PyTorch: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`
   - Check GPU drivers are installed
   - Run `python test_gpu.py` for detailed diagnosis
4. **Dataset download fails**: Check internet connection
5. **Permission errors**: Ensure write access to results directories

### GPU-Specific Troubleshooting:

- **"CUDA out of memory"**: Reduce batch size or number of devices
- **"No CUDA capable device"**: Install NVIDIA drivers and CUDA toolkit
- **Slow GPU performance**: Check if other processes are using GPU
- **Mixed precision errors**: Use `--device cpu` for compatibility

### Performance Tips:

- **GPU Available**: Use `--device cuda` for 5-10x speedup
- **Large GPU Memory (8GB+)**: Use larger experiments with `--num_devices 200` and `--batch_size 64`
- **Limited GPU Memory (4GB)**: Use `--batch_size 16` and `--num_devices 50`
- **CPU Only**: Start with small experiments (20 devices, 10 rounds)
- Use GPU when available for faster training
- Run `python test_gpu.py` to get personalized recommendations
- Reduce dataset size for quick testing
- Monitor memory usage with large device counts

## 📈 Key Features Demonstrated

The implementation includes all key features from the ContinuumFL paper:

✅ **Dynamic Zone Discovery**: Multi-dimensional device clustering
✅ **Hierarchical Aggregation**: Two-tier spatial-aware aggregation  
✅ **Communication Optimization**: Compression, caching, delta encoding
✅ **Fairness-Aware Weighting**: Balanced zone representation
✅ **Baseline Comparisons**: FedAvg, FedProx, HierFL, ClusterFL
✅ **Rich Visualizations**: Comprehensive analysis plots
✅ **Multiple Datasets**: CIFAR-100, FEMNIST, Shakespeare
✅ **Configurable Parameters**: Extensive customization options

## 📝 Example Commands

```bash
# Quick test
python main.py --num_devices 10 --num_rounds 5

# CIFAR-100 with compression
python main.py --dataset cifar100 --compression_rate 0.05

# Large scale experiment
python main.py --num_devices 500 --num_zones 50 --num_rounds 300

# Baseline comparison
python main.py --run_baselines --create_visualizations

# Custom configuration
python main.py --config_file my_config.json

# GPU training
python main.py --device cuda --batch_size 64

# High spatial regularization
python main.py --spatial_regularization 0.2 --spatial_weight 0.6
```

## 🎯 Next Steps

1. **Run the test**: `python test_continuumfl.py`
2. **Basic experiment**: `python main.py --num_devices 20 --num_rounds 10`
3. **Full experiment**: Use `run_experiments.bat` or shell script
4. **Customize**: Modify `config.py` or create custom configuration files
5. **Analyze**: Check visualization outputs in results directory

## 📞 Support

If you encounter issues:
1. Check the console output for error messages
2. Verify all dependencies are installed
3. Try with smaller dataset sizes first
4. Ensure adequate disk space for datasets and results

---

**You're now ready to run ContinuumFL experiments! 🚀**