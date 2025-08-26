@echo off
REM ContinuumFL Experimental Run Scripts for Windows
REM This file contains different experimental scenarios for ContinuumFL

echo 🚀 ContinuumFL Experimental Run Scripts
echo ========================================

echo 🧪 Available Experiments:
echo 1. Quick Test
echo 2. Standard CIFAR-100 Experiment
echo 3. Large Scale Experiment  
echo 4. Communication Efficiency Study
echo 5. Baseline Comparison
echo 6. Test Framework

set /p choice="Select experiment (1-6): "

if "%choice%"=="1" goto quick_test
if "%choice%"=="2" goto standard_cifar100
if "%choice%"=="3" goto large_scale
if "%choice%"=="4" goto comm_study
if "%choice%"=="5" goto baseline_comparison
if "%choice%"=="6" goto test_framework
goto invalid_choice

:quick_test
echo Running Quick Test...
mkdir "results\quick_test" 2>nul
python main.py --dataset cifar100 --num_devices 20 --num_zones 5 --num_rounds 10 --local_epochs 2 --batch_size 16 --create_visualizations --results_dir "results\quick_test"
goto end

:standard_cifar100
echo Running Standard CIFAR-100 Experiment...
mkdir "results\standard_cifar100" 2>nul
python main.py --dataset cifar100 --num_devices 100 --num_zones 20 --num_rounds 200 --local_epochs 5 --learning_rate 0.01 --spatial_regularization 0.1 --compression_rate 0.1 --create_visualizations --results_dir "results\standard_cifar100"
goto end

:large_scale
echo Running Large Scale Experiment...
mkdir "results\large_scale" 2>nul
python main.py --dataset cifar100 --num_devices 500 --num_zones 50 --num_rounds 300 --local_epochs 3 --batch_size 64 --spatial_regularization 0.15 --create_visualizations --results_dir "results\large_scale"
goto end

:comm_study
echo Running Communication Efficiency Study...
mkdir "results\comm_study_005" 2>nul
python main.py --dataset cifar100 --num_devices 100 --num_zones 20 --num_rounds 150 --compression_rate 0.05 --create_visualizations --results_dir "results\comm_study_005"

mkdir "results\comm_study_01" 2>nul
python main.py --dataset cifar100 --num_devices 100 --num_zones 20 --num_rounds 150 --compression_rate 0.1 --create_visualizations --results_dir "results\comm_study_01"

mkdir "results\comm_study_02" 2>nul
python main.py --dataset cifar100 --num_devices 100 --num_zones 20 --num_rounds 150 --compression_rate 0.2 --create_visualizations --results_dir "results\comm_study_02"
goto end

:baseline_comparison
echo Running Baseline Comparison...
mkdir "results\baseline_comparison" 2>nul
python main.py --dataset cifar100 --num_devices 100 --num_zones 20 --num_rounds 200 --run_baselines --create_visualizations --results_dir "results\baseline_comparison"
goto end

:test_framework
echo Testing ContinuumFL Framework...
python test_continuumfl.py
goto end

:invalid_choice
echo ❌ Invalid choice. Please select 1-6.
pause
exit /b 1

:end
echo 🎉 Experiment completed!
echo 📁 Results available in .\results\
echo.
echo 📊 To view results:
echo    - Check .\results\[experiment_name]\ for detailed results
echo    - View .png files for visualizations
echo    - Check training logs in the results directory
echo.
echo 🔧 To run custom experiments:
echo    python main.py --help
echo.
pause