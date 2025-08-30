"""
Visualization module for ContinuumFL framework.
Provides comprehensive plotting and analysis capabilities.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import os
from collections import defaultdict
import json

class ContinuumFLVisualizer:
    """
    Comprehensive visualization for ContinuumFL training and analysis.
    """
    
    def __init__(self, config, save_dir: str = "./results"):
        self.config = config
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Set plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Figure parameters
        self.fig_size = (12, 8)
        self.dpi = 300
    
    def plot_training_curves(self, training_history: List[Dict[str, Any]], 
                           baseline_results: Optional[Dict[str, Dict[str, Any]]] = None,
                           save_name: str = "training_curves.png"):
        """Plot training accuracy and loss curves"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Extract ContinuumFL data
        rounds = [entry["round"] for entry in training_history]
        accuracies = [entry["global_accuracy"] for entry in training_history]
        losses = [entry["global_loss"] for entry in training_history]
        
        # Plot ContinuumFL curves
        ax1.plot(rounds, accuracies, label='ContinuumFL', linewidth=2, marker='o', markersize=3)
        ax2.plot(rounds, losses, label='ContinuumFL', linewidth=2, marker='o', markersize=3)
        
        # Plot baseline comparisons if available
        if baseline_results:
            for method_name, results in baseline_results.items():
                if "accuracies" in results and "losses" in results:
                    baseline_rounds = range(len(results["accuracies"]))
                    ax1.plot(baseline_rounds, results["accuracies"], 
                            label=method_name, linewidth=2, alpha=0.8)
                    ax2.plot(baseline_rounds, results["losses"], 
                            label=method_name, linewidth=2, alpha=0.8)
        
        # Customize accuracy plot
        ax1.set_xlabel('Round')
        ax1.set_ylabel('Global Accuracy')
        ax1.set_title('Training Accuracy Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Customize loss plot
        ax2.set_xlabel('Round')
        ax2.set_ylabel('Global Loss')
        ax2.set_title('Training Loss Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=self.dpi, bbox_inches='tight')
        plt.show()
    
    def plot_zone_performance(self, training_history: List[Dict[str, Any]], 
                            save_name: str = "zone_performance.png"):
        """Plot zone-level performance over time"""
        
        # Extract zone metrics
        zone_data = defaultdict(list)
        rounds = []
        
        for entry in training_history:
            if "zone_metrics" in entry:
                rounds.append(entry["round"])
                for zone_id, metrics in entry["zone_metrics"].items():
                    zone_data[zone_id].append(metrics.get("accuracy", 0.0))
        
        if not zone_data:
            print("No zone performance data available")
            return
        
        plt.figure(figsize=self.fig_size)
        
        # Plot each zone's accuracy
        for zone_id, accuracies in zone_data.items():
            # Pad with zeros if zone didn't participate in all rounds
            if len(accuracies) < len(rounds):
                accuracies.extend([0.0] * (len(rounds) - len(accuracies)))
            
            plt.plot(rounds[:len(accuracies)], accuracies, 
                    label=f'Zone {zone_id}', linewidth=2, marker='o', markersize=3)
        
        plt.xlabel('Round')
        plt.ylabel('Zone Accuracy')
        plt.title('Zone-Level Performance Over Time')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=self.dpi, bbox_inches='tight')
        plt.show()
    
    def plot_spatial_distribution(self, devices: Dict[str, Any], zones: Dict[str, Any], 
                                save_name: str = "spatial_distribution.png"):
        """Plot spatial distribution of devices and zones"""
        
        plt.figure(figsize=self.fig_size)
        
        # Create color map for zones
        zone_colors = plt.cm.Set3(np.linspace(0, 1, len(zones)))
        zone_color_map = {zone_id: color for zone_id, color in zip(zones.keys(), zone_colors)}
        
        # Plot devices colored by zone
        for device_id, device in devices.items():
            zone_id = device.zone_id
            color = zone_color_map.get(zone_id, 'gray')
            
            plt.scatter(device.location[0], device.location[1], 
                       c=[color], s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
        
        # Plot zone centroids
        for zone_id, zone in zones.items():
            if zone.centroid:
                plt.scatter(zone.centroid[0], zone.centroid[1], 
                           c=[zone_color_map[zone_id]], s=200, marker='*', 
                           edgecolors='black', linewidth=2, label=f'Zone {zone_id}')
        
        plt.xlabel('X Coordinate (km)')
        plt.ylabel('Y Coordinate (km)')
        plt.title('Spatial Distribution of Devices and Zones')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=self.dpi, bbox_inches='tight')
        plt.show()
    
    def plot_communication_costs(self, training_history: List[Dict[str, Any]], 
                               save_name: str = "communication_costs.png"):
        """Plot communication costs over time"""
        
        rounds = [entry["round"] for entry in training_history]
        comm_costs = [entry.get("communication_cost_mb", 0.0) for entry in training_history]
        
        plt.figure(figsize=self.fig_size)
        
        # Plot communication cost per round
        plt.subplot(2, 1, 1)
        plt.plot(rounds, comm_costs, linewidth=2, marker='o', markersize=3)
        plt.xlabel('Round')
        plt.ylabel('Communication Cost (MB)')
        plt.title('Communication Cost per Round')
        plt.grid(True, alpha=0.3)
        
        # Plot cumulative communication cost
        plt.subplot(2, 1, 2)
        cumulative_costs = np.cumsum(comm_costs)
        plt.plot(rounds, cumulative_costs, linewidth=2, marker='s', markersize=3, color='orange')
        plt.xlabel('Round')
        plt.ylabel('Cumulative Communication Cost (MB)')
        plt.title('Cumulative Communication Cost')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=self.dpi, bbox_inches='tight')
        plt.show()
    
    def plot_device_participation(self, training_history: List[Dict[str, Any]], 
                                device_participation: Dict[str, int],
                                save_name: str = "device_participation.png"):
        """Plot device participation statistics"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot participation over time
        rounds = [entry["round"] for entry in training_history]
        participating_devices = [entry.get("participating_devices", 0) for entry in training_history]
        participating_zones = [entry.get("participating_zones", 0) for entry in training_history]
        
        ax1.plot(rounds, participating_devices, label='Devices', linewidth=2, marker='o', markersize=3)
        ax1.plot(rounds, participating_zones, label='Zones', linewidth=2, marker='s', markersize=3)
        ax1.set_xlabel('Round')
        ax1.set_ylabel('Number of Participants')
        ax1.set_title('Participation Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot device participation histogram
        participation_counts = list(device_participation.values())
        ax2.hist(participation_counts, bins=20, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Number of Rounds Participated')
        ax2.set_ylabel('Number of Devices')
        ax2.set_title('Device Participation Distribution')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=self.dpi, bbox_inches='tight')
        plt.show()
    
    def plot_zone_aggregation_weights(self, aggregation_stats: Dict[str, Any], 
                                    save_name: str = "zone_weights.png"):
        """Plot zone aggregation weights over time"""
        
        if "current_zone_weights" not in aggregation_stats:
            print("No zone weight data available")
            return
        
        zone_weights = aggregation_stats["current_zone_weights"]
        zone_ids = list(zone_weights.keys())
        weights = list(zone_weights.values())
        
        plt.figure(figsize=self.fig_size)
        
        # Bar plot of current zone weights
        bars = plt.bar(zone_ids, weights, alpha=0.7, edgecolor='black')
        plt.xlabel('Zone ID')
        plt.ylabel('Aggregation Weight')
        plt.title('Current Zone Aggregation Weights')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=self.dpi, bbox_inches='tight')
        plt.show()
    
    def plot_spatial_correlations(self, zones: Dict[str, Any], 
                                save_name: str = "spatial_correlations.png"):
        """Plot spatial correlation matrix between zones"""
        
        zone_ids = list(zones.keys())
        num_zones = len(zone_ids)
        
        # Create correlation matrix
        correlation_matrix = np.zeros((num_zones, num_zones))
        
        for i, zone_i_id in enumerate(zone_ids):
            zone_i = zones[zone_i_id]
            for j, zone_j_id in enumerate(zone_ids):
                if i == j:
                    correlation_matrix[i, j] = 1.0
                elif zone_j_id in zone_i.spatial_correlations:
                    correlation_matrix[i, j] = zone_i.spatial_correlations[zone_j_id]
        
        plt.figure(figsize=(10, 8))
        
        # Create heatmap
        mask = correlation_matrix == 0
        sns.heatmap(correlation_matrix, annot=True, fmt='.3f', 
                   xticklabels=zone_ids, yticklabels=zone_ids,
                   cmap='coolwarm', center=0, mask=mask,
                   square=True, cbar_kws={'label': 'Spatial Correlation'})
        
        plt.title('Zone Spatial Correlation Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=self.dpi, bbox_inches='tight')
        plt.show()
    
    def plot_data_distribution_analysis(self, distribution_analysis: Dict[str, Any],
                                      save_name: str = "data_distribution.png"):
        """Plot data distribution analysis across zones and devices"""

        if "zone_stats" not in distribution_analysis:
            print("No data distribution analysis available")
            return

        zone_stats = distribution_analysis["zone_stats"]

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Plot 1: Devices per zone
        zone_ids = list(zone_stats.keys())
        devices_per_zone = [zone_stats[zone_id]["devices"] for zone_id in zone_ids]

        ax1.bar(zone_ids, devices_per_zone, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Zone ID')
        ax1.set_ylabel('Number of Devices')
        ax1.set_title('Devices per Zone')
        ax1.tick_params(axis='x', rotation=45)

        # Plot 2: Samples per zone
        samples_per_zone = [zone_stats[zone_id]["total_samples"] for zone_id in zone_ids]

        ax2.bar(zone_ids, samples_per_zone, alpha=0.7, edgecolor='black', color='orange')
        ax2.set_xlabel('Zone ID')
        ax2.set_ylabel('Total Samples')
        ax2.set_title('Data Samples per Zone')
        ax2.tick_params(axis='x', rotation=45)

        # Plot 3: Class distribution across zones (if available)
        if zone_ids and "class_distribution" in zone_stats[zone_ids[0]]:
            class_distributions = []
            all_classes = set()

            for zone_id in zone_ids:
                class_dist = zone_stats[zone_id]["class_distribution"]
                class_distributions.append(class_dist)
                all_classes.update(class_dist.keys())

            # Create stacked bar chart for class distribution
            all_classes = sorted(list(all_classes))[:10]  # Show top 10 classes
            bottom = np.zeros(len(zone_ids))

            colors = plt.cm.Set3(np.linspace(0, 1, len(all_classes)))

            for i, class_id in enumerate(all_classes):
                class_counts = []
                for zone_id in zone_ids:
                    count = zone_stats[zone_id]["class_distribution"].get(class_id, 0)
                    class_counts.append(count)

                ax3.bar(zone_ids, class_counts, bottom=bottom,
                       label=f'Class {class_id}', color=colors[i], alpha=0.8)
                bottom += np.array(class_counts)

            ax3.set_xlabel('Zone ID')
            ax3.set_ylabel('Number of Samples')
            ax3.set_title('Class Distribution per Zone')
            ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax3.tick_params(axis='x', rotation=45)

        # Plot 4: Device resource distribution
        if "device_stats" in distribution_analysis:
            device_stats = distribution_analysis["device_stats"]
            sample_counts = [stats["total_samples"] for stats in device_stats.values()]

            ax4.hist(sample_counts, bins=20, alpha=0.7, edgecolor='black')
            ax4.set_xlabel('Samples per Device')
            ax4.set_ylabel('Number of Devices')
            ax4.set_title('Device Sample Distribution')
            ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=self.dpi, bbox_inches='tight')
        plt.show()
    
    def plot_convergence_analysis(self, training_history: List[Dict[str, Any]], 
                                baseline_results: Optional[Dict[str, Dict[str, Any]]] = None,
                                save_name: str = "convergence_analysis.png"):
        """Plot convergence analysis and comparison"""
        
        plt.figure(figsize=self.fig_size)
        
        # ContinuumFL convergence
        rounds = [entry["round"] for entry in training_history]
        accuracies = [entry["global_accuracy"] for entry in training_history]
        
        # Find convergence point (accuracy stabilization)
        convergence_round = self._find_convergence_point(accuracies)
        
        plt.plot(rounds, accuracies, label='ContinuumFL', linewidth=3, marker='o', markersize=4)
        
        if convergence_round > 0:
            plt.axvline(x=convergence_round, color='red', linestyle='--', alpha=0.7,
                       label=f'ContinuumFL Convergence (Round {convergence_round})')
        
        # Baseline convergence
        if baseline_results:
            for method_name, results in baseline_results.items():
                if "accuracies" in results:
                    baseline_rounds = range(len(results["accuracies"]))
                    plt.plot(baseline_rounds, results["accuracies"], 
                            label=method_name, linewidth=2, alpha=0.8)
                    
                    # Find baseline convergence
                    baseline_conv = self._find_convergence_point(results["accuracies"])
                    if baseline_conv > 0:
                        plt.axvline(x=baseline_conv, linestyle=':', alpha=0.5,
                                   label=f'{method_name} Conv. (Round {baseline_conv})')
        
        plt.xlabel('Round')
        plt.ylabel('Global Accuracy')
        plt.title('Convergence Analysis Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=self.dpi, bbox_inches='tight')
        plt.show()
    
    def _find_convergence_point(self, accuracies: List[float], window_size: int = 10, 
                              threshold: float = 0.001) -> int:
        """Find convergence point based on accuracy stabilization"""
        if len(accuracies) < window_size:
            return -1
        
        for i in range(window_size, len(accuracies)):
            recent_acc = accuracies[i-window_size:i]
            acc_variance = np.var(recent_acc)
            
            if acc_variance < threshold:
                return i - window_size
        
        return -1
    
    def create_comprehensive_report(self, coordinator: Any, baseline_results: Optional[Dict] = None):
        """Create a comprehensive visualization report"""
        
        print("Creating comprehensive visualization report...")

        print(f"Train History: {coordinator.training_history}")

        # Get system status and data
        training_history = list(coordinator.training_history)
        aggregation_stats = coordinator.aggregator.get_aggregation_stats()

        # 1. Training curves
        self.plot_training_curves(training_history, baseline_results, "01_training_curves.png")
        
        # 2. Zone performance
        self.plot_zone_performance(training_history, "02_zone_performance.png")
        
        # 3. Spatial distribution
        self.plot_spatial_distribution(coordinator.devices, coordinator.zones, "03_spatial_distribution.png")
        
        # 4. Communication costs
        self.plot_communication_costs(training_history, "04_communication_costs.png")
        
        # 5. Device participation
        self.plot_device_participation(training_history, coordinator.device_participation, 
                                     "05_device_participation.png")
        
        # 6. Zone weights
        self.plot_zone_aggregation_weights(aggregation_stats, "06_zone_weights.png")
        
        # 7. Spatial correlations
        self.plot_spatial_correlations(coordinator.zones, "07_spatial_correlations.png")
        
        # 8. Data distribution
        if hasattr(coordinator.dataset, 'analyze_data_distribution'):
            distribution_analysis = coordinator.dataset.analyze_data_distribution(zones=coordinator.zones)
            with open(os.path.join(self.save_dir, "distribution_analysis.json"), 'w') as f:
                json.dump(distribution_analysis, f, indent=2, default=str)
            self.plot_data_distribution_analysis(distribution_analysis, "08_data_distribution.png")
        
        # 9. Convergence analysis
        self.plot_convergence_analysis(training_history, baseline_results, "09_convergence_analysis.png")
        
        # Save summary statistics
        self._save_summary_statistics(coordinator, baseline_results)
        
        print(f"Comprehensive report saved to {self.save_dir}")
    
    def _save_summary_statistics(self, coordinator: Any, baseline_results: Optional[Dict] = None):
        """Save summary statistics to JSON file"""
        
        summary = {
            "system_configuration": {
                "num_devices": len(coordinator.devices),
                "num_zones": len(coordinator.zones),
                "num_rounds": coordinator.current_round,
                "dataset": coordinator.config.dataset_name
            },
            "final_performance": {
                "continuum_fl": {
                    "accuracy": coordinator.accuracies[-1] if coordinator.accuracies else 0.0,
                    "loss": coordinator.losses[-1] if coordinator.losses else float('inf'),
                    "total_communication_cost": sum(coordinator.communication_costs)
                }
            },
            "aggregation_statistics": coordinator.aggregator.get_aggregation_stats(),
            "zone_discovery_statistics": coordinator.zone_discovery.get_discovery_stats()
        }
        
        if baseline_results:
            summary["baseline_comparison"] = baseline_results
        
        with open(os.path.join(self.save_dir, "summary_statistics.json"), 'w') as f:
            json.dump(summary, f, indent=2, default=str)  # default=str to handle non-serializable types