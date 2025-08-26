"""
Zone Discovery module for ContinuumFL framework.
Implements the dynamic zone discovery algorithm from Section 4.1 of the paper.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Set, Optional, Any
from collections import defaultdict, deque
import networkx as nx
from sklearn.cluster import KMeans
import time

from .device import EdgeDevice
from .zone import Zone

class ZoneDiscovery:
    """
    Implements the adaptive zone discovery algorithm from Algorithm 1.
    
    Clusters devices based on multi-dimensional similarity:
    - Spatial proximity
    - Data distribution similarity 
    - Network characteristics
    """
    
    def __init__(self, config):
        self.config = config
        
        # Similarity weights (ω1, ω2, ω3)
        self.spatial_weight = config.similarity_weights['spatial']
        self.data_weight = config.similarity_weights['data'] 
        self.network_weight = config.similarity_weights['network']
        
        # Clustering parameters
        self.similarity_threshold = config.similarity_threshold  # θ
        self.min_zone_size = config.min_zone_size              # n_min
        self.max_zone_size = config.max_zone_size              # n_max
        self.distance_scaling = config.distance_scaling        # σ
        
        # Stability parameters
        self.stability_tradeoff = config.stability_tradeoff    # γ
        self.stability_threshold = config.stability_threshold  # θ_stability
        self.stability_window = config.stability_window        # W
        
        # History tracking for stability
        self.migration_history = defaultdict(lambda: defaultdict(deque))  # H(z_k, z_j)
        self.zone_assignments_history = deque(maxlen=self.stability_window)
        self.current_round = 0
    
    def compute_device_similarity(self, device_i: EdgeDevice, 
                                device_j: EdgeDevice) -> float:
        """
        Compute multi-dimensional similarity between two devices.
        
        Implements Equation (5):
        Sim(d_i, d_j) = ω1·S_spatial + ω2·S_data + ω3·S_network
        """
        # Spatial similarity - Equation (6)
        distance = device_i.compute_spatial_distance(device_j)
        spatial_sim = np.exp(-distance / self.distance_scaling)
        
        # Data similarity - Equation (7) 
        data_sim = device_i.compute_gradient_similarity(device_j)
        
        # Network similarity - described after Equation (7)
        network_sim = device_i.compute_network_similarity(device_j)
        
        # Combined similarity
        total_sim = (self.spatial_weight * spatial_sim + 
                    self.data_weight * data_sim + 
                    self.network_weight * network_sim)
        
        return total_sim
    
    def compute_similarity_matrix(self, devices: List[EdgeDevice]) -> np.ndarray:
        """Compute pairwise similarity matrix for all devices"""
        n_devices = len(devices)
        similarity_matrix = np.zeros((n_devices, n_devices))
        
        for i, device_i in enumerate(devices):
            for j, device_j in enumerate(devices):
                if i != j:
                    similarity_matrix[i, j] = self.compute_device_similarity(device_i, device_j)
                else:
                    similarity_matrix[i, j] = 1.0  # Self-similarity
        
        return similarity_matrix
    
    def hierarchical_clustering(self, devices: List[EdgeDevice]) -> List[Set[int]]:
        """
        Perform hierarchical clustering based on device similarity.
        
        Implements the core of Algorithm 1.
        """
        n_devices = len(devices)
        
        # Initialize each device as a singleton cluster
        clusters = [set([i]) for i in range(n_devices)]
        
        while len(clusters) > self.config.num_zones:
            # Compute similarity matrix for current clusters
            cluster_similarities = self._compute_cluster_similarities(clusters, devices)
            
            # Find most similar cluster pair
            max_sim = -1
            best_pair = None
            
            for i, cluster_i in enumerate(clusters):
                for j, cluster_j in enumerate(clusters):
                    if i < j:  # Avoid duplicates
                        sim = cluster_similarities[i][j]
                        new_size = len(cluster_i) + len(cluster_j)
                        
                        if (sim > max_sim and 
                            sim > self.similarity_threshold and 
                            new_size <= self.max_zone_size):
                            max_sim = sim
                            best_pair = (i, j)
            
            # If no valid merge found, break
            if best_pair is None:
                break
            
            # Merge the best pair
            i, j = best_pair
            merged_cluster = clusters[i].union(clusters[j])
            
            # Remove old clusters and add merged one
            new_clusters = []
            for k, cluster in enumerate(clusters):
                if k != i and k != j:
                    new_clusters.append(cluster)
            new_clusters.append(merged_cluster)
            clusters = new_clusters
        
        return clusters
    
    def _compute_cluster_similarities(self, clusters: List[Set[int]], 
                                    devices: List[EdgeDevice]) -> List[List[float]]:
        """Compute similarity between clusters using average linkage"""
        n_clusters = len(clusters)
        similarities = [[0.0 for _ in range(n_clusters)] for _ in range(n_clusters)]
        
        for i, cluster_i in enumerate(clusters):
            for j, cluster_j in enumerate(clusters):
                if i != j:
                    total_sim = 0.0
                    count = 0
                    
                    for device_idx_i in cluster_i:
                        for device_idx_j in cluster_j:
                            sim = self.compute_device_similarity(
                                devices[device_idx_i], devices[device_idx_j]
                            )
                            total_sim += sim
                            count += 1
                    
                    if count > 0:
                        similarities[i][j] = total_sim / count
                else:
                    similarities[i][j] = 1.0
        
        return similarities
    
    def split_large_clusters(self, clusters: List[Set[int]], 
                           devices: List[EdgeDevice]) -> List[Set[int]]:
        """Split clusters that exceed maximum size using k-means"""
        final_clusters = []
        
        for cluster in clusters:
            if len(cluster) > self.max_zone_size:
                # Extract device locations for k-means
                locations = np.array([devices[i].location for i in cluster])
                
                # Determine number of sub-clusters
                n_subclusters = int(np.ceil(len(cluster) / self.max_zone_size))
                
                # Apply k-means clustering
                kmeans = KMeans(n_clusters=n_subclusters, random_state=42)
                labels = kmeans.fit_predict(locations)
                
                # Create sub-clusters
                subclusters = defaultdict(set)
                for idx, (device_idx, label) in enumerate(zip(cluster, labels)):
                    subclusters[label].add(device_idx)
                
                final_clusters.extend(subclusters.values())
            else:
                final_clusters.append(cluster)
        
        return final_clusters
    
    def merge_small_clusters(self, clusters: List[Set[int]], 
                           devices: List[EdgeDevice]) -> List[Set[int]]:
        """Merge clusters that are below minimum size"""
        final_clusters = []
        small_clusters = []
        
        # Separate small and normal clusters
        for cluster in clusters:
            if len(cluster) < self.min_zone_size:
                small_clusters.append(cluster)
            else:
                final_clusters.append(cluster)
        
        # Merge small clusters with nearest neighbors
        while small_clusters:
            small_cluster = small_clusters.pop(0)
            
            if not final_clusters:
                # If no normal clusters exist, merge with another small cluster
                if small_clusters:
                    merge_with = small_clusters.pop(0)
                    merged = small_cluster.union(merge_with)
                    if len(merged) >= self.min_zone_size:
                        final_clusters.append(merged)
                    else:
                        small_clusters.append(merged)
                else:
                    final_clusters.append(small_cluster)  # Keep as is
                continue
            
            # Find nearest normal cluster
            best_cluster_idx = 0
            best_similarity = -1
            
            for i, normal_cluster in enumerate(final_clusters):
                sim = self._compute_cluster_similarity(small_cluster, normal_cluster, devices)
                if sim > best_similarity:
                    best_similarity = sim
                    best_cluster_idx = i
            
            # Merge with best cluster if size constraint allows
            best_cluster = final_clusters[best_cluster_idx]
            if len(best_cluster) + len(small_cluster) <= self.max_zone_size:
                final_clusters[best_cluster_idx] = best_cluster.union(small_cluster)
            else:
                # Try to merge with other small clusters
                merged = False
                for i, other_small in enumerate(small_clusters):
                    if len(small_cluster) + len(other_small) <= self.max_zone_size:
                        small_clusters[i] = small_cluster.union(other_small)
                        merged = True
                        break
                
                if not merged:
                    final_clusters.append(small_cluster)  # Keep as separate zone
        
        return final_clusters
    
    def _compute_cluster_similarity(self, cluster_i: Set[int], cluster_j: Set[int],
                                  devices: List[EdgeDevice]) -> float:
        """Compute similarity between two clusters"""
        total_sim = 0.0
        count = 0
        
        for device_idx_i in cluster_i:
            for device_idx_j in cluster_j:
                sim = self.compute_device_similarity(devices[device_idx_i], devices[device_idx_j])
                total_sim += sim
                count += 1
        
        return total_sim / max(count, 1)
    
    def compute_reassignment_cost(self, device: EdgeDevice, 
                                from_zone: str, to_zone: str,
                                zones: Dict[str, Zone]) -> float:
        """
        Compute cost of reassigning device between zones.
        
        Implements Equation (8):
        Cost(d_i: z_k → z_j) = (1 - Sim(d_i, z_j)) + γ·H(z_k, z_j)
        """
        if to_zone not in zones:
            return float('inf')
        
        target_zone = zones[to_zone]
        
        # Compute similarity to target zone - Equation (9)
        if target_zone.devices:
            zone_similarity = np.mean([
                self.compute_device_similarity(device, other_device)
                for other_device in target_zone.devices.values()
                if other_device.is_active
            ])
        else:
            zone_similarity = 0.0
        
        # Historical stability term - Equation (11)
        stability_cost = 0.0
        if from_zone in self.migration_history and to_zone in self.migration_history[from_zone]:
            recent_migrations = list(self.migration_history[from_zone][to_zone])
            if recent_migrations:
                stability_cost = np.mean(recent_migrations)
        
        total_cost = (1.0 - zone_similarity) + self.stability_tradeoff * stability_cost
        return total_cost
    
    def should_reassign_device(self, device: EdgeDevice, 
                             current_zone: str, candidate_zone: str,
                             zones: Dict[str, Zone]) -> bool:
        """Check if device should be reassigned based on cost"""
        reassignment_cost = self.compute_reassignment_cost(
            device, current_zone, candidate_zone, zones
        )
        
        staying_cost = self.compute_reassignment_cost(
            device, current_zone, current_zone, zones
        )
        
        return reassignment_cost < staying_cost - self.stability_threshold
    
    def update_migration_history(self, old_assignments: Dict[str, str], 
                               new_assignments: Dict[str, str]):
        """Update migration history for stability tracking"""
        migrations = defaultdict(lambda: defaultdict(int))
        
        for device_id in old_assignments:
            old_zone = old_assignments.get(device_id)
            new_zone = new_assignments.get(device_id)
            
            if old_zone and new_zone and old_zone != new_zone:
                migrations[old_zone][new_zone] += 1
        
        # Update rolling history
        for from_zone, to_zones in migrations.items():
            for to_zone, count in to_zones.items():
                migration_count = count / max(len(old_assignments), 1)
                self.migration_history[from_zone][to_zone].append(migration_count)
                
                # Maintain window size
                if len(self.migration_history[from_zone][to_zone]) > self.stability_window:
                    self.migration_history[from_zone][to_zone].popleft()
    
    def discover_zones(self, devices: List[EdgeDevice], 
                      existing_zones: Optional[Dict[str, Zone]] = None) -> Dict[str, Zone]:
        """
        Main zone discovery algorithm.
        
        Implements Algorithm 1 from the paper.
        """
        start_time = time.time()
        
        # Store old assignments for stability tracking
        old_assignments = {}
        if existing_zones:
            for zone in existing_zones.values():
                for device_id in zone.device_ids:
                    old_assignments[device_id] = zone.zone_id
        
        # Phase 1: Initial clustering
        active_devices = [d for d in devices if d.is_active]
        
        if len(active_devices) < self.min_zone_size:
            # Not enough devices for clustering
            if existing_zones:
                return existing_zones
            else:
                # Create single zone
                zone = Zone(zone_id="zone_0", edge_server_id="server_0")
                for device in active_devices:
                    zone.add_device(device)
                return {"zone_0": zone}
        
        # Hierarchical clustering
        clusters = self.hierarchical_clustering(active_devices)
        
        # Phase 2: Size constraint enforcement
        clusters = self.split_large_clusters(clusters, active_devices)
        clusters = self.merge_small_clusters(clusters, active_devices)
        
        # Phase 3: Create zones from clusters
        zones = {}
        new_assignments = {}
        
        for i, cluster in enumerate(clusters):
            zone_id = f"zone_{i}"
            edge_server_id = f"server_{i}"
            
            zone = Zone(zone_id=zone_id, edge_server_id=edge_server_id)
            
            for device_idx in cluster:
                device = active_devices[device_idx]
                zone.add_device(device)
                new_assignments[device.device_id] = zone_id
            
            zones[zone_id] = zone
        
        # Phase 4: Update spatial correlations
        for zone in zones.values():
            zone.update_spatial_correlations(zones)
            zone.identify_neighbor_zones(zones)
        
        # Update migration history for stability
        self.update_migration_history(old_assignments, new_assignments)
        
        # Store assignment history
        self.zone_assignments_history.append(new_assignments.copy())
        self.current_round += 1
        
        discovery_time = time.time() - start_time
        
        print(f"Zone discovery completed in {discovery_time:.2f}s")
        print(f"Created {len(zones)} zones with sizes: {[len(z) for z in zones.values()]}")
        
        return zones
    
    def adaptive_zone_update(self, devices: List[EdgeDevice], 
                           zones: Dict[str, Zone]) -> Dict[str, Zone]:
        """
        Incrementally update zone assignments based on changing conditions.
        
        This is more efficient than full rediscovery when only minor changes are needed.
        """
        reassignments = []
        
        # Check each device for potential reassignment
        for device in devices:
            if not device.is_active or not device.zone_id:
                continue
            
            current_zone_id = device.zone_id
            
            # Evaluate alternative zones
            for candidate_zone_id, candidate_zone in zones.items():
                if candidate_zone_id != current_zone_id:
                    if self.should_reassign_device(device, current_zone_id, 
                                                 candidate_zone_id, zones):
                        reassignments.append((device.device_id, current_zone_id, candidate_zone_id))
        
        # Apply reassignments
        for device_id, from_zone_id, to_zone_id in reassignments:
            if from_zone_id in zones and to_zone_id in zones:
                device = zones[from_zone_id].devices[device_id]
                zones[from_zone_id].remove_device(device_id)
                zones[to_zone_id].add_device(device)
        
        # Update correlations if reassignments occurred
        if reassignments:
            for zone in zones.values():
                zone.update_spatial_correlations(zones)
                zone.identify_neighbor_zones(zones)
        
        return zones
    
    def get_discovery_stats(self) -> Dict[str, Any]:
        """Get statistics about the zone discovery process"""
        if not self.zone_assignments_history:
            return {}
        
        # Calculate stability metrics
        recent_assignments = list(self.zone_assignments_history)[-5:]  # Last 5 rounds
        stability_scores = []
        
        for i in range(1, len(recent_assignments)):
            prev_assign = recent_assignments[i-1]
            curr_assign = recent_assignments[i]
            
            # Calculate percentage of devices that didn't change zones
            stable_devices = sum(1 for device_id in prev_assign 
                               if curr_assign.get(device_id) == prev_assign[device_id])
            stability = stable_devices / max(len(prev_assign), 1)
            stability_scores.append(stability)
        
        avg_stability = np.mean(stability_scores) if stability_scores else 1.0
        
        return {
            "rounds_completed": self.current_round,
            "average_stability": avg_stability,
            "total_migrations": sum(
                sum(sum(migration_counts.values()) for migration_counts in from_zone.values())
                for from_zone in self.migration_history.values()
            ),
            "migration_history_size": len(self.migration_history)
        }