"""
Baseline federated learning methods for comparison with ContinuumFL.
Implements FedAvg, FedProx, HierFL, and ClusterFL.
"""

import torch
import torch.nn as nn
import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from sklearn.cluster import KMeans
import copy
import torch.nn.functional as _F

class BaselineFLMethods:
    """
    Implements baseline federated learning methods for comparison.
    """
    
    def __init__(self, config):
        self.config = config
        self.baseline_results = {}
    
    def run_method(self, method_name: str, devices: Dict[str, Any], 
                  global_model: nn.Module, dataset: Any) -> Dict[str, Any]:
        """Run a specific baseline method"""
        
        if method_name.lower() == 'fedavg':
            return self._run_fedavg(devices, global_model, dataset)
        elif method_name.lower() == 'fedprox':
            return self._run_fedprox(devices, global_model, dataset)
        elif method_name.lower() == 'hierfl':
            return self._run_hierfl(devices, global_model, dataset)
        elif method_name.lower() == 'clusterfl':
            return self._run_clusterfl(devices, global_model, dataset)
        else:
            raise ValueError(f"Unknown baseline method: {method_name}")
    
    def _run_fedavg(self, devices: Dict[str, Any], global_model: nn.Module, 
                   dataset: Any) -> Dict[str, Any]:
        """
        Run FedAvg baseline.
        
        Standard federated averaging without spatial awareness.
        """
        print("Running FedAvg baseline...")
        
        # Initialize tracking
        accuracies = []
        losses = []
        start_time = time.time()
        
        # Training loop
        try:
            for round_num in range(self.config.num_rounds):
                round_time = time.time()

                # Sample participating devices
                participating_devices = self._sample_devices(devices, 0.7)

                # Collect device updates
                device_updates = []
                device_weights = []

                for device_id in participating_devices:
                    device = devices[device_id]
                    if not device.is_active or not device.local_dataset:
                        continue

                    # Local training
                    local_model = copy.deepcopy(global_model)
                    local_model.load_state_dict(global_model.state_dict())

                    # Train locally
                    training_result = self._train_local_model(
                        local_model, device.local_dataloader,
                        self.config.local_epochs, self.config.learning_rate
                    )

                    if training_result["success"]:
                        device_updates.append(training_result["model_weights"])
                        device_weights.append(device.dataset_size)

                # FedAvg aggregation
                if device_updates:
                    global_weights = self._fedavg_aggregate(device_updates, device_weights)
                    global_model.load_state_dict(global_weights)

                # Evaluate
                accuracy, loss = self._evaluate_model(global_model, dataset)
                accuracies.append(accuracy)
                losses.append(loss)
                print(f"FedAvg Round {round_num + 1}: Accuracy={accuracy:.4f}, Loss={loss:.4f}, Time={time.time()-round_time:.2f}")
        except KeyboardInterrupt:
            print(f"Training (FedProx) interrupted by user")
        total_time = time.time() - start_time
        
        return {
            "method": "FedAvg",
            "final_accuracy": accuracies[-1] if accuracies else 0.0,
            "final_loss": losses[-1] if losses else float('inf'),
            "accuracies": accuracies,
            "losses": losses,
            "total_time": total_time,
            "convergence_round": self._find_convergence(accuracies)
        }
    
    def _run_fedprox(self, devices: Dict[str, Any], global_model: nn.Module, 
                    dataset: Any) -> Dict[str, Any]:
        """
        Run FedProx baseline.
        
        Federated learning with proximal term for heterogeneous devices.
        """
        print("Running FedProx baseline...")
        
        mu = 0.01  # Proximal term coefficient
        accuracies = []
        losses = []
        start_time = time.time()

        try:
            for round_num in range(self.config.num_rounds):
                round_time = time.time()

                participating_devices = self._sample_devices(devices, 0.7)
                device_updates = []
                device_weights = []

                for device_id in participating_devices:
                    device = devices[device_id]
                    if not device.is_active or not device.local_dataset:
                        continue

                    # Local training with proximal term
                    local_model = copy.deepcopy(global_model)
                    global_weights = {name: param.clone() for name, param in global_model.named_parameters()}

                    training_result = self._train_local_model_fedprox(
                        local_model, device.local_dataloader, global_weights,
                        self.config.local_epochs, self.config.learning_rate, mu
                    )

                    if training_result["success"]:
                        device_updates.append(training_result["model_weights"])
                        device_weights.append(device.dataset_size)

                # Standard aggregation
                if device_updates:
                    global_weights = self._fedavg_aggregate(device_updates, device_weights)
                    global_model.load_state_dict(global_weights)

                # Evaluate
                accuracy, loss = self._evaluate_model(global_model, dataset)
                accuracies.append(accuracy)
                losses.append(loss)

                print(f"FedProx Round {round_num + 1}: Accuracy={accuracy:.4f}, Loss={loss:.4f}, Time={time.time()-round_time:.2f}")
        except KeyboardInterrupt:
            print(f"Training (FedProx) interrupted by user")
        total_time = time.time() - start_time
        
        return {
            "method": "FedProx",
            "final_accuracy": accuracies[-1] if accuracies else 0.0,
            "final_loss": losses[-1] if losses else float('inf'),
            "accuracies": accuracies,
            "losses": losses,
            "total_time": total_time,
            "convergence_round": self._find_convergence(accuracies)
        }
    
    def _run_hierfl(self, devices: Dict[str, Any], global_model: nn.Module, 
                   dataset: Any) -> Dict[str, Any]:
        """
        Run HierFL baseline.
        
        Static hierarchical aggregation based on network topology.
        """
        print("Running HierFL baseline...")
        
        # Create static clusters (zones) based on device IDs
        num_clusters = min(self.config.num_zones, len(devices) // 3)
        device_list = list(devices.keys())
        cluster_size = len(device_list) // num_clusters
        
        clusters = []
        for i in range(num_clusters):
            start_idx = i * cluster_size
            end_idx = start_idx + cluster_size if i < num_clusters - 1 else len(device_list)
            clusters.append(device_list[start_idx:end_idx])
        
        accuracies = []
        losses = []
        start_time = time.time()
        try:
            for round_num in range(self.config.num_rounds):
                round_time = time.time()

                # Two-level aggregation
                cluster_models = []
                cluster_weights = []

                for cluster in clusters:
                    # Intra-cluster aggregation
                    cluster_device_updates = []
                    cluster_device_weights = []

                    for device_id in cluster:
                        device = devices[device_id]
                        if not device.is_active or not device.local_dataset:
                            continue

                        if np.random.random() < 0.7:  # Participation probability
                            local_model = copy.deepcopy(global_model)
                            training_result = self._train_local_model(
                                local_model, device.local_dataloader,
                                self.config.local_epochs, self.config.learning_rate
                            )

                            if training_result["success"]:
                                cluster_device_updates.append(training_result["model_weights"])
                                cluster_device_weights.append(device.dataset_size)

                    # Aggregate within cluster
                    if cluster_device_updates:
                        cluster_model = self._fedavg_aggregate(cluster_device_updates, cluster_device_weights)
                        cluster_models.append(cluster_model)
                        cluster_weights.append(sum(cluster_device_weights))

                # Inter-cluster aggregation
                if cluster_models:
                    global_weights = self._fedavg_aggregate(cluster_models, cluster_weights)
                    global_model.load_state_dict(global_weights)

                # Evaluate
                accuracy, loss = self._evaluate_model(global_model, dataset)
                accuracies.append(accuracy)
                losses.append(loss)

                print(f"HierFL Round {round_num + 1}: Accuracy={accuracy:.4f}, Loss={loss:.4f}, Time={time.time()-round_time:.2f}")
        except KeyboardInterrupt:
            print(f"Training (HierFL) interrupted by user")
        total_time = time.time() - start_time
        
        return {
            "method": "HierFL",
            "final_accuracy": accuracies[-1] if accuracies else 0.0,
            "final_loss": losses[-1] if losses else float('inf'),
            "accuracies": accuracies,
            "losses": losses,
            "total_time": total_time,
            "convergence_round": self._find_convergence(accuracies),
            "num_clusters": num_clusters
        }

    def _run_clusterfl(self, devices, global_model, dataset):
        """
        Simple ClusterFL baseline:
        - participation rate (default 0.7)
        - softmax outputs → KL divergence → similarity matrix F
        - proximal updates toward weighted neighbors
        - FedAvg aggregation
        """

        participation_rate = 0.7
        rho = 0.01
        prox_step = 1.0
        temperature = 1.0
        recluster_every = 1
        batch_eval_size = 64

        start_time = time.time()
        accuracies, losses = [], []

        # template for converting vectors ↔ state_dict
        template_state_dict = global_model.state_dict()

        W_dim = None
        last_F = None

        try:
            for round_num in range(self.config.num_rounds):
                round_time = time.time()

                # --- Sample participants ---
                participating = np.random.choice(list(devices.keys()),
                                                 int(len(devices) * participation_rate),
                                                 replace=False)
                local_state_dicts = {}
                device_weights = []

                # --- Local training ---
                for cid in participating:
                    device = devices[cid]
                    if not device.is_active or not device.local_dataset:
                        continue

                    local_model = copy.deepcopy(global_model)
                    res = self._train_local_model(local_model, device.local_dataloader,
                                            self.config.local_epochs, self.config.learning_rate)
                    if not res.get("success", False):
                        continue

                    local_state_dicts[cid] = res["model_weights"]
                    device_weights.append(device.dataset_size)

                if not local_state_dicts:
                    acc, loss = self._evaluate_model(global_model, dataset)
                    accuracies.append(acc)
                    losses.append(loss)
                    continue

                client_ids = list(local_state_dicts.keys())
                state_dicts = [local_state_dicts[cid] for cid in client_ids]

                # --- Convert to vectors ---
                W = np.vstack([self.state_dict_to_vector(sd) for sd in state_dicts])
                if W_dim is None:
                    W_dim = W.shape[1]

                # --- Compute similarity F ---
                if (round_num % recluster_every == 0) or last_F is None:
                    # compute average softmax outputs
                    outputs = []
                    for sd in state_dicts:
                        m = copy.deepcopy(global_model)
                        m.load_state_dict(sd)
                        logits = self.predict_logits(m, dataset, batch_eval_size)
                        outputs.append(torch.softmax(torch.tensor(logits) / temperature, dim=1).numpy())
                    m = len(outputs)
                    F_mat = np.zeros((m, m))
                    for i in range(m):
                        for j in range(m):
                            p, q = outputs[i], outputs[j]
                            F_mat[i, j] = np.mean(np.sum(p * np.log((p + 1e-10) / (q + 1e-10)), axis=1))
                    # similarity = 1 - normalized KL
                    F_mat = 1 - (F_mat / (F_mat.max() + 1e-10))
                    F_mat = np.clip(F_mat, 0, 1)
                    # normalize columns
                    F_mat = F_mat / (F_mat.sum(axis=0, keepdims=True) + 1e-10)
                    last_F = F_mat
                else:
                    F_mat = last_F

                # --- Proximal update toward neighbors ---
                W_new = W - prox_step * rho * (W - F_mat.T @ W)

                # --- Convert back & aggregate (FedAvg) ---
                updated_state_dicts = [self.vector_to_state_dict(vec, template_state_dict) for vec in W_new]
                global_weights = self._fedavg_aggregate(updated_state_dicts, device_weights)
                global_model.load_state_dict(global_weights)

                # --- Eval ---
                acc, loss = self._evaluate_model(global_model, dataset)
                accuracies.append(acc)
                losses.append(loss)
                if round_num % 10 == 0:
                    print(f"Round {round_num + 1}: Accuracy={acc:.4f}, Loss={loss:.4f}, Time={time.time()-round_time:.2f}")
        except KeyboardInterrupt:
            print("Training (ClusterFL) interrupted by user")
        return {
            "method": "ClusterFL",
            "final_accuracy": accuracies[-1] if accuracies else 0.0,
            "final_loss": losses[-1] if losses else float('inf'),
            "accuracies": accuracies,
            "losses": losses,
            "total_time": time.time() - start_time
        }

    def state_dict_to_vector(self, state_dict: dict) -> np.ndarray:
        vecs = []
        for key in state_dict:
            vecs.append(state_dict[key].detach().cpu().numpy().ravel())
        return np.concatenate(vecs)

    def vector_to_state_dict(self, vec: np.ndarray, template_state_dict: dict) -> dict:
        new_state_dict = {}
        pointer = 0
        for key, param in template_state_dict.items():
            shape = param.shape
            numel = param.numel()
            new_state_dict[key] = torch.tensor(vec[pointer:pointer+numel].reshape(shape))
            pointer += numel
        return new_state_dict

    def predict_logits(self, model, dataset, batch_size=64):
        model.eval()
        logits_list = []
        test_dataloader = dataset.get_global_dataloader(batch_size=64, is_train=False)
        device = self.config.device

        with torch.no_grad():
            for data, target in test_dataloader:
                if device == 'cuda':
                    data, target = data.cuda(), target.cuda()
                if self.config.dataset_name == 'shakespeare':
                    out, _ = model(data)
                else:
                    out = model(data)
                logits_list.append(out.detach().cpu().numpy())
        return np.vstack(logits_list)

    def _sample_devices(self, devices: Dict[str, Any], participation_rate: float) -> List[str]:
        """Sample devices for participation"""
        available_devices = [
            device_id for device_id, device in devices.items()
            if device.is_active and device.local_dataset
        ]
        
        num_participants = max(1, int(participation_rate * len(available_devices)))
        return np.random.choice(available_devices, size=num_participants, replace=False).tolist()
    
    def _train_local_model(self, model: nn.Module, dataloader: torch.utils.data.DataLoader,
                          epochs: int, learning_rate: float) -> Dict[str, Any]:
        """Train local model using standard SGD"""
        if dataloader is None or len(dataloader) == 0:
            return {"success": False}
        
        # Move model to appropriate device
        device = self.config.device
        if device == 'cuda' and torch.cuda.is_available():
            model = model.cuda()
        else:
            model = model.cpu()
            device = 'cpu'
        
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        # Move criterion to same device
        if device == 'cuda':
            criterion = criterion.cuda()
        
        try:
            for epoch in range(epochs):
                for batch_idx, (data, target) in enumerate(dataloader):
                    # Move data to device
                    if device == 'cuda':
                        data, target = data.cuda(), target.cuda()
                    
                    optimizer.zero_grad()
                    output = model(data)
                    
                    # Handle different model outputs
                    if isinstance(output, tuple):
                        output = output[0]
                    
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
            
            # Move model back to CPU for state dict
            if device == 'cuda':
                model = model.cpu()
            
            return {
                "success": True,
                "model_weights": model.state_dict()
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _train_local_model_fedprox(self, model: nn.Module, dataloader: torch.utils.data.DataLoader,
                                  global_weights: Dict[str, torch.Tensor], epochs: int, 
                                  learning_rate: float, mu: float) -> Dict[str, Any]:
        """Train local model with FedProx proximal term"""
        if dataloader is None or len(dataloader) == 0:
            return {"success": False}
        
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        try:
            for epoch in range(epochs):
                for batch_idx, (data, target) in enumerate(dataloader):
                    if self.config.device == 'cuda' and torch.cuda.is_available():
                        data, target = data.cuda(), target.cuda()
                    
                    optimizer.zero_grad()
                    output = model(data)
                    
                    if isinstance(output, tuple):
                        output = output[0]
                    
                    # Standard loss
                    loss = criterion(output, target)
                    
                    # Add proximal term
                    proximal_term = 0.0
                    for name, param in model.named_parameters():
                        if name in global_weights:
                            proximal_term += torch.sum((param - global_weights[name]) ** 2)
                    
                    total_loss = loss + (mu / 2) * proximal_term
                    total_loss.backward()
                    optimizer.step()
            
            return {
                "success": True,
                "model_weights": model.state_dict()
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _fedavg_aggregate(self, model_updates: List[Dict[str, torch.Tensor]], 
                         weights: List[float]) -> Dict[str, torch.Tensor]:
        """Perform FedAvg aggregation"""
        if not model_updates:
            return {}
        
        # Normalize weights
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        # Initialize aggregated model
        aggregated_weights = {}
        first_model = model_updates[0]
        
        for param_name in first_model.keys():
            aggregated_weights[param_name] = torch.zeros_like(first_model[param_name], dtype=torch.float32)
        
        # Weighted aggregation
        for model_update, weight in zip(model_updates, normalized_weights):
            for param_name, param_tensor in model_update.items():
                if param_name in aggregated_weights:
                    aggregated_weights[param_name] += weight * param_tensor
        
        return aggregated_weights
    
    def _evaluate_model(self, model: nn.Module, dataset: Any) -> Tuple[float, float]:
        """Evaluate model on test dataset"""
        # Move model to appropriate device
        device = self.config.device
        if device == 'cuda' and torch.cuda.is_available():
            model = model.cuda()
        else:
            model = model.cpu()
            device = 'cpu'
        
        model.eval()
        
        try:
            test_dataloader = dataset.get_global_dataloader(batch_size=64, is_train=False)
            
            total_loss = 0.0
            correct = 0
            total = 0
            criterion = nn.CrossEntropyLoss()
            
            # Move criterion to same device
            if device == 'cuda':
                criterion = criterion.cuda()
            
            with torch.no_grad():
                for data, target in test_dataloader:
                    # Move data to device
                    if device == 'cuda':
                        data, target = data.cuda(), target.cuda()
                    
                    output = model(data)
                    
                    if isinstance(output, tuple):
                        output = output[0]
                    
                    loss = criterion(output, target)
                    total_loss += loss.detach().item()
                    
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
                    total += target.size(0)
            
            accuracy = correct / max(total, 1)
            avg_loss = total_loss / max(len(test_dataloader), 1)
            
        except Exception as e:
            print(f"Evaluation error: {e}")
            accuracy = 0.0
            avg_loss = float('inf')
        
        model.train()
        return accuracy, avg_loss
    
    def _find_convergence(self, accuracies: List[float], window_size: int = 10) -> int:
        """Find convergence round based on accuracy stabilization"""
        if len(accuracies) < window_size:
            return -1
        
        for i in range(window_size, len(accuracies)):
            recent_acc = accuracies[i-window_size:i]
            acc_variance = np.var(recent_acc)
            
            if acc_variance < 0.001:  # Convergence threshold
                return i - window_size
        
        return -1  # Not converged
    
    def compare_methods(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Compare results from different baseline methods"""
        comparison = {
            "method_comparison": {},
            "best_method": "",
            "best_accuracy": 0.0,
            "convergence_comparison": {},
            "efficiency_comparison": {}
        }
        
        best_accuracy = 0.0
        best_method = ""
        
        for method_name, result in results.items():
            final_acc = result.get("final_accuracy", 0.0)
            final_loss = result.get("final_loss", float('inf'))
            total_time = result.get("total_time", 0.0)
            conv_round = result.get("convergence_round", -1)
            
            comparison["method_comparison"][method_name] = {
                "final_accuracy": final_acc,
                "final_loss": final_loss,
                "total_time": total_time,
                "convergence_round": conv_round
            }
            
            if final_acc > best_accuracy:
                best_accuracy = final_acc
                best_method = method_name
            
            # Efficiency metrics
            if conv_round > 0:
                comparison["convergence_comparison"][method_name] = conv_round
            
            comparison["efficiency_comparison"][method_name] = {
                "time_per_round": total_time / self.config.num_rounds,
                "accuracy_per_time": final_acc / max(total_time, 1e-6)
            }
        
        comparison["best_method"] = best_method
        comparison["best_accuracy"] = best_accuracy
        
        return comparison