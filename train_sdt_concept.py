"""
Soft Decision Tree with Concept Distillation for MiniGrid
- Uses MinigridFeaturesExtractor to extract concept features
- Trains SDT on top of extracted features
- Total loss = SDT loss + concept losses
- Flexible: load from file or collect from environment
"""

import os
import shutil
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from minigrid.wrappers import ImgObsWrapper
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import json
import pickle

from minigrid_features_extractor import MinigridFeaturesExtractor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor


# -----------------------
# Soft Decision Tree with Concept Support
# -----------------------
class SoftDecisionTree(nn.Module):
    def __init__(self, input_dim, output_dim, depth=4, lambda_1=0.05, lambda_2=0.004, lambda_3=2.0):
        super(SoftDecisionTree, self).__init__()
        self.depth = depth
        self.num_leaves = 2 ** depth
        self.num_inner_nodes = 2 ** depth - 1
        
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        
        # Routing decisions at inner nodes
        self.inner_nodes = nn.Linear(input_dim, self.num_inner_nodes)
        
        # Action distributions at leaves
        self.leaves = nn.Parameter(torch.randn(self.num_leaves, output_dim))
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Probability of going right at each node
        p_right = torch.sigmoid(self.inner_nodes(x))
        p_left = 1.0 - p_right
        
        path_probs = torch.ones(batch_size, self.num_leaves, device=x.device)
        
        # Compute path probabilities
        for i in range(self.num_leaves):
            node_idx = 0
            path_prob = torch.ones(batch_size, device=x.device)
            for d in range(self.depth):
                is_right = (i >> (self.depth - 1 - d)) & 1
                if is_right:
                    path_prob = path_prob * p_right[:, node_idx]
                    node_idx = 2 * node_idx + 2
                else:
                    path_prob = path_prob * p_left[:, node_idx]
                    node_idx = 2 * node_idx + 1
            path_probs[:, i] = path_prob
            
        leaf_probs = torch.softmax(self.leaves, dim=-1)
        out = torch.matmul(path_probs, leaf_probs)
        return torch.log(out + 1e-8)


# -----------------------
# Wrapper compatible with test_agent
# -----------------------
class SDTWrapper:
    def __init__(self, model):
        self.model = model
        self.device = next(model.parameters()).device
        
    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            if not isinstance(X, torch.Tensor):
                X = torch.tensor(X, dtype=torch.float32)
            X = X.to(self.device)
            log_probs = self.model(X)
            actions = torch.argmax(log_probs, dim=1)
            return actions.cpu().numpy()


# -----------------------
# Environment helpers
# -----------------------
def make_env(env_id, seed=0):
    def _init():
        env = gym.make(env_id, render_mode="rgb_array")
        env = ImgObsWrapper(env)
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init


# -----------------------
# Data collection from environment
# -----------------------
def collect_data_from_env(env_id, n_episodes=100, n_envs=4, seed=42, 
                         n_concepts=4, concept_mode=1):
    """
    Collect observations and actions from environment rollouts
    using MinigridFeaturesExtractor to extract features
    """
    print(f"Collecting data from {env_id}...")
    
    # Create environments
    envs = DummyVecEnv([make_env(env_id, seed+i) for i in range(n_envs)])
    
    # Initialize feature extractor with concept distillation
    extractor = MinigridFeaturesExtractor(
        features_dim=128,
        concept_distilling=True,
        n_concepts=n_concepts,
        concept_mode=concept_mode
    )
    extractor.eval()
    
    features_list = []
    actions_list = []
    
    obs, _ = envs.reset()
    steps = 0
    episodes = 0
    
    with torch.no_grad():
        while episodes < n_episodes:
            obs_images = obs["image"]
            obs_tensor = torch.tensor(obs_images, dtype=torch.float32) / 255.0
            
            # Extract features using MinigridFeaturesExtractor
            features = extractor(obs_tensor)
            
            # Random actions
            actions = np.array([envs.single_action_space.sample() for _ in range(n_envs)])
            
            # Store
            features_list.append(features.cpu().numpy())
            actions_list.append(actions.copy())
            
            # Step
            obs, rewards, dones, truncs, info = envs.step(actions)
            steps += 1
            
            # Reset on done
            for i in range(n_envs):
                if dones[i] or truncs[i]:
                    episodes += 1
                    obs[i], _ = envs.reset()
            
            if steps % 500 == 0:
                print(f"  Collected {episodes} episodes, {steps} steps...")
    
    envs.close()
    
    # Stack data
    X = np.vstack(features_list)
    y = np.concatenate(actions_list)
    
    print(f"✓ Collected {len(X)} samples from {episodes} episodes")
    print(f"  X shape: {X.shape}, y shape: {y.shape}")
    
    return X, y, extractor


# -----------------------
# Training function
# -----------------------
def train_sdt(
        env_id="MiniGrid-Empty-5x5-v0",
        n_episodes=100,
        n_envs=4,
        n_concepts=4,
        concept_mode=1,
        seed=42,
        device="cuda",
        depth=5,
        learning_rate=0.01,
        epochs=500,
        lambda_1=0.05,     # Orthogonality
        lambda_2=0.004,    # Sparsity
        lambda_3=2.0,      # L1
        use_file_data=False,
        data_file="dt_training_data.pkl"
):
    """
    Train SDT with concept distillation losses
    
    Args:
        use_file_data: If True, load from data_file; If False, collect from environment
        data_file: Path to pickle file with X, y data
    """
    output_dir = "dt_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load or collect data
    if use_file_data:
        print(f"Loading data from {data_file}...")
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
        X_all = data['X']
        y_all = data['y']
        feature_extractor = None
        print(f"✓ Loaded {len(X_all)} samples")
    else:
        X_all, y_all, feature_extractor = collect_data_from_env(
            env_id,
            n_episodes=n_episodes,
            n_envs=n_envs,
            seed=seed,
            n_concepts=n_concepts,
            concept_mode=concept_mode
        )
        device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        feature_extractor = feature_extractor.to(device_obj)
        feature_extractor.eval()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42
    )
    
    input_dim = X_train.shape[1]
    output_dim = len(np.unique(y_all))
    
    print(f"\nTraining SDT with Concept Distillation:")
    print(f"  Input dim: {input_dim}")
    print(f"  Output dim: {output_dim}")
    print(f"  Depth: {depth}")
    print(f"  Lambda (ortho, spar, l1): ({lambda_1}, {lambda_2}, {lambda_3})")
    
    # Prepare device and tensors
    device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device_obj)
    y_train_t = torch.tensor(y_train, dtype=torch.long).to(device_obj)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device_obj)
    y_test_t = torch.tensor(y_test, dtype=torch.long).to(device_obj)
    
    # Initialize model
    model = SoftDecisionTree(
        input_dim, output_dim, depth=depth,
        lambda_1=lambda_1, lambda_2=lambda_2, lambda_3=lambda_3
    ).to(device_obj)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()
    
    concept_losses_history = []
    
    print(f"\nTraining for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        output = model(X_train_t)
        loss = criterion(output, y_train_t)
        
        # ✅ Add concept losses if feature extractor available
        concept_loss = torch.tensor(0.0, device=device_obj)
        if feature_extractor is not None and hasattr(feature_extractor, "last_concept_losses"):
            if feature_extractor.last_concept_losses is not None:
                L_ortho, L_spar, L_l1 = feature_extractor.last_concept_losses
                concept_loss = lambda_1 * L_ortho + lambda_2 * L_spar + lambda_3 * L_l1
                concept_losses_history.append({
                    'epoch': epoch,
                    'L_ortho': L_ortho.item(),
                    'L_spar': L_spar.item(),
                    'L_l1': L_l1.item()
                })
        
        total_loss = loss + concept_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Evaluation every 50 epochs
        if (epoch + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                test_output = model(X_test_t)
                test_loss = criterion(test_output, y_test_t)
                
                y_pred = torch.argmax(test_output, dim=1).cpu().numpy()
                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                
                print(f"Epoch {epoch+1:3d}/{epochs} | Train Loss: {total_loss.item():.4f} | "
                      f"Test Loss: {test_loss.item():.4f} | Acc: {acc:.4f} | F1: {f1:.4f}")
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        y_pred = torch.argmax(model(X_test_t), dim=1).cpu().numpy()
        final_acc = accuracy_score(y_test, y_pred)
        final_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print(f"\n{'='*60}")
    print(f"✓ FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Test Accuracy: {final_acc:.4f}")
    print(f"F1 Score:     {final_f1:.4f}")
    print(f"{'='*60}\n")
    
    # Save model (compatible with test_agent)
    wrapped_model = SDTWrapper(model)
    model_path = os.path.join(output_dir, "dt_policy.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': wrapped_model,
            'evaluation': {'accuracy': final_acc, 'weighted_f1': final_f1},
            'hyperparams': {
                'depth': depth,
                'learning_rate': learning_rate,
                'epochs': epochs,
                'lambda_1': lambda_1,
                'lambda_2': lambda_2,
                'lambda_3': lambda_3
            }
        }, f)
    
    print(f"✓ Model saved to {model_path}")
    
    # Save training metadata
    metadata = {
        'env_id': env_id if not use_file_data else None,
        'n_episodes': n_episodes if not use_file_data else None,
        'depth': depth,
        'learning_rate': learning_rate,
        'epochs': epochs,
        'lambda_1': lambda_1,
        'lambda_2': lambda_2,
        'lambda_3': lambda_3,
        'accuracy': float(final_acc),
        'f1_score': float(final_f1),
        'timestamp': datetime.now().isoformat(),
        'concept_losses_samples': concept_losses_history[:10] if concept_losses_history else []
    }
    
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Metadata saved to {metadata_path}")
    
    return model, final_acc, final_f1


# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    # Option 1: Train from environment (with concept distillation)
    model, acc, f1 = train_sdt(
        env_id="MiniGrid-Empty-5x5-v0",
        n_episodes=100,
        n_envs=4,
        seed=42,
        device="cuda",
        depth=5,
        lambda_1=0.05,
        lambda_2=0.004,
        lambda_3=2.0,
        use_file_data=False
    )
    
    # Option 2: Train from file (uncomment to use)
    # model, acc, f1 = train_sdt(
    #     depth=5,
    #     lambda_1=0.05,
    #     lambda_2=0.004,
    #     lambda_3=2.0,
    #     use_file_data=True,
    #     data_file="dt_training_data.pkl"
    # )
