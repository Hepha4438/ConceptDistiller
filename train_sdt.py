"""
Soft Decision Tree (SDT) Training Script for MiniGrid Environments
Based on "Distilling a Neural Network Into a Soft Decision Tree"

SDT learns from concepts extracted by the CNN feature extractor,
similar to PPO-Concept but with a tree-based policy.
"""

import os
import re
import gymnasium as gym
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback, 
    CheckpointCallback, 
    EvalCallback,
    CallbackList
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn
import torch.nn.functional as F
from minigrid_features_extractor import MinigridFeaturesExtractor
import numpy as np
from typing import Dict, List, Tuple, Any


def get_next_model_number(save_dir, prefix="sdt_minigrid"):
    """
    Find the next available model number (000-999) in the save directory.
    Scans for existing files matching pattern: {prefix}_XXX.zip
    Returns the next number after the highest found.
    """
    if not os.path.exists(save_dir):
        return 0

    # Find all files matching the pattern
    pattern = re.compile(rf"{re.escape(prefix)}_(\d{{3}})\.zip")
    existing_numbers = []

    for filename in os.listdir(save_dir):
        match = pattern.match(filename)
        if match:
            existing_numbers.append(int(match.group(1)))

    # Return next number (0 if none found)
    if not existing_numbers:
        return 0

    return max(existing_numbers) + 1


class SoftDecisionTree(nn.Module):
    """
    Soft Decision Tree implementation.
    Each internal node computes a soft routing probability using sigmoid.
    Leaf nodes output action distributions.
    """
    def __init__(self, input_dim: int, output_dim: int, depth: int = 5, temperature: float = 1.0):
        """
        Args:
            input_dim: Feature dimension from CNN backbone
            output_dim: Number of actions
            depth: Tree depth (depth=5 means 2^5=32 leaf nodes)
            temperature: Temperature for soft routing (lower = harder decisions)
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.depth = depth
        self.temperature = temperature
        
        # Number of internal nodes and leaf nodes
        self.n_internal_nodes = 2 ** depth - 1
        self.n_leaf_nodes = 2 ** depth
        
        # Internal node decision functions (linear + sigmoid)
        # Each node: linear transformation of features
        self.internal_nodes = nn.ModuleList([
            nn.Linear(input_dim, 1) for _ in range(self.n_internal_nodes)
        ])
        
        # Leaf node outputs (action logits)
        self.leaf_nodes = nn.Parameter(
            torch.randn(self.n_leaf_nodes, output_dim) * 0.1
        )
        
        # Penalty coefficient for regularization
        self.penalty_alpha = 0.5
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through soft decision tree.
        
        Args:
            x: Input features [batch_size, input_dim]
            
        Returns:
            action_logits: [batch_size, output_dim]
            info: Dictionary with routing probabilities and penalties
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Compute all internal node decisions
        # Shape: [batch_size, n_internal_nodes]
        node_decisions = torch.zeros(batch_size, self.n_internal_nodes, device=device)
        for i, node in enumerate(self.internal_nodes):
            node_decisions[:, i] = torch.sigmoid(node(x).squeeze(-1) / self.temperature)
        
        # Compute probability of reaching each leaf
        # Each path in the tree is a product of decisions
        leaf_probs = torch.ones(batch_size, self.n_leaf_nodes, device=device)
        
        for leaf_idx in range(self.n_leaf_nodes):
            # Trace path from root to this leaf
            path = self._get_path_to_leaf(leaf_idx)
            
            for node_idx, go_right in path:
                if go_right:
                    # Take right branch (decision = 1)
                    leaf_probs[:, leaf_idx] *= node_decisions[:, node_idx]
                else:
                    # Take left branch (decision = 0)
                    leaf_probs[:, leaf_idx] *= (1 - node_decisions[:, node_idx])
        
        # Weighted sum of leaf outputs
        # [batch_size, output_dim] = [batch_size, n_leaves] @ [n_leaves, output_dim]
        action_logits = torch.matmul(leaf_probs, self.leaf_nodes)
        
        # Compute regularization: encourage confident decisions at internal nodes
        # Penalty is higher when decisions are uncertain (close to 0.5)
        decision_penalty = torch.mean(torch.sum(
            node_decisions * (1 - node_decisions), dim=1
        ))
        
        info = {
            'leaf_probs': leaf_probs,
            'node_decisions': node_decisions,
            'decision_penalty': decision_penalty,
        }
        
        return action_logits, info
    
    def _get_path_to_leaf(self, leaf_idx: int) -> List[Tuple[int, bool]]:
        """
        Get the path from root to a specific leaf.
        Returns list of (node_idx, go_right) tuples.
        """
        path = []
        node_idx = 0  # Start at root
        
        for depth in range(self.depth):
            # Get bit at this depth (from left to right)
            bit = (leaf_idx >> (self.depth - 1 - depth)) & 1
            go_right = (bit == 1)
            path.append((node_idx, go_right))
            
            # Move to next node in tree
            if go_right:
                node_idx = 2 * node_idx + 2  # Right child
            else:
                node_idx = 2 * node_idx + 1  # Left child
        
        return path
    
    def get_tree_structure(self) -> Dict[str, Any]:
        """Get tree structure information for visualization."""
        return {
            'depth': self.depth,
            'n_internal_nodes': self.n_internal_nodes,
            'n_leaf_nodes': self.n_leaf_nodes,
            'temperature': self.temperature,
        }


class SDTMLPExtractor(nn.Module):
    """
    Wrapper for SDT to be compatible with Stable Baselines3's mlp_extractor interface.
    """
    def __init__(self, n_concepts: int, action_dim: int, tree_depth: int, tree_temperature: float):
        super().__init__()
        
        self.n_concepts = n_concepts
        # SB3 expects these dimensions for creating action_net and value_net
        # For SDT, we pass concepts through directly
        self.latent_dim_pi = n_concepts  # Output features for policy
        self.latent_dim_vf = n_concepts  # Output features for value
        
        # Create Soft Decision Tree for policy (will be used as action_net)
        self.sdt_policy = SoftDecisionTree(
            input_dim=n_concepts,
            output_dim=action_dim,
            depth=tree_depth,
            temperature=tree_temperature
        )
        
        # Store action dim for later
        self.action_dim = action_dim
        
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass, returns (policy_latent, value_latent).
        Both are the concepts themselves.
        """
        # Return concepts for both policy and value
        # SB3 will pass these through action_net and value_net
        return features, features
    
    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass for actor only."""
        return features
    
    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass for critic only."""
        return features


class SDTPolicy(ActorCriticPolicy):
    """
    Custom policy using Soft Decision Tree for action selection.
    SDT learns from concepts extracted from CNN feature extractor.
    """
    def __init__(self, *args, n_concepts: int = 4, tree_depth: int = 5, tree_temperature: float = 1.0, 
                 penalty_coef: float = 0.01, **kwargs):
        self.n_concepts = n_concepts
        self.tree_depth = tree_depth
        self.tree_temperature = tree_temperature
        self.penalty_coef = penalty_coef
        
        super().__init__(*args, **kwargs)
        
    def _build_mlp_extractor(self) -> None:
        """Override to use SDT wrapper instead of standard MLP."""
        self.mlp_extractor = SDTMLPExtractor(
            n_concepts=self.n_concepts,
            action_dim=self.action_space.n,
            tree_depth=self.tree_depth,
            tree_temperature=self.tree_temperature
        )
    
    def _build(self, lr_schedule) -> None:
        """
        Override _build to create SDT-based action and value networks.
        """
        # Build features extractor first
        self._build_mlp_extractor()
        
        # For policy: SDT outputs action logits directly
        # Create a linear layer from concepts to action logits
        self.action_net = self.mlp_extractor.sdt_policy
        
        # For value: simple MLP from concepts to value
        self.value_net = nn.Sequential(
            nn.Linear(self.n_concepts, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # Setup optimizer
        if self.optimizer_class is not None:
            self.optimizer = self.optimizer_class(
                self.parameters(),
                lr=lr_schedule(1),
                **self.optimizer_kwargs
            )
        
    def extract_concept_features(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Extract concepts from the feature extractor.
        This assumes the feature extractor has concept_distilling=True.
        """
        # Get the feature extractor
        extractor = self.features_extractor
        
        # Extract concepts
        if hasattr(extractor, 'concept_distilling') and extractor.concept_distilling:
            # Forward through CNN to get concepts
            _ = extractor(obs)  # This populates last_concepts
            
            # The concept vector should be stored in the extractor
            if hasattr(extractor, 'last_concepts'):
                concepts = extractor.last_concepts
                return concepts
        
        # Fallback: if no concepts available, extract features normally
        return self.extract_features(obs)
    
    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor):
        """
        Create action distribution from latent policy (action logits for SDT).
        """
        from stable_baselines3.common.distributions import CategoricalDistribution
        return CategoricalDistribution(self.action_space.n).proba_distribution(latent_pi)
    
    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the policy.
        
        Returns:
            actions, values, log_probs
        """
        # Extract concepts from feature extractor
        concepts = self.extract_concept_features(obs)
        
        # Get latent vectors from mlp_extractor
        latent_pi, latent_vf = self.mlp_extractor(concepts)
        
        # Get action logits from SDT (action_net is sdt_policy)
        action_logits, _ = self.action_net(latent_pi)
        
        # Get values from value network
        values = self.value_net(latent_vf)
        
        # Create action distribution from logits
        distribution = self._get_action_dist_from_latent(action_logits)
        actions = distribution.get_actions(deterministic=deterministic)
        log_probs = distribution.log_prob(actions)
        
        return actions, values.flatten(), log_probs
    
    def _predict(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Predict action without computing value and log_prob."""
        concepts = self.extract_concept_features(observation)
        
        # Get policy latent from mlp_extractor
        latent_pi = self.mlp_extractor.forward_actor(concepts)
        
        # Get action logits from SDT
        action_logits, _ = self.action_net(latent_pi)
        
        distribution = self._get_action_dist_from_latent(action_logits)
        return distribution.get_actions(deterministic=deterministic)
    
    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions for training."""
        concepts = self.extract_concept_features(obs)
        
        # Get latent vectors
        latent_pi, latent_vf = self.mlp_extractor(concepts)
        
        # Get action logits from SDT
        action_logits, _ = self.action_net(latent_pi)
        
        # Get values
        values = self.value_net(latent_vf)
        
        # Get log probabilities
        distribution = self._get_action_dist_from_latent(action_logits)
        log_probs = distribution.log_prob(actions)
        
        return values.flatten(), log_probs, distribution.entropy()
    
    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Get the estimated values according to the current policy.
        Override to use concept features instead of raw CNN features.
        """
        concepts = self.extract_concept_features(obs)
        
        # Get value latent from mlp_extractor
        latent_vf = self.mlp_extractor.forward_critic(concepts)
        
        # Get values from value network
        return self.value_net(latent_vf)
    
    def get_tree_penalty(self) -> torch.Tensor:
        """Get the tree decision penalty for regularization."""
        # Access the sdt_policy which is self.action_net
        if hasattr(self.action_net, '_last_tree_info') and 'decision_penalty' in self.action_net._last_tree_info:
            return self.action_net._last_tree_info['decision_penalty']
        return torch.tensor(0.0)


class SDTLoggingCallback(BaseCallback):
    """
    Callback for logging SDT-specific metrics.
    """
    def __init__(self, verbose=0, log_freq=100):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0

    def _on_step(self) -> bool:
        # Log tree penalty
        if hasattr(self.model.policy, 'get_tree_penalty'):
            tree_penalty = self.model.policy.get_tree_penalty()
            self.logger.record("sdt/tree_penalty", tree_penalty.item())
        
        # Get rewards and dones from the environment
        if len(self.locals.get("rewards", [])) > 0:
            reward = self.locals["rewards"][0]
            done = self.locals["dones"][0]
            
            self.current_episode_reward += reward
            self.current_episode_length += 1
            
            if done:
                self.episode_rewards.append(self.current_episode_reward)
                self.episode_lengths.append(self.current_episode_length)
                
                if self.verbose > 0:
                    print(f"Episode finished: Reward={self.current_episode_reward:.2f}, Length={self.current_episode_length}")
                
                # Log to tensorboard
                self.logger.record("episode/reward", self.current_episode_reward)
                self.logger.record("episode/length", self.current_episode_length)
                
                if len(self.episode_rewards) >= 10:
                    self.logger.record("episode/mean_reward_10", np.mean(self.episode_rewards[-10:]))
                    self.logger.record("episode/mean_length_10", np.mean(self.episode_lengths[-10:]))
                
                self.current_episode_reward = 0
                self.current_episode_length = 0
        
        return True


def make_env(env_id, seed=0):
    """
    Create and wrap MiniGrid environment.
    """
    def _init():
        env = gym.make(env_id, render_mode="rgb_array")
        env = ImgObsWrapper(env)
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init


def train_sdt(
    env_id="MiniGrid-Empty-5x5-v0",
    total_timesteps=100000,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    n_envs=4,
    n_concepts=4,
    concept_mode=5,
    lambda_1=0.05,
    lambda_2=0.004,
    lambda_3=2.0,
    constraint_lambda=1.0,
    n_continuous_concepts=1,
    tree_depth=5,
    tree_temperature=1.0,
    penalty_coef=0.01,
    seed=0,
    device="cuda",
):
    """
    Train SDT agent on MiniGrid environment.
    SDT learns from concepts extracted by CNN feature extractor.
    
    Args:
        env_id: MiniGrid environment ID
        total_timesteps: Total training timesteps
        learning_rate: Learning rate for optimizer
        n_steps: Number of steps to run for each environment per update
        batch_size: Minibatch size
        n_epochs: Number of epoch when optimizing the surrogate loss
        gamma: Discount factor
        gae_lambda: Factor for trade-off of bias vs variance for GAE
        clip_range: Clipping parameter for PPO
        ent_coef: Entropy coefficient for loss calculation
        vf_coef: Value function coefficient for loss calculation
        max_grad_norm: Max norm for gradient clipping
        n_envs: Number of parallel environments
        n_concepts: Number of concepts to extract from CNN
        concept_mode: Concept extraction mode (1-5)
        lambda_1: Orthogonality regularization weight
        lambda_2: Sparsity regularization weight
        lambda_3: L1 regularization weight
        constraint_lambda: Mode 5 constraint loss weight
        n_continuous_concepts: Mode 5 number of continuous concepts
        tree_depth: Depth of soft decision tree (2^depth leaf nodes)
        tree_temperature: Temperature for soft routing decisions
        penalty_coef: Coefficient for tree decision penalty
        seed: Random seed
        device: Device to use (cpu, cuda, mps)
    """
    
    # Create directories
    save_dir = f"models/{env_id}/sdt"
    os.makedirs(save_dir, exist_ok=True)
    
    # Checkpoint directory for intermediate saves (cleared each training run)
    checkpoint_dir = f"{save_dir}/last_train"
    if os.path.exists(checkpoint_dir):
        import shutil
        shutil.rmtree(checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Get next model number
    model_number = get_next_model_number(save_dir, prefix="sdt_minigrid")
    model_name = f"sdt_minigrid_{model_number:03d}"
    best_model_path = f"{save_dir}/{model_name}"
    
    # Create tensorboard log directory
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tensorboard_log = f"minigrid_tensorboard/{env_id}/sdt"
    run_name = f"{model_name}_{timestamp}"
    os.makedirs(tensorboard_log, exist_ok=True)
    
    print(f"Best model will be saved to: {best_model_path}.zip")
    print(f"Checkpoints will be saved to: {checkpoint_dir}/")
    print(f"TensorBoard logs: {tensorboard_log}/{run_name}")
    
    # Create training environments
    print(f"Creating {n_envs} parallel training environments: {env_id}")
    train_env = DummyVecEnv([make_env(env_id, seed=seed+i) for i in range(n_envs)])
    train_env = VecMonitor(train_env)
    
    # Create evaluation environment
    print("Creating evaluation environment...")
    eval_env = DummyVecEnv([make_env(env_id, seed=seed+1000)])
    eval_env = VecMonitor(eval_env)
    
    # Define policy kwargs with SDT
    # ✅ Use concept distilling like PPO-Concept
    policy_kwargs = dict(
        features_extractor_class=MinigridFeaturesExtractor,
        features_extractor_kwargs=dict(
            features_dim=128,
            concept_distilling=True,  # ✅ Enable concept extraction
            n_concepts=n_concepts,
            concept_mode=concept_mode,
            constraint_lambda=constraint_lambda,
            n_continuous_concepts=n_continuous_concepts
        ),
        n_concepts=n_concepts,  # ✅ Pass to SDTPolicy
        tree_depth=tree_depth,
        tree_temperature=tree_temperature,
        penalty_coef=penalty_coef,
    )
    
    # Create PPO model with SDT policy
    print("Creating SDT model...")
    print(f"Concepts: {n_concepts} (mode {concept_mode})")
    print(f"Tree depth: {tree_depth} (2^{tree_depth} = {2**tree_depth} leaf nodes)")
    print(f"Tree temperature: {tree_temperature}")
    print(f"TensorBoard logs will be saved to: {tensorboard_log}/{run_name}")
    
    model = PPO(
        SDTPolicy,
        train_env,
        policy_kwargs=policy_kwargs,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        use_sde=False,
        sde_sample_freq=-1,
        target_kl=None,
        tensorboard_log=tensorboard_log,
        verbose=1,
        seed=seed,
        device=device,
    )
    
    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=max(10000 // n_envs, 1),
        save_path=checkpoint_dir,
        name_prefix="sdt_checkpoint",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=checkpoint_dir,
        log_path=checkpoint_dir,
        eval_freq=max(5000 // n_envs, 1),
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )
    
    logging_callback = SDTLoggingCallback(verbose=1, log_freq=100)
    
    callback = CallbackList([checkpoint_callback, eval_callback, logging_callback])
    
    # Train the model
    print(f"\n{'='*60}")
    print(f"Training SDT agent on {env_id}")
    print(f"Total timesteps: {total_timesteps}")
    print(f"Number of parallel environments: {n_envs}")
    print(f"Device: {device}")
    print(f"Model: {model_name}")
    print(f"TensorBoard: tensorboard --logdir minigrid_tensorboard/{env_id}/sdt")
    print(f"{'='*60}\n")
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True,
        tb_log_name=run_name,
    )
    
    # Copy best model
    best_model_in_last_train = f"{checkpoint_dir}/best_model.zip"
    if os.path.exists(best_model_in_last_train):
        import shutil
        shutil.copy2(best_model_in_last_train, f"{best_model_path}.zip")
        print(f"\n✓ Best model copied to: {best_model_path}.zip")
    else:
        print(f"\n⚠ Warning: best_model.zip not found in {checkpoint_dir}/")
    
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Best model: {best_model_path}.zip")
    print(f"Training files: {checkpoint_dir}/")
    print(f"Model number: {model_number:03d}")
    print(f"Tree structure: depth={tree_depth}, leaves={2**tree_depth}")
    print(f"{'='*60}\n")
    
    # Cleanup
    train_env.close()
    eval_env.close()
    
    return model


if __name__ == "__main__":
    # Training example
    print("Training SDT on MiniGrid-Empty-5x5-v0...")
    train_sdt(
        env_id="MiniGrid-Empty-5x5-v0",
        total_timesteps=100000,
        learning_rate=3e-4,
        n_steps=2048,
        n_envs=4,
        n_concepts=3,  # Number of concepts to learn
        concept_mode=5,  # Mode 5: FC-bottleneck with STE
        tree_depth=4,  # 2^4 = 16 leaf nodes
        tree_temperature=1.0,
        penalty_coef=0.01,
        seed=42,
    )
