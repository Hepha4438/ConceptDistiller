"""
Post-hoc Concept Extraction from Pretrained PPO Models

Extract interpretable concepts from a trained PPO model using 
encoder-decoder architecture with gated concepts (g ⊙ v).

Architecture:
    Pretrained PPO (frozen):
        Obs → CNN → h [256] → Policy Head → Actions
    
    Concept Model (trainable):
        h → Encoder: g=σ(h), v=ReLU(h), z=g⊙v → Decoder: ĥ → Actions
        
Loss:
    L = λ_rec·||h-ĥ||² + λ_p·KL(π(h)||π(ĥ)) + λ_o·L_ortho + λ_s·L_sparse + λ_b·L_binary
"""

import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
import gymnasium as gym
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from config import get_posthoc_concept_config, ENV_DIFFICULTY


def get_next_model_number(save_dir, prefix="posthoc_minigrid"):
    """Find the next available model number."""
    if not os.path.exists(save_dir):
        return 0
    pattern = re.compile(rf"{re.escape(prefix)}_(\d{{3}})\.zip")
    existing_numbers = [int(m.group(1)) for f in os.listdir(save_dir) if (m:=pattern.match(f))]
    return max(existing_numbers)+1 if existing_numbers else 0


class ConceptEncoderDecoder(nn.Module):
    """
    Encoder-Decoder with gated concepts: z = tanh(g ⊙ v)
    - g: sigmoid gates (first n_continuous continuous, rest pushed to binary via loss)
    - v: ReLU values
    - z: gated concepts
    """
    def __init__(self, h_dim: int = 256, n_concepts: int = 5, n_continuous: int = 1):
        super().__init__()
        self.h_dim = h_dim
        self.n_concepts = n_concepts
        self.n_continuous = n_continuous
        self.n_binary = n_concepts - n_continuous
        
        # Encoder: h → g, v
        self.encoder_g = nn.Linear(h_dim, n_concepts)  # Gates
        self.encoder_v = nn.Linear(h_dim, n_concepts)  # Values
        
        # Decoder: z → ĥ
        self.decoder = nn.Sequential(
            nn.Linear(n_concepts, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim)
        )
        
    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            h: hidden features [B, h_dim]
        Returns:
            z: gated concepts [B, n_concepts]
            h_recon: reconstructed h [B, h_dim]
            g: gates for loss computation [B, n_concepts]
        """
        # Encode
        g = torch.sigmoid(self.encoder_g(h))  # [B, K] gates
        v = F.relu(self.encoder_v(h))          # [B, K] values
        
        # Gate: z = g ⊙ v
        z = g * v  # Element-wise multiplication
        
        # Decode
        h_recon = self.decoder(z)
        
        return z, h_recon, g
    
    def extract_concepts(self, h: torch.Tensor) -> torch.Tensor:
        """Extract only concepts without reconstruction."""
        g = torch.sigmoid(self.encoder_g(h))
        v = F.relu(self.encoder_v(h))
        z = g * v
        return z


class PostHocConceptFeaturesExtractor(BaseFeaturesExtractor):
    """
    Features extractor that wraps pretrained CNN + first MLP layer + concept encoder-decoder.
    This allows IG to trace through the concept bottleneck.
    
    Architecture:
        obs → pretrained_cnn[256] → pretrained_first_layer (Linear+Tanh) → h[256] 
            → concept_encoder → z[K] → decoder → h'[256]
    
    Returns h' (reconstructed h after decoder) which goes to pretrained second_layer.
    """
    def __init__(
        self,
        observation_space: gym.Space,
        pretrained_cnn: nn.Module,
        pretrained_first_layer: nn.Module,
        concept_encoder: ConceptEncoderDecoder,
        features_dim: int = 256  # Always h_dim, not n_concepts
    ):
        # features_dim = h_dim (256) for h_recon
        super().__init__(observation_space, features_dim)
        
        self.pretrained_cnn = pretrained_cnn
        self.pretrained_first_layer = pretrained_first_layer  # Only first layer: Linear(256,256)+Tanh
        self.concept_encoder = concept_encoder
        self.last_concepts = None  # Store for visualization
        
        # Freeze pretrained CNN and first MLP layer
        for param in self.pretrained_cnn.parameters():
            param.requires_grad = False
        for param in self.pretrained_first_layer.parameters():
            param.requires_grad = False
        self.pretrained_cnn.eval()
        self.pretrained_first_layer.eval()
        
        # Set concept encoder to eval mode (no training)
        self.concept_encoder.eval()
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: obs → CNN → first_layer → h → encoder → z → decoder → h'
        Returns h_recon [B, 256] for pretrained second_layer → action_net
        """
        # Extract features from pretrained CNN
        # NOTE: Remove torch.no_grad() to allow gradient flow for IG visualization
        # Pretrained layers have requires_grad=False so weights won't update
        cnn_features = self.pretrained_cnn(observations)  # [B, 256]
        # Extract h from first layer only
        h = self.pretrained_first_layer(cnn_features)  # [B, 256]
        
        # Extract concepts and reconstruct h
        z = self.concept_encoder.extract_concepts(h)  # [B, K]
        h_recon = self.concept_encoder.decoder(z)     # [B, 256]
        
        # Store concepts for visualization
        # NOTE: Don't detach() to allow gradient flow for IG
        self.last_concepts = z
        # DEBUG: Print to verify forward is called
        # print(f"[DEBUG] PostHocConceptFeaturesExtractor.forward() called, concepts: {z[0, :3]}")
        
        return h_recon  # Return h', not z


class PostHocConceptModel(nn.Module):
    """
    Full post-hoc concept model combining pretrained PPO + concept encoder-decoder.
    """
    def __init__(
        self,
        pretrained_model: PPO,
        n_concepts: int = 5,
        n_continuous: int = 1,
        h_dim: int = 256
    ):
        super().__init__()
        
        # Freeze pretrained PPO
        self.pretrained_policy = pretrained_model.policy
        for param in self.pretrained_policy.parameters():
            param.requires_grad = False
        self.pretrained_policy.eval()
        
        # Trainable concept encoder-decoder
        self.concept_model = ConceptEncoderDecoder(
            h_dim=h_dim,
            n_concepts=n_concepts,
            n_continuous=n_continuous
        )
        
        self.n_concepts = n_concepts
        self.n_continuous = n_continuous
        self.n_binary = n_concepts - n_continuous
        
    def extract_h(self, obs: torch.Tensor) -> torch.Tensor:
        """Extract hidden representation h from pretrained model."""
        with torch.no_grad():
            # Forward through features extractor
            features = self.pretrained_policy.extract_features(obs)
            
            # Get h from FIRST LAYER ONLY of mlp_extractor.policy_net
            # policy_net is Sequential: [Linear(256,256), Tanh, Linear(256,256), Tanh]
            # We only want output after first Linear + Tanh: policy_net[0:2]
            h = self.pretrained_policy.mlp_extractor.policy_net[0](features)  # Linear
            h = self.pretrained_policy.mlp_extractor.policy_net[1](h)         # Tanh
            
            return h  # h after first layer only [B, 256]
    
    def forward(self, obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Full forward pass for training.
        Returns dict with all necessary outputs for loss computation.
        
        Pipeline: h → encoder → z → decoder → h' → second_layer → actions
        Loss: KL(policy(h) || policy(h'))
        """
        # Extract h from pretrained model (frozen)
        h = self.extract_h(obs)
        
        # Get original actions from h through second_layer
        with torch.no_grad():
            # h → second_layer (policy_net[2:4]) → action_net
            h_latent = self.pretrained_policy.mlp_extractor.policy_net[2](h)  # Linear
            h_latent = self.pretrained_policy.mlp_extractor.policy_net[3](h_latent)  # Tanh
            action_logits_original = self.pretrained_policy.action_net(h_latent)
        
        # Pass through concept model: h → z → h_recon
        z, h_recon, g = self.concept_model(h)
        
        # Get actions from h_recon through second_layer
        # NOTE: NO torch.no_grad() here! Gradient must flow through h_recon for KL loss training
        # Second layer and action_net are already frozen, so their weights won't update
        # h' → second_layer (policy_net[2:4]) → action_net
        h_recon_latent = self.pretrained_policy.mlp_extractor.policy_net[2](h_recon)  # Linear
        h_recon_latent = self.pretrained_policy.mlp_extractor.policy_net[3](h_recon_latent)  # Tanh
        action_logits_recon = self.pretrained_policy.action_net(h_recon_latent)
        
        return {
            'h': h,
            'h_recon': h_recon,
            'z': z,
            'g': g,
            'action_logits_original': action_logits_original,
            'action_logits_recon': action_logits_recon
        }


def compute_losses(
    outputs: Dict[str, torch.Tensor],
    n_binary: int,
    lambda_rec: float = 1.0,
    lambda_p: float = 0.5,
    lambda_o: float = 0.05,
    lambda_s: float = 0.004,
    lambda_b: float = 0.1
) -> Dict[str, torch.Tensor]:
    """
    Compute all loss components.
    
    Loss = λ_rec·L_rec + λ_p·L_policy + λ_o·L_ortho + λ_s·L_sparse + λ_b·L_binary
    """
    h = outputs['h']
    h_recon = outputs['h_recon']
    z = outputs['z']
    g = outputs['g']
    action_logits_original = outputs['action_logits_original']
    action_logits_recon = outputs['action_logits_recon']
    
    # 1. Reconstruction loss
    L_rec = F.mse_loss(h_recon, h)
    
    # 2. Policy matching loss (KL divergence)
    # KL(π_original || π_recon)
    log_probs_original = F.log_softmax(action_logits_original, dim=-1)
    log_probs_recon = F.log_softmax(action_logits_recon, dim=-1)
    probs_original = F.softmax(action_logits_original, dim=-1)
    L_policy = (probs_original * (log_probs_original - log_probs_recon)).sum(dim=-1).mean()
    
    # Action agreement metric (for debugging)
    pred_actions_original = action_logits_original.argmax(dim=-1)
    pred_actions_recon = action_logits_recon.argmax(dim=-1)
    action_agreement = (pred_actions_original == pred_actions_recon).float().mean()
    
    # 3. Orthogonality loss (concepts should be decorrelated)
    B, K = z.shape
    if B > 1:
        z_centered = z - z.mean(dim=0, keepdim=True)
        cov = (z_centered.T @ z_centered) / (B - 1)
        L_ortho = (cov.sum() - torch.diag(cov).sum()).abs()
    else:
        L_ortho = torch.tensor(0.0, device=z.device)
    
    # 4. Sparsity loss (Hoyer sparsity on concepts)
    n = z.numel()
    l1_norm = torch.norm(z.flatten(), p=1)
    l2_norm = torch.norm(z.flatten(), p=2)
    eps = 1e-8
    sqrt_n = torch.sqrt(torch.tensor(n, dtype=z.dtype, device=z.device))
    sparsity = (sqrt_n - l1_norm / (l2_norm + eps)) / (sqrt_n - 1.0 + eps)
    L_sparse = 1.0 - sparsity  # Minimize to maximize sparsity
    
    # 5. Binary enforcement loss (push binary concepts to 0 or 1)
    # Apply only to last n_binary concepts
    if n_binary > 0:
        g_binary = g[:, -n_binary:]  # Last n_binary gates
        L_binary = torch.mean((g_binary ** 2) * ((1 - g_binary) ** 2))  # Minimize g^2(1-g)^2
    else:
        L_binary = torch.tensor(0.0, device=g.device)
    
    # Total loss
    loss = (
        lambda_rec * L_rec +
        lambda_p * L_policy +
        lambda_o * L_ortho +
        lambda_s * L_sparse +
        lambda_b * L_binary
    )
    
    return {
        'loss': loss,
        'L_rec': L_rec,
        'L_policy': L_policy,
        'L_ortho': L_ortho,
        'L_sparse': L_sparse,
        'L_binary': L_binary,
        'action_agreement': action_agreement  # NEW: % actions match
    }


def collect_dataset(
    pretrained_model: PPO,
    env,
    n_samples: int,
    device: str = "cuda"
) -> torch.Tensor:
    """
    Collect dataset of observations from pretrained model.
    """
    print(f"Collecting dataset: {n_samples} samples...")
    
    observations = []
    obs = env.reset()
    
    # Handle VecEnv reset return (could be tuple or just obs)
    if isinstance(obs, tuple):
        obs = obs[0]  # Extract obs from (obs, info)
    
    steps = 0
    while steps < n_samples:
        # VecEnv returns observations with shape [n_envs, H, W, C]
        # Extract single environment observation: [H, W, C]
        if obs.ndim == 4:  # [n_envs, H, W, C]
            current_obs = obs[0]
        else:  # Already [H, W, C]
            current_obs = obs
        
        observations.append(current_obs)
        
        # Step environment with pretrained model
        action, _ = pretrained_model.predict(obs, deterministic=False)
        obs, reward, done, info = env.step(action)
        
        steps += 1
        
        # VecEnv returns done as array [n_envs], check if any episode is done
        done_flag = done[0] if isinstance(done, (list, np.ndarray)) else done
        if done_flag:
            obs = env.reset()
            # Handle VecEnv reset return
            if isinstance(obs, tuple):
                obs = obs[0]
            
        if steps % 10000 == 0:
            print(f"  Collected {steps}/{n_samples} samples...")
    
    # Convert to tensor and transpose to PyTorch format
    observations = np.array(observations)  # [N, H, W, C]
    observations = np.transpose(observations, (0, 3, 1, 2))  # [N, C, H, W]
    observations = torch.from_numpy(observations).float().to(device)
    
    print(f"✓ Dataset collected: {observations.shape} (format: [N, C, H, W])")
    return observations


def collect_posthoc_dataset(
    pretrained_model_path: str,
    env_id: str,
    n_samples: int = 500000,
    device: str = "cuda",
    seed: int = 42,
    save_dir: str = "posthoc_datasets"
) -> str:
    """
    Collect dataset from pretrained PPO model and save to disk.
    
    Returns:
        Path to saved dataset file
    """
    import os
    
    print(f"\n{'='*60}")
    print(f"Collecting Post-hoc Dataset")
    print(f"{'='*60}")
    print(f"Environment: {env_id}")
    print(f"Pretrained model: {pretrained_model_path}")
    print(f"Samples: {n_samples}")
    print(f"{'='*60}\n")
    
    # Load pretrained PPO model
    print("Loading pretrained PPO model...")
    pretrained_model = PPO.load(pretrained_model_path, device=device)
    print("✓ Pretrained model loaded")
    
    # Create environment for data collection
    def _make_env():
        def _init():
            env = gym.make(env_id, render_mode="rgb_array")
            env = ImgObsWrapper(env)  # Converts Dict obs to Box(3,7,7) RGB
            # Don't use Monitor here - VecMonitor will wrap it later
            env.reset(seed=seed)  # ← IMPORTANT: DummyVecEnv needs this to detect obs space
            return env
        return _init
    
    env = DummyVecEnv([_make_env()])
    env = VecMonitor(env)
    # Note: seed is set during collection via env.reset(seed=...) if needed
    
    # Collect dataset
    dataset = collect_dataset(
        pretrained_model,
        env,
        n_samples=n_samples,
        device=device
    )
    
    env.close()
    
    # Save dataset
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate filename: {env_id}_{model_name}_{n_samples}.pt
    model_name = os.path.basename(pretrained_model_path).replace('.zip', '')
    dataset_filename = f"{env_id}_{model_name}_{n_samples}.pt"
    dataset_path = os.path.join(save_dir, dataset_filename)
    
    print(f"\nSaving dataset to: {dataset_path}")
    torch.save({
        'observations': dataset.cpu(),  # Move to CPU for storage
        'env_id': env_id,
        'pretrained_model_path': pretrained_model_path,
        'n_samples': n_samples,
        'seed': seed
    }, dataset_path)
    
    print(f"✓ Dataset saved ({dataset.shape[0]} samples)")
    print(f"\n{'='*60}")
    print(f"Dataset collection completed!")
    print(f"File: {dataset_path}")
    print(f"{'='*60}\n")
    
    return dataset_path


def train_posthoc_concepts(
    pretrained_model_path: str,
    env_id: str,
    n_concepts: int = 5,
    n_continuous: int = 1,
    dataset_path: str = None,  # ← NEW: Path to pre-collected dataset
    collection_timesteps: int = 500000,  # Only used if dataset_path is None
    training_epochs: int = 100,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    lambda_rec: float = 1.0,
    lambda_p: float = 0.5,
    lambda_o: float = 0.05,
    lambda_s: float = 0.004,
    lambda_b: float = 0.1,
    device: str = "cuda",
    seed: int = 42
):
    """
    Train post-hoc concept extraction model from pretrained PPO.
    
    Args:
        dataset_path: Path to pre-collected dataset. If None, will collect new dataset.
        collection_timesteps: Only used if dataset_path is None.
    """
    
    print(f"\n{'='*60}")
    print(f"Post-hoc Concept Extraction")
    print(f"{'='*60}")
    print(f"Environment: {env_id}")
    print(f"Pretrained model: {pretrained_model_path}")
    print(f"Concepts: {n_concepts} total ({n_continuous} continuous, {n_concepts-n_continuous} binary)")
    print(f"Training epochs: {training_epochs}")
    print(f"{'='*60}\n")
    
    # Load pretrained PPO model
    print("Loading pretrained PPO model...")
    pretrained_model = PPO.load(pretrained_model_path, device=device)
    print("✓ Pretrained model loaded")
    
    # Create environment for model creation (not for collection)
    def _make_env():
        def _init():
            env = gym.make(env_id, render_mode="rgb_array")
            env = ImgObsWrapper(env)  # Converts Dict obs to Box(3,7,7) RGB
            env = Monitor(env)
            env.reset(seed=seed)  # ← IMPORTANT: DummyVecEnv needs this to detect obs space
            return env
        return _init
    
    # Load or collect dataset
    if dataset_path is not None and os.path.exists(dataset_path):
        # Load pre-collected dataset
        print(f"\nLoading dataset from: {dataset_path}")
        dataset_data = torch.load(dataset_path, map_location='cpu')
        dataset = dataset_data['observations'].to(device)
        print(f"✓ Dataset loaded: {dataset.shape} (format: [N, C, H, W])")
        print(f"  From: {dataset_data.get('pretrained_model_path', 'unknown')}")
        print(f"  Samples: {dataset_data.get('n_samples', len(dataset))}")
    else:
        # Collect new dataset
        if dataset_path is not None:
            print(f"⚠ Dataset file not found: {dataset_path}")
        print(f"\nCollecting new dataset ({collection_timesteps} samples)...")
        
        env = DummyVecEnv([_make_env()])
        env = VecMonitor(env)
        # Note: seed is set during collection via env.reset(seed=...) if needed
        
        dataset = collect_dataset(
            pretrained_model,
            env,
            n_samples=collection_timesteps,
            device=device
        )
        
        env.close()
    
    # Create post-hoc concept model
    print("\nCreating concept model...")
    model = PostHocConceptModel(
        pretrained_model=pretrained_model,
        n_concepts=n_concepts,
        n_continuous=n_continuous,
        h_dim=256  # SB3 default MLP hidden size
    ).to(device)
    print(f"✓ Concept model created")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.concept_model.parameters(), lr=learning_rate)
    
    # Training loop
    print(f"\nTraining for {training_epochs} epochs...")
    n_samples = len(dataset)
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    best_loss = float('inf')
    
    for epoch in range(training_epochs):
        epoch_losses = []
        
        # Shuffle dataset
        indices = torch.randperm(n_samples)
        dataset_shuffled = dataset[indices]
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_samples)
            batch_obs = dataset_shuffled[start_idx:end_idx]
            
            # Forward pass
            outputs = model(batch_obs)
            
            # Compute losses
            losses = compute_losses(
                outputs,
                n_binary=model.n_binary,
                lambda_rec=lambda_rec,
                lambda_p=lambda_p,
                lambda_o=lambda_o,
                lambda_s=lambda_s,
                lambda_b=lambda_b
            )
            
            # Backward pass
            optimizer.zero_grad()
            losses['loss'].backward()
            optimizer.step()
            
            epoch_losses.append(losses['loss'].item())
        
        # Epoch stats
        mean_loss = np.mean(epoch_losses)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{training_epochs}: Loss = {mean_loss:.6f}")
            
            # Print detailed losses
            with torch.no_grad():
                sample_batch = dataset[:batch_size]
                outputs = model(sample_batch)
                losses = compute_losses(
                    outputs,
                    n_binary=model.n_binary,
                    lambda_rec=lambda_rec,
                    lambda_p=lambda_p,
                    lambda_o=lambda_o,
                    lambda_s=lambda_s,
                    lambda_b=lambda_b
                )
                print(f"  L_rec={losses['L_rec'].item():.4f}, "
                      f"L_policy={losses['L_policy'].item():.4f}, "
                      f"L_ortho={losses['L_ortho'].item():.4f}, "
                      f"L_sparse={losses['L_sparse'].item():.4f}, "
                      f"L_binary={losses['L_binary'].item():.4f}")
                print(f"  Action Agreement: {losses['action_agreement'].item()*100:.1f}%")
        
        # Save best model
        if mean_loss < best_loss:
            best_loss = mean_loss
    
    # =========================================================================
    # Create new PPO model with integrated concept encoder
    # =========================================================================
    print("\nCreating PPO model with integrated concepts...")
    
    # Extract pretrained components
    pretrained_cnn = pretrained_model.policy.features_extractor
    pretrained_mlp_policy_net = pretrained_model.policy.mlp_extractor.policy_net
    pretrained_mlp_value_net = pretrained_model.policy.mlp_extractor.value_net
    
    # Extract first layer (Linear + Tanh) from policy_net
    # policy_net is Sequential: [Linear(256,256), Tanh, Linear(256,256), Tanh]
    # First layer: policy_net[0:2]
    pretrained_first_layer = nn.Sequential(
        pretrained_mlp_policy_net[0],  # Linear(256, 256)
        pretrained_mlp_policy_net[1]   # Tanh
    )
    
    # Extract second layer (Linear + Tanh) from policy_net and value_net
    # Second layer: policy_net[2:4]
    pretrained_second_layer_pi = nn.Sequential(
        pretrained_mlp_policy_net[2],  # Linear(256, 256)
        pretrained_mlp_policy_net[3]   # Tanh
    )
    pretrained_second_layer_vf = nn.Sequential(
        pretrained_mlp_value_net[2],   # Linear(256, 256)
        pretrained_mlp_value_net[3]    # Tanh
    )
    
    # Extract action and value nets
    pretrained_action_net = pretrained_model.policy.action_net
    pretrained_value_net = pretrained_model.policy.value_net
    
    # Create dummy environment for model initialization
    dummy_env = DummyVecEnv([_make_env()])
    
    # Get policy kwargs from pretrained model to ensure CNN architecture compatibility
    policy_kwargs = {
        'features_extractor_class': type(pretrained_cnn),
        'features_extractor_kwargs': {},
        'net_arch': dict(pi=[256], vf=[256])  # Single layer (second_layer already Linear+Tanh)
    }
    
    # Create new PPO model with compatible CNN policy
    posthoc_ppo = PPO(
        "CnnPolicy",
        dummy_env,
        policy_kwargs=policy_kwargs,
        verbose=0,
        device=device
    )
    
    # Replace features extractor with PostHocConceptFeaturesExtractor
    # This returns h_recon [256] after decoder
    features_extractor = PostHocConceptFeaturesExtractor(
        observation_space=dummy_env.observation_space,
        pretrained_cnn=pretrained_cnn,
        pretrained_first_layer=pretrained_first_layer,
        concept_encoder=model.concept_model,
        features_dim=256  # h_dim, not n_concepts
    ).to(device)
    
    # Set ALL THREE extractors (features, pi, vf) to same instance
    # CnnPolicy has separate extractors for policy and value networks
    posthoc_ppo.policy.features_extractor = features_extractor
    if hasattr(posthoc_ppo.policy, 'pi_features_extractor'):
        posthoc_ppo.policy.pi_features_extractor = features_extractor
    if hasattr(posthoc_ppo.policy, 'vf_features_extractor'):
        posthoc_ppo.policy.vf_features_extractor = features_extractor
    
    # Use pretrained second layers as MLP extractor
    from stable_baselines3.common.torch_layers import MlpExtractor
    
    # Create MLP extractor structure with feature_dim=256 (h_dim)
    posthoc_ppo.policy.mlp_extractor = MlpExtractor(
        feature_dim=256,
        net_arch=dict(pi=[256], vf=[256]),
        activation_fn=nn.Identity,  # No activation needed (already in second_layer)
        device=device
    )
    
    # Replace with pretrained second layers (frozen)
    posthoc_ppo.policy.mlp_extractor.policy_net = pretrained_second_layer_pi
    posthoc_ppo.policy.mlp_extractor.value_net = pretrained_second_layer_vf
    posthoc_ppo.policy.mlp_extractor.latent_dim_pi = 256
    posthoc_ppo.policy.mlp_extractor.latent_dim_vf = 256
    
    # Use pretrained action and value nets (frozen)
    posthoc_ppo.policy.action_net = pretrained_action_net
    posthoc_ppo.policy.value_net = pretrained_value_net
    
    # Freeze all pretrained components
    for param in posthoc_ppo.policy.parameters():
        param.requires_grad = False
    
    # CRITICAL: Set to eval mode for BatchNorm layers
    posthoc_ppo.policy.eval()
    
    print("✓ PPO model with concepts created")
    
    # Save model to separate posthoc_concept folder
    save_dir = f"models/{env_id}/posthoc_concept"
    os.makedirs(save_dir, exist_ok=True)
    
    model_number = get_next_model_number(save_dir)
    model_name = f"posthoc_minigrid_{model_number:03d}"
    save_path = f"{save_dir}/{model_name}"
    
    # Save the new PPO model with integrated concepts
    posthoc_ppo.save(f"{save_path}.zip")
    
    # Also save concept info separately for reference
    torch.save({
        'concept_encoder_decoder': model.concept_model.state_dict(),
        'n_concepts': n_concepts,
        'n_continuous': n_continuous,
        'is_posthoc': True,  # Flag to identify posthoc models
        'config': {
            'env_id': env_id,
            'pretrained_model_path': pretrained_model_path,
            'lambda_rec': lambda_rec,
            'lambda_p': lambda_p,
            'lambda_o': lambda_o,
            'lambda_s': lambda_s,
            'lambda_b': lambda_b,
        }
    }, f"{save_path}_concepts.pt")
    
    dummy_env.close()
    
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Model saved to: {save_path}.zip")
    print(f"Concepts saved to: {save_path}_concepts.pt")
    print(f"Best loss: {best_loss:.6f}")
    print(f"{'='*60}\n")
    
    return posthoc_ppo, save_path


def load_posthoc_model(model_path: str, env):
    """
    Load a posthoc concept model from .zip file.
    
    Args:
        model_path: Path to .zip model file
        env: Environment instance
    
    Returns:
        Loaded PPO model with PostHocConceptFeaturesExtractor
    """
    import zipfile
    import tempfile
    import pickle
    
    # Load concept encoder info from _concepts.pt
    model_name_without_ext = os.path.splitext(model_path)[0]
    concepts_file = f"{model_name_without_ext}_concepts.pt"
    
    if not os.path.exists(concepts_file):
        raise FileNotFoundError(f"Concepts file not found: {concepts_file}")
    
    concepts_data = torch.load(concepts_file, map_location='cpu')
    n_concepts = concepts_data['n_concepts']
    n_continuous = concepts_data['n_continuous']
    
    # Extract and load model data from zip
    with zipfile.ZipFile(model_path, 'r') as archive:
        # Load pytorch state dict
        with tempfile.TemporaryDirectory() as tmpdir:
            archive.extract('pytorch_variables.pth', tmpdir)
            state_dict = torch.load(os.path.join(tmpdir, 'pytorch_variables.pth'), 
                                   map_location='cpu')
    
    # Check if state_dict has 'policy' key or is already the policy state dict
    if 'policy' in state_dict:
        policy_state_dict = state_dict['policy']
    else:
        # state_dict IS the policy state dict
        policy_state_dict = state_dict
    
    # Create concept encoder and load weights
    concept_encoder = ConceptEncoderDecoder(h_dim=256, n_concepts=n_concepts,
                                           n_continuous=n_continuous)
    concept_encoder.load_state_dict(concepts_data['concept_encoder_decoder'])
    
    # Load pretrained PPO model to get the EXACT CNN architecture
    # Cannot create MinigridFeaturesExtractor from scratch - need features_dim from original model
    pretrained_model_path = concepts_data['config']['pretrained_model_path']
    
    # Try alternate paths if original absolute path (e.g. from another machine) doesn't exist
    if not os.path.exists(pretrained_model_path) and not os.path.exists(f"{pretrained_model_path}.zip"):
        env_id = concepts_data['config'].get('env_id')
        if not env_id and hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'spec') and env.unwrapped.spec:
            env_id = env.unwrapped.spec.id
            
        basename = os.path.basename(pretrained_model_path)
        if not basename.endswith('.zip'):
            basename_zip = f"{basename}.zip"
        else:
            basename_zip = basename
            
        alt_paths = [
            basename_zip,
            os.path.join("models", env_id, "ppo", basename_zip) if env_id else None
        ]
        
        for p in alt_paths:
            if p and (os.path.exists(p) or os.path.exists(p.replace('.zip', ''))):
                print(f"Original path {pretrained_model_path} not found.")
                print(f"Using alternative path: {p}")
                pretrained_model_path = p
                break
                
    print(f"Loading original pretrained model from: {pretrained_model_path}")
    pretrained_model = PPO.load(pretrained_model_path, device='cpu')
    
    # Extract pretrained CNN with correct features_dim
    pretrained_cnn = pretrained_model.policy.features_extractor
    print(f"✓ Pretrained CNN loaded (features_dim={pretrained_cnn.features_dim})")
    
    # Extract pretrained first MLP layer from policy_net
    # MLP structure: [Linear(features_dim, 256), Tanh, Linear(256, 256), Tanh]
    # First layer: policy_net[0:2]
    pretrained_mlp_policy_net = pretrained_model.policy.mlp_extractor.policy_net
    pretrained_first_layer = nn.Sequential(
        pretrained_mlp_policy_net[0],  # Linear(features_dim, 256)
        pretrained_mlp_policy_net[1]   # Tanh
    )
    print(f"✓ Pretrained first layer loaded: {pretrained_mlp_policy_net[0]}")
    
    # Extract pretrained second layer from policy_net and value_net
    # Second layer: policy_net[2:4]
    pretrained_second_layer_pi = nn.Sequential(
        pretrained_mlp_policy_net[2],  # Linear(256, 256)
        pretrained_mlp_policy_net[3]   # Tanh
    )
    pretrained_mlp_value_net = pretrained_model.policy.mlp_extractor.value_net
    pretrained_second_layer_vf = nn.Sequential(
        pretrained_mlp_value_net[2],   # Linear(256, 256)
        pretrained_mlp_value_net[3]    # Tanh
    )
    
    # Extract action and value nets
    pretrained_action_net = pretrained_model.policy.action_net
    pretrained_value_net = pretrained_model.policy.value_net
    
    print(f"✓ All pretrained components extracted from original model")
    
    # Create PostHocConceptFeaturesExtractor
    features_extractor = PostHocConceptFeaturesExtractor(
        observation_space=env.observation_space,
        pretrained_cnn=pretrained_cnn,
        pretrained_first_layer=pretrained_first_layer,
        concept_encoder=concept_encoder,
        features_dim=256  # h_dim, not n_concepts
    )
    
    # Create new PPO model with correct architecture
    from stable_baselines3.common.torch_layers import MlpExtractor
    from minigrid_features_extractor import MinigridFeaturesExtractor
    
    # Create PPO with MinigridFeaturesExtractor to avoid size mismatch
    # We'll replace it immediately anyway
    model = PPO(
        policy="CnnPolicy",
        env=env,
        policy_kwargs={
            'features_extractor_class': MinigridFeaturesExtractor,
            'features_extractor_kwargs': {'features_dim': 256},
            'net_arch': dict(pi=[256], vf=[256])
        },
        verbose=0
    )
    
    # Now manually replace features extractor with PostHocConceptFeaturesExtractor
    # CnnPolicy has separate pi_features_extractor and vf_features_extractor
    model.policy.features_extractor = features_extractor
    
    # Also set pi_features_extractor and vf_features_extractor to same instance
    # (they need to share concepts and weights)
    if hasattr(model.policy, 'pi_features_extractor'):
        model.policy.pi_features_extractor = features_extractor
    if hasattr(model.policy, 'vf_features_extractor'):
        model.policy.vf_features_extractor = features_extractor
    
    # Rebuild MLP extractor with correct input dim (256 = h_dim)
    model.policy.mlp_extractor = MlpExtractor(
        feature_dim=256,
        net_arch=dict(pi=[256], vf=[256]),
        activation_fn=nn.Identity  # No activation needed
    )
    
    # Replace with pretrained second layers
    model.policy.mlp_extractor.policy_net = pretrained_second_layer_pi
    model.policy.mlp_extractor.value_net = pretrained_second_layer_vf
    model.policy.mlp_extractor.latent_dim_pi = 256
    model.policy.mlp_extractor.latent_dim_vf = 256
    
    # Use pretrained action and value nets
    model.policy.action_net = pretrained_action_net
    model.policy.value_net = pretrained_value_net
    
    # CRITICAL: Set entire model to eval mode for inference
    model.policy.eval()
    
    return model


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Post-hoc Concept Extraction")
    parser.add_argument("--mode", type=str, choices=["collect", "train", "all"], default="all",
                       help="Mode: 'collect' (only collect dataset), 'train' (only train), 'all' (collect + train)")
    parser.add_argument("--pretrained-model", type=str, required=True,
                       help="Path to pretrained PPO model (.zip)")
    parser.add_argument("--env-id", type=str, default="MiniGrid-Empty-5x5-v0")
    parser.add_argument("--dataset-path", type=str, default=None,
                       help="Path to pre-collected dataset (only for 'train' mode)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    # Get config based on environment
    config = get_posthoc_concept_config(args.env_id)
    
    if args.mode == "collect":
        # Only collect dataset
        print("\n[MODE: Collect Dataset Only]\n")
        dataset_path = collect_posthoc_dataset(
            pretrained_model_path=args.pretrained_model,
            env_id=args.env_id,
            n_samples=config.get('collection_timesteps', 500000),
            device=args.device,
            seed=args.seed
        )
        print(f"\n✓ Dataset saved: {dataset_path}")
        print(f"  Use this path with --mode train --dataset-path {dataset_path}")
        
    elif args.mode == "train":
        # Only train (requires dataset_path)
        print("\n[MODE: Train Only]\n")
        if args.dataset_path is None:
            print("ERROR: --dataset-path is required for 'train' mode")
            print("  Run with --mode collect first to generate dataset")
            return
        
        train_posthoc_concepts(
            pretrained_model_path=args.pretrained_model,
            env_id=args.env_id,
            dataset_path=args.dataset_path,  # Use pre-collected dataset
            device=args.device,
            seed=args.seed,
            **config
        )
        
    else:  # args.mode == "all"
        # Collect + Train (original behavior)
        print("\n[MODE: Collect + Train]\n")
        train_posthoc_concepts(
            pretrained_model_path=args.pretrained_model,
            env_id=args.env_id,
            dataset_path=None,  # Will collect new dataset
            device=args.device,
            seed=args.seed,
            **config
        )


if __name__ == "__main__":
    main()
