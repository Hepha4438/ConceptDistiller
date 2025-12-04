import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


# ----------------------------
# Concept Layer
# ----------------------------
class ConceptLayer(nn.Module):
    """
    Concept layer producing K concept maps [B, K, H, W] from feature map [B, C, H, W].
    Two branches:
    - concept_map [B,K,H,W] for policy/value
    - concept_vector [B,K] (patch-wise pooled) for loss calculation
    """
    def __init__(self, in_channels, n_concepts=8, l1_lambda=1e-4, patch_pool_size=2, n_bins=10):
        super().__init__()
        self.n_concepts = n_concepts
        self.l1_lambda = l1_lambda
        self.patch_pool_size = patch_pool_size
        self.n_bins = n_bins

        # 1x1 conv to map C -> K concept maps
        self.conv1x1 = nn.Conv2d(in_channels, n_concepts, kernel_size=1)

    def forward(self, x):
        """
        x: [B, C, H, W]
        Returns:
            concept_map: [B,K,H,W] (padded to patch_pool_size multiples)
            concept_vector: [B,K] (pooled for losses)
        """
        B, _, H, W = x.shape
        concept_map = torch.sigmoid(self.conv1x1(x))  # [B,K,H,W]

        # Pad to patch_pool_size multiples (needed for consistent architecture)
        p = self.patch_pool_size
        H_pad = (p - H % p) % p
        W_pad = (p - W % p) % p
        if H_pad > 0 or W_pad > 0:
            concept_map = F.pad(concept_map, (0, W_pad, 0, H_pad))  # pad right,bottom

        # For concept_vector: use global average pooling (simpler and correct)
        # This aggregates all spatial information into [B, K]
        concept_vector = concept_map.mean(dim=[2, 3])  # [B,K]

        return concept_map, concept_vector

    def compute_losses(self, concept_vector):
        """
        Compute 3 losses: L_otho, L_spar, L_l1
        concept_vector: [B, K]
        """
        B, K = concept_vector.shape
        device = concept_vector.device

        # --------------------------
        # 1) L_otho: sum off-diagonal covariance
        # --------------------------
        C_centered = concept_vector - concept_vector.mean(dim=0, keepdim=True)
        cov = (C_centered.T @ C_centered) / (B - 1)  # [K,K]
        L_otho = cov.sum() - torch.diag(cov).sum()   # sum off-diagonal

        # --------------------------
        # 2) L_spar: Hoyer sparsity (DIFFERENTIABLE!)
        # --------------------------
        # Sparsity = (sqrt(n) - ||x||_1/||x||_2) / (sqrt(n) - 1)
        # Loss = 1 - sparsity (minimize to maximize sparsity)
        n = concept_vector.numel()
        l1_norm = torch.norm(concept_vector.flatten(), p=1)
        l2_norm = torch.norm(concept_vector.flatten(), p=2)
        
        eps = 1e-8
        sqrt_n = torch.sqrt(torch.tensor(n, dtype=concept_vector.dtype, device=device))
        sparsity = (sqrt_n - l1_norm / (l2_norm + eps)) / (sqrt_n - 1.0 + eps)
        L_spar = 1.0 - sparsity  # Minimize to maximize sparsity

        # --------------------------
        # 3) L1 penalty on conv1x1 weights
        # --------------------------
        L_l1 = self.l1_lambda * torch.norm(self.conv1x1.weight, p=1)

        return L_otho, L_spar, L_l1


# -------------------------------------------------------------------
# MinigridFeaturesExtractor with ConceptLayer
# -------------------------------------------------------------------
class MinigridFeaturesExtractor(BaseFeaturesExtractor):
    """
    CNN feature extractor for MiniGrid with optional concept distillation.
    """
    def __init__(
            self,
            observation_space: gym.Space,
            features_dim: int = 128,
            normalized_image: bool = False,
            n_concepts: int = 8,
            concept_distilling: bool = False,
            patch_pool_size: int = 2,
            n_bins: int = 10
    ):
        super().__init__(observation_space, features_dim)
        self.concept_distilling = concept_distilling
        self.n_concepts = n_concepts
        self.patch_pool_size = patch_pool_size
        self.n_bins = n_bins

        n_input_channels = observation_space.shape[0]

        # -----------------------------
        # CNN trunk
        # -----------------------------
        self.cnn = nn.Sequential(
            # First conv layer: extract basic features
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # Second conv layer: extract complex features
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # Third conv layer: high-level features
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # Compute CNN output channels, H, W
        with torch.no_grad():
            sample_obs = torch.as_tensor(observation_space.sample()[None]).float()
            cnn_out = self.cnn(sample_obs)
            _, C, H, W = cnn_out.shape
            self.cnn_output_dim = (C, H, W)

        # -----------------------------
        # Concept Layer
        # -----------------------------
        if self.concept_distilling:
            self.concept_layer = ConceptLayer(
                in_channels=C,
                n_concepts=n_concepts,
                patch_pool_size=patch_pool_size,
                n_bins=n_bins
            )
            self.last_concept_losses = None
            
            # ✅ FIX: Compute n_flatten from ACTUAL concept_map output shape
            # (because ConceptLayer may pad the spatial dimensions)
            with torch.no_grad():
                concept_map, _ = self.concept_layer(cnn_out)
                _, K_out, H_out, W_out = concept_map.shape
                n_flatten = K_out * H_out * W_out
        else:
            n_flatten = C * H * W

        # -----------------------------
        # Fully connected layers (keep FC same as old to allow checkpoint load)
        # -----------------------------
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim),
            nn.ReLU()
        )

    # --------------------------------------------------
    # Forward
    # --------------------------------------------------
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = self.cnn(observations)  # [B,C,H,W]

        if self.concept_distilling:
            concept_map, concept_vector = self.concept_layer(x)
            L_otho, L_spar, L_l1 = self.concept_layer.compute_losses(concept_vector)
            self.last_concept_losses = (L_otho, L_spar, L_l1)
            # Policy/value branch: flatten concept_map before FC
            x_fc = concept_map.flatten(start_dim=1)  # [B, K*H*W]
        else:
            x_fc = x.flatten(start_dim=1)  # [B, C*H*W]

        return self.linear(x_fc)
