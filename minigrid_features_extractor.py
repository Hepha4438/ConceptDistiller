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

        NOTE: Requires B >= 2 for covariance computation!
        """
        B, K = concept_vector.shape
        device = concept_vector.device

        # --------------------------
        # 1) L_otho: sum off-diagonal covariance
        # --------------------------
        C_centered = concept_vector - concept_vector.mean(dim=0, keepdim=True)
        cov = (C_centered.T @ C_centered) / (B - 1)  # [K,K] - requires B > 1!
        L_otho = cov.sum() - torch.diag(cov).sum()   # sum off-diagonal

        # --------------------------
        # 2) L_spar: Hoyer sparsity (DIFFERENTIABLE!)
        # --------------------------
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
    
    Concept Modes:
    - Mode 1 (default): concept_map [B,K,H,W] -> flatten -> FC
    - Mode 2: concept_map [B,K,H,W] -> global average pool -> concept_vector [B,K] -> FC
    - Mode 3: concept_map [B,K,H,W] -> global max pool -> concept_vector [B,K] -> FC
    - Mode 4: concept_map [B,K,H,W] -> flatten -> FC1 -> concept bottleneck [B,K] -> FC2
    - Mode 5: Like Mode 4, but only 1st concept uses sigmoid, rest use STE (binary 0/1)
    """
    def __init__(
            self,
            observation_space: gym.Space,
            features_dim: int = 128,
            normalized_image: bool = False,
            n_concepts: int = 8,
            concept_distilling: bool = False,
            concept_mode: int = 1,
            patch_pool_size: int = 2,
            n_bins: int = 10,
            constraint_lambda: float = 1.0  # Weight for Mode 5 constraint loss
    ):
        super().__init__(observation_space, features_dim)
        self.concept_distilling = concept_distilling
        self.concept_mode = concept_mode  # 1: flatten, 2: avg pool, 3: max pool, 4: FC-concept-FC, 5: FC-concept-FC with STE
        self.n_concepts = n_concepts
        self.patch_pool_size = patch_pool_size
        self.n_bins = n_bins
        self.constraint_lambda = constraint_lambda
        
        # Initialize constraint loss storage (for Mode 5)
        self.last_constraint_loss = torch.tensor(0.0)
        
        # Calculate upper limit for Mode 5: max(1, floor(2/3 * (n-1)))
        # This ensures at most ~66% of STE concepts are active per state
        # Good balance: sparse enough but allows flexibility for complex patterns
        if self.concept_mode == 5 and self.n_concepts > 1:
            self.active_limit = max(1, int((2.0 / 3.0) * (self.n_concepts - 1)))
            # Use float for soft constraint (2/3 * (n-1) + 0.5 for smooth gradient)
            self.active_limit_soft = (2.0 / 3.0) * (self.n_concepts - 1) + 0.5
        else:
            self.active_limit = 1
            self.active_limit_soft = 1.0

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
            # All modes use ConceptLayer after CNN
            self.concept_layer = ConceptLayer(
                in_channels=C,
                n_concepts=n_concepts,
                patch_pool_size=patch_pool_size,
                n_bins=n_bins
            )
            self.last_concept_losses = None
            
            # ✅ Compute n_flatten based on concept_mode
            if self.concept_mode in [1, 4, 5]:
                # Mode 1/4/5: Flatten concept_map [B,K,H,W] -> [B,K*H*W]
                with torch.no_grad():
                    concept_map, _ = self.concept_layer(cnn_out)
                    _, K_out, H_out, W_out = concept_map.shape
                    n_flatten = K_out * H_out * W_out
            elif self.concept_mode in [2, 3]:
                # Mode 2/3: Use concept_vector [B,K] directly (avg or max pool)
                n_flatten = n_concepts
            else:
                raise ValueError(f"Invalid concept_mode: {self.concept_mode}. Must be 1, 2, 3, 4, or 5.")
            
            # FC layers based on concept_mode
            if self.concept_mode in [4, 5]:
                # Mode 4/5: Flatten -> FC1 -> concept bottleneck -> FC2
                self.fc1 = nn.Linear(n_flatten, features_dim)
                self.concept_bottleneck = nn.Linear(features_dim, n_concepts)
                self.fc2 = nn.Sequential(
                    nn.Linear(n_concepts, features_dim),
                    nn.ReLU()
                )
            else:
                # Mode 1/2/3: Standard FC layers
                self.linear = nn.Sequential(
                    nn.Linear(n_flatten, features_dim),
                    nn.ReLU(),
                    nn.Linear(features_dim, features_dim),
                    nn.ReLU()
                )
        else:
            n_flatten = C * H * W
            # FC layers for non-concept mode
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
            # All modes: Get concept_map and concept_vector from ConceptLayer
            concept_map, concept_vector = self.concept_layer(x)
            L_otho, L_spar, L_l1 = self.concept_layer.compute_losses(concept_vector)
            self.last_concept_losses = (L_otho, L_spar, L_l1)
            
            # Choose branch based on concept_mode
            if self.concept_mode == 1:
                # Mode 1: Flatten concept_map [B,K,H,W] -> [B,K*H*W] -> FC
                x_fc = concept_map.flatten(start_dim=1)
                return self.linear(x_fc)
                
            elif self.concept_mode == 2:
                # Mode 2: Global average pooling -> concept_vector [B,K] -> FC
                x_fc = concept_vector
                return self.linear(x_fc)
                
            elif self.concept_mode == 3:
                # Mode 3: Global max pooling -> concept_vector [B,K] -> FC
                x_fc = concept_map.flatten(2).max(dim=2)[0]  # [B,K]
                return self.linear(x_fc)
                
            elif self.concept_mode == 4:
                # Mode 4: Flatten -> FC1 -> concept bottleneck [B,K] -> FC2
                x_flat = concept_map.flatten(start_dim=1)  # [B,K*H*W]
                h = F.relu(self.fc1(x_flat))  # [B, features_dim]
                
                # Concept bottleneck with sigmoid activation
                concept_bottleneck_vector = torch.sigmoid(self.concept_bottleneck(h))  # [B,K]
                
                # Store additional concept bottleneck for visualization/analysis
                self.last_concept_bottleneck = concept_bottleneck_vector
                
                # Continue through FC2
                return self.fc2(concept_bottleneck_vector)
                
            elif self.concept_mode == 5:
                # Mode 5: Like Mode 4, but only 1st concept uses sigmoid, rest use STE
                x_flat = concept_map.flatten(start_dim=1)  # [B,K*H*W]
                h = F.relu(self.fc1(x_flat))  # [B, features_dim]
                
                # Concept bottleneck
                concept_raw = self.concept_bottleneck(h)  # [B,K]
                concept_sigmoid = torch.sigmoid(concept_raw)  # [B,K]
                
                # Split: 1st concept = sigmoid (soft), rest = STE (binary)
                concept_first = concept_sigmoid[:, 0:1]  # [B,1] - Keep sigmoid (soft)
                
                if self.n_concepts > 1:
                    concept_rest_sigmoid = concept_sigmoid[:, 1:]  # [B,K-1]
                    
                    # --- Compute Constraint Loss on Soft Probabilities ---
                    # Soft count of active concepts (sum of sigmoid probabilities)
                    active_soft_count = concept_rest_sigmoid.sum(dim=1)  # [B]
                    
                    # Constraint 1: At least 1 active (sum >= 1.0)
                    # Penalty if sum < 1.0
                    loss_min = F.relu(1.0 - active_soft_count).mean()
                    
                    # Constraint 2: At most active_limit_soft active (sum <= limit)
                    # Penalty if sum > limit
                    loss_max = F.relu(active_soft_count - self.active_limit_soft).mean()
                    
                    # Store weighted constraint loss
                    self.last_constraint_loss = self.constraint_lambda * (loss_min + loss_max)
                    # -----------------------------------------------------
                    
                    # Apply STE: hard threshold but gradient flows through sigmoid
                    concept_rest_hard = (concept_rest_sigmoid > 0.5).float()  # [B,K-1] - binary 0/1
                    concept_rest_ste = concept_rest_sigmoid + (concept_rest_hard - concept_rest_sigmoid).detach()
                    
                    # Concatenate: [sigmoid, STE, STE, ...]
                    concept_bottleneck_vector = torch.cat([concept_first, concept_rest_ste], dim=1)  # [B,K]
                else:
                    # Only 1 concept: just use sigmoid, no constraint
                    concept_bottleneck_vector = concept_first
                    self.last_constraint_loss = torch.tensor(0.0, device=concept_first.device)
                
                # Store for visualization/analysis
                self.last_concept_bottleneck = concept_bottleneck_vector
                
                # Continue through FC2
                return self.fc2(concept_bottleneck_vector)
            else:
                raise ValueError(f"Invalid concept_mode: {self.concept_mode}")
        else:
            x_fc = x.flatten(start_dim=1)  # [B, C*H*W]
            return self.linear(x_fc)
