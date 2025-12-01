import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class MinigridFeaturesExtractor(BaseFeaturesExtractor):
    """
    Custom CNN feature extractor for MiniGrid environments.
    Based on best practices from DI-engine and gym-minigrid repo.
    
    The architecture is designed to handle the partially observable nature
    of MiniGrid environments with a 7x7 agent view.
    """
    def __init__(self, observation_space: gym.Space, features_dim: int = 128, normalized_image: bool = False) -> None:
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        
        # Improved CNN architecture for MiniGrid
        # MiniGrid observations are typically 7x7x3 when using ImgObsWrapper
        self.cnn = nn.Sequential(
            # First conv layer: extract basic features
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Second conv layer: extract complex features
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Third conv layer: high-level features
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            sample_obs = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample_obs).shape[1]

        # Fully connected layers
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Normalize observations to [0, 1] if not already normalized
        # MiniGrid observations are typically in range [0, 255]
        if observations.max() > 1.0:
            observations = observations / 255.0
        return self.linear(self.cnn(observations))
