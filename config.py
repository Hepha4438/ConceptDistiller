"""
Hyperparameter configurations for different MiniGrid environments.
Based on research standards from:
- Mnih et al. (2015): "Human-level control through deep reinforcement learning" (Nature DQN)
- Hessel et al. (2018): "Rainbow: Combining Improvements in Deep Reinforcement Learning"
- Schulman et al. (2017): "Proximal Policy Optimization Algorithms"
- Engstrom et al. (2020): OpenAI Baselines PPO implementation
"""

# DQN Hyperparameters - Based on Nature DQN and Rainbow DQN standards
DQN_CONFIGS = {
    # Easy environments (5x5 grids)
    "easy": {
        "total_timesteps": 100000,
        "learning_rate": 1e-4,  # Mnih et al. (2015): 1e-4 for Adam
        "buffer_size": 100000,  # Replay buffer size
        "learning_starts": 1000,  # Start learning after initial exploration
        "batch_size": 32,  # Nature DQN standard batch size
        "gamma": 0.99,  # Standard discount factor
        "target_update_interval": 1000,  # Hard update frequency (Nature DQN)
        "exploration_fraction": 0.1,  # 10% of training for epsilon decay
        "exploration_final_eps": 0.01,  # Mnih et al. (2015): final epsilon = 0.01
    },
    
    # Medium environments (8x8 to 16x16 grids)
    "medium": {
        "total_timesteps": 300000,
        "learning_rate": 1e-4,
        "buffer_size": 200000,
        "learning_starts": 2000,
        "batch_size": 32,
        "gamma": 0.99,
        "target_update_interval": 1000,
        "exploration_fraction": 0.2,  # More exploration for complex tasks
        "exploration_final_eps": 0.01,
    },
    
    # Hard environments (complex tasks, large grids)
    "hard": {
        "total_timesteps": 1000000,
        "learning_rate": 1e-4,
        "buffer_size": 500000,
        "learning_starts": 5000,
        "batch_size": 32,
        "gamma": 0.99,
        "target_update_interval": 1000,
        "exploration_fraction": 0.3,  # Extended exploration
        "exploration_final_eps": 0.01,
    }
}

# PPO Hyperparameters - Based on Schulman et al. (2017) and OpenAI Baselines standards
PPO_CONFIGS = {
    # Easy environments (5x5 grids)
    "easy": {
        "total_timesteps": 100000,
        "learning_rate": 3e-4,  # Schulman et al. (2017): 3e-4 is standard for PPO
        "n_steps": 2048,  # Rollout length (Schulman et al.: 128-2048)
        "batch_size": 64,  # Minibatch size for SGD
        "n_epochs": 10,  # Schulman et al. (2017): 3-10 epochs per update
        "gamma": 0.99,  # Standard discount factor
        "gae_lambda": 0.95,  # Schulman et al. (2016, GAE): 0.95
        "clip_range": 0.2,  # Schulman et al. (2017): 0.2 clipping
        "ent_coef": 0.0,  # Entropy coefficient (0.0 for no exploration bonus, 0.01 for exploration)
        "vf_coef": 0.5,  # Value function coefficient
        "max_grad_norm": 0.5,  # Gradient clipping
        "n_envs": 4,  # Parallel environments
    },
    
    # Medium environments (8x8 to 16x16 grids)
    "medium": {
        "total_timesteps": 300000,
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.0,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "n_envs": 8,  # More parallel environments for efficiency
    },
    
    # Hard environments (complex tasks, large grids)
    "hard": {
        "total_timesteps": 1000000,
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.0,  # Pure exploitation for hard tasks
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "n_envs": 8,
    }
}

# Environment difficulty mapping
ENV_DIFFICULTY = {
    # Easy environments
    "MiniGrid-Empty-5x5-v0": "easy",
    "MiniGrid-Empty-6x6-v0": "easy",
    "MiniGrid-Empty-8x8-v0": "easy",
    "MiniGrid-FourRooms-v0": "easy",
    
    # Medium environments
    "MiniGrid-DoorKey-5x5-v0": "medium",
    "MiniGrid-DoorKey-6x6-v0": "medium",
    "MiniGrid-DoorKey-8x8-v0": "medium",
    "MiniGrid-MultiRoom-N2-S4-v0": "medium",
    "MiniGrid-MultiRoom-N4-S5-v0": "medium",
    
    # Hard environments
    "MiniGrid-DoorKey-16x16-v0": "hard",
    "MiniGrid-MultiRoom-N6-v0": "hard",
    "MiniGrid-KeyCorridorS3R3-v0": "hard",
    "MiniGrid-ObstructedMaze-2Dlh-v0": "hard",
    "MiniGrid-RedBlueDoors-6x6-v0": "hard",
    "MiniGrid-LockedRoom-v0": "hard",
}


def get_dqn_config(env_id):
    """Get DQN hyperparameters for a specific environment."""
    difficulty = ENV_DIFFICULTY.get(env_id, "medium")
    return DQN_CONFIGS[difficulty].copy()


def get_ppo_config(env_id):
    """Get PPO hyperparameters for a specific environment."""
    difficulty = ENV_DIFFICULTY.get(env_id, "medium")
    return PPO_CONFIGS[difficulty].copy()


def print_config(env_id, algorithm="DQN"):
    """Print the recommended configuration for an environment."""
    if algorithm.upper() == "DQN":
        config = get_dqn_config(env_id)
    elif algorithm.upper() == "PPO":
        config = get_ppo_config(env_id)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    difficulty = ENV_DIFFICULTY.get(env_id, "medium")
    
    print(f"\n{'='*60}")
    print(f"Recommended {algorithm} Configuration for {env_id}")
    print(f"Difficulty: {difficulty.upper()}")
    print(f"{'='*60}")
    for key, value in config.items():
        print(f"{key:30s}: {value}")
    print(f"{'='*60}\n")
    
    return config


if __name__ == "__main__":
    # Example usage
    print("DQN Configuration Examples:")
    print_config("MiniGrid-Empty-5x5-v0", "DQN")
    print_config("MiniGrid-DoorKey-8x8-v0", "DQN")
    print_config("MiniGrid-DoorKey-16x16-v0", "DQN")
    
    print("\nPPO Configuration Examples:")
    print_config("MiniGrid-Empty-5x5-v0", "PPO")
    print_config("MiniGrid-DoorKey-8x8-v0", "PPO")
    print_config("MiniGrid-DoorKey-16x16-v0", "PPO")
