"""
DQN Training Script for MiniGrid Environments
Based on best practices from:
- OpenDILab DI-engine: https://opendilab.github.io/DI-engine/13_envs/minigrid.html
- gym-minigrid: https://github.com/maximecb/gym-minigrid
"""

import os
import re
import gymnasium as gym
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import (
    BaseCallback, 
    CheckpointCallback, 
    EvalCallback,
    CallbackList
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from minigrid_features_extractor import MinigridFeaturesExtractor
import numpy as np


def get_next_model_number(save_dir, prefix="dqn_minigrid"):
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


class DetailedLoggingCallback(BaseCallback):
    """
    Callback for detailed logging during training.
    Logs actions, rewards, and episode statistics.
    """
    def __init__(self, verbose=0, log_freq=100):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0

    def _on_step(self) -> bool:
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
        
        # Log exploration rate
        if self.num_timesteps % self.log_freq == 0:
            if hasattr(self.model, 'exploration_rate'):
                self.logger.record("train/exploration_rate", self.model.exploration_rate)
        
        return True


def make_env(env_id, seed=0):
    """
    Create and wrap MiniGrid environment.
    """
    def _init():
        env = gym.make(env_id, render_mode="rgb_array")
        # Use ImgObsWrapper to get image observations
        env = ImgObsWrapper(env)
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init


def train_dqn(
    env_id="MiniGrid-Empty-5x5-v0",
    total_timesteps=100000,
    learning_rate=1e-4,  # Mnih et al. (2015): 1e-4 for Adam, 2.5e-4 for RMSprop
    buffer_size=100000,  # Standard: 100k-1M depending on task complexity
    learning_starts=1000,  # Start learning after collecting initial experience
    batch_size=32,  # Standard: 32 (Nature DQN, Rainbow DQN)
    gamma=0.99,  # Standard discount factor in RL literature
    target_update_interval=1000,  # Hard update every 1000 steps (Nature DQN: 10k frames = 2.5k steps with frameskip)
    exploration_fraction=0.1,  # Linearly decay epsilon over 10% of training
    exploration_final_eps=0.01,  # Mnih et al. (2015): final epsilon = 0.01 (not 0.1)
    seed=0,
    device="cuda",
):
    """
    Train DQN agent on MiniGrid environment.
    
    Args:
        env_id: MiniGrid environment ID
        total_timesteps: Total training timesteps
        learning_rate: Learning rate for optimizer
        buffer_size: Size of replay buffer
        learning_starts: Steps before training starts
        batch_size: Batch size for training
        gamma: Discount factor
        target_update_interval: Steps between target network updates
        exploration_fraction: Fraction of training for exploration
        exploration_final_eps: Final exploration epsilon
        seed: Random seed
        device: Device to use (cpu, cuda, mps)
    """
    
    # Create directories
    save_dir = f"models/{env_id}/dqn"
    os.makedirs(save_dir, exist_ok=True)
    
    # Checkpoint directory for intermediate saves (cleared each training run)
    checkpoint_dir = f"{save_dir}/last_train"
    if os.path.exists(checkpoint_dir):
        import shutil
        shutil.rmtree(checkpoint_dir)  # Clear old checkpoints
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Get next model number for best model naming
    model_number = get_next_model_number(save_dir, prefix="dqn_minigrid")
    model_name = f"dqn_minigrid_{model_number:03d}"
    best_model_path = f"{save_dir}/{model_name}"
    
    # Create tensorboard log directory with structure: minigrid_tensorboard/env_id/algo/model_name
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tensorboard_log = f"minigrid_tensorboard/{env_id}/dqn"
    run_name = f"{model_name}_{timestamp}"
    os.makedirs(tensorboard_log, exist_ok=True)
    
    print(f"Best model will be saved to: {best_model_path}.zip")
    print(f"Checkpoints will be saved to: {checkpoint_dir}/")
    print(f"TensorBoard logs: {tensorboard_log}/{run_name}")
    
    # Create training environment
    print(f"Creating training environment: {env_id}")
    train_env = DummyVecEnv([make_env(env_id, seed=seed)])
    train_env = VecMonitor(train_env)
    
    # Create evaluation environment
    print("Creating evaluation environment...")
    eval_env = DummyVecEnv([make_env(env_id, seed=seed+100)])
    eval_env = VecMonitor(eval_env)
    
    # Define policy kwargs with custom feature extractor
    policy_kwargs = dict(
        features_extractor_class=MinigridFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=128),
        net_arch=[256, 256],  # Additional fully connected layers
    )
    
    # Create DQN model with research-grade hyperparameters
    # Based on: Mnih et al. (Nature 2015), Hessel et al. (Rainbow DQN, 2018)
    print("Creating DQN model...")
    print(f"TensorBoard logs will be saved to: {tensorboard_log}/{run_name}")
    model = DQN(
        "CnnPolicy",
        train_env,
        policy_kwargs=policy_kwargs,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        learning_starts=learning_starts,
        batch_size=batch_size,
        tau=1.0,  # Hard update (standard for DQN)
        gamma=gamma,
        train_freq=4,  # Update every 4 steps (Nature DQN)
        gradient_steps=1,  # 1 gradient step per update
        target_update_interval=target_update_interval,
        exploration_fraction=exploration_fraction,
        exploration_initial_eps=1.0,
        exploration_final_eps=exploration_final_eps,
        max_grad_norm=10,  # Gradient clipping (standard)
        tensorboard_log=tensorboard_log,
        verbose=1,
        seed=seed,
        device=device,
    )
    
    # Setup callbacks
    # Checkpoint callback - saves intermediate checkpoints to last_train/
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=checkpoint_dir,  # Save to last_train/ subdirectory
        name_prefix="dqn_checkpoint",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    
    # Evaluation callback - saves BEST model to last_train/ during training
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=checkpoint_dir,  # Save to last_train/ during training
        log_path=checkpoint_dir,  # evaluations.npz also in last_train/
        eval_freq=5000,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )
    
    # Detailed logging callback
    logging_callback = DetailedLoggingCallback(verbose=1, log_freq=100)
    
    # Combine all callbacks
    callback = CallbackList([checkpoint_callback, eval_callback, logging_callback])
    
    # Train the model
    print(f"\n{'='*60}")
    print(f"Training DQN agent on {env_id}")
    print(f"Total timesteps: {total_timesteps}")
    print(f"Device: {device}")
    print(f"Model: {model_name}")
    print(f"TensorBoard: tensorboard --logdir minigrid_tensorboard/{env_id}/dqn")
    print(f"{'='*60}\n")
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True,
        tb_log_name=run_name,  # Use custom run name with timestamp
    )
    
    # Copy best_model.zip from last_train/ to final location with numbered name
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
    print(f"{'='*60}\n")
    
    # Cleanup
    train_env.close()
    eval_env.close()
    
    return model


if __name__ == "__main__":
    # Training configurations for different environments
    
    # Easy environment for quick testing
    print("Training on MiniGrid-Empty-5x5-v0...")
    train_dqn(
        env_id="MiniGrid-Empty-5x5-v0",
        total_timesteps=100000,
        learning_rate=1e-4,
        exploration_fraction=0.1,
        seed=42,
    )
    
    # Uncomment to train on more complex environments
    # print("\nTraining on MiniGrid-DoorKey-5x5-v0...")
    # train_dqn(
    #     env_id="MiniGrid-DoorKey-5x5-v0",
    #     total_timesteps=300000,
    #     learning_rate=1e-4,
    #     buffer_size=200000,
    #     exploration_fraction=0.2,
    #     seed=42,
    # )
    
    # print("\nTraining on MiniGrid-DoorKey-16x16-v0...")
    # train_dqn(
    #     env_id="MiniGrid-DoorKey-16x16-v0",
    #     total_timesteps=1000000,
    #     learning_rate=1e-4,
    #     buffer_size=500000,
    #     exploration_fraction=0.3,
    #     seed=42,
    # )
