"""
PPO Training Script for MiniGrid Environments
Based on best practices from:
- OpenDILab DI-engine: https://opendilab.github.io/DI-engine/13_envs/minigrid.html
- gym-minigrid: https://github.com/maximecb/gym-minigrid
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
from minigrid_features_extractor import MinigridFeaturesExtractor
import numpy as np


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


def train_ppo(
    env_id="MiniGrid-Empty-5x5-v0",
    total_timesteps=100000,
    learning_rate=3e-4,  # Schulman et al. (2017): 3e-4, standard for PPO
    n_steps=2048,  # Schulman et al. (2017): 2048 steps per rollout (can be 128-2048)
    batch_size=64,  # Minibatch size: typically 32-128
    n_epochs=10,  # Schulman et al. (2017): 10 epochs (can be 3-10)
    gamma=0.99,  # Standard discount factor
    gae_lambda=0.95,  # Schulman et al. (2016, GAE paper): 0.95
    clip_range=0.2,  # Schulman et al. (2017): 0.2 (can be 0.1-0.3)
    ent_coef=0.0,  # Entropy coefficient: 0.0-0.01 (0.01 for exploration, 0.0 for exploitation)
    vf_coef=0.5,  # Value function coefficient: 0.5 (standard)
    max_grad_norm=0.5,  # Gradient clipping: 0.5 (standard)
    n_envs=4,  # Number of parallel environments
    seed=0,
    device="mps",
):
    """
    Train PPO agent on MiniGrid environment.
    
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
        seed: Random seed
        device: Device to use (cpu, cuda, mps)
    """
    
    # Create directories
    save_dir = f"models/{env_id}/ppo"
    os.makedirs(save_dir, exist_ok=True)
    
    # Checkpoint directory for intermediate saves (cleared each training run)
    checkpoint_dir = f"{save_dir}/last_train"
    if os.path.exists(checkpoint_dir):
        import shutil
        shutil.rmtree(checkpoint_dir)  # Clear old checkpoints
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Get next model number for best model naming
    model_number = get_next_model_number(save_dir, prefix="ppo_minigrid")
    model_name = f"ppo_minigrid_{model_number:03d}"
    best_model_path = f"{save_dir}/{model_name}"
    
    # Create tensorboard log directory with structure: minigrid_tensorboard/env_id/algo/model_name
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tensorboard_log = f"minigrid_tensorboard/{env_id}/ppo"
    run_name = f"{model_name}_{timestamp}"
    os.makedirs(tensorboard_log, exist_ok=True)
    
    print(f"Best model will be saved to: {best_model_path}.zip")
    print(f"Checkpoints will be saved to: {checkpoint_dir}/")
    print(f"TensorBoard logs: {tensorboard_log}/{run_name}")
    
    # Create training environments (parallel for better sample efficiency)
    print(f"Creating {n_envs} parallel training environments: {env_id}")
    if n_envs > 1:
        # Use SubprocVecEnv for true parallel execution (if not on macOS)
        # On macOS, DummyVecEnv is safer due to multiprocessing issues
        train_env = DummyVecEnv([make_env(env_id, seed=seed+i) for i in range(n_envs)])
    else:
        train_env = DummyVecEnv([make_env(env_id, seed=seed)])
    train_env = VecMonitor(train_env)
    
    # Create evaluation environment
    print("Creating evaluation environment...")
    eval_env = DummyVecEnv([make_env(env_id, seed=seed+1000)])
    eval_env = VecMonitor(eval_env)
    
    # Define policy kwargs with custom feature extractor
    policy_kwargs = dict(
        features_extractor_class=MinigridFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=128),
        net_arch=dict(pi=[256, 256], vf=[256, 256]),  # Separate networks for policy and value
    )
    
    # Create PPO model with research-grade hyperparameters
    # Based on: Schulman et al. (2017), Engstrom et al. (2020) - OpenAI Baselines
    print("Creating PPO model...")
    print(f"TensorBoard logs will be saved to: {tensorboard_log}/{run_name}")
    model = PPO(
        "CnnPolicy",
        train_env,
        policy_kwargs=policy_kwargs,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        clip_range_vf=None,  # No value function clipping (standard)
        normalize_advantage=True,  # Standard in PPO implementations
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        use_sde=False,  # State-dependent exploration (optional)
        sde_sample_freq=-1,
        target_kl=None,  # Early stopping based on KL divergence (can use 0.01-0.05)
        tensorboard_log=tensorboard_log,
        verbose=1,
        seed=seed,
        device=device,
    )
    
    # Setup callbacks
    # Checkpoint callback - saves intermediate checkpoints to last_train/
    checkpoint_callback = CheckpointCallback(
        save_freq=max(10000 // n_envs, 1),  # Adjust for number of parallel environments
        save_path=checkpoint_dir,  # Save to last_train/ subdirectory
        name_prefix="ppo_checkpoint",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )
    
    # Evaluation callback - saves BEST model to last_train/ during training
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=checkpoint_dir,  # Save to last_train/ during training
        log_path=checkpoint_dir,  # evaluations.npz also in last_train/
        eval_freq=max(5000 // n_envs, 1),
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
    print(f"Training PPO agent on {env_id}")
    print(f"Total timesteps: {total_timesteps}")
    print(f"Number of parallel environments: {n_envs}")
    print(f"Device: {device}")
    print(f"Model: {model_name}")
    print(f"TensorBoard: tensorboard --logdir minigrid_tensorboard/{env_id}/ppo")
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
    train_ppo(
        env_id="MiniGrid-Empty-5x5-v0",
        total_timesteps=100000,
        learning_rate=3e-4,
        n_steps=2048,
        n_envs=4,
        seed=42,
    )
    
    # Uncomment to train on more complex environments
    # print("\nTraining on MiniGrid-DoorKey-5x5-v0...")
    # train_ppo(
    #     env_id="MiniGrid-DoorKey-5x5-v0",
    #     total_timesteps=300000,
    #     learning_rate=3e-4,
    #     n_steps=2048,
    #     n_envs=8,
    #     seed=42,
    # )
    
    # print("\nTraining on MiniGrid-DoorKey-16x16-v0...")
    # train_ppo(
    #     env_id="MiniGrid-DoorKey-16x16-v0",
    #     total_timesteps=1000000,
    #     learning_rate=3e-4,
    #     n_steps=2048,
    #     batch_size=128,
    #     n_envs=8,
    #     seed=42,
    # )
