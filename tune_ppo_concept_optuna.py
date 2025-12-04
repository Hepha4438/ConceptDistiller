"""
Optuna Hyperparameter Tuning for PPO_CONCEPT
Automatically tune lambda_1, lambda_2, lambda_3, and n_concepts
"""

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import numpy as np
import os
import shutil
import json
from train_ppo_concept import train_ppo_concept, ConceptPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
from minigrid.wrappers import ImgObsWrapper
import torch
from config import N_CONCEPTS_RANGES, ENV_DIFFICULTY, get_optuna_tuning_config
import sys


# Global callback for UI logging
_ui_log_callback = None

def set_ui_log_callback(callback):
    """Set callback function for UI logging"""
    global _ui_log_callback
    _ui_log_callback = callback

def log_to_ui(message):
    """Log message to UI if callback is set"""
    global _ui_log_callback
    if _ui_log_callback:
        _ui_log_callback(message)
    print(message, end='')  # Also print to stdout


def make_env(env_id, seed=0):
    def _init():
        env = gym.make(env_id, render_mode="rgb_array")
        env = ImgObsWrapper(env)
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init


def objective(trial, env_id, total_timesteps, n_envs, seed, device, is_trial=True):
    """
    Objective function for Optuna
    Returns: combined score (reward + concept quality penalties)

    Objective = alpha * reward - beta * L_ortho - gamma * L_spar - delta * L_l1
    Where concept losses are normalized by their typical ranges
    
    Args:
        is_trial: If True, save to trials/ directory; if False, save to normal directory
    """

    # Get difficulty and n_concepts range from config
    difficulty = ENV_DIFFICULTY.get(env_id, "medium")
    n_concepts_range = N_CONCEPTS_RANGES[difficulty]
    
    # ✅ Suggest hyperparameters
    # Adjusted ranges to center around new defaults:
    # lambda_1=0.05 → range [0.01, 0.2]
    # lambda_2=0.002 → range [1e-4, 0.01] (giữ nguyên)
    # lambda_3=0.01 → range [0.001, 0.05]
    lambda_1 = trial.suggest_float('lambda_1', 0.01, 0.2, log=True)
    lambda_2 = trial.suggest_float('lambda_2', 1e-4, 0.01, log=True)
    lambda_3 = trial.suggest_float('lambda_3', 0.001, 0.05, log=True)
    n_concepts = trial.suggest_categorical('n_concepts', n_concepts_range)
    
    log_to_ui(f"\n{'='*60}\n")
    log_to_ui(f"Trial {trial.number} (Difficulty: {difficulty})\n")
    log_to_ui(f"{'='*60}\n")
    log_to_ui(f"  lambda_1 (orthogonality): {lambda_1:.6f}\n")
    log_to_ui(f"  lambda_2 (sparsity):      {lambda_2:.6f}\n")
    log_to_ui(f"  lambda_3 (L1):            {lambda_3:.6f}\n")
    log_to_ui(f"  n_concepts:               {n_concepts} (from {n_concepts_range})\n")
    log_to_ui(f"{'='*60}\n\n")
    
    try:
        # Train model with suggested hyperparameters
        # ✅ Pass is_trial and trial_number to control saving behavior
        model = train_ppo_concept(
            env_id=env_id,
            total_timesteps=total_timesteps,
            n_envs=n_envs,
            n_concepts=n_concepts,
            seed=seed + trial.number,  # Different seed per trial
            device=device,
            lambda_1=lambda_1,
            lambda_2=lambda_2,
            lambda_3=lambda_3,
            is_trial=is_trial,
            trial_number=trial.number
        )
        
        # ✅ GET CONCEPT LOSSES FROM BEST MODEL (BEFORE cleanup!)
        # Read from JSON file saved when best model was updated
        last_train_dir = f"models/{env_id}/ppo_concept/last_train"
        concept_losses_path = os.path.join(last_train_dir, 'best_model_concept_losses.json')
        
        if os.path.exists(concept_losses_path):
            with open(concept_losses_path, 'r') as f:
                concept_losses_data = json.load(f)
            
            mean_L_ortho = concept_losses_data['L_ortho']
            mean_L_spar = concept_losses_data['L_spar']
            mean_L_l1 = concept_losses_data['L_l1']
            
            log_to_ui(f"  ✓ Loaded concept losses from best model checkpoint\n")
            log_to_ui(f"    (at timestep {concept_losses_data['timestep']}, reward={concept_losses_data['mean_reward']:.3f})\n")
        else:
            log_to_ui(f"  ⚠ Best model concept losses not found at {concept_losses_path}\n")
            log_to_ui(f"  Using fallback: 0.0 for all losses\n")
            mean_L_ortho = 0.0
            mean_L_spar = 0.0
            mean_L_l1 = 0.0
        
        # ✅ NOW clean up: Remove checkpoint files from last_train (AFTER reading!)
        if is_trial:
            if os.path.exists(last_train_dir):
                shutil.rmtree(last_train_dir)
                log_to_ui(f"  ✓ Cleaned up trial checkpoints\n")

        # ✅ Evaluate reward (still need to run episodes)
        eval_env = DummyVecEnv([make_env(env_id, seed=seed + 10000 + trial.number)])
        eval_env = VecMonitor(eval_env)
        
        n_eval_episodes = 20
        episode_rewards = []
        
        for episode_idx in range(n_eval_episodes):
            obs = eval_env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _ = eval_env.step(action)
                episode_reward += reward[0]
            
            episode_rewards.append(episode_reward)
        
        eval_env.close()
        
        # ✅ Compute statistics
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        
        # ✅ Concept losses already extracted from training logs above
        # (mean_L_ortho, mean_L_spar, mean_L_l1 already set)

        # Validate - if losses are 0, might be error reading tensorboard
        if mean_L_ortho == 0.0 and mean_L_spar == 0.0 and mean_L_l1 == 0.0:
            log_to_ui(f"  ⚠ WARNING: All concept losses are 0 - check tensorboard logs!\n")
        
        # ✅ Combined objective with normalized losses
        # Weights for balancing: prioritize reward, but penalize bad concept quality
        alpha = 1.0      # reward weight (maximize)
        beta = 0.05      # orthogonality penalty (L_ortho typically 0-1)
        gamma = 0.02     # sparsity penalty (L_spar typically 0-1)
        delta = 0.01     # L1 penalty (L_l1 typically 0-0.1)

        objective_value = (alpha * mean_reward 
                          - beta * mean_L_ortho
                          - gamma * mean_L_spar
                          - delta * mean_L_l1)

        # ✅ Final validation of objective value
        if np.isnan(objective_value) or np.isinf(objective_value):
            log_to_ui(f"⚠️  WARNING: Invalid objective value detected!\n")
            log_to_ui(f"  Using reward-only fallback: {mean_reward:.3f}\n")
            objective_value = mean_reward
        
        log_to_ui(f"\nTrial {trial.number} Results:\n")
        log_to_ui(f"  Mean Reward:      {mean_reward:.3f} ± {std_reward:.3f}\n")
        log_to_ui(f"  Min/Max Reward:   {np.min(episode_rewards):.3f} / {np.max(episode_rewards):.3f}\n")
        log_to_ui(f"  Concept Losses (from training logs):\n")
        log_to_ui(f"    - Mean L_ortho: {mean_L_ortho:.6f} (penalty: {beta * mean_L_ortho:.6f})\n")
        log_to_ui(f"    - Mean L_spar:  {mean_L_spar:.6f} (penalty: {gamma * mean_L_spar:.6f})\n")
        log_to_ui(f"    - Mean L_l1:    {mean_L_l1:.6f} (penalty: {delta * mean_L_l1:.6f})\n")
        log_to_ui(f"  Combined Score:   {objective_value:.3f}\n")

        # Report intermediate value for pruning
        trial.report(objective_value, step=0)
        
        # Check if trial should be pruned
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        return objective_value
        
    except Exception as e:
        print(f"\n❌ Trial {trial.number} failed: {str(e)}")
        import traceback
        traceback.print_exc()
        # Return very low score for failed trials
        return -1000.0


def optimize_hyperparameters(
    env_id="MiniGrid-Empty-5x5-v0",
    n_trials=50,
    total_timesteps=50000,  # Shorter for tuning
    n_envs=4,
    seed=42,
    device="cuda",
    study_name=None,
    storage=None
):
    """
    Run Optuna optimization
    
    Args:
        env_id: Environment ID
        n_trials: Number of trials to run
        total_timesteps: Timesteps per trial (should be shorter for tuning)
        n_envs: Number of parallel environments
        seed: Random seed
        device: Device (cuda/cpu/mps)
        study_name: Name for the study (for resuming)
        storage: Storage URL (e.g., 'sqlite:///optuna_study.db')
    """
    
    difficulty = ENV_DIFFICULTY.get(env_id, "medium")
    n_concepts_range = N_CONCEPTS_RANGES[difficulty]
    
    # Get tuning config for n_startup_trials
    tuning_config = get_optuna_tuning_config(env_id)
    n_startup_trials = tuning_config.get('n_startup_trials', 5)  # Default to 5 if not in config
    
    log_to_ui("="*60 + "\n")
    log_to_ui("OPTUNA HYPERPARAMETER TUNING FOR PPO_CONCEPT\n")
    log_to_ui("="*60 + "\n")
    log_to_ui(f"Environment:     {env_id}\n")
    log_to_ui(f"Difficulty:      {difficulty}\n")
    log_to_ui(f"Trials:          {n_trials}\n")
    log_to_ui(f"Startup trials:  {n_startup_trials} (random sampling)\n")
    log_to_ui(f"Timesteps/trial: {total_timesteps:,}\n")
    log_to_ui(f"Device:          {device}\n")
    log_to_ui("="*60 + "\n")
    log_to_ui("\n🎯 Optimization Objective:\n")
    log_to_ui("   Combined Score = 1.0*reward - 0.05*L_ortho - 0.02*L_spar - 0.01*L_l1\n")
    log_to_ui("   (Maximize reward while minimizing concept losses)\n")
    log_to_ui("\n🔍 Searching for optimal:\n")
    log_to_ui("  - lambda_1 (orthogonality):  0.01 to 0.2\n")
    log_to_ui("  - lambda_2 (sparsity):       1e-4 to 0.01\n")
    log_to_ui("  - lambda_3 (L1):             0.001 to 0.05\n")
    log_to_ui(f"  - n_concepts:                {n_concepts_range}\n")
    log_to_ui("="*60 + "\n")
    log_to_ui("\n💾 Trials will be saved to:\n")
    log_to_ui(f"   - models/{env_id}/ppo_concept/trials/\n")
    log_to_ui(f"   - minigrid_tensorboard/{env_id}/ppo_concept/trials/\n")
    log_to_ui("="*60 + "\n\n")
    
    # Create study
    if study_name is None:
        study_name = f"ppo_concept_{env_id}_{seed}"
    
    sampler = TPESampler(seed=seed, n_startup_trials=n_startup_trials)
    pruner = MedianPruner(n_startup_trials=n_startup_trials, n_warmup_steps=0)
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        direction='maximize',  # Maximize combined score (reward - concept_losses)
        load_if_exists=True
    )
    
    # Optimize
    study.optimize(
        lambda trial: objective(trial, env_id, total_timesteps, n_envs, seed, device),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    # Print results
    log_to_ui("\n" + "="*60 + "\n")
    log_to_ui("OPTIMIZATION COMPLETE\n")
    log_to_ui("="*60 + "\n")
    
    log_to_ui(f"\nNumber of finished trials: {len(study.trials)}\n")
    
    log_to_ui("\n📊 Best trial:\n")
    trial = study.best_trial
    log_to_ui(f"  Value (combined score): {trial.value:.3f}\n")
    log_to_ui(f"\n  Params:\n")
    for key, value in trial.params.items():
        log_to_ui(f"    {key}: {value}\n")
    
    print("\n📈 All trials:")
    df = study.trials_dataframe()
    print(df[['number', 'value', 'params_lambda_1', 'params_lambda_2',
              'params_lambda_3', 'params_n_concepts']].to_string())
    
    # Save to CSV
    csv_path = f"optuna_results_{env_id}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n💾 Results saved to: {csv_path}")
    
    # Visualization (if possible)
    try:
        import optuna.visualization as vis
        
        # Optimization history
        fig = vis.plot_optimization_history(study)
        fig.write_html(f"optuna_history_{env_id}.html")
        print(f"📊 Optimization history: optuna_history_{env_id}.html")

        # Parameter importances
        fig = vis.plot_param_importances(study)
        fig.write_html(f"optuna_importances_{env_id}.html")
        print(f"📊 Parameter importances: optuna_importances_{env_id}.html")

        # Parallel coordinate plot
        fig = vis.plot_parallel_coordinate(study)
        fig.write_html(f"optuna_parallel_{env_id}.html")
        print(f"📊 Parallel coordinates: optuna_parallel_{env_id}.html")
        
    except ImportError:
        print("\n⚠️  Install plotly for visualizations: pip install plotly")
    
    return study


def train_with_best_params(study, env_id, total_timesteps, n_envs, seed, device):
    """
    Train a final model with best parameters found by Optuna
    """
    best_params = study.best_params
    
    print("\n" + "="*60)
    print("TRAINING FINAL MODEL WITH BEST PARAMS")
    print("="*60)
    print(f"  lambda_1: {best_params['lambda_1']:.6f}")
    print(f"  lambda_2: {best_params['lambda_2']:.6f}")
    print(f"  lambda_3: {best_params['lambda_3']:.6f}")
    print(f"  n_concepts: {best_params['n_concepts']}")
    print("="*60 + "\n")
    
    model = train_ppo_concept(
        env_id=env_id,
        total_timesteps=total_timesteps,
        n_envs=n_envs,
        n_concepts=best_params['n_concepts'],
        seed=seed,
        device=device,
        lambda_1=best_params['lambda_1'],
        lambda_2=best_params['lambda_2'],
        lambda_3=best_params['lambda_3']
    )
    
    print(f"\n✅ Final model trained and saved!")
    return model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Tune PPO_CONCEPT hyperparameters with Optuna")
    parser.add_argument("--env", type=str, default="MiniGrid-Empty-5x5-v0",
                        help="Environment ID")
    parser.add_argument("--trials", type=int, default=50,
                        help="Number of Optuna trials")
    parser.add_argument("--timesteps", type=int, default=50000,
                        help="Timesteps per trial (use shorter for tuning)")
    parser.add_argument("--n-envs", type=int, default=4,
                        help="Number of parallel environments")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda/cpu/mps)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--storage", type=str, default=None,
                        help="Optuna storage (e.g., sqlite:///optuna.db)")
    parser.add_argument("--train-final", action="store_true",
                        help="Train final model with full timesteps after optimization")
    parser.add_argument("--final-timesteps", type=int, default=100000,
                        help="Timesteps for final training")
    
    args = parser.parse_args()
    
    # Run optimization
    study = optimize_hyperparameters(
        env_id=args.env,
        n_trials=args.trials,
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        seed=args.seed,
        device=args.device,
        storage=args.storage
    )
    
    # Optionally train final model with best params
    if args.train_final:
        train_with_best_params(
            study=study,
            env_id=args.env,
            total_timesteps=args.final_timesteps,
            n_envs=args.n_envs,
            seed=args.seed,
            device=args.device
        )
