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
from train_ppo_concept import train_ppo_concept, ConceptPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
from minigrid.wrappers import ImgObsWrapper
import torch
from config import N_CONCEPTS_RANGES, ENV_DIFFICULTY, get_optuna_tuning_config


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
    Returns: mean reward on evaluation episodes
    
    Args:
        is_trial: If True, save to trials/ directory; if False, save to normal directory
    """
    
    # Get difficulty and n_concepts range from config
    difficulty = ENV_DIFFICULTY.get(env_id, "medium")
    n_concepts_range = N_CONCEPTS_RANGES[difficulty]
    
    # ✅ Suggest hyperparameters
    lambda_1 = trial.suggest_float('lambda_1', 1e-4, 0.1, log=True)
    lambda_2 = trial.suggest_float('lambda_2', 1e-5, 0.01, log=True)
    lambda_3 = trial.suggest_float('lambda_3', 1e-6, 0.001, log=True)
    n_concepts = trial.suggest_categorical('n_concepts', n_concepts_range)
    
    print(f"\n{'='*60}")
    print(f"Trial {trial.number} (Difficulty: {difficulty})")
    print(f"{'='*60}")
    print(f"  lambda_1 (orthogonality): {lambda_1:.6f}")
    print(f"  lambda_2 (sparsity):      {lambda_2:.6f}")
    print(f"  lambda_3 (L1):            {lambda_3:.6f}")
    print(f"  n_concepts:               {n_concepts} (from {n_concepts_range})")
    print(f"{'='*60}\n")
    
    try:
        # ✅ Setup directories for trials
        if is_trial:
            # Save trials to separate directory
            original_save_dir = f"models/{env_id}/ppo_concept"
            trial_save_dir = f"models/{env_id}/ppo_concept/trials"
            os.makedirs(trial_save_dir, exist_ok=True)
            
            # Temporarily modify train_ppo_concept to use trials directory
            # We'll pass this through by modifying the save path
            
        # Train model with suggested hyperparameters
        model = train_ppo_concept(
            env_id=env_id,
            total_timesteps=total_timesteps,
            n_envs=n_envs,
            n_concepts=n_concepts,
            seed=seed + trial.number,  # Different seed per trial
            device=device,
            lambda_1=lambda_1,
            lambda_2=lambda_2,
            lambda_3=lambda_3
        )
        
        # ✅ Move trial results to trials directory
        if is_trial:
            # Move last_train directory to trials
            last_train_src = f"models/{env_id}/ppo_concept/last_train"
            last_train_dst = f"models/{env_id}/ppo_concept/trials/trial_{trial.number:03d}"
            if os.path.exists(last_train_src):
                if os.path.exists(last_train_dst):
                    shutil.rmtree(last_train_dst)
                shutil.move(last_train_src, last_train_dst)
            
            # Move tensorboard logs to trials
            tb_base = f"minigrid_tensorboard/{env_id}/ppo_concept"
            if os.path.exists(tb_base):
                # Find the most recent run
                runs = [d for d in os.listdir(tb_base) if os.path.isdir(os.path.join(tb_base, d))]
                if runs:
                    latest_run = max(runs, key=lambda d: os.path.getctime(os.path.join(tb_base, d)))
                    tb_src = os.path.join(tb_base, latest_run)
                    tb_trials_dir = f"{tb_base}/trials"
                    os.makedirs(tb_trials_dir, exist_ok=True)
                    tb_dst = os.path.join(tb_trials_dir, f"trial_{trial.number:03d}_{latest_run}")
                    if os.path.exists(tb_dst):
                        shutil.rmtree(tb_dst)
                    shutil.move(tb_src, tb_dst)
        
        # Evaluate the trained model
        eval_env = DummyVecEnv([make_env(env_id, seed=seed + 10000 + trial.number)])
        eval_env = VecMonitor(eval_env)
        
        n_eval_episodes = 20
        episode_rewards = []
        
        for _ in range(n_eval_episodes):
            obs = eval_env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _ = eval_env.step(action)
                episode_reward += reward[0]
            
            episode_rewards.append(episode_reward)
        
        eval_env.close()
        
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        
        print(f"\nTrial {trial.number} Results:")
        print(f"  Mean Reward: {mean_reward:.3f} ± {std_reward:.3f}")
        print(f"  Min: {np.min(episode_rewards):.3f}, Max: {np.max(episode_rewards):.3f}")
        
        # Report intermediate value for pruning
        trial.report(mean_reward, step=0)
        
        # Check if trial should be pruned
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        return mean_reward
        
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
    
    print("="*60)
    print("OPTUNA HYPERPARAMETER TUNING FOR PPO_CONCEPT")
    print("="*60)
    print(f"Environment:     {env_id}")
    print(f"Difficulty:      {difficulty}")
    print(f"Trials:          {n_trials}")
    print(f"Startup trials:  {n_startup_trials} (random sampling)")
    print(f"Timesteps/trial: {total_timesteps:,}")
    print(f"Device:          {device}")
    print("="*60)
    print("\nSearching for optimal:")
    print("  - lambda_1 (orthogonality):  1e-4 to 0.1")
    print("  - lambda_2 (sparsity):       1e-5 to 0.01")
    print("  - lambda_3 (L1):             1e-6 to 0.001")
    print(f"  - n_concepts:                {n_concepts_range}")
    print("="*60)
    print("\n💾 Trials will be saved to:")
    print(f"   - models/{env_id}/ppo_concept/trials/")
    print(f"   - minigrid_tensorboard/{env_id}/ppo_concept/trials/")
    print("="*60 + "\n")
    
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
        direction='maximize',  # Maximize reward
        load_if_exists=True
    )
    
    # Optimize
    study.optimize(
        lambda trial: objective(trial, env_id, total_timesteps, n_envs, seed, device),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    # Print results
    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE")
    print("="*60)
    
    print(f"\nNumber of finished trials: {len(study.trials)}")
    
    print("\n📊 Best trial:")
    trial = study.best_trial
    print(f"  Value (mean reward): {trial.value:.3f}")
    print(f"\n  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
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
