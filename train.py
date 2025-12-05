"""
Universal Training Script for MiniGrid Environments
Supports both DQN and PPO with automatic hyperparameter selection
"""

import argparse
from train_dqn import train_dqn
from train_ppo import train_ppo
from train_ppo_concept import train_ppo_concept
from config import get_dqn_config, get_ppo_config, get_ppo_concept_config, print_config


def main():
    parser = argparse.ArgumentParser(
        description="Train RL agents on MiniGrid environments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train PPO on Empty environment
  python train.py --env MiniGrid-Empty-5x5-v0 --algo PPO
  
  # Train DQN on DoorKey environment with custom timesteps
  python train.py --env MiniGrid-DoorKey-5x5-v0 --algo DQN --timesteps 200000
  
  # Train with custom learning rate
  python train.py --env MiniGrid-DoorKey-16x16-v0 --algo PPO --lr 3e-4
  
  # Show recommended config without training
  python train.py --env MiniGrid-DoorKey-8x8-v0 --algo DQN --show-config
        """
    )
    
    # Required arguments
    parser.add_argument("--env", type=str, required=True,
                        help="MiniGrid environment ID (e.g., MiniGrid-Empty-5x5-v0)")
    parser.add_argument("--algo", type=str, required=True, choices=["DQN", "PPO", "PPO_CONCEPT", "dqn", "ppo", "ppo_concept"],
                        help="Algorithm to use (DQN, PPO, or PPO_CONCEPT)")
    
    # Optional arguments
    parser.add_argument("--timesteps", type=int, default=None,
                        help="Total timesteps for training (default: auto-detect based on env)")
    parser.add_argument("--lr", "--learning-rate", type=float, default=None, dest="lr",
                        help="Learning rate (default: auto-detect based on env)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cpu", "cuda", "mps"],
                        help="Device to use for training (default: cuda)")
    
    # DQN specific
    parser.add_argument("--buffer-size", type=int, default=None,
                        help="DQN replay buffer size (default: auto-detect)")
    parser.add_argument("--exploration-fraction", type=float, default=None,
                        help="DQN exploration fraction (default: auto-detect)")
    
    # PPO specific
    parser.add_argument("--n-envs", type=int, default=None,
                        help="PPO number of parallel environments (default: auto-detect)")
    parser.add_argument("--n-steps", type=int, default=None,
                        help="PPO steps per update (default: auto-detect)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="PPO batch size (default: auto-detect)")
    parser.add_argument("--n-epochs", type=int, default=None,
                        help="PPO number of epochs (default: auto-detect)")
    parser.add_argument("--gamma", type=float, default=None,
                        help="PPO discount factor (default: auto-detect)")
    parser.add_argument("--gae-lambda", type=float, default=None,
                        help="PPO GAE lambda (default: auto-detect)")
    parser.add_argument("--clip-range", type=float, default=None,
                        help="PPO clip range (default: auto-detect)")
    parser.add_argument("--ent-coef", type=float, default=None,
                        help="PPO entropy coefficient (default: auto-detect)")
    parser.add_argument("--vf-coef", type=float, default=None,
                        help="PPO value function coefficient (default: auto-detect)")
    parser.add_argument("--max-grad-norm", type=float, default=None,
                        help="PPO max gradient norm (default: auto-detect)")
    
    # PPO_CONCEPT specific
    parser.add_argument("--n-concepts", type=int, default=None,
                        help="PPO_CONCEPT number of concepts (default: auto-detect based on env difficulty)")
    parser.add_argument("--concept-mode", type=int, default=None, choices=[1, 2, 3, 4],
                        help="PPO_CONCEPT extraction mode: 1=flatten, 2=avg pool, 3=max pool, 4=FC-bottleneck (default: 1)")
    parser.add_argument("--lambda-1", type=float, default=None,
                        help="PPO_CONCEPT orthogonality regularization weight (default: 0.05)")
    parser.add_argument("--lambda-2", type=float, default=None,
                        help="PPO_CONCEPT sparsity regularization weight (default: 0.004)")
    parser.add_argument("--lambda-3", type=float, default=None,
                        help="PPO_CONCEPT L1 regularization weight (default: 2.0)")
    
    # Utility
    parser.add_argument("--show-config", action="store_true",
                        help="Show recommended configuration and exit")
    
    args = parser.parse_args()
    
    # Normalize algorithm name
    algo = args.algo.upper()
    
    # Get base configuration
    if algo == "DQN":
        config = get_dqn_config(args.env)
    elif algo == "PPO":
        config = get_ppo_config(args.env)
    elif algo == "PPO_CONCEPT":
        config = get_ppo_concept_config(args.env)
    else:
        raise ValueError(f"Unknown algorithm: {algo}")
    
    # Override with command line arguments if provided
    if args.timesteps is not None:
        config["total_timesteps"] = args.timesteps
    if args.lr is not None:
        config["learning_rate"] = args.lr
    config["seed"] = args.seed
    config["device"] = args.device
    
    # Algorithm-specific overrides
    if algo == "DQN":
        if args.buffer_size is not None:
            config["buffer_size"] = args.buffer_size
        if args.exploration_fraction is not None:
            config["exploration_fraction"] = args.exploration_fraction
    elif algo == "PPO" or algo == "PPO_CONCEPT":
        if args.n_envs is not None:
            config["n_envs"] = args.n_envs
        if args.n_steps is not None:
            config["n_steps"] = args.n_steps
        if args.batch_size is not None:
            config["batch_size"] = args.batch_size
        if args.n_epochs is not None:
            config["n_epochs"] = args.n_epochs
        if args.gamma is not None:
            config["gamma"] = args.gamma
        if args.gae_lambda is not None:
            config["gae_lambda"] = args.gae_lambda
        if args.clip_range is not None:
            config["clip_range"] = args.clip_range
        if args.ent_coef is not None:
            config["ent_coef"] = args.ent_coef
        if args.vf_coef is not None:
            config["vf_coef"] = args.vf_coef
        if args.max_grad_norm is not None:
            config["max_grad_norm"] = args.max_grad_norm
        
        # PPO_CONCEPT specific
        if algo == "PPO_CONCEPT":
            if args.n_concepts is not None:
                config["n_concepts"] = args.n_concepts
            if args.concept_mode is not None:
                config["concept_mode"] = args.concept_mode
            if args.lambda_1 is not None:
                config["lambda_1"] = args.lambda_1
            if args.lambda_2 is not None:
                config["lambda_2"] = args.lambda_2
            if args.lambda_3 is not None:
                config["lambda_3"] = args.lambda_3
    
    # Show configuration
    if args.show_config:
        print_config(args.env, algo)
        return
    
    # Print configuration before training
    print(f"\n{'='*60}")
    print(f"Training Configuration")
    print(f"{'='*60}")
    print(f"Environment: {args.env}")
    print(f"Algorithm: {algo}")
    print(f"Device: {args.device}")
    print(f"Seed: {args.seed}")
    print(f"{'='*60}")
    print("Hyperparameters:")
    for key, value in config.items():
        print(f"  {key:30s}: {value}")
    print(f"{'='*60}\n")
    
    # Train the agent
    if algo == "DQN":
        model = train_dqn(env_id=args.env, **config)
    elif algo == "PPO":
        model = train_ppo(env_id=args.env, **config)
    elif algo == "PPO_CONCEPT":
        model = train_ppo_concept(env_id=args.env, **config)
    else:
        raise ValueError(f"Unknown algorithm: {algo}")
    
    print(f"\n{'='*60}")
    print(f"Training completed successfully!")
    print(f"Model saved in: models/{args.env}/{algo.lower()}/")
    print(f"{'='*60}\n")
    
    # Print testing command
    print("To test the trained agent, run:")
    print(f"  python test_agent.py --env {args.env} --algo {algo}")
    print("\nTo view training logs in TensorBoard, run:")
    print(f"  tensorboard --logdir ./{algo.lower()}_minigrid_tensorboard/")


if __name__ == "__main__":
    main()
