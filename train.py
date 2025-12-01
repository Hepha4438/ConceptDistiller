"""
Universal Training Script for MiniGrid Environments
Supports both DQN and PPO with automatic hyperparameter selection
"""

import argparse
from train_dqn import train_dqn
from train_ppo import train_ppo
from config import get_dqn_config, get_ppo_config, print_config


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
    parser.add_argument("--algo", type=str, required=True, choices=["DQN", "PPO", "dqn", "ppo"],
                        help="Algorithm to use (DQN or PPO)")
    
    # Optional arguments
    parser.add_argument("--timesteps", type=int, default=None,
                        help="Total timesteps for training (default: auto-detect based on env)")
    parser.add_argument("--lr", type=float, default=None,
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
    
    # Utility
    parser.add_argument("--show-config", action="store_true",
                        help="Show recommended configuration and exit")
    
    args = parser.parse_args()
    
    # Normalize algorithm name
    algo = args.algo.upper()
    
    # Get base configuration
    if algo == "DQN":
        config = get_dqn_config(args.env)
    else:
        config = get_ppo_config(args.env)
    
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
    else:  # PPO
        if args.n_envs is not None:
            config["n_envs"] = args.n_envs
        if args.n_steps is not None:
            config["n_steps"] = args.n_steps
    
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
    else:
        model = train_ppo(env_id=args.env, **config)
    
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
