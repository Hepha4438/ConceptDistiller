"""
Test Script for Trained MiniGrid Agents
Tests PPO and DQN agents on various MiniGrid environments
"""

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO, DQN
from train_ppo_concept import ConceptPPO
import numpy as np
import argparse
import os
import re


def test_agent(model_path, env_id="MiniGrid-Empty-5x5-v0", algorithm="PPO", 
               num_episodes=10, render=True, deterministic=True, max_steps=1000,
               save_video=False):
    """
    Test a trained agent on MiniGrid environment
    
    Args:
        model_path: Path to the saved model
        env_id: MiniGrid environment ID
        algorithm: "PPO" or "DQN"
        num_episodes: Number of episodes to test
        render: Whether to render the environment
        deterministic: Whether to use deterministic actions
        max_steps: Maximum steps per episode (safety limit)
        save_video: Whether to save video of best episode
    """
    # Extract model name from path for video directory
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    
    # Create environment
    render_mode = "human" if render else "rgb_array"
    env = gym.make(env_id, render_mode="rgb_array" if save_video else render_mode)
    env = ImgObsWrapper(env)
    
    # Setup video recording if requested
    video_dir = None
    if save_video:
        # Create video directory: ./video/env_id/algo/model_name/
        video_dir = os.path.join("./video", env_id, algorithm.lower(), model_name)
        os.makedirs(video_dir, exist_ok=True)
        
        # We'll record manually by storing frames of best episode
        best_episode_frames = []
    
    # Load the model
    try:
        if algorithm.upper() == "PPO":
            model = PPO.load(model_path, env=env)
        elif algorithm.upper() == "DQN":
            model = DQN.load(model_path, env=env)
        elif algorithm.upper() == "PPO_CONCEPT":
            model = ConceptPPO.load(model_path, env=env)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        return
    
    print(f"\n{'='*60}")
    print(f"Testing {algorithm} agent on {env_id}")
    print(f"Model: {model_path}")
    print(f"Episodes: {num_episodes}")
    print(f"Deterministic: {deterministic}")
    if save_video:
        print(f"Video will be saved to: {video_dir}")
    print(f"{'='*60}\n")
    
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    best_reward = float('-inf')
    best_episode_idx = -1
    all_episodes_frames = []  # Store frames for all episodes
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        steps = 0
        episode_frames = []
        
        while not done and steps < max_steps:
            # Capture frame if saving video
            if save_video:
                frame = env.unwrapped.get_frame()
                episode_frames.append(frame)
            
            action, _states = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
        
        # Capture final frame
        if save_video:
            frame = env.unwrapped.get_frame()
            episode_frames.append(frame)
            all_episodes_frames.append(episode_frames)
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        
        # Track best episode
        if total_reward > best_reward:
            best_reward = total_reward
            best_episode_idx = episode
        
        # Count success (reward > 0 typically means success in MiniGrid)
        if total_reward > 0:
            success_count += 1
        
        print(f"Episode {episode + 1}: Reward = {total_reward:.2f}, Steps = {steps}, Success = {total_reward > 0}")
    
    # Save video of best episode
    if save_video and best_episode_idx >= 0:
        try:
            # For moviepy 2.x
            from moviepy import ImageSequenceClip
            best_frames = all_episodes_frames[best_episode_idx]
            video_path = os.path.join(video_dir, f"best_episode_reward_{best_reward:.2f}.mp4")
            
            # Create video clip (10 fps for smooth playback)
            clip = ImageSequenceClip(best_frames, fps=10)
            clip.write_videofile(video_path, codec='libx264', audio=False, logger=None)
            print(f"\n✓ Video of best episode (reward={best_reward:.2f}) saved to: {video_path}")
        except ImportError as e:
            print(f"\n⚠ Warning: moviepy not available")
            print(f"   Error: {e}")
            print(f"   Install with: pip install moviepy")
        except Exception as e:
            print(f"\n⚠ Warning: Could not save video: {str(e)}")
    
    # Calculate statistics
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)
    std_length = np.std(episode_lengths)
    success_rate = (success_count / num_episodes) * 100
    
    print(f"\n{'='*60}")
    print(f"Test Results:")
    print(f"{'='*60}")
    print(f"Average Reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Best Reward: {best_reward:.2f} (Episode {best_episode_idx + 1})")
    print(f"Average Episode Length: {mean_length:.2f} ± {std_length:.2f}")
    print(f"Success Rate: {success_rate:.1f}% ({success_count}/{num_episodes})")
    print(f"{'='*60}\n")
    
    env.close()
    
    return {
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'best_reward': best_reward,
        'best_episode': best_episode_idx,
        'mean_length': mean_length,
        'std_length': std_length,
        'success_rate': success_rate,
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths
    }


def find_best_model(env_id, algorithm):
    """
    Find the best model for a given environment and algorithm
    """
    model_dir = f"models/{env_id}/{algorithm.lower()}"
    
    # Look for best_model.zip first (from EvalCallback)
    best_model_path = os.path.join(model_dir, "best_model.zip")
    if os.path.exists(best_model_path):
        return best_model_path
    
    # Otherwise, look for the final model
    final_model_path = os.path.join(model_dir, f"{algorithm.lower()}_minigrid.zip")
    if os.path.exists(final_model_path):
        return final_model_path
    
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test trained MiniGrid agents")
    parser.add_argument("--env", type=str, default="MiniGrid-Empty-5x5-v0",
                        help="MiniGrid environment ID")
    parser.add_argument("--algo", type=str, default="PPO", choices=["PPO", "DQN", "PPO_CONCEPT", "ppo", "dqn", "ppo_concept"],
                        help="Algorithm (PPO, DQN, or PPO_CONCEPT)")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to model file (optional, will auto-detect if not provided)")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of test episodes")
    parser.add_argument("--render", action="store_true",
                        help="Render the environment")
    parser.add_argument("--stochastic", action="store_true",
                        help="Use stochastic actions instead of deterministic")
    parser.add_argument("--save-video", action="store_true",
                        help="Save video of best episode to ./video/env_id/algo/model/")
    
    args = parser.parse_args()
    
    # Find model path if not provided
    if args.model is None:
        args.model = find_best_model(args.env, args.algo)
        if args.model is None:
            print(f"Error: No trained model found for {args.env} with {args.algo}")
            print(f"Please train a model first or specify --model path")
            exit(1)
    
    # Test the agent
    test_agent(
        model_path=args.model,
        env_id=args.env,
        algorithm=args.algo,
        num_episodes=args.episodes,
        render=args.render,
        deterministic=not args.stochastic,
        save_video=args.save_video
    )
    
    # Example tests for different environments (commented out)
    # print("\n" + "="*60)
    # print("Testing on Multiple Environments")
    # print("="*60 + "\n")
    #
    # environments = [
    #     "MiniGrid-Empty-5x5-v0",
    #     "MiniGrid-DoorKey-5x5-v0",
    #     "MiniGrid-DoorKey-16x16-v0"
    # ]
    #
    # for env_id in environments:
    #     for algo in ["PPO", "DQN"]:
    #         model_path = find_best_model(env_id, algo)
    #         if model_path:
    #             test_agent(
    #                 model_path=model_path,
    #                 env_id=env_id,
    #                 algorithm=algo,
    #                 num_episodes=5,
    #                 render=False
    #             )
