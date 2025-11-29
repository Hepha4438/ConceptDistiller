import gymnasium as gym
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO, DQN
import numpy as np

def test_agent(model_path, algorithm="PPO", num_episodes=5, render=True):
    """
    Test a trained agent on MiniGrid environment
    
    Args:
        model_path: Path to the saved model
        algorithm: "PPO" or "DQN"
        num_episodes: Number of episodes to test
        render: Whether to render the environment
    """
    # Create environment
    render_mode = "human" if render else "rgb_array"
    env = gym.make("MiniGrid-DoorKey-5x5-v0", render_mode=render_mode)
    env = ImgObsWrapper(env)
    
    # Load the model
    if algorithm == "PPO":
        model = PPO.load(model_path, env=env)
    elif algorithm == "DQN":
        model = DQN.load(model_path, env=env)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    print(f"Testing {algorithm} agent for {num_episodes} episodes...")
    
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        print(f"Episode {episode + 1}: Reward = {total_reward:.2f}, Steps = {steps}")
    
    print(f"\nAverage Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average Episode Length: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")
    
    env.close()

if __name__ == "__main__":
    # Test PPO agent
    print("=" * 50)
    print("Testing PPO Agent")
    print("=" * 50)
    try:
        test_agent("ppo_minigrid", algorithm="PPO", num_episodes=5, render=False)
    except Exception as e:
        print(f"Could not test PPO agent: {e}")
    
    print("\n" + "=" * 50)
    print("Testing DQN Agent")
    print("=" * 50)
    try:
        test_agent("dqn_minigrid", algorithm="DQN", num_episodes=5, render=False)
    except Exception as e:
        print(f"Could not test DQN agent: {e}")
