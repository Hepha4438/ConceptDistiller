import gymnasium as gym
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import DQN
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from minigrid_features_extractor import MinigridFeaturesExtractor

# Create environment
env = gym.make("MiniGrid-DoorKey-5x5-v0", render_mode="rgb_array")
env = ImgObsWrapper(env)

# Define policy kwargs with custom feature extractor
policy_kwargs = dict(
    features_extractor_class=MinigridFeaturesExtractor,
    features_extractor_kwargs=dict(features_dim=128),
)

# Create DQN model
model = DQN(
    "CnnPolicy",
    env,
    policy_kwargs=policy_kwargs,
    verbose=1,
    learning_rate=1e-4,
    buffer_size=100000,
    learning_starts=1000,
    batch_size=64,
    tau=1.0,
    gamma=0.99,
    train_freq=4,
    gradient_steps=1,
    target_update_interval=1000,
    exploration_fraction=0.2,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.05,
    max_grad_norm=10,
    tensorboard_log="./dqn_minigrid_tensorboard/",
)

# Train the model
print("Training DQN agent on MiniGrid...")
model.learn(total_timesteps=100000, progress_bar=True)

# Save the model
model.save("dqn_minigrid")
print("Model saved as dqn_minigrid")

env.close()
