import gymnasium as gym
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO
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

# Create PPO model
model = PPO(
    "CnnPolicy",
    env,
    policy_kwargs=policy_kwargs,
    verbose=1,
    learning_rate=1e-4,
    n_steps=128,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    tensorboard_log="./ppo_minigrid_tensorboard/",
    device="cuda",
)

# Train the model
print("Training PPO agent on MiniGrid...")
model.learn(total_timesteps=100000, progress_bar=True)

# Save the model
model.save("ppo_minigrid")
print("Model saved as ppo_minigrid")

env.close()
