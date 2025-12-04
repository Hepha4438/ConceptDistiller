"""
PPO with Concept Distillation for MiniGrid
- Total loss = policy + value + entropy + concept losses
- Full callbacks: checkpoint, eval, logging
- Gradient clipping supported
- Allows passing LAMBDA_1,2,3 when calling train_ppo_concept()
"""

import os
import shutil
from datetime import datetime
import numpy as np
import torch
import gymnasium as gym
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback, CallbackList
import json


class BestModelConceptCallback(EvalCallback):
    """
    Custom EvalCallback that saves concept losses when best model is updated
    """
    def __init__(self, *args, concept_logging_callback=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.concept_logging_callback = concept_logging_callback
        self.prev_best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        result = super()._on_step()

        # ✅ Check if best model was updated (best_mean_reward changed)
        if self.best_mean_reward > self.prev_best_mean_reward:
            # Best model was just updated!
            self.prev_best_mean_reward = self.best_mean_reward

            # ✅ Get concept losses from ConceptLoggingCallback (from training rollout)
            if self.concept_logging_callback and self.concept_logging_callback.latest_concept_losses:
                L_ortho, L_spar, L_l1 = self.concept_logging_callback.latest_concept_losses

                # Save to file alongside best_model.zip
                concept_losses_path = os.path.join(self.best_model_save_path, 'best_model_concept_losses.json')
                concept_losses_data = {
                    'L_ortho': float(L_ortho),
                    'L_spar': float(L_spar),
                    'L_l1': float(L_l1),
                    'timestep': self.num_timesteps,
                    'mean_reward': float(self.best_mean_reward)
                }

                try:
                    with open(concept_losses_path, 'w') as f:
                        json.dump(concept_losses_data, f, indent=2)
                    print(f"✓ Saved concept losses to {concept_losses_path}")
                except Exception as e:
                    print(f"⚠ Failed to save concept losses: {e}")
            else:
                print(f"⚠ No concept losses available from training rollout")

        return result



from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.monitor import Monitor
from minigrid_features_extractor import MinigridFeaturesExtractor

# -----------------------
# Environment helpers
# -----------------------
def make_env(env_id, seed=0):
    def _init():
        env = gym.make(env_id, render_mode="rgb_array")
        env = ImgObsWrapper(env)
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init

def get_next_model_number(save_dir, prefix="ppo_concept_minigrid"):
    if not os.path.exists(save_dir):
        return 0
    import re
    pattern = re.compile(rf"{re.escape(prefix)}_(\d{{3}})\.zip")
    existing_numbers = [int(m.group(1)) for f in os.listdir(save_dir) if (m:=pattern.match(f))]
    return max(existing_numbers)+1 if existing_numbers else 0

# -----------------------
# Custom PPO class
# -----------------------
class PPOWithConcept(PPO):
    """
    PPO that adds concept layer losses to total loss
    """
    def __init__(self, *args, lambda_1=0.005, lambda_2=0.0015, lambda_3=0.00015, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3

    def train(self) -> None:
        """
        Override train() to add concept losses to PPO loss before backward
        """
        from gymnasium import spaces
        import torch.nn.functional as F

        # Switch to train mode
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        concept_losses_list = []  # Track concept losses

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy
                ratio = torch.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = torch.mean((torch.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                # Value loss
                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + torch.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss
                if entropy is None:
                    entropy_loss = -torch.mean(-log_prob)
                else:
                    entropy_loss = -torch.mean(entropy)
                entropy_losses.append(entropy_loss.item())

                # ✅ Standard PPO loss
                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # ✅ ADD concept losses
                concept_loss = torch.tensor(0.0, device=self.device)
                features_extractor = self.policy.features_extractor
                if hasattr(features_extractor, "last_concept_losses") and features_extractor.last_concept_losses is not None:
                    L_otho, L_spar, L_l1 = features_extractor.last_concept_losses
                    concept_loss = self.lambda_1 * L_otho + self.lambda_2 * L_spar + self.lambda_3 * L_l1
                    concept_losses_list.append(concept_loss.item())
                
                # ✅ Total loss with concept
                loss = loss + concept_loss

                # Calculate approximate KL divergence for early stopping
                with torch.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        # Logging
        from stable_baselines3.common.utils import explained_variance
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())
        
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self, "clip_range"):
            self.logger.record("train/clip_range", clip_range)
        
        # ✅ Log concept losses
        if concept_losses_list:
            # Chỉ log tổng concept loss (chi tiết được log bởi ConceptLoggingCallback)
            self.logger.record("train/concept_loss", np.mean(concept_losses_list))

# -----------------------
# Callbacks
# -----------------------
class ConceptLoggingCallback(BaseCallback):
    """
    Callback để log chi tiết các thành phần của concept loss và activation statistics
    Chạy sau mỗi rollout để tránh overhead cao
    """
    def __init__(self):
        super().__init__()
        self.best_concept_losses = None  # Track concept losses at best model
        self.latest_concept_losses = None  # Track latest concept losses from rollout

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        """
        Log chi tiết concept losses và statistics sau mỗi rollout
        """
        if hasattr(self.model.policy, "features_extractor"):
            extractor = self.model.policy.features_extractor
            
            # Chỉ log nếu concept distilling được bật
            if extractor.concept_distilling and hasattr(extractor, "last_concept_losses"):
                if extractor.last_concept_losses is not None:
                    L_otho, L_spar, L_l1 = extractor.last_concept_losses
                    
                    # ✅ Store latest losses (will be used by BestModelConceptCallback)
                    self.latest_concept_losses = (L_otho.item(), L_spar.item(), L_l1.item())
                    
                    # ✅ Log chi tiết từng component loss
                    self.logger.record("concept_detail/orthogonality_loss", L_otho.item())
                    self.logger.record("concept_detail/sparsity_loss", L_spar.item())
                    self.logger.record("concept_detail/l1_loss", L_l1.item())
                    
                    # ✅ Log lambda weights để dễ tune
                    self.logger.record("concept_detail/lambda_1_ortho", self.model.lambda_1)
                    self.logger.record("concept_detail/lambda_2_spar", self.model.lambda_2)
                    self.logger.record("concept_detail/lambda_3_l1", self.model.lambda_3)
                    
                    # ✅ Log weighted contributions
                    self.logger.record("concept_detail/weighted_ortho", (self.model.lambda_1 * L_otho).item())
                    self.logger.record("concept_detail/weighted_spar", (self.model.lambda_2 * L_spar).item())
                    self.logger.record("concept_detail/weighted_l1", (self.model.lambda_3 * L_l1).item())
                
                # ✅ Log concept activation statistics
                if hasattr(extractor, "concept_layer"):
                    try:
                        with torch.no_grad():
                            # Get last batch of observations from rollout buffer
                            obs = self.model.rollout_buffer.observations[-1]
                            if isinstance(obs, dict):
                                obs = obs["image"]
                            
                            # Forward through CNN and concept layer
                            features = extractor.cnn(obs).flatten(1)
                            concept_features = extractor.concept_layer(features)
                            
                            # Activation statistics
                            self.logger.record("concept_stats/mean_activation", concept_features.mean().item())
                            self.logger.record("concept_stats/std_activation", concept_features.std().item())
                            self.logger.record("concept_stats/max_activation", concept_features.max().item())
                            self.logger.record("concept_stats/min_activation", concept_features.min().item())
                            
                            # Sparsity metrics
                            sparsity_001 = (concept_features.abs() < 0.01).float().mean().item()
                            sparsity_01 = (concept_features.abs() < 0.1).float().mean().item()
                            self.logger.record("concept_stats/sparsity_ratio_0.01", sparsity_001)
                            self.logger.record("concept_stats/sparsity_ratio_0.1", sparsity_01)
                            
                            # Active concepts count
                            active_concepts = (concept_features.abs() > 0.1).float().sum(dim=1).mean().item()
                            self.logger.record("concept_stats/active_concepts_mean", active_concepts)
                            
                            # Weight statistics
                            weights = extractor.concept_layer.weight.data
                            self.logger.record("concept_weights/mean", weights.mean().item())
                            self.logger.record("concept_weights/std", weights.std().item())
                            self.logger.record("concept_weights/max_abs", weights.abs().max().item())
                            self.logger.record("concept_weights/l1_norm", weights.abs().mean().item())
                            
                    except Exception as e:
                        # Không crash nếu có lỗi trong logging
                        pass

# -----------------------
# Training function
# -----------------------
def train_ppo_concept(
        env_id="MiniGrid-Empty-5x5-v0",
        total_timesteps=100000,
        n_envs=4,
        n_concepts=4,
        seed=42,
        device="cuda",
        learning_rate=7e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        lambda_1=0.05,     # [Tăng] Orthogonality
        lambda_2=0.002,    # [Giữ nguyên] Sparsity
        lambda_3=0.01,     # [Tăng mạnh] L1
        is_trial=False,
        trial_number=None
):

    # Directories
    save_dir = f"models/{env_id}/ppo_concept"
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_dir = f"{save_dir}/last_train"
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)

    model_number = get_next_model_number(save_dir)
    model_name = f"ppo_concept_minigrid_{model_number:03d}"
    
    # ✅ For trials: save tensorboard directly to trials directory
    if is_trial and trial_number is not None:
        tensorboard_log = f"minigrid_tensorboard/{env_id}/ppo_concept/trials"
        run_name = f"trial_{trial_number:03d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        tensorboard_log = f"minigrid_tensorboard/{env_id}/ppo_concept"
        run_name = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    os.makedirs(tensorboard_log, exist_ok=True)

    # Environments
    train_env = DummyVecEnv([make_env(env_id, seed+i) for i in range(n_envs)]) if n_envs>1 else DummyVecEnv([make_env(env_id, seed)])
    train_env = VecMonitor(train_env)
    eval_env = DummyVecEnv([make_env(env_id, seed+1000)])
    eval_env = VecMonitor(eval_env)

    # Policy kwargs with concept distilling
    policy_kwargs = dict(
        features_extractor_class=MinigridFeaturesExtractor,
        features_extractor_kwargs=dict(
            features_dim=128,
            concept_distilling=True,
            n_concepts=n_concepts
        ),
        net_arch=dict(pi=[256,256], vf=[256,256])
    )

    # Instantiate PPO with concept
    model = PPOWithConcept(
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
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        tensorboard_log=tensorboard_log,
        verbose=1,
        seed=seed,
        device=device,
        lambda_1=lambda_1,
        lambda_2=lambda_2,
        lambda_3=lambda_3
    )

    # Callbacks
    concept_logging_cb = ConceptLoggingCallback()  # Log chi tiết concept losses
    checkpoint_cb = CheckpointCallback(save_freq=max(5000 // n_envs, 1), save_path=checkpoint_dir, name_prefix="ppo_concept_checkpoint")
    
    # ✅ Use custom callback that saves concept losses at best model
    eval_cb = BestModelConceptCallback(
        eval_env, 
        best_model_save_path=checkpoint_dir, 
        log_path=checkpoint_dir,
        eval_freq=max(5000 // n_envs, 1), 
        n_eval_episodes=10, 
        deterministic=True,
        concept_logging_callback=concept_logging_cb
    )
    callback = CallbackList([concept_logging_cb, checkpoint_cb, eval_cb])

    # Train
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True,
        tb_log_name=run_name
    )

    # ✅ Only save best model when NOT in trial mode
    if not is_trial:
        # Save best model
        best_model_in_last_train = f"{checkpoint_dir}/best_model.zip"
        if os.path.exists(best_model_in_last_train):
            shutil.copy2(best_model_in_last_train, f"{save_dir}/{model_name}.zip")
            print(f"✓ Best model saved to {save_dir}/{model_name}.zip")
        else:
            print(f"⚠ best_model.zip not found in {checkpoint_dir}/")
    else:
        # In trial mode: don't save final model
        print(f"ℹ Trial mode: skipping final model save (only tensorboard kept)")

    train_env.close()
    eval_env.close()
    return model

# -----------------------
# Alias for compatibility
# -----------------------
ConceptPPO = PPOWithConcept  # Alias for external imports

# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    train_ppo_concept(
        env_id="MiniGrid-Empty-5x5-v0",
        total_timesteps=100000,
        n_envs=4,
        seed=42,
        device="cuda",
        lambda_1=0.01,   # ví dụ tùy chỉnh khi gọi
        lambda_2=0.002,
        lambda_3=0.0002
    )