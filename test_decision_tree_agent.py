"""
Test Decision Tree Policy Agent in MiniGrid Environment

Deploys trained DT policy: Observation → CNN (frozen) → Concepts → DT → Action
Compares performance with original MLP policy
"""

import numpy as np
import torch
import gymnasium as gym
import pickle
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple

from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO
from train_ppo_concept import ConceptPPO

from parse_concept_actions import ACTION_NAMES


class DecisionTreeAgent:
    """
    Agent that uses Decision Tree policy on concept vectors
    Reuses CNN from trained PPO model for feature extraction
    """
    
    def __init__(self, ppo_model_path: str, dt_model_path: str, device: str = "cpu", epsilon: float = 0.0):
        """
        Args:
            ppo_model_path: Path to trained PPO model (for CNN)
            dt_model_path: Path to trained DT policy
            device: torch device (cpu/cuda/mps)
            epsilon: Probability of random action (0.0 = no randomness, 0.1 = 10% random)
        """
        
        print(f"\n{'='*60}")
        print(f"🤖 Initializing Decision Tree Agent")
        print(f"{'='*60}")
        
        # Load DT policy FIRST to extract metadata and environment
        print(f"📂 Loading DT policy from: {dt_model_path}")
        with open(dt_model_path, 'rb') as f:
            dt_data = pickle.load(f)
            
        self.dt_model = dt_data['model']
        self.metadata = dt_data.get('metadata', {})
        self.action_names = dt_data.get('action_names', ACTION_NAMES)
        self.training_stats = dt_data.get('training_stats', {})
        
        env_id = self.metadata.get('environment', 'MiniGrid-DoorKey-6x6-v0')
        print(f"\n✓ DT Model loaded successfully! Environment: {env_id}")
        
        # Load PPO model (for CNN feature extraction)
        print(f"📂 Loading PPO model from: {ppo_model_path}")
        if 'posthoc' in str(ppo_model_path).lower():
            print("  Detected PostHoc concept model...")
            from train_concept_posthoc import load_posthoc_model
            # Create a dummy env since posthoc model requires it for observation_space
            dummy_env = ImgObsWrapper(gym.make(env_id))
            self.ppo_model = load_posthoc_model(ppo_model_path, dummy_env)
        else:
            self.ppo_model = ConceptPPO.load(ppo_model_path, device=device)
            
        self.device = device
        self.epsilon = epsilon  # For epsilon-greedy exploration
        
        print(f"\n✓ Model loaded successfully!")
        print(f"  Concept dimensions: {self.metadata.get('n_concepts', 'N/A')}")
        print(f"  Continuous concepts: {self.metadata.get('n_continuous_concepts', 'N/A')}")
        print(f"  Environment: {self.metadata.get('environment', 'N/A')}")
        
        # Print training statistics if available
        if self.training_stats:
            print(f"\n📊 Training Data Statistics:")
            print(f"  Total training states: {self.training_stats.get('total_states', 'N/A')}")
            print(f"  Unique training states: {self.training_stats.get('unique_states', 'N/A')}")
            print(f"  State diversity: {self.training_stats.get('state_diversity', 0)*100:.1f}%")
        
        # Verify concept extraction matches training format
        self._verify_concept_extraction()
        
        print(f"{'='*60}\n")
    
    def _verify_concept_extraction(self):
        """Quick verification that concept extraction uses correct format (with STE)"""
        fx = self.ppo_model.policy.features_extractor
        
        # Check if model uses Mode 4/5 (with concept bottleneck) or Posthoc
        has_concept_bottleneck = hasattr(fx, 'last_concept_bottleneck')
        has_posthoc_concepts = hasattr(fx, 'last_concepts')
        
        if not has_concept_bottleneck and not has_posthoc_concepts:
            print(f"⚠️  Concept extraction: last_concept_bottleneck and last_concepts not available")
            print(f"    Model may not be using Mode 4/5 or Posthoc concept bottleneck")
            print(f"    → Expected success rate may be low")
            return
            
        print(f"✅ Concept extraction verification passed")
        
        # Posthoc models don't have concept_mode
        if has_posthoc_concepts:
            return
            
        concept_mode = getattr(fx, 'concept_mode', None)
        if concept_mode not in [4, 5]:
            print(f"\n⚠️  Model is using concept_mode={concept_mode}, expected Mode 4 or 5")
            print(f"    DT training expects Mode 5 (mixed continuous/binary with STE)")
            print(f"    → Performance may be degraded")
            return
        
        # Mode 5 check
        if concept_mode == 5:
            print(f"\n✅ Concept extraction: Using last_concept_bottleneck (Mode 5 with STE) - Correct!")
            print(f"    Binary concepts will be discrete {{0,1}} as in training data")
            n_continuous = getattr(fx, 'n_continuous_concepts', 1)
            n_concepts = getattr(fx, 'n_concepts', 'N/A')
            print(f"    Concept split: {n_continuous} continuous, {n_concepts-n_continuous if isinstance(n_concepts, int) else 'N/A'} binary (STE)")
        else:
            print(f"\n⚠️  Model is using Mode 4 (all sigmoid, no STE)")
            print(f"    Training data expects Mode 5 with binary STE concepts")
            print(f"    → May cause distribution mismatch")
    
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> Tuple[int, np.ndarray]:
        """
        Predict action from observation
        
        Args:
            observation: Environment observation (uint8, shape [H, W, C])
            deterministic: If True, use argmax; if False, sample from probabilities
        
        Returns:
            action: Selected action index
            concept_vector: Extracted concept vector
        """
        
        # Convert observation from [H, W, C] to [C, H, W] for PyTorch
        if observation.ndim == 3:
            observation = np.transpose(observation, (2, 0, 1))  # HWC -> CHW
        
        # Convert to torch tensor and normalize to float
        obs_tensor = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Extract features through CNN
        with torch.no_grad():
            features = self.ppo_model.policy.features_extractor(obs_tensor)
            
            # ✅ Use concept_bottleneck (with STE already applied in Mode 5)
            # This matches "Concept Vector (bottleneck (after STE))" from training logs
            fx = self.ppo_model.policy.features_extractor
            
            if hasattr(fx, 'last_concept_bottleneck') and fx.last_concept_bottleneck is not None:
                concept_vector = fx.last_concept_bottleneck.cpu().numpy().flatten()
            elif hasattr(fx, 'last_concepts') and fx.last_concepts is not None:
                concept_vector = fx.last_concepts.cpu().numpy().flatten()
            else:
                raise RuntimeError(
                    "Model does not have concept bottleneck (Mode 4/5 required) or Posthoc concepts!\n"
                    "DT policy requires a PPO model with concepts extracted.\n"
                )
            
            # ⚠️  CRITICAL: Round to .6f to match training data precision
            # Training data was logged with 6 decimal places
            # Without this, tree comparisons may go to wrong branches!
            concept_vector = np.round(concept_vector, decimals=6)
        
        # Epsilon-greedy: random action with probability epsilon
        if self.epsilon > 0 and np.random.random() < self.epsilon:
            # Random action from action space
            n_actions = len(self.dt_model.classes_)
            action = np.random.randint(0, n_actions)
            return int(action), concept_vector
        
        # Decision tree prediction
        if deterministic:
            action = self.dt_model.predict(concept_vector.reshape(1, -1))[0]
        else:
            # Sample from probabilities
            probs = self.dt_model.predict_proba(concept_vector.reshape(1, -1))[0]
            action = np.random.choice(len(probs), p=probs)
        
        return int(action), concept_vector


def test_agent(agent: DecisionTreeAgent,
              env_id: str,
              num_episodes: int = 10,
              render: bool = False,
              verbose: bool = True,
              log_dir: str = None) -> Dict:
    """
    Test agent in environment
    
    Args:
        log_dir: Optional directory to save episode logs (success_runs.txt, failed_runs.txt)
    
    Returns:
        results: Dictionary with episode statistics
    """
    
    print(f"\n{'='*60}")
    print(f"🎮 Testing Decision Tree Agent")
    print(f"{'='*60}")
    print(f"Environment: {env_id}")
    print(f"Episodes: {num_episodes}")
    print(f"Render: {render}")
    if log_dir:
        print(f"Log directory: {log_dir}")
    print(f"{'='*60}\n")
    
    # Create environment
    render_mode = "human" if render else None
    env = gym.make(env_id, render_mode=render_mode)
    env = ImgObsWrapper(env)
    
    # Statistics
    episode_rewards = []
    episode_lengths = []
    successes = []
    
    # Concept usage tracking
    concept_activations = []
    action_counts = {action: 0 for action in ACTION_NAMES.keys()}
    
    # Track unique concept states for distribution analysis
    concept_states_seen = []
    
    # Episode logging data
    success_episodes = []
    failed_episodes = []
    
    start_time = datetime.now()
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        episode_concepts = []
        episode_actions = []
        episode_steps = []  # Store (concept_vector, action) for logging
        
        while not done:
            # Get action from DT agent
            action, concept_vector = agent.predict(obs, deterministic=True)
            
            # Track
            episode_concepts.append(concept_vector)
            episode_actions.append(action)
            episode_steps.append((concept_vector.copy(), action))
            action_counts[action] += 1
            concept_states_seen.append(concept_vector)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1
        
        # Record episode statistics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Check success (environment-dependent)
        success = reward > 0  # Simple heuristic: positive final reward
        successes.append(success)
        
        # Store concept activations
        concept_activations.extend(episode_concepts)
        
        # Store episode data for logging
        episode_data = {
            'episode': episode + 1,
            'steps': episode_steps,
            'reward': episode_reward,
            'length': episode_length,
            'success': success
        }
        if success:
            success_episodes.append(episode_data)
        else:
            failed_episodes.append(episode_data)
        
        if verbose:
            status = "✓ SUCCESS" if success else "✗ FAILED"
            print(f"Episode {episode+1:3d}: Reward={episode_reward:6.2f}, "
                  f"Steps={episode_length:3d}, {status}")
    
    end_time = datetime.now()
    
    env.close()
    
    # Compute statistics
    avg_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    avg_length = np.mean(episode_lengths)
    std_length = np.std(episode_lengths)
    success_rate = (sum(successes) / len(successes)) * 100
    
    # Concept statistics
    concept_activations = np.array(concept_activations)
    concept_means = np.mean(concept_activations, axis=0)
    concept_stds = np.std(concept_activations, axis=0)
    
    # Analyze concept distribution
    concept_states_seen = np.array(concept_states_seen)
    unique_concept_states = len(np.unique(concept_states_seen, axis=0))
    total_concept_states = len(concept_states_seen)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"📊 Test Results Summary")
    print(f"{'='*60}")
    print(f"Episodes: {num_episodes}")
    print(f"Test time: {(end_time - start_time).total_seconds():.2f}s")
    print(f"\n🎯 Performance:")
    print(f"  Success rate:   {success_rate:6.2f}%")
    print(f"  Avg reward:     {avg_reward:6.2f} ± {std_reward:.2f}")
    print(f"  Avg steps:      {avg_length:6.2f} ± {std_length:.2f}")
    
    print(f"\n🔍 Distribution Analysis:")
    print(f"  Total states visited:     {total_concept_states}")
    print(f"  Unique concept states:    {unique_concept_states}")
    print(f"  State diversity ratio:    {unique_concept_states/total_concept_states*100:.1f}%")
    print(f"\n🎬 Action Distribution:")
    total_actions = sum(action_counts.values())
    for action_idx in sorted(action_counts.keys()):
        action_name = ACTION_NAMES.get(action_idx, f"Unknown_{action_idx}")
        count = action_counts[action_idx]
        percentage = (count / total_actions) * 100 if total_actions > 0 else 0
        print(f"  {action_name:<20s}: {count:5d} ({percentage:5.2f}%)")
    
    print(f"\n🧠 Concept Activation Summary:")
    n_concepts = len(concept_means)
    n_continuous = agent.metadata.get('n_continuous_concepts', 1)
    
    for i in range(n_concepts):
        concept_type = "continuous" if i < n_continuous else "binary"
        print(f"  C{i+1} ({concept_type:10s}): mean={concept_means[i]:.3f}, std={concept_stds[i]:.3f}")
    
    print(f"{'='*60}\n")
    
    results = {
        'env_id': env_id,
        'num_episodes': num_episodes,
        'success_rate': success_rate,
        'avg_reward': avg_reward,
        'std_reward': std_reward,
        'avg_length': avg_length,
        'std_length': std_length,
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'successes': successes,
        'action_counts': action_counts,
        'concept_means': concept_means.tolist(),
        'concept_stds': concept_stds.tolist(),
        'test_time': (end_time - start_time).total_seconds(),
        'unique_concept_states': unique_concept_states,
        'total_concept_states': total_concept_states,
        'concept_diversity_ratio': unique_concept_states/total_concept_states
    }
    
    # Write episode logs if requested
    if log_dir:
        from pathlib import Path
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        # Write success_runs.txt
        success_path = log_path / "success_runs.txt"
        with open(success_path, 'w') as f:
            f.write(f"DT Agent Test - Success Episodes\n")
            f.write(f"Environment: {env_id}\n")
            f.write(f"Total Episodes: {num_episodes}\n")
            f.write(f"Success Episodes: {len(success_episodes)}\n")
            f.write(f"Success Rate: {success_rate:.2f}%\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*70}\n\n")
            
            for ep_data in success_episodes:
                f.write(f"Episode {ep_data['episode']}: SUCCESS (Reward={ep_data['reward']:.2f}, Steps={ep_data['length']})\n")
                f.write(f"{'-'*70}\n")
                
                for step_idx, (concept_vec, action) in enumerate(ep_data['steps']):
                    # Format concept vector with 6 decimals (matching training precision)
                    concept_str = "[" + ", ".join([f"{v:.6f}" for v in concept_vec]) + "]"
                    action_name = ACTION_NAMES.get(action, f"UNKNOWN_{action}")
                    
                    f.write(f"Step {step_idx:4d}:\n")
                    f.write(f"  Concept Vector (bottleneck): {concept_str}\n")
                    f.write(f"  Action: {action} ({action_name})\n")
                    f.write("\n")
                
                f.write(f"{'-'*70}\n\n")
        
        print(f"✓ Saved success episodes to: {success_path}")
        
        # Write failed_runs.txt
        failed_path = log_path / "failed_runs.txt"
        with open(failed_path, 'w') as f:
            f.write(f"DT Agent Test - Failed Episodes\n")
            f.write(f"Environment: {env_id}\n")
            f.write(f"Total Episodes: {num_episodes}\n")
            f.write(f"Failed Episodes: {len(failed_episodes)}\n")
            f.write(f"Failure Rate: {100 - success_rate:.2f}%\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*70}\n\n")
            
            for ep_data in failed_episodes:
                f.write(f"Episode {ep_data['episode']}: FAILED (Reward={ep_data['reward']:.2f}, Steps={ep_data['length']})\n")
                f.write(f"{'-'*70}\n")
                
                for step_idx, (concept_vec, action) in enumerate(ep_data['steps']):
                    # Format concept vector with 6 decimals (matching training precision)
                    concept_str = "[" + ", ".join([f"{v:.6f}" for v in concept_vec]) + "]"
                    action_name = ACTION_NAMES.get(action, f"UNKNOWN_{action}")
                    
                    f.write(f"Step {step_idx:4d}:\n")
                    f.write(f"  Concept Vector (bottleneck): {concept_str}\n")
                    f.write(f"  Action: {action} ({action_name})\n")
                    f.write("\n")
                
                f.write(f"{'-'*70}\n\n")
        
        print(f"✓ Saved failed episodes to: {failed_path}")
    
    return results


def compare_with_mlp(dt_agent: DecisionTreeAgent,
                     ppo_model_path: str,
                     env_id: str,
                     num_episodes: int = 10,
                     log_dir: str = None) -> Dict:
    """
    Compare DT policy with original MLP policy
    
    Args:
        log_dir: Optional directory to save episode logs (success_runs.txt, failed_runs.txt)
    
    Returns:
        comparison: Dictionary with both results
    """
    
    print(f"\n{'='*60}")
    print(f"⚖️  Comparing DT Policy vs MLP Policy")
    print(f"{'='*60}\n")
    
    # Test DT policy
    print("Testing DT Policy...")
    dt_results = test_agent(dt_agent, env_id, num_episodes, render=False, verbose=False, log_dir=log_dir)
    
    # Test MLP policy
    print("\nTesting MLP Policy...")
    
    env = gym.make(env_id)
    env = ImgObsWrapper(env)
    
    if 'posthoc' in str(ppo_model_path).lower():
        from train_concept_posthoc import load_posthoc_model
        mlp_model = load_posthoc_model(ppo_model_path, env)
    else:
        mlp_model = ConceptPPO.load(ppo_model_path)
    
    mlp_rewards = []
    mlp_lengths = []
    mlp_successes = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        
        while not done:
            action, _ = mlp_model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1
        
        mlp_rewards.append(episode_reward)
        mlp_lengths.append(episode_length)
        mlp_successes.append(reward > 0)
    
    env.close()
    
    mlp_success_rate = (sum(mlp_successes) / len(mlp_successes)) * 100
    mlp_avg_reward = np.mean(mlp_rewards)
    mlp_avg_length = np.mean(mlp_lengths)
    
    # Print comparison
    print(f"\n{'='*60}")
    print(f"📊 Comparison Results")
    print(f"{'='*60}")
    print(f"{'Metric':<25s} {'DT Policy':>15s} {'MLP Policy':>15s} {'Difference':>15s}")
    print(f"{'-'*75}")
    
    # Success rate
    diff_success = dt_results['success_rate'] - mlp_success_rate
    print(f"{'Success Rate (%)':<25s} {dt_results['success_rate']:>15.2f} {mlp_success_rate:>15.2f} {diff_success:>+15.2f}")
    
    # Average reward
    diff_reward = dt_results['avg_reward'] - mlp_avg_reward
    print(f"{'Avg Reward':<25s} {dt_results['avg_reward']:>15.2f} {mlp_avg_reward:>15.2f} {diff_reward:>+15.2f}")
    
    # Average length
    diff_length = dt_results['avg_length'] - mlp_avg_length
    print(f"{'Avg Steps':<25s} {dt_results['avg_length']:>15.2f} {mlp_avg_length:>15.2f} {diff_length:>+15.2f}")
    
    # Performance ratio
    performance_ratio = (dt_results['success_rate'] / mlp_success_rate * 100) if mlp_success_rate > 0 else 0
    print(f"\n🎯 DT achieves {performance_ratio:.1f}% of MLP's success rate")
    
    # Diagnostic info
    print(f"\n🔍 DT Policy Diagnostics:")
    print(f"  States visited: {dt_results['total_concept_states']}")
    print(f"  Unique states:  {dt_results['unique_concept_states']}")
    print(f"  Diversity:      {dt_results['concept_diversity_ratio']*100:.1f}%")
    
    # Compare with training distribution if available
    if hasattr(dt_agent, 'training_stats') and dt_agent.training_stats:
        train_unique = dt_agent.training_stats.get('unique_states', 0)
        train_total = dt_agent.training_stats.get('total_states', 1)
        
        # Estimate how many test states were out-of-distribution
        # (this is a rough estimate since we can't directly match states)
        state_coverage_ratio = dt_results['unique_concept_states'] / train_unique if train_unique > 0 else 0
        
        print(f"\n📊 Training vs Test Distribution:")
        print(f"  Training unique states: {train_unique}")
        print(f"  Test unique states:     {dt_results['unique_concept_states']}")
        
        if state_coverage_ratio > 1.5:
            print(f"\n⚠️  WARNING: Test saw {state_coverage_ratio:.1f}x more unique states than training!")
            print(f"    DT is encountering many OUT-OF-DISTRIBUTION states")
            print(f"    → This causes compounding errors and poor performance")
            print(f"\n💡 Suggestions:")
            print(f"    1. Collect more diverse training data (increase failed_sample_ratio)")
            print(f"    2. Use DAgger: collect failed trajectories, label with MLP, retrain")
            print(f"    3. Increase tree max_depth to handle more state variations")
    
    # Check action distribution for issues
    total_actions = sum(dt_results['action_counts'].values())
    dominant_action = max(dt_results['action_counts'].items(), key=lambda x: x[1])
    dominant_pct = (dominant_action[1] / total_actions * 100) if total_actions > 0 else 0
    
    if dominant_pct > 70:
        print(f"\n⚠️  WARNING: DT is heavily biased toward action {ACTION_NAMES.get(dominant_action[0])} ({dominant_pct:.1f}%)")
        print(f"    This suggests the DT is not adapting to different states!")
        
        # Compare with training action distribution
        if hasattr(dt_agent, 'training_stats') and dt_agent.training_stats:
            train_action_dist = dt_agent.training_stats.get('action_distribution', {})
            train_total_states = dt_agent.training_stats.get('total_states', 1)
            if train_action_dist:
                print(f"\n    Training action distribution:")
                for action_idx, count in sorted(train_action_dist.items()):
                    train_pct = count / train_total_states * 100
                    test_count = dt_results['action_counts'].get(action_idx, 0)
                    test_pct = test_count / total_actions * 100 if total_actions > 0 else 0
                    diff = test_pct - train_pct
                    print(f"      {ACTION_NAMES.get(action_idx)}: train={train_pct:.1f}%, test={test_pct:.1f}%, diff={diff:+.1f}%")
    
    print(f"{'='*60}\n")
    
    comparison = {
        'dt_results': dt_results,
        'mlp_success_rate': mlp_success_rate,
        'mlp_avg_reward': mlp_avg_reward,
        'mlp_avg_length': mlp_avg_length,
        'performance_ratio': performance_ratio
    }
    
    return comparison


def verify_dt_predictions(agent: DecisionTreeAgent, model_path: str):
    """
    Verify DT predictions match training accuracy on test data
    Checks for discrepancies between training metrics and actual predictions
    """
    print(f"\n{'='*60}")
    print(f"🔬 Verifying DT Predictions Against Training Data")
    print(f"{'='*60}\n")
    
    # Load training/test data from model
    model_dir = Path(model_path).parent
    training_data_file = model_dir / "dt_training_data.pkl"
    
    if not training_data_file.exists():
        print(f"❌ Training data not found: {training_data_file}")
        print(f"   Cannot verify predictions without original training data")
        return
    
    # Load data
    with open(training_data_file, 'rb') as f:
        data = pickle.load(f)
    
    X = data['X']
    y_true = data['y']
    
    # Sample subset for verification
    n_samples = min(500, len(X))
    indices = np.random.choice(len(X), n_samples, replace=False)
    X_sample = X[indices]
    y_sample = y_true[indices]
    
    print(f"Verifying on {n_samples} samples from training data...")
    
    # Get DT predictions
    y_pred = agent.dt_model.predict(X_sample)
    
    # Overall agreement
    agreement_rate = np.mean(y_pred == y_sample)
    
    print(f"\n📊 Prediction Agreement:")
    print(f"  DT vs True Actions: {agreement_rate:.2%}")
    print(f"  Expected (from training accuracy): ~99%")
    
    if agreement_rate < 0.95:
        print(f"\n⚠️  WARNING: Agreement rate significantly below training accuracy!")
        print(f"     This suggests an issue with model serialization or concept extraction.")
    
    # Per-action breakdown
    print(f"\n📋 Per-Action Agreement:")
    print(f"{'Action':<20s} {'Total':<10s} {'Correct':<10s} {'Agreement':<12s}")
    print(f"{'-'*60}")
    
    action_agreements = {}
    for action_idx in np.unique(y_sample):
        mask = (y_sample == action_idx)
        if mask.sum() > 0:
            action_name = ACTION_NAMES.get(action_idx, f"Unknown_{action_idx}")
            total = mask.sum()
            correct = np.sum(y_pred[mask] == y_sample[mask])
            agreement = correct / total
            action_agreements[action_idx] = agreement
            
            status = "✓" if agreement > 0.9 else ("⚠️" if agreement > 0.7 else "❌")
            print(f"{action_name:<20s} {total:<10d} {correct:<10d} {agreement:<11.2%} {status}")
    
    # Check for severe discrepancies
    min_agreement = min(action_agreements.values()) if action_agreements else 1.0
    if min_agreement < 0.5:
        print(f"\n❌ CRITICAL: Some actions have <50% agreement!")
        print(f"   Model is not reproducing training behavior correctly.")
        print(f"\n💡 Possible causes:")
        print(f"   1. Concept extraction differs between training and inference")
        print(f"   2. Data preprocessing inconsistency")
        print(f"   3. Model loaded incorrectly")
    elif min_agreement < 0.8:
        print(f"\n⚠️  Some actions have low agreement (<80%)")
        print(f"   Rare actions may not be learned well due to class imbalance.")
    else:
        print(f"\n✅ All actions have good agreement (>80%)")
        print(f"   Model is correctly reproducing training behavior.")
    
    print(f"{'='*60}\n")


def analyze_distribution_shift(agent: DecisionTreeAgent, env_id: str, num_episodes: int = 100):
    """
    Analyze action distribution shift from training to test
    Identifies if DT is overusing certain actions in deployment
    """
    print(f"\n{'='*60}")
    print(f"📊 Analyzing Action Distribution Shift")
    print(f"{'='*60}\n")
    
    # Get training distribution
    if not hasattr(agent, 'training_stats') or not agent.training_stats:
        print(f"❌ Training statistics not available in model")
        return
    
    train_action_dist = agent.training_stats.get('action_distribution', {})
    train_total = agent.training_stats.get('total_states', 1)
    
    if not train_action_dist:
        print(f"❌ Training action distribution not found")
        return
    
    print(f"Training data: {train_total} total actions")
    print(f"\nTraining action distribution:")
    for action_idx, count in sorted(train_action_dist.items()):
        pct = count / train_total * 100
        print(f"  {ACTION_NAMES.get(action_idx, f'A{action_idx}'):<20s}: {count:>5d} ({pct:>5.1f}%)")
    
    # Test in environment
    print(f"\nTesting in {env_id} for {num_episodes} episodes...")
    test_results = test_agent(agent, env_id, num_episodes, render=False, verbose=False)
    
    test_action_counts = test_results['action_counts']
    test_total = sum(test_action_counts.values())
    
    # Compare distributions
    print(f"\n{'='*60}")
    print(f"📈 Distribution Comparison")
    print(f"{'='*60}")
    print(f"{'Action':<20s} {'Training %':>12s} {'Test %':>12s} {'Difference':>15s} {'Status':>8s}")
    print(f"{'-'*75}")
    
    large_shifts = []
    for action_idx in range(7):
        action_name = ACTION_NAMES.get(action_idx, f"A{action_idx}")
        
        train_count = train_action_dist.get(action_idx, 0)
        train_pct = train_count / train_total * 100
        
        test_count = test_action_counts.get(action_idx, 0)
        test_pct = test_count / test_total * 100 if test_total > 0 else 0
        
        diff = test_pct - train_pct
        
        # Determine status
        if abs(diff) > 15:
            status = "❌ LARGE"
            large_shifts.append((action_name, train_pct, test_pct, diff))
        elif abs(diff) > 8:
            status = "⚠️  MEDIUM"
        else:
            status = "✓ OK"
        
        print(f"{action_name:<20s} {train_pct:>11.1f}% {test_pct:>11.1f}% {diff:>+14.1f}% {status:>8s}")
    
    # Analysis
    print(f"\n{'='*60}")
    if large_shifts:
        print(f"⚠️  LARGE DISTRIBUTION SHIFTS DETECTED:")
        for action_name, train_pct, test_pct, diff in large_shifts:
            if diff > 0:
                print(f"  • {action_name}: OVERUSED by {diff:+.1f}% (train {train_pct:.1f}% → test {test_pct:.1f}%)")
            else:
                print(f"  • {action_name}: UNDERUSED by {diff:+.1f}% (train {train_pct:.1f}% → test {test_pct:.1f}%)")
        
        print(f"\n💡 Interpretation:")
        print(f"  DT is NOT deploying the same behavior it learned in training!")
        print(f"  This indicates:")
        print(f"  1. Test states differ from training states (out-of-distribution)")
        print(f"  2. DT generalizes poorly to unseen states")
        print(f"  3. Compounding errors lead to unexpected action patterns")
    else:
        print(f"✅ Distribution shifts are reasonable (<8% for all actions)")
        print(f"   DT is deploying similar behavior to training.")
    
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Test Decision Tree Policy Agent")
    parser.add_argument("--model", type=str, required=True,
                       help="Path to trained DT model (.pkl)")
    parser.add_argument("--ppo-model", type=str, required=True,
                       help="Path to trained PPO model (.zip) for CNN extraction")
    parser.add_argument("--env", type=str, default=None,
                       help="Environment ID (default: from metadata)")
    parser.add_argument("--episodes", type=int, default=10,
                       help="Number of test episodes (default: 10)")
    parser.add_argument("--render", action="store_true",
                       help="Render environment during testing")
    parser.add_argument("--compare-mlp", action="store_true",
                       help="Compare with MLP policy")
    parser.add_argument("--verify-predictions", action="store_true",
                       help="Verify DT predictions match training data")
    parser.add_argument("--analyze-distribution", action="store_true",
                       help="Analyze action distribution shift from training to test")
    parser.add_argument("--device", type=str, default="cpu",
                       choices=["cpu", "cuda", "mps"],
                       help="Device for torch (default: cpu)")
    parser.add_argument("--epsilon", type=float, default=0.0,
                       help="Epsilon-greedy: probability of random action (default: 0.0, recommended: 0.05-0.15)")
    parser.add_argument("--output", type=str, default=None,
                       help="Save results to pickle file")
    parser.add_argument("--log-dir", type=str, default=None,
                       help="Directory to save episode logs (success_runs.txt, failed_runs.txt)")
    
    args = parser.parse_args()
    
    # Initialize agent
    agent = DecisionTreeAgent(
        ppo_model_path=args.ppo_model,
        dt_model_path=args.model,
        device=args.device,
        epsilon=args.epsilon
    )
    
    if args.epsilon > 0:
        print(f"🎲 Using epsilon-greedy with ε={args.epsilon:.2f} ({args.epsilon*100:.0f}% random actions)")
        print(f"   This helps break out of action loops and explore better paths\n")
    
    # Get environment
    env_id = args.env if args.env else agent.metadata.get('environment', 'MiniGrid-DoorKey-6x6-v0')
    
    # Verification mode
    if args.verify_predictions:
        verify_dt_predictions(agent, args.model)
        return
    
    # Distribution analysis mode
    if args.analyze_distribution:
        analyze_distribution_shift(agent, env_id, args.episodes)
        return
    
    # Test agent
    if args.compare_mlp:
        results = compare_with_mlp(agent, args.ppo_model, env_id, args.episodes, log_dir=args.log_dir)
    else:
        results = test_agent(agent, env_id, args.episodes, args.render, verbose=True, log_dir=args.log_dir)
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'wb') as f:
            pickle.dump(results, f)
        print(f"✓ Saved results to: {output_path}")
    
    print(f"\n✅ Testing Complete!\n")


if __name__ == "__main__":
    main()
