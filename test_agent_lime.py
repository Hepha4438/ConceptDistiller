#!/usr/bin/env python3
"""
test_agent_lime.py

LIME (Local Interpretable Model-agnostic Explanations) for MiniGrid agents.
Uses LIME to explain which parts of the observation contribute to the agent's decision.
"""

import os
import argparse
from datetime import datetime

import numpy as np
import torch

# Set matplotlib backend before importing pyplot to avoid GUI issues in threads
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import gymnasium as gym

from PIL import Image
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO, DQN
from train_ppo_concept import ConceptPPO

# Try to import lime
try:
    from lime import lime_image
    from skimage.segmentation import mark_boundaries
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("⚠ WARNING: lime-ml not available. Install with: pip install lime")


def rotate_obs_to_god_view(obs, agent_dir):
    """
    Rotate observation to align with God View.
    
    In MiniGrid:
    - Model Input: Agent always faces UP (ego-centric)
    - God View: Agent can face any direction (world-centric)
    - agent_dir: 0=Right, 1=Down, 2=Left, 3=Up
    
    NEW ROTATION RULE (discovered empirically):
    Only use rot90 (rotate) and fliplr (horizontal flip)
    
    Args:
        obs: observation array (H, W, C) or (H, W)
        agent_dir: agent direction (0-3)
    
    Returns:
        rotated observation (H, W, C) or (H, W)
    """
    if agent_dir == 0:  # Facing Right
        result = np.fliplr(obs)     # Only horizontal flip
        return result
    elif agent_dir == 1:  # Facing Down
        result = np.rot90(obs, k=1)   # 90° CCW
        result = np.fliplr(result)     # Then horizontal flip
        return result
    elif agent_dir == 2:  # Facing Left
        result = np.rot90(obs, k=2)    # 180° rotation
        result = np.fliplr(result)     # Then horizontal flip
        return result
    elif agent_dir == 3:  # Facing Up
        result = np.rot90(obs, k=-1)   # 90° CW
        result = np.fliplr(result)     # Then horizontal flip
        return result
    else:
        return obs


def get_predict_fn(model, device):
    """
    Create a prediction function for LIME.
    LIME expects a function that takes (n_samples, height, width, channels) and returns (n_samples, n_classes)
    """
    def predict_fn(images):
        """
        Args:
            images: numpy array of shape (n_samples, H, W, C) in [0, 255] range
                   (scaled from MiniGrid's [0-5] range for better LIME visualization)
        Returns:
            probabilities: numpy array of shape (n_samples, n_actions)
        """
        # Debug: print input shape
        if images.shape[0] <= 2:  # Only print for first few calls
            print(f"[DEBUG] predict_fn input shape: {images.shape}, range: [{images.min()}, {images.max()}]")
        
        # Scale back to [0-5] range (MiniGrid's actual encoding)
        # [0-255] -> [0-5]
        images_rescaled = (images.astype(np.float32) / 51).astype(np.uint8)  # 255/51 ≈ 5
        
        if images.shape[0] <= 2:
            print(f"[DEBUG] After rescaling: range [{images_rescaled.min()}, {images_rescaled.max()}]")
        
        # Now images_rescaled is in correct format: (N, H, W, C) uint8 [0-5]
        # This matches what model.predict() expects from ImgObsWrapper
        
        # Get probabilities for each sample
        probs_list = []
        for i in range(images_rescaled.shape[0]):
            obs = images_rescaled[i]  # (H, W, C) uint8 [0-5]
            
            # Get action probabilities from policy
            with torch.no_grad():
                # Convert obs to tensor: (H,W,C) -> (C,H,W) for PyTorch
                obs_tensor = torch.from_numpy(obs).float().to(device)
                obs_tensor = obs_tensor.permute(2, 0, 1)  # (H,W,C) -> (C,H,W)
                obs_tensor = obs_tensor.unsqueeze(0)  # Add batch: (C,H,W) -> (1,C,H,W)
                
                if hasattr(model, 'policy'):
                    # PPO or ConceptPPO
                    distribution = model.policy.get_distribution(obs_tensor)
                    probs = torch.softmax(distribution.distribution.logits, dim=-1)
                else:
                    # DQN
                    q_values = model.q_net(obs_tensor)
                    probs = torch.softmax(q_values, dim=-1)
                
                probs_list.append(probs.cpu().numpy()[0])
        
        return np.array(probs_list)
    
    return predict_fn


def explain_with_lime(obs_rgb, model, device, num_samples=1000, num_features=10):
    """
    Generate LIME explanation for a single observation.
    
    Args:
        obs_rgb: numpy array (H, W, C) in [0, 255] range
        model: trained model
        device: torch device
        num_samples: number of perturbed samples for LIME
        num_features: number of top features to show
    
    Returns:
        explanation: LIME explanation object
        mask: binary mask showing important regions
    """
    if not LIME_AVAILABLE:
        raise ImportError("LIME not available. Install with: pip install lime")
    
    # Debug: print observation shape
    print(f"[DEBUG] obs_rgb shape: {obs_rgb.shape}, dtype: {obs_rgb.dtype}, range: [{obs_rgb.min()}, {obs_rgb.max()}]")
    
    # Create LIME explainer for images
    explainer = lime_image.LimeImageExplainer()
    
    # Get prediction function
    predict_fn = get_predict_fn(model, device)
    
    # Get the predicted action first
    probs = predict_fn(obs_rgb[np.newaxis, ...])[0]
    predicted_action = int(np.argmax(probs))
    
    # Generate explanation for the predicted action specifically
    # obs_rgb should be (H, W, C) in [0, 255] range
    explanation = explainer.explain_instance(
        obs_rgb.astype(np.uint8),
        predict_fn,
        top_labels=1,  # Only explain the predicted action
        labels=[predicted_action],  # Specify which action to explain
        hide_color=0,  # Color to use for hidden regions
        num_samples=num_samples,
        batch_size=32
    )
    
    # Get explanation for the predicted action
    try:
        temp, mask = explanation.get_image_and_mask(
            predicted_action,
            positive_only=True,
            num_features=num_features,
            hide_rest=False
        )
    except KeyError:
        # If action not in explanation, try to get available labels
        available_labels = list(explanation.local_exp.keys())
        if available_labels:
            # Use first available label
            label = available_labels[0]
            print(f"[WARNING] Action {predicted_action} not in explanation, using label {label}")
            temp, mask = explanation.get_image_and_mask(
                label,
                positive_only=True,
                num_features=num_features,
                hide_rest=False
            )
        else:
            # Create empty mask as fallback
            print(f"[WARNING] No labels in explanation, creating empty mask")
            temp = obs_rgb.copy()
            mask = np.zeros(obs_rgb.shape[:2], dtype=bool)
    
    return explanation, mask, predicted_action, probs


def create_lime_heatmap(explanation, action):
    """
    Create a heatmap from LIME explanation weights.
    
    Args:
        explanation: LIME explanation object
        action: Action index to explain
        
    Returns:
        heatmap: 2D numpy array of weights
    """
    # Get segments (superpixel mask)
    segments = explanation.segments
    
    # Get weights for the specific action
    # local_exp is a list of (segment_id, weight) tuples
    if action not in explanation.local_exp:
        raise KeyError(f"Action {action} not in explanation")
        
    exp = explanation.local_exp[action]
    
    # Create heatmap initialized with zeros
    heatmap = np.zeros(segments.shape, dtype=np.float32)
    
    # Fill heatmap with weights
    for seg_id, weight in exp:
        mask = (segments == seg_id)
        heatmap[mask] = weight
        
    return heatmap


def visualize_lime_explanation(obs_rgb, explanation, mask, action, probs, save_path=None, frame_rgb=None, agent_dir=None):
    """
    Visualize LIME explanation.
    
    Args:
        obs_rgb: original observation (H, W, C) in [0, 255] - what model sees (7x7x3) ego-centric
        explanation: LIME explanation object
        mask: binary mask of important regions (ego-centric)
        action: predicted action
        probs: action probabilities
        save_path: path to save the visualization
        frame_rgb: optional rendered game frame (H, W, C) for prettier display
        agent_dir: agent direction (0-3) for rotating obs_rgb and mask to align with God View
    """
    fig, axes = plt.subplots(1, 4 if frame_rgb is not None else 3, 
                            figsize=(20 if frame_rgb is not None else 15, 5))
    
    ax_idx = 0
    
    # Show rendered game frame if available (prettier)
    if frame_rgb is not None:
        axes[ax_idx].imshow(frame_rgb.astype(np.uint8))
        axes[ax_idx].set_title('God View\n(What human sees)', fontsize=12, fontweight='bold')
        axes[ax_idx].axis('off')
        ax_idx += 1
    
    # Rotate observation to align with God View if agent_dir provided
    if agent_dir is not None:
        obs_rgb_rotated = rotate_obs_to_god_view(obs_rgb, agent_dir)
    else:
        obs_rgb_rotated = obs_rgb
    
    # Model observation (rotated to match God View)
    axes[ax_idx].imshow(obs_rgb_rotated.astype(np.uint8))
    axes[ax_idx].set_title('Model Input (Aligned)\n(7x7 grid)', fontsize=12, fontweight='bold')
    axes[ax_idx].axis('off')
    ax_idx += 1
    
    # LIME Heatmap (Weights) - Apply same rotation as model input
    try:
        # Create heatmap from weights
        heatmap = create_lime_heatmap(explanation, action)
        
        # Rotate heatmap if agent_dir provided
        if agent_dir is not None:
            heatmap_rotated = rotate_obs_to_god_view(heatmap, agent_dir)
        else:
            heatmap_rotated = heatmap
            
        # Display heatmap
        # Use RdBu_r colormap (Red=Positive, Blue=Negative) centered at 0
        max_abs = np.max(np.abs(heatmap))
        if max_abs > 0:
            im = axes[ax_idx].imshow(heatmap_rotated, cmap='RdBu_r', 
                                    vmin=-max_abs, vmax=max_abs,
                                    interpolation='nearest')
        else:
            im = axes[ax_idx].imshow(heatmap_rotated, cmap='RdBu_r',
                                    interpolation='nearest')
        
        # Add colorbar with proper size
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(axes[ax_idx])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        
        # Keep axis visible for heatmap (don't call axis('off') here)
        axes[ax_idx].set_xticks([])
        axes[ax_idx].set_yticks([])
        
    except (KeyError, Exception) as e:
        # Fallback: show rotated original image
        axes[ax_idx].imshow(obs_rgb_rotated.astype(np.uint8))
        axes[ax_idx].text(0.5, 0.5, f'Error: {e}', ha='center', va='center', 
                    transform=axes[ax_idx].transAxes, color='red', fontsize=10)
        axes[ax_idx].set_xticks([])
        axes[ax_idx].set_yticks([])
        
    axes[ax_idx].set_title(f'LIME Heatmap (Weights)\nAction: {action}', fontsize=12, fontweight='bold')
    ax_idx += 1
    
    # LIME Overlay (Boundaries) - Apply same rotation as model input
    try:
        temp, mask_from_exp = explanation.get_image_and_mask(
            action,
            positive_only=True,
            num_features=10,
            hide_rest=False
        )
        # Rotate explanation and mask if agent_dir provided
        if agent_dir is not None:
            temp_rotated = rotate_obs_to_god_view(temp, agent_dir)
            mask_rotated = rotate_obs_to_god_view(mask_from_exp, agent_dir)
        else:
            temp_rotated = temp
            mask_rotated = mask_from_exp
        
        axes[ax_idx].imshow(mark_boundaries(temp_rotated / 255.0, mask_rotated))
    except (KeyError, Exception) as e:
        # Fallback: show rotated original image
        axes[ax_idx].imshow(obs_rgb_rotated.astype(np.uint8))
        axes[ax_idx].text(0.5, 0.5, f'Error: {e}', ha='center', va='center',
                    transform=axes[ax_idx].transAxes, color='red', fontsize=10)
    axes[ax_idx].set_title('LIME Overlay (Boundaries)', fontsize=12, fontweight='bold')
    axes[ax_idx].axis('off')
    
    # Add action probabilities as text
    action_names = ['Turn Left', 'Turn Right', 'Forward', 'Pickup', 'Drop', 'Toggle', 'Done']
    prob_text = '\n'.join([f'{name}: {p:.3f}' for name, p in zip(action_names[:len(probs)], probs)])
    fig.text(0.5, 0.02, prob_text, ha='center', fontsize=10, family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Use tight_layout with proper padding to avoid overlap
    plt.tight_layout(rect=[0, 0.08, 1, 0.98], pad=2.0, w_pad=1.5, h_pad=1.5)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
    else:
        plt.show()


def test_agent_lime(
    model_path,
    env_id="MiniGrid-Empty-5x5-v0",
    episodes=5,
    device="cpu",
    save_best_only=True,
    num_samples=1000,
    num_features=10,
    outdir=None,
    fps=2
):
    """
    Test agent and generate LIME explanations.
    
    Args:
        model_path: path to saved model
        env_id: environment ID
        episodes: number of episodes to run
        device: device to use
        save_best_only: if True, only save LIME for best episode
        num_samples: LIME num_samples parameter
        num_features: LIME num_features parameter
        outdir: output directory for LIME visualizations
        fps: frames per second for video (default: 2 for LIME, slower than GradCAM)
    """
    if not LIME_AVAILABLE:
        print("❌ ERROR: LIME not available. Install with: pip install lime")
        return
    
    device = torch.device(device)
    
    # Load model
    print(f"Loading model from {model_path}...")
    if "concept" in model_path.lower():
        model = ConceptPPO.load(model_path, device=device)
        algo = "ppo_concept"
    elif "ppo" in model_path.lower():
        model = PPO.load(model_path, device=device)
        algo = "ppo"
    elif "dqn" in model_path.lower():
        model = DQN.load(model_path, device=device)
        algo = "dqn"
    else:
        # Try to infer
        try:
            model = ConceptPPO.load(model_path, device=device)
            algo = "ppo_concept"
        except:
            try:
                model = PPO.load(model_path, device=device)
                algo = "ppo"
            except:
                model = DQN.load(model_path, device=device)
                algo = "dqn"
    
    # Set output directory
    if outdir is None:
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        outdir = f"lime_out/{env_id}/{algo}/{model_name}"
    
    os.makedirs(outdir, exist_ok=True)
    print(f"Output directory: {outdir}")
    
    # Create environment
    env = gym.make(env_id, render_mode="rgb_array")
    env = ImgObsWrapper(env)
    
    # Run episodes
    episode_rewards = []
    episode_data = []  # Store (episode_num, frames, rewards_per_step)
    
    for episode in range(episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        frames = []
        step = 0
        
        while not done:
            # obs from ImgObsWrapper is already (H, W, C) uint8 in range [0-5]
            # For LIME, we need [0-255] range for better visualization
            # Scale: [0-5] -> [0-255]
            obs_for_lime = (obs.astype(np.float32) * 51).astype(np.uint8)  # 5*51=255
            
            # Debug: print shapes on first step
            if step == 0:
                print(f"[DEBUG] First step - obs shape: {obs.shape}, range: [{obs.min()}, {obs.max()}]")
                print(f"[DEBUG] obs_for_lime shape: {obs_for_lime.shape}, range: [{obs_for_lime.min()}, {obs_for_lime.max()}]")
            
            # Get RGB frame and agent direction
            frame_rgb = env.unwrapped.get_frame()
            agent_dir = env.unwrapped.agent_dir
            
            frames.append((step, obs_for_lime.copy(), obs.copy(), frame_rgb.copy(), agent_dir))
            
            # Get action
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            step += 1
        
        episode_rewards.append(total_reward)
        episode_data.append((episode, frames, total_reward))
        print(f"Episode {episode + 1}/{episodes}: Reward = {total_reward:.3f}, Steps = {step}")
    
    env.close()
    
    # Determine which episodes to save
    if save_best_only:
        best_idx = np.argmax(episode_rewards)
        episodes_to_save = [best_idx]
        print(f"\n✓ Best episode: {best_idx + 1} with reward {episode_rewards[best_idx]:.3f}")
    else:
        episodes_to_save = range(len(episode_data))
    
    # Generate LIME explanations for selected episodes
    for ep_idx in episodes_to_save:
        episode_num, frames, total_reward = episode_data[ep_idx]
        
        if save_best_only:
            episode_dir = os.path.join(outdir, "best_run")
        else:
            episode_dir = os.path.join(outdir, f"episode_{episode_num:03d}")
        
        os.makedirs(episode_dir, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"Generating LIME explanations for episode {episode_num + 1}")
        print(f"Episode reward: {total_reward:.3f}")
        print(f"Number of frames: {len(frames)}")
        print(f"Output directory: {episode_dir}")
        print(f"{'='*60}\n")
        
        saved_frames = []
        
        for step, obs_for_lime, obs_original, frame_rgb, agent_dir in frames:
            # Generate LIME explanation
            try:
                print(f"  Processing step {step:04d}...", end=" ", flush=True)
                
                # Use obs_for_lime (the actual observation model sees) for LIME
                explanation, mask, action, probs = explain_with_lime(
                    obs_for_lime, model, device, num_samples, num_features
                )
                
                # Save visualization with both model obs and game render, rotated to align
                save_path = os.path.join(episode_dir, f"lime_step_{step:04d}.png")
                visualize_lime_explanation(obs_for_lime, explanation, mask, action, probs, 
                                         save_path, frame_rgb=frame_rgb, agent_dir=agent_dir)
                
                saved_frames.append(save_path)
                print(f"✓ Action: {action}, Saved: {os.path.basename(save_path)}")
            except Exception as e:
                print(f"❌ Error: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"  ✓ Saved {len(saved_frames)}/{len(frames)} frames")
        
        # Create video from saved frames
        if len(saved_frames) > 0:
            try:
                # Debug: print first few frame paths
                print(f"  📋 Frame paths sample:")
                for i, frame_path in enumerate(saved_frames[:3]):
                    exists = "✓" if os.path.exists(frame_path) else "✗"
                    print(f"     {exists} {frame_path}")
                if len(saved_frames) > 3:
                    print(f"     ... and {len(saved_frames) - 3} more frames")
                
                # Import moviepy and PIL
                try:
                    from moviepy import ImageSequenceClip
                    from PIL import Image
                    print(f"  ✓ moviepy and PIL imported successfully")
                except ImportError as e:
                    print(f"  ❌ Failed to import required libraries: {e}")
                    print(f"     Install with: pip install moviepy pillow")
                    raise
                
                # Check frame sizes and find max dimensions
                print(f"  📐 Checking frame sizes...")
                max_width = 0
                max_height = 0
                sizes = []
                for frame_path in saved_frames:
                    img = Image.open(frame_path)
                    w, h = img.size
                    sizes.append((w, h))
                    max_width = max(max_width, w)
                    max_height = max(max_height, h)
                    img.close()
                
                # Check if all frames have same size
                unique_sizes = set(sizes)
                if len(unique_sizes) > 1:
                    print(f"  ⚠ Found {len(unique_sizes)} different frame sizes: {unique_sizes}")
                    print(f"  🔧 Resizing all frames to: {max_width}x{max_height}")
                    
                    # Create temporary directory for resized frames
                    import tempfile
                    temp_dir = tempfile.mkdtemp(prefix="lime_video_")
                    resized_frames = []
                    
                    for i, frame_path in enumerate(saved_frames):
                        img = Image.open(frame_path)
                        
                        # Create new image with max size and white background
                        new_img = Image.new('RGB', (max_width, max_height), (255, 255, 255))
                        
                        # Paste original image (centered if needed)
                        x_offset = (max_width - img.size[0]) // 2
                        y_offset = (max_height - img.size[1]) // 2
                        new_img.paste(img, (x_offset, y_offset))
                        
                        # Save resized frame
                        resized_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
                        new_img.save(resized_path)
                        resized_frames.append(resized_path)
                        
                        img.close()
                        new_img.close()
                    
                    print(f"  ✓ Resized {len(resized_frames)} frames")
                    frames_to_use = resized_frames
                else:
                    print(f"  ✓ All frames have same size: {sizes[0]}")
                    frames_to_use = saved_frames
                
                print(f"  🎬 Creating video from {len(frames_to_use)} frames at {fps} fps...")
                
                # Create clip
                clip = ImageSequenceClip(frames_to_use, fps=fps)
                print(f"  ✓ Clip created: duration={clip.duration:.2f}s")
                
                # Write video
                video_path = os.path.join(episode_dir, "lime_episode.mp4")
                print(f"  ⏳ Writing video to: {video_path}")
                clip.write_videofile(video_path, codec="libx264", audio=False, logger=None)
                
                # Clean up temp directory if created
                if len(unique_sizes) > 1:
                    import shutil
                    shutil.rmtree(temp_dir)
                    print(f"  🗑️  Cleaned up temporary files")
                
                # Verify video was created
                if os.path.exists(video_path):
                    video_size = os.path.getsize(video_path)
                    print(f"  ✓ Video saved: {video_path} ({video_size:,} bytes)")
                else:
                    print(f"  ❌ Video file not created at: {video_path}")
                    
            except ImportError as e:
                print(f"  ⚠ Warning: Required libraries not available, skipping video creation")
                print(f"     Error: {e}")
                print(f"     Install with: pip install moviepy pillow")
            except Exception as e:
                import traceback
                print(f"  ❌ Error creating video: {e}")
                print(f"  Traceback:")
                traceback.print_exc()
        else:
            print(f"  ⚠ No frames saved, skipping video creation")
        
        print(f"✓ LIME explanations saved to: {episode_dir}")
    
    print(f"\n{'='*60}")
    print(f"LIME Analysis Complete!")
    print(f"Average reward: {np.mean(episode_rewards):.3f} ± {np.std(episode_rewards):.3f}")
    print(f"Output directory: {outdir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test agent with LIME explanations")
    parser.add_argument("--model", type=str, required=True, help="Path to saved model")
    parser.add_argument("--env", type=str, default="MiniGrid-Empty-5x5-v0", help="Environment ID")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to run")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"], help="Device")
    parser.add_argument("--save-all", action="store_true", help="Save LIME for all episodes (default: best only)")
    parser.add_argument("--num-samples", type=int, default=1000, help="LIME num_samples parameter")
    parser.add_argument("--num-features", type=int, default=10, help="LIME num_features parameter")
    parser.add_argument("--fps", type=int, default=2, help="FPS for video (default: 2)")
    parser.add_argument("--outdir", type=str, default=None, help="Output directory (default: auto)")
    
    args = parser.parse_args()
    
    test_agent_lime(
        model_path=args.model,
        env_id=args.env,
        episodes=args.episodes,
        device=args.device,
        save_best_only=not args.save_all,
        num_samples=args.num_samples,
        num_features=args.num_features,
        outdir=args.outdir,
        fps=args.fps
    )