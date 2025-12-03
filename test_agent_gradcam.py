#!/usr/bin/env python3
"""
test_agent_gradcam.py

Full fixed version — corrected:
- compute_concept_gradcams now returns (orig_image, cams)
- generate_gradcam_for_best now unpacks correctly
- original frame image no longer black (use env frame instead of obs tensor)
"""

import os
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import gymnasium as gym

from PIL import Image
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO, DQN
from train_ppo_concept import ConceptPPO

# ============================================================
# Helpers
# ============================================================

def normalize_to_0_1(x, eps=1e-8):
    x_min = x.min()
    x_max = x.max()
    denom = (x_max - x_min) if (x_max - x_min) > eps else eps
    return (x - x_min) / denom

def heatmap_apply_colormap(cam: np.ndarray, cmap_name="jet"):
    cmap = plt.get_cmap(cmap_name)
    colored = cmap(cam)[:, :, :3]
    colored = (colored * 255).astype(np.uint8)
    return colored


def create_concept_values_bar(concept_values, width=200, height=None):
    """
    Create a vertical bar chart showing concept neuron output values.
    Bar chart fills 60%+ of the image, with small title at bottom.
    
    Args:
        concept_values: numpy array of shape (K,) with concept values
        width: width of output image
        height: height of output image (if None, uses same as width)
    
    Returns:
        PIL Image of the bar chart
    """
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    
    K = len(concept_values)
    if height is None:
        height = width
    
    # Create figure with tight layout to maximize bar chart space
    fig = plt.figure(figsize=(width/100, height/100), dpi=100)
    
    # Create axes with position that leaves space only at bottom for title
    # [left, bottom, width, height] in figure coordinates (0-1)
    ax = fig.add_axes([0.15, 0.12, 0.75, 0.80])  # Bar chart takes 80% height, 75% width
    
    # Create horizontal bar chart
    y_pos = np.arange(K)
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, K))
    
    # Create bars with good thickness
    bars = ax.barh(y_pos, concept_values, height=0.7, color=colors, edgecolor='black', linewidth=1.0)
    
    # Styling - maximize bar chart area
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f'C{i+1}' for i in range(K)], fontsize=10, fontweight='bold')
    ax.set_xlim(0, 1.0)  # Assuming sigmoid output [0, 1]
    ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Remove spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add value labels - always black text with white background for visibility
    for i, (bar, v) in enumerate(zip(bars, concept_values)):
        # Position text to the right of bar for consistency
        x_pos = v + 0.02
        ha = 'left'
        
        # Use exact y position of bar center
        y_pos = i
        
        # Always use black text with white background box for maximum visibility
        ax.text(x_pos, y_pos, f'{v:.3f}', 
                va='center', ha=ha, 
                fontsize=9, fontweight='bold', 
                color='black',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', linewidth=0.5))
    
    # Add small title at the very bottom of figure (outside axes)
    fig.text(0.5, 0.02, 'Concept Values', ha='center', va='bottom', fontsize=8, fontstyle='italic')
    
    # Convert to PIL Image
    fig.canvas.draw()
    
    # Use buffer_rgba() for newer matplotlib or tostring_rgb() for older
    try:
        buf = fig.canvas.buffer_rgba()
        img_array = np.asarray(buf)[:, :, :3]  # Drop alpha channel
    except AttributeError:
        # Fallback for older matplotlib
        img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close(fig)
    
    return Image.fromarray(img_array)


# ============================================================
# GradCAM
# ============================================================

def compute_concept_gradcams(features_extractor, obs_tensor, device):
    """
    RETURNS:
        img_rgb:       HxWx3 uint8 original (from obs_tensor)
        cams_list:     list[K] of Hc x Wc floats in [0,1]
        concept_values: numpy array of K concept neuron outputs (floats)
    """
    features_extractor.to(device)
    features_extractor.eval()

    obs_tensor = obs_tensor.float().to(device)

    # CNN forward
    cnn_out = features_extractor.cnn(obs_tensor)

    # concept maps + sigmoid
    concept_map, concept_vector = features_extractor.concept_layer(cnn_out)
    concept_map.retain_grad()

    B, K, Hc, Wc = concept_map.shape
    cams = []

    for k in range(K):
        features_extractor.zero_grad()
        if concept_map.grad is not None:
            concept_map.grad.zero_()

        target = concept_vector[0, k]
        target.backward(retain_graph=True)

        grad_k = concept_map.grad[0, k]
        alpha_k = grad_k.mean()

        fmap_k = concept_map[0, k].detach()
        cam = F.relu(alpha_k * fmap_k)

        cam_np = cam.cpu().numpy().astype(np.float32)
        cam_np = normalize_to_0_1(cam_np)
        cams.append(cam_np)

    # ---------------------------------------------------------
    # Extract concept vector values (K outputs)
    # ---------------------------------------------------------
    # concept_vector should be [B, K] but let's verify and handle edge cases
    concept_values = concept_vector[0].detach().cpu().numpy()  # Shape: (K,) or more?
    
    # DEBUG: Check if we have exactly K values
    # If concept_values has more than K elements, it might be flattened incorrectly
    # Take only first K values to match K concept maps
    if len(concept_values) > K:
        print(f"WARNING: concept_vector has {len(concept_values)} values but K={K}")
        print(f"  concept_vector shape: {concept_vector.shape}")
        print(f"  Taking only first {K} values")
        concept_values = concept_values[:K]
    elif len(concept_values) < K:
        print(f"ERROR: concept_vector has only {len(concept_values)} values but K={K}")
        # Pad with zeros if needed
        concept_values = np.pad(concept_values, (0, K - len(concept_values)))
    
    # Ensure concept_values is 1D array of length K
    concept_values = concept_values.flatten()[:K]

    # ---------------------------------------------------------
    # Build original RGB image (from obs_tensor)
    # ---------------------------------------------------------
    obs_cpu = obs_tensor.detach().cpu().squeeze(0)

    if obs_cpu.shape[0] == 3:
        img = np.transpose(obs_cpu.numpy(), (1, 2, 0))
    else:
        gray = obs_cpu.numpy()
        img = np.stack([gray, gray, gray], axis=-1)

    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    else:
        img = img.astype(np.uint8)

    return img, cams, concept_values


# ============================================================
# Composite output
# ============================================================

def composite_and_save(img_rgb_uint8, cams_list, concept_values, out_path, cmap="jet", spacing=5, target_size=300):
    """
    Create composite image: [Original] [Heatmap1] [Heatmap2] ... [HeatmapK] [Bar Chart]
    
    Args:
        img_rgb_uint8: Original image (H, W, 3)
        cams_list: List of K heatmaps
        concept_values: Numpy array of K concept values
        out_path: Output file path
        cmap: Colormap for heatmaps
        spacing: Pixels of white space between columns (default: 5)
        target_size: Target size for each panel (default: 300) - larger = less blur
    """
    # Use target_size for all panels instead of original size
    W = target_size
    H = target_size
    
    # Resize original image to target size
    orig = Image.fromarray(img_rgb_uint8).resize((W, H), Image.LANCZOS)

    # Create heatmap images at target size
    cam_ims = []
    for cam in cams_list:
        cam_rgb = heatmap_apply_colormap(cam)
        cam_rgb = Image.fromarray(cam_rgb).resize((W, H), Image.LANCZOS)
        cam_ims.append(cam_rgb)

    # Create bar chart at a fixed large size (e.g., 400x400) to maintain quality
    # Then resize to target size - this preserves the layout better than creating at small size
    FIXED_SIZE = 400
    bar_chart_large = create_concept_values_bar(concept_values, width=FIXED_SIZE, height=FIXED_SIZE)
    
    # Resize to target size using high-quality resampling
    bar_chart = bar_chart_large.resize((W, H), Image.LANCZOS)

    # Total width: original + K heatmaps + bar chart + spacing between each
    num_panels = 1 + len(cam_ims) + 1
    total_w = W * num_panels + spacing * (num_panels - 1)
    canvas = Image.new("RGB", (total_w, H), color=(255, 255, 255))  # White background

    # Paste images with spacing
    x = 0
    canvas.paste(orig, (x, 0))
    x += W + spacing

    for c in cam_ims:
        canvas.paste(c, (x, 0))
        x += W + spacing
    
    # Paste bar chart at the end
    canvas.paste(bar_chart, (x, 0))

    canvas.save(out_path)


# ============================================================
# Run episodes & pick best
# ============================================================

def run_and_collect_best_episode(model_path,
                                 env_id="MiniGrid-Empty-5x5-v0",
                                 algorithm="PPO_CONCEPT",
                                 num_episodes=10,
                                 deterministic=True,
                                 device="cpu",
                                 out_dir="gradcam_out",
                                 max_steps=1000):

    env = gym.make(env_id, render_mode="rgb_array")
    env = ImgObsWrapper(env)

    if algorithm.upper() == "PPO_CONCEPT":
        model = ConceptPPO.load(model_path, env=env, device=device)
    elif algorithm.upper().startswith("PPO"):
        model = PPO.load(model_path, env=env, device=device)
    elif algorithm.upper() == "DQN":
        model = DQN.load(model_path, env=env, device=device)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    best_reward = -1e9
    best_obs = None
    best_frames = None

    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        obs_list = []
        frame_list = []
        steps = 0

        # ✅ Capture initial state BEFORE any step
        obs_list.append(np.array(obs))
        frame_list.append(env.unwrapped.get_frame())

        while not done and steps < max_steps:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, _ = env.step(action)

            # ✅ Capture state AFTER each step (including final state)
            obs_list.append(np.array(obs))
            frame_list.append(env.unwrapped.get_frame())

            ep_reward += reward
            done = terminated or truncated
            steps += 1

        print(f"Episode {ep+1}: reward={ep_reward:.3f}, steps={steps}, frames={len(frame_list)}")

        if ep_reward > best_reward:
            best_reward = ep_reward
            best_obs = obs_list
            best_frames = frame_list

    env.close()
    os.makedirs(out_dir, exist_ok=True)

    # ✅ Verify that we captured all frames correctly
    if best_obs is not None and best_frames is not None:
        print(f"\nBest episode: {len(best_obs)} observations, {len(best_frames)} frames")
        if len(best_obs) != len(best_frames):
            print(f"⚠️  WARNING: Mismatch between obs and frames!")
        else:
            print(f"✓ Frame count verified: {len(best_frames)} frames for {len(best_frames)-1} steps")

    return model, best_obs, best_frames, best_reward, out_dir


# ============================================================
# GradCAM generator for best episode
# ============================================================

def generate_gradcam_for_best(model, best_obs, frames, out_dir, device="cpu", fps=6):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(out_dir, f"gradcam_{timestamp}")
    png_dir = os.path.join(run_dir, "frames")

    os.makedirs(png_dir, exist_ok=True)

    fx = model.policy.features_extractor
    fx.to(device)
    fx.eval()

    saved = []

    print(f"\nGenerating GradCAM for {len(best_obs)} frames...")
    
    for i, obs_raw in enumerate(best_obs):
        obs_np = np.array(obs_raw)

        if obs_np.ndim == 3 and obs_np.shape[-1] == 3:
            obs_np = np.transpose(obs_np, (2, 0, 1))

        obs_t = torch.from_numpy(obs_np).float().unsqueeze(0)

        try:
            img_input, cams, concept_vals = compute_concept_gradcams(fx, obs_t, device)
        except Exception as e:
            print("GradCAM error:", e)
            raise

        # IMPORTANT:
        # Use frame from env, not obs, to avoid black obs
        orig = frames[i]

        out_file = os.path.join(png_dir, f"frame_{i:04d}.png")
        composite_and_save(orig, cams, concept_vals, out_file)
        saved.append(out_file)

        if (i+1) % 10 == 0:
            print(f"Saved {i+1}/{len(best_obs)}")

    print(f"✓ Saved all {len(saved)} frames (frame_0000.png to frame_{len(saved)-1:04d}.png)")

    # Make video
    try:
        from moviepy import ImageSequenceClip
        print(f"\nCreating video from {len(saved)} frames at {fps} FPS...")
        clip = ImageSequenceClip(saved, fps=fps)
        vid_path = os.path.join(run_dir, "gradcam_episode.mp4")
        clip.write_videofile(vid_path, codec="libx264", audio=False, logger=None)
        print(f"✓ Video saved: {vid_path}")
    except Exception as e:
        vid_path = None
        print(f"⚠️  Warning: Could not create video: {e}")

    return run_dir


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--env", default="MiniGrid-Empty-5x5-v0")
    parser.add_argument("--algo", default="PPO_CONCEPT")
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--outdir", default="gradcam_out")
    parser.add_argument("--fps", type=int, default=6)

    args = parser.parse_args()

    model, best_obs, frames, best_reward, out_dir = run_and_collect_best_episode(
        args.model, args.env, args.algo,
        args.episodes, True, args.device, args.outdir
    )

    print(f"\nBest reward = {best_reward}. Running Grad-CAM...")

    run_dir = generate_gradcam_for_best(
        model, best_obs, frames, out_dir,
        device=args.device, fps=args.fps
    )

    print("\nDONE. Results saved in:", run_dir)


if __name__ == "__main__":
    main()
