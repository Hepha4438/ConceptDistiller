#!/usr/bin/env python3
"""
test_agent_ig.py

Integrated Gradients (IG) visualization for MiniGrid agents with concept-based models.
IG provides pixel-level attribution for better understanding of model decisions.
"""

import os
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

# Set matplotlib backend before importing pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import gymnasium as gym
from PIL import Image, ImageDraw, ImageFont
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO, DQN
from train_ppo_concept import ConceptPPO

# Try to import captum for Integrated Gradients
try:
    from captum.attr import IntegratedGradients
    CAPTUM_AVAILABLE = True
except ImportError:
    CAPTUM_AVAILABLE = False
    print("⚠ WARNING: captum not available. Install with: pip install captum")


# ============================================================
# Helpers (same as GradCAM)
# ============================================================

def rotate_obs_to_god_view(obs, agent_dir):
    """
    Rotate observation to align with God View.
    MUST match GradCAM rotation logic exactly!
    
    In MiniGrid:
    - Model Input: Agent always faces UP (ego-centric)
    - God View: Agent can face any direction (world-centric)
    - agent_dir: 0=Right, 1=Down, 2=Left, 3=Up
    
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
        result = np.flipud(obs)        # Vertical flip (FIXED: was rot90+fliplr)
        return result
    elif agent_dir == 3:  # Facing Up
        result = np.rot90(obs, k=-1)   # 90° CW
        result = np.fliplr(result)     # Then horizontal flip
        return result
    else:
        return obs


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
    """Create vertical bar chart for concept values."""
    import matplotlib
    matplotlib.use('Agg')
    
    K = len(concept_values)
    if height is None:
        height = width
    
    fig = plt.figure(figsize=(width/100, height/100), dpi=100)
    ax = fig.add_axes([0.15, 0.25, 0.7, 0.65])
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, K))
    bars = ax.bar(range(K), concept_values, color=colors, edgecolor='black', linewidth=1.5)
    
    ax.set_ylim([0, max(1.0, concept_values.max() * 1.1)])
    ax.set_xticks(range(K))
    ax.set_xticklabels([f'C{i+1}' for i in range(K)], fontsize=9, fontweight='bold')
    ax.set_ylabel('Activation', fontsize=10, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    for bar, val in zip(bars, concept_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}', ha='center', va='bottom', fontsize=7, fontweight='bold')
    
    fig.text(0.5, 0.08, 'Concept Values', ha='center', fontsize=9, fontweight='bold')
    
    # Draw the figure to ensure canvas is ready
    fig.canvas.draw()
    
    # Get image from canvas (compatible with newer matplotlib versions)
    try:
        # Method 1: Modern matplotlib (3.5+)
        img_array = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        img_array = img_array[:, :, :3]  # Remove alpha channel
    except (AttributeError, ValueError):
        try:
            # Method 2: Older matplotlib with tostring_rgb
            img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        except AttributeError:
            # Method 3: Fallback using PIL directly from canvas
            from io import BytesIO
            buf = BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
            buf.seek(0)
            img = Image.open(buf)
            img_array = np.array(img)
            buf.close()
            # Ensure RGB (remove alpha if present)
            if img_array.shape[-1] == 4:
                img_array = img_array[:, :, :3]
    
    plt.close(fig)
    return Image.fromarray(img_array)


# ============================================================
# Integrated Gradients
# ============================================================

def compute_concept_integrated_gradients(features_extractor, obs_tensor, device, n_steps=50):
    """
    Compute Integrated Gradients for concept-based model.
    
    Args:
        features_extractor: Model's feature extractor
        obs_tensor: Input observation tensor (1, C, H, W)
        device: torch device
        n_steps: Number of steps for IG approximation (default: 50)
    
    Returns:
        img_rgb: HxWx3 uint8 original image
        attributions_list: list[K] of attribution maps (H, W) in [0, 1]
        concept_values: numpy array of K concept neuron outputs
    """
    if not CAPTUM_AVAILABLE:
        raise ImportError("Captum not available. Install with: pip install captum")
    
    features_extractor.to(device)
    features_extractor.eval()
    
    # Ensure input is contiguous and float
    obs_tensor = obs_tensor.float().to(device).contiguous()
    
    print(f"DEBUG: obs_tensor shape = {obs_tensor.shape}, contiguous = {obs_tensor.is_contiguous()}")
    
    # Get concept values first
    with torch.no_grad():
        # For Mode 4/5: need to run full forward to get concept_bottleneck
        concept_mode = getattr(features_extractor, 'concept_mode', 1)
        
        if concept_mode in [4, 5]:
            # Run full forward pass to populate last_concept_bottleneck
            _ = features_extractor(obs_tensor)
            
            # Use concept bottleneck vector (what the policy actually sees)
            if hasattr(features_extractor, 'last_concept_bottleneck'):
                concept_vector_for_vis = features_extractor.last_concept_bottleneck
                print(f"DEBUG: Using concept_bottleneck from Mode {concept_mode}")
            else:
                # Fallback to ConceptLayer output
                cnn_out = features_extractor.cnn(obs_tensor)
                concept_map, concept_vector_for_vis = features_extractor.concept_layer(cnn_out)
                print(f"DEBUG: Fallback to ConceptLayer output")
        else:
            # Mode 1/2/3: use ConceptLayer output
            cnn_out = features_extractor.cnn(obs_tensor)
            concept_map, concept_vector_for_vis = features_extractor.concept_layer(cnn_out)
            print(f"DEBUG: Using ConceptLayer output for Mode {concept_mode}")
        
        # For heatmap computation, we still need concept_map from ConceptLayer
        cnn_out = features_extractor.cnn(obs_tensor)
        concept_map, _ = features_extractor.concept_layer(cnn_out)
    
    print(f"DEBUG: concept_map shape = {concept_map.shape}")
    print(f"DEBUG: concept_vector_for_vis shape = {concept_vector_for_vis.shape}")
    print(f"DEBUG: concept_vector_for_vis values = {concept_vector_for_vis[0].cpu().numpy()}")
    
    B, K, Hc, Wc = concept_map.shape
    
    # Create baseline - ensure it's contiguous
    # For MiniGrid: use BLACK BASELINE (all zeros)
    # This is standard in IG literature and represents "no information" state
    # Note: MiniGrid uses discrete encoding, so interpolation creates invalid states anyway
    # Black baseline is more interpretable than random mid-range values
    baseline = torch.zeros_like(obs_tensor).contiguous()
    
    print(f"DEBUG: Using BLACK baseline (all zeros), contiguous={baseline.is_contiguous()}")
    
    # Compute IG for each concept
    attributions_list = []
    
    for k in range(K):
        print(f"  Computing IG for concept {k+1}/{K}...", end="\r")
        
        # Create a simple wrapper model that returns scalar output
        class ConceptKWrapper(torch.nn.Module):
            def __init__(self, extractor, concept_idx, use_bottleneck=False):
                super().__init__()
                self.extractor = extractor
                self.k = concept_idx
                self.use_bottleneck = use_bottleneck
            
            def forward(self, x):
                # Aggressively ensure contiguity
                x = x.contiguous()
                
                if self.use_bottleneck:
                    # Mode 4/5: Use concept bottleneck (what policy actually uses)
                    # Run full forward to get concept_bottleneck
                    features = self.extractor(x)  # This populates last_concept_bottleneck
                    
                    if hasattr(self.extractor, 'last_concept_bottleneck'):
                        concept_vec = self.extractor.last_concept_bottleneck
                    else:
                        # Fallback
                        cnn_out = self.extractor.cnn(x).contiguous()
                        _, concept_vec = self.extractor.concept_layer(cnn_out)
                else:
                    # Mode 1/2/3: Use ConceptLayer output
                    cnn_out = self.extractor.cnn(x).contiguous()
                    _, concept_vec = self.extractor.concept_layer(cnn_out)
                
                concept_vec = concept_vec.contiguous()
                
                # Use index_select which creates new tensor (not a view)
                output = torch.index_select(concept_vec, 1, torch.tensor([self.k], device=concept_vec.device))
                output = output.squeeze(1)
                
                return output
        
        # Create wrapper - use bottleneck for Mode 4/5
        concept_mode = getattr(features_extractor, 'concept_mode', 1)
        use_bottleneck = (concept_mode in [4, 5])
        wrapper = ConceptKWrapper(features_extractor, k, use_bottleneck=use_bottleneck)
        wrapper.eval()
        
        # Initialize IG with wrapper
        ig = IntegratedGradients(wrapper)
        
        # Compute attributions
        try:
            attributions = ig.attribute(
                obs_tensor,
                baselines=baseline,
                n_steps=n_steps,
                return_convergence_delta=False,
                internal_batch_size=1  # Process one interpolation at a time to avoid view issues
            )
            
            # Process attributions - ensure contiguous before numpy conversion
            attributions = attributions.contiguous()
            attr_np = attributions.squeeze(0).cpu().detach().numpy()  # (C, H, W)
            
            # Debug: Print raw attribution stats
            print(f"\n    [DEBUG C{k+1}] Raw attribution: shape={attr_np.shape}, min={attr_np.min():.6f}, max={attr_np.max():.6f}, mean={attr_np.mean():.6f}")
            
            # Aggregate across channels
            # For IG, we want to see the most important regions
            # Use max of absolute values instead of sum to preserve signal strength
            if attr_np.ndim == 3:
                # Take absolute value first, then max across channels
                attr_np = np.abs(attr_np).max(axis=0)  # (H, W)
            
            # Debug: Print after aggregation
            print(f"    [DEBUG C{k+1}] After aggregation (max): min={attr_np.min():.6f}, max={attr_np.max():.6f}, mean={attr_np.mean():.6f}")
            
            # ✅ Special handling for Mode 5 STE concepts with zero attributions
            # If attribution is nearly zero but concept is active, use concept_map activation instead
            concept_mode = getattr(features_extractor, 'concept_mode', 1)
            if concept_mode == 5 and attr_np.max() < 1e-6:
                # Check if this concept is active in bottleneck
                with torch.no_grad():
                    _ = features_extractor(obs_tensor)
                    if hasattr(features_extractor, 'last_concept_bottleneck'):
                        bottleneck_value = features_extractor.last_concept_bottleneck[0, k].item()
                        
                        if bottleneck_value > 0.5:  # Concept is active
                            print(f"    [MODE 5 FALLBACK C{k+1}] Concept is active (value={bottleneck_value:.2f}) but IG≈0.")
                            print(f"                             Using concept_map activation instead of IG.")
                            
                            # Use concept_map activation as heatmap
                            cnn_out = features_extractor.cnn(obs_tensor)
                            concept_map_temp, _ = features_extractor.concept_layer(cnn_out)
                            activation_map = concept_map_temp[0, k].detach().cpu().numpy()
                            
                            # Normalize activation map
                            attr_np = normalize_to_0_1(activation_map)
                            print(f"                             Activation map: min={attr_np.min():.6f}, max={attr_np.max():.6f}")
            
            # Apply small epsilon to avoid all-zero maps
            if attr_np.max() < 1e-6:
                print(f"    [WARNING C{k+1}] Attribution map is nearly zero! Creating uniform map.")
                attr_np = np.ones_like(attr_np) * 0.5
            
            # Normalize to [0, 1]
            attr_np = normalize_to_0_1(attr_np)
            
            # Debug: Print after normalization
            print(f"    [DEBUG C{k+1}] After normalize: min={attr_np.min():.6f}, max={attr_np.max():.6f}, mean={attr_np.mean():.6f}")
            
            attributions_list.append(attr_np)
            
        except Exception as e:
            print(f"\n  ⚠ Error computing IG for concept {k}: {e}")
            # Fallback: create zero attribution
            attr_np = np.zeros((Hc, Wc), dtype=np.float32)
            attributions_list.append(attr_np)
    
    print(f"  ✓ Computed IG for all {K} concepts          ")
    
    # Get concept values for visualization (use bottleneck for Mode 4/5)
    concept_values = concept_vector_for_vis[0].detach().cpu().numpy()
    
    # Handle frame stacking (same as GradCAM)
    if len(concept_values) > K:
        print(f"INFO: concept_vector has {len(concept_values)} values, aggregating to K={K}")
        n_frames = len(concept_values) // K
        if len(concept_values) % K == 0:
            concept_values_reshaped = concept_values.reshape(n_frames, K)
            concept_values = concept_values_reshaped.max(axis=0)
            print(f"  ✓ Aggregated {n_frames} frames using max pooling")
        else:
            print(f"  WARNING: Cannot reshape evenly. Taking first {K} values.")
            concept_values = concept_values[:K]
    elif len(concept_values) < K:
        print(f"WARNING: concept_vector has only {len(concept_values)} values but K={K}")
        concept_values = np.pad(concept_values, (0, K - len(concept_values)))
    
    concept_values = concept_values.flatten()[:K]
    
    # Build original RGB image
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
    
    return img, attributions_list, concept_values


# ============================================================
# Composite output
# ============================================================

def composite_and_save(img_rgb_uint8, attributions_list, concept_values, out_path, 
                      cmap="jet", spacing=5, target_size=300, model_obs=None, agent_dir=None):
    """
    Create composite image: [God View] [Model Input] [IG Map1] [IG Map2] ... [IG MapK] [Bar Chart]
    
    Same structure as GradCAM but using Integrated Gradients attributions instead.
    """
    W = target_size
    H = target_size
    
    # Resize God View
    god_view = Image.fromarray(img_rgb_uint8).resize((W, H), Image.LANCZOS)
    
    # Add agent direction indicator
    if agent_dir is not None:
        from PIL import ImageDraw, ImageFont
        god_view_with_dir = god_view.copy()
        draw = ImageDraw.Draw(god_view_with_dir)
        
        dir_names = {0: "RIGHT →", 1: "DOWN ↓", 2: "LEFT ←", 3: "UP ↑"}
        dir_name = dir_names.get(agent_dir, f"DIR={agent_dir}")
        text = f"Agent: {dir_name}"
        
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        except:
            font = ImageFont.load_default()
        
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        padding = 5
        bg_box = [5, 5, 5 + text_width + padding*2, 5 + text_height + padding*2]
        draw.rectangle(bg_box, fill=(0, 0, 0, 180))
        draw.text((5 + padding, 5 + padding), text, fill=(255, 255, 0), font=font)
        
        god_view = god_view_with_dir
    
    # Prepare model observation
    model_view = None
    if model_obs is not None:
        if agent_dir is not None:
            model_obs_rotated = rotate_obs_to_god_view(model_obs, agent_dir)
        else:
            model_obs_rotated = model_obs
        
        model_obs_scaled = (model_obs_rotated.astype(np.float32) * 51).astype(np.uint8)
        model_view = Image.fromarray(model_obs_scaled).resize((W, H), Image.NEAREST)
    
    # Create IG attribution images
    attr_ims = []
    for i, attr in enumerate(attributions_list):
        # Debug: Check attribution shape before rotation
        # print(f"  [DEBUG] Concept {i+1} attr before rotation: shape={attr.shape}, min={attr.min():.4f}, max={attr.max():.4f}")
        
        # Rotate attribution using same rule as model_obs to align with God View
        if agent_dir is not None:
            attr_rotated = rotate_obs_to_god_view(attr, agent_dir)
            # print(f"  [DEBUG] After rotation (dir={agent_dir}): shape={attr_rotated.shape}")
        else:
            attr_rotated = attr
        
        # Apply colormap and resize
        attr_rgb = heatmap_apply_colormap(attr_rotated, cmap)
        attr_rgb = Image.fromarray(attr_rgb).resize((W, H), Image.LANCZOS)
        attr_ims.append(attr_rgb)
    
    # Create bar chart
    FIXED_SIZE = 400
    bar_chart_large = create_concept_values_bar(concept_values, width=FIXED_SIZE, height=FIXED_SIZE)
    bar_chart = bar_chart_large.resize((W, H), Image.LANCZOS)
    
    # Composite
    num_panels = 1 + (1 if model_view else 0) + len(attr_ims) + 1
    total_w = W * num_panels + spacing * (num_panels - 1)
    canvas = Image.new("RGB", (total_w, H), color=(255, 255, 255))
    
    x = 0
    canvas.paste(god_view, (x, 0))
    x += W + spacing
    
    if model_view is not None:
        canvas.paste(model_view, (x, 0))
        x += W + spacing
    
    for attr_im in attr_ims:
        canvas.paste(attr_im, (x, 0))
        x += W + spacing
    
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
                                 out_dir="ig_out",
                                 max_steps=1000):
    """Run episodes and collect best one. Same as GradCAM version."""
    
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
    best_agent_dirs = None
    
    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        obs_list = []
        frame_list = []
        agent_dir_list = []
        steps = 0
        
        obs_list.append(np.array(obs))
        frame_list.append(env.unwrapped.get_frame())
        agent_dir_list.append(env.unwrapped.agent_dir)
        
        while not done and steps < max_steps:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, _ = env.step(action)
            
            obs_list.append(np.array(obs))
            frame_list.append(env.unwrapped.get_frame())
            agent_dir_list.append(env.unwrapped.agent_dir)
            
            ep_reward += reward
            done = terminated or truncated
            steps += 1
        
        print(f"Episode {ep+1}: reward={ep_reward:.3f}, steps={steps}, frames={len(frame_list)}")
        
        if ep_reward > best_reward:
            best_reward = ep_reward
            best_obs = obs_list
            best_frames = frame_list
            best_agent_dirs = agent_dir_list
    
    env.close()
    os.makedirs(out_dir, exist_ok=True)
    
    if best_obs is not None and best_frames is not None:
        print(f"\nBest episode: {len(best_obs)} observations, {len(best_frames)} frames, {len(best_agent_dirs)} directions")
        if len(best_obs) != len(best_frames) != len(best_agent_dirs):
            print(f"⚠️  WARNING: Mismatch between obs, frames, and directions!")
        else:
            print(f"✓ Data count verified: {len(best_frames)} frames")
    
    return model, best_obs, best_frames, best_agent_dirs, best_reward, out_dir


# ============================================================
# IG generator for best episode
# ============================================================

def generate_ig_for_best(model, best_obs, frames, agent_dirs, out_dir, device="cpu", fps=6, n_steps=50):
    """Generate Integrated Gradients visualizations for best episode."""
    
    if not CAPTUM_AVAILABLE:
        print("❌ ERROR: Captum not available. Install with: pip install captum")
        return None
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(out_dir, f"ig_{timestamp}")
    png_dir = os.path.join(run_dir, "frames")
    
    os.makedirs(png_dir, exist_ok=True)
    
    fx = model.policy.features_extractor
    fx.to(device)
    fx.eval()
    
    saved = []
    
    print(f"\n{'='*60}")
    print(f"Generating Integrated Gradients for {len(best_obs)} frames...")
    print(f"IG steps: {n_steps} (higher = more accurate but slower)")
    print(f"{'='*60}\n")
    
    for i, obs_raw in enumerate(best_obs):
        obs_np = np.array(obs_raw)
        
        if obs_np.ndim == 3 and obs_np.shape[-1] == 3:
            obs_np = np.transpose(obs_np, (2, 0, 1))
        
        obs_t = torch.from_numpy(obs_np).float().unsqueeze(0)
        
        try:
            img_input, attributions, concept_vals = compute_concept_integrated_gradients(
                fx, obs_t, device, n_steps=n_steps
            )
        except Exception as e:
            print(f"IG error at frame {i}: {e}")
            continue
        
        orig = frames[i]
        agent_dir = agent_dirs[i]
        
        out_file = os.path.join(png_dir, f"frame_{i:04d}.png")
        composite_and_save(orig, attributions, concept_vals, out_file, 
                          model_obs=obs_raw, agent_dir=agent_dir)
        saved.append(out_file)
        
        if (i+1) % 10 == 0:
            print(f"Saved {i+1}/{len(best_obs)}")
    
    print(f"\n✓ Saved all {len(saved)} frames")
    
    # Make video
    try:
        from moviepy import ImageSequenceClip
        print(f"\n🎬 Creating video from {len(saved)} frames at {fps} fps...")
        clip = ImageSequenceClip(saved, fps=fps)
        vid_path = os.path.join(run_dir, "ig_episode.mp4")
        clip.write_videofile(vid_path, codec="libx264", audio=False, logger=None)
        print(f"✓ Video saved: {vid_path}")
    except Exception as e:
        print(f"⚠ Warning: Could not create video: {e}")
    
    return run_dir


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Test agent with Integrated Gradients visualization")
    parser.add_argument("--model", type=str, required=True, help="Path to model .zip file")
    parser.add_argument("--env", type=str, default="MiniGrid-Empty-5x5-v0", help="Environment ID")
    parser.add_argument("--algo", type=str, default="PPO_CONCEPT", 
                       choices=["PPO", "PPO_CONCEPT", "DQN", "ppo", "ppo_concept", "dqn"],
                       help="Algorithm type")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to run")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--fps", type=int, default=6, help="FPS for output video")
    parser.add_argument("--n-steps", type=int, default=50, 
                       help="Number of IG steps (higher = more accurate but slower)")
    parser.add_argument("--outdir", type=str, default=None, help="Output directory")
    
    args = parser.parse_args()
    
    if not CAPTUM_AVAILABLE:
        print("\n" + "="*60)
        print("❌ ERROR: Captum library not installed!")
        print("="*60)
        print("\nIntegrated Gradients requires the Captum library.")
        print("Install it with: pip install captum")
        print("\n" + "="*60)
        return
    
    # Determine output directory
    if args.outdir is None:
        model_name = os.path.splitext(os.path.basename(args.model))[0]
        out_dir = f"ig_out/{args.env}/{args.algo.lower()}/{model_name}"
    else:
        out_dir = args.outdir
    
    print("\n" + "="*60)
    print("INTEGRATED GRADIENTS VISUALIZATION")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Environment: {args.env}")
    print(f"Algorithm: {args.algo}")
    print(f"Device: {args.device}")
    print(f"IG Steps: {args.n_steps}")
    print(f"Output: {out_dir}")
    print("="*60 + "\n")
    
    # Run and collect best episode
    model, best_obs, frames, agent_dirs, best_reward, out_dir = run_and_collect_best_episode(
        args.model, args.env, args.algo,
        args.episodes, True, args.device, out_dir
    )
    
    print(f"\n✓ Best reward = {best_reward}")
    print(f"\nRunning Integrated Gradients visualization...")
    
    # Generate IG visualizations
    run_dir = generate_ig_for_best(
        model, best_obs, frames, agent_dirs, out_dir,
        device=args.device, fps=args.fps, n_steps=args.n_steps
    )
    
    print("\n" + "="*60)
    print("✓ DONE! Results saved in:", run_dir)
    print("="*60 + "\n")


def test_agent_ig(model_path, env_id, algorithm="PPO_CONCEPT", num_episodes=10,
                 device="cpu", outdir="ig_out", fps=6, n_steps=50, save_mode="best_run"):
    """
    Wrapper function for UI integration.
    
    Args:
        model_path: Path to model file
        env_id: Environment ID
        algorithm: Algorithm name (PPO_CONCEPT, PPO, DQN)
        num_episodes: Number of episodes to run
        device: Device (cpu, cuda, mps)
        outdir: Output directory
        fps: Video FPS
        n_steps: Number of IG steps
        save_mode: "best_run" or "all_episodes"
    """
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    
    if save_mode == "best_run":
        # Original behavior: save only best run
        model, best_obs, frames, agent_dirs, best_reward, _ = run_and_collect_best_episode(
            model_path, env_id, algorithm,
            num_episodes, True, device, outdir
        )
        
        print(f"✓ Best reward = {best_reward}")
        
        run_dir = generate_ig_for_best(
            model, best_obs, frames, agent_dirs, outdir,
            device=device, fps=fps, n_steps=n_steps
        )
        
        print(f"✓ Saved best run to: {run_dir}")
        
    else:  # all_episodes
        # New behavior: save all episodes
        env = gym.make(env_id, render_mode="rgb_array")
        env = ImgObsWrapper(env)
        
        if algorithm.upper() == "PPO_CONCEPT":
            model = ConceptPPO.load(model_path, env=env, device=device)
        elif algorithm.upper().startswith("PPO"):
            model = PPO.load(model_path, env=env, device=device)
        elif algorithm.upper() == "DQN":
            model = DQN.load(model_path, env=env, device=device)
        
        for ep in range(num_episodes):
            print(f"\n{'='*60}")
            print(f"Episode {ep+1}/{num_episodes}")
            print(f"{'='*60}")
            
            obs, _ = env.reset()
            done = False
            ep_reward = 0
            obs_list = []
            frame_list = []
            agent_dir_list = []
            steps = 0
            max_steps = 1000
            
            obs_list.append(np.array(obs))
            frame_list.append(env.unwrapped.get_frame())
            agent_dir_list.append(env.unwrapped.agent_dir)
            
            while not done and steps < max_steps:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                
                obs_list.append(np.array(obs))
                frame_list.append(env.unwrapped.get_frame())
                agent_dir_list.append(env.unwrapped.agent_dir)
                
                ep_reward += reward
                done = terminated or truncated
                steps += 1
            
            print(f"✓ Episode {ep+1} reward: {ep_reward}, steps: {steps}")
            
            # Generate IG for this episode
            ep_out_dir = f"{outdir}/episode_{ep+1:03d}"
            run_dir = generate_ig_for_best(
                model, obs_list, frame_list, agent_dir_list, ep_out_dir,
                device=device, fps=fps, n_steps=n_steps
            )
            
            print(f"✓ Saved episode {ep+1} to: {run_dir}")
        
        env.close()
        print(f"\n✓ All {num_episodes} episodes saved to: {outdir}")


if __name__ == "__main__":
    main()
