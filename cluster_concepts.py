"""
Clustering concepts for representative sample selection
- Remove duplicate frames from failed runs
- C1 (continuous): 3 bins (low/med/high) with N samples each
- C2-Cn (binary): activated + inactive, N samples each
"""

import numpy as np
from pathlib import Path
from PIL import Image
import shutil
from sklearn.cluster import KMeans
from typing import List, Tuple, Dict
import hashlib


def compute_image_hash(img_path: Path) -> str:
    """Compute perceptual hash for duplicate detection"""
    img = Image.open(img_path)
    # Resize to small size for fast comparison
    img = img.resize((8, 8), Image.LANCZOS)
    # Convert to grayscale
    img = img.convert('L')
    # Get pixel data
    pixels = np.array(img).flatten()
    # Hash
    return hashlib.md5(pixels.tobytes()).hexdigest()


def remove_duplicate_frames(frame_paths: List[Path]) -> List[Path]:
    """Remove visually similar frames using perceptual hashing"""
    unique_frames = []
    seen_hashes = set()
    
    print(f"  Checking {len(frame_paths)} frames for duplicates...")
    
    for frame_path in frame_paths:
        frame_hash = compute_image_hash(frame_path)
        if frame_hash not in seen_hashes:
            seen_hashes.add(frame_hash)
            unique_frames.append(frame_path)
    
    print(f"  ✓ Kept {len(unique_frames)}/{len(frame_paths)} unique frames")
    return unique_frames


def extract_heatmap_features(img_path: Path) -> np.ndarray:
    """Extract features from heatmap for clustering"""
    img = Image.open(img_path)
    
    # If composite image (God View + Model Input + Heatmaps), extract heatmap region
    w, h = img.size
    
    # Heatmap is usually in the right portion
    # For our composite: [God View | Model Input | C1 | C2 | ... | Bar Chart]
    # Each section is FIXED_SIZE (e.g., 256x256)
    FIXED_SIZE = 256
    
    # Try to detect if this is a composite or single heatmap
    if w > h * 1.5:  # Composite (wide)
        # Extract one heatmap region (skip first 2: God View, Model Input)
        # Use the specific concept's heatmap
        heatmap_start = FIXED_SIZE * 2  # Skip God View and Model Input
        img_crop = img.crop((heatmap_start, 0, heatmap_start + FIXED_SIZE, FIXED_SIZE))
    else:  # Single heatmap
        img_crop = img
    
    # Resize to fixed size for feature extraction
    img_crop = img_crop.resize((32, 32), Image.LANCZOS)
    img_array = np.array(img_crop)
    
    # If RGB, average channels or use specific channel
    if len(img_array.shape) == 3:
        # Use red channel for heatmap (typically hot colors)
        img_array = img_array[:, :, 0]
    
    # Flatten to feature vector
    features = img_array.flatten().astype(np.float32) / 255.0
    
    return features


def cluster_images(img_paths: List[Path], n_clusters: int, concept_name: str) -> List[List[Path]]:
    """Cluster images and return grouped paths"""
    
    if len(img_paths) <= n_clusters:
        # Not enough images to cluster, return all
        return [[img] for img in img_paths]
    
    print(f"  Extracting features from {len(img_paths)} images for {concept_name}...")
    
    # Extract features
    features = []
    valid_paths = []
    
    for img_path in img_paths:
        try:
            feat = extract_heatmap_features(img_path)
            features.append(feat)
            valid_paths.append(img_path)
        except Exception as e:
            print(f"    ⚠ Error processing {img_path.name}: {e}")
            continue
    
    if len(valid_paths) == 0:
        return []
    
    features = np.array(features)
    
    # KMeans clustering
    print(f"  Clustering into {n_clusters} groups...")
    kmeans = KMeans(n_clusters=min(n_clusters, len(valid_paths)), random_state=42)
    labels = kmeans.fit_predict(features)
    
    # Group by cluster
    clusters = [[] for _ in range(n_clusters)]
    for img_path, label in zip(valid_paths, labels):
        if label < n_clusters:
            clusters[label].append(img_path)
    
    return clusters


def select_representative_samples(clusters: List[List[Path]], n_samples_per_cluster: int) -> List[Path]:
    """Select N representative samples from each cluster"""
    
    representatives = []
    
    for cluster_id, cluster_imgs in enumerate(clusters):
        if not cluster_imgs:
            continue
        
        # If cluster has fewer images than requested, take all
        if len(cluster_imgs) <= n_samples_per_cluster:
            representatives.extend(cluster_imgs)
        else:
            # Evenly sample from cluster
            step = len(cluster_imgs) / n_samples_per_cluster
            indices = [int(i * step) for i in range(n_samples_per_cluster)]
            representatives.extend([cluster_imgs[i] for i in indices])
    
    return representatives


def cluster_concept_continuous(ig_out_dir: Path, concept_name: str, concept_idx: int, n_samples_per_bin: int) -> Dict[str, List[Path]]:
    """
    Cluster continuous concept into 3 bins: low, medium, high
    
    Args:
        ig_out_dir: Path to IG output directory
        concept_name: Name like "C1", "C2", etc.
        concept_idx: 0-based index (C1=0, C2=1, ...)
        n_samples_per_bin: Number of samples per bin
    
    Returns dict: {"low": [paths], "medium": [paths], "high": [paths]}
    """
    
    print(f"\n📊 Clustering {concept_name} (continuous concept)...")
    
    frame_to_value = {}
    
    # Parse success_runs.txt and failed_runs.txt
    # Strategy: Match episodes by ORDER (chronological)
    # Episodes in txt appear in same order as episode_XXX folders
    
    all_episode_dirs = sorted(ig_out_dir.glob("episode_*"))
    episode_dir_idx = 0  # Track current episode folder index
    
    for txt_file in [ig_out_dir / "success_runs.txt", ig_out_dir / "failed_runs.txt"]:
        if not txt_file.exists():
            print(f"  ⚠ {txt_file.name} not found")
            continue
        
        print(f"  Parsing {txt_file.name}...")
        
        with open(txt_file, 'r') as f:
            content = f.read()
        
        # Split by episode separator
        import re
        episode_blocks = content.split("======================================================================")
        
        episodes_in_txt = 0
        
        for episode_block in episode_blocks:
            if "Step" not in episode_block:
                continue
            
            # Extract all steps and concept vectors
            steps = re.findall(
                r"Step\s+(\d+):\s+Concept Vector.*?\[([\d.,\s]+)\]",
                episode_block,
                re.DOTALL
            )
            
            if not steps:
                continue
            
            episodes_in_txt += 1
            
            # Match with NEXT episode folder in order
            if episode_dir_idx >= len(all_episode_dirs):
                print(f"    ⚠ No more episode folders (txt has more episodes)")
                break
            
            episode_dir = all_episode_dirs[episode_dir_idx]
            episode_dir_idx += 1
            
            frames_dir = episode_dir / "frames"
            if not frames_dir.exists():
                print(f"    ⚠ {episode_dir.name}/frames not found")
                continue
            
            frame_files = sorted(frames_dir.glob("frame_*.png"))
            
            # Verify frame count roughly matches steps (allow ±1 difference)
            if abs(len(frame_files) - len(steps)) > 1:
                print(f"    ⚠ Frame/step mismatch in {episode_dir.name}: {len(frame_files)} frames vs {len(steps)} steps")
            
            # Map frames to concept values (use min length to avoid index errors)
            for i in range(min(len(frame_files), len(steps))):
                step_num, vector_str = steps[i]
                vector = [float(v.strip()) for v in vector_str.split(",")]
                if len(vector) <= concept_idx:
                    continue
                
                concept_val = vector[concept_idx]
                frame_to_value[frame_files[i]] = concept_val
        
        print(f"    Found {episodes_in_txt} episodes in {txt_file.name}")
    
    print(f"  Mapped {len(frame_to_value)} frames to {concept_name} values")
    
    if not frame_to_value:
        print(f"  ⚠ No {concept_name} values mapped - cannot cluster")
        return {"low": [], "medium": [], "high": []}
    
    # Bin frames by concept value
    bins = {"low": [], "medium": [], "high": []}
    
    for frame_path, concept_val in frame_to_value.items():
        if concept_val < 0.33:
            bins["low"].append(frame_path)
        elif concept_val < 0.67:
            bins["medium"].append(frame_path)
        else:
            bins["high"].append(frame_path)
    
    print(f"  Bins: low={len(bins['low'])}, medium={len(bins['medium'])}, high={len(bins['high'])}")
    
    # Remove duplicates within each bin
    for bin_name in bins:
        if bins[bin_name]:
            original_count = len(bins[bin_name])
            bins[bin_name] = remove_duplicate_frames(bins[bin_name])
            print(f"    {bin_name}: {len(bins[bin_name])}/{original_count} unique frames")
    
    # Cluster each bin separately
    clustered_bins = {}
    
    for bin_name, bin_frames in bins.items():
        if not bin_frames:
            print(f"  ⚠ No frames in {bin_name} bin")
            clustered_bins[bin_name] = []
            continue
        
        print(f"\n  Clustering {bin_name} bin ({len(bin_frames)} frames)...")
        
        # Use 2-3 clusters per bin
        n_clusters = min(3, max(2, len(bin_frames) // 5))
        clusters = cluster_images(bin_frames, n_clusters, f"{concept_name}_{bin_name}")
        
        # Select representatives
        representatives = select_representative_samples(clusters, n_samples_per_bin)
        clustered_bins[bin_name] = representatives
        
        print(f"  ✓ Selected {len(representatives)} representatives from {bin_name} bin")
    
    return clustered_bins


def cluster_concept_binary(ig_out_dir: Path, concept_name: str, n_samples_per_state: int) -> Dict[str, List[Path]]:
    """
    Cluster binary concept (C2, C3, C4, ...) into activated and inactive states
    Returns dict: {"activated": [paths], "inactive": [paths]}
    """
    
    print(f"\n📊 Clustering {concept_name} (binary concept)...")
    
    # Collect frames from all episodes
    activated_frames = []
    inactive_frames = []
    
    for episode_dir in sorted(ig_out_dir.glob("episode_*")):
        concept_dir = episode_dir / concept_name
        if not concept_dir.exists():
            continue
        
        # Activated frames
        act_dir = concept_dir / "activated"
        if act_dir.exists():
            activated_frames.extend(list(act_dir.glob("frame_*.png")))
        
        # Inactive frames
        inact_dir = concept_dir / "inactive"
        if inact_dir.exists():
            inactive_frames.extend(list(inact_dir.glob("frame_*.png")))
    
    print(f"  Found {len(activated_frames)} activated, {len(inactive_frames)} inactive frames")
    
    # Remove duplicates
    activated_frames = remove_duplicate_frames(activated_frames)
    inactive_frames = remove_duplicate_frames(inactive_frames)
    
    results = {}
    
    # Cluster activated frames
    if activated_frames:
        print(f"\n  Clustering ACTIVATED frames ({len(activated_frames)})...")
        n_clusters = min(5, max(2, len(activated_frames) // 10))  # 2-5 clusters
        clusters = cluster_images(activated_frames, n_clusters, f"{concept_name}_activated")
        representatives = select_representative_samples(clusters, n_samples_per_state)
        results["activated"] = representatives
        print(f"  ✓ Selected {len(representatives)} activated representatives")
    else:
        results["activated"] = []
    
    # Cluster inactive frames
    if inactive_frames:
        print(f"\n  Clustering INACTIVE frames ({len(inactive_frames)})...")
        n_clusters = min(5, max(2, len(inactive_frames) // 10))  # 2-5 clusters
        clusters = cluster_images(inactive_frames, n_clusters, f"{concept_name}_inactive")
        representatives = select_representative_samples(clusters, n_samples_per_state)
        results["inactive"] = representatives
        print(f"  ✓ Selected {len(representatives)} inactive representatives")
    else:
        results["inactive"] = []
    
    return results


def create_clustering_output(ig_out_dir: Path, n_samples: int = 5):
    """
    Main function to cluster all concepts and create output folders
    
    Detects continuous vs binary concepts from:
    1. Model metadata in success_runs.txt (preferred)
    2. Folder structure as fallback
    
    Args:
        ig_out_dir: Path to ig_YYYYMMDD_HHMMSS folder
        n_samples: Number of sample images per cluster/bin
    """
    
    print(f"\n{'='*60}")
    print(f"🎯 Clustering Concepts for Representative Samples")
    print(f"{'='*60}")
    print(f"Input: {ig_out_dir}")
    print(f"Samples per cluster: {n_samples}")
    
    # Create clustering output directory
    cluster_dir = ig_out_dir / "clustered_samples"
    cluster_dir.mkdir(exist_ok=True)
    
    # Try to get model metadata from success_runs.txt
    success_txt = ig_out_dir / "success_runs.txt"
    n_continuous_concepts = 1  # Default fallback
    n_total_concepts = 0
    
    if success_txt.exists():
        import re
        with open(success_txt, 'r') as f:
            content = f.read(1000)  # Read first 1000 chars to find metadata
        
        # Try to parse model metadata
        model_info_match = re.search(r'Model: (\d+) concepts, mode (\d+), (\d+) continuous', content)
        if model_info_match:
            n_total_concepts = int(model_info_match.group(1))
            n_continuous_concepts = int(model_info_match.group(3))
            print(f"  ✓ Detected from metadata: {n_total_concepts} concepts, {n_continuous_concepts} continuous")
    
    # If metadata was not found, try to detect from folder structure
    first_episode = next(ig_out_dir.glob("episode_*"), None)
    if not first_episode:
        print("  ⚠ No episodes found!")
        return cluster_dir
    
    # If we have metadata, use it to determine concept count
    if n_total_concepts == 0:
        # Fallback: count concept folders in first episode
        all_concept_folders = [d.name for d in first_episode.iterdir() 
                              if d.is_dir() and d.name.startswith("C") and d.name[1:].isdigit()]
        n_total_concepts = len(all_concept_folders)
        if n_total_concepts == 0:
            print("  ⚠ No concept folders found!")
            return cluster_dir
    
    # Generate concept names based on total count from metadata
    # NOTE: IG output only creates folders for BINARY concepts (activated/inactive)
    # Continuous concepts don't have folders in episode_XXX, but we still cluster them from frames
    all_concepts = [f"C{i}" for i in range(1, n_total_concepts + 1)]
    
    # Classify concepts based on metadata
    continuous_concepts = []
    binary_concepts = []
    
    for concept_name in all_concepts:
        concept_idx = int(concept_name[1:]) - 1  # C1 -> 0, C2 -> 1, ...
        
        # Check if continuous based on index from metadata
        if concept_idx < n_continuous_concepts:
            continuous_concepts.append(concept_name)
        else:
            binary_concepts.append(concept_name)
    
    print(f"\n  Detected continuous concepts: {continuous_concepts}")
    print(f"  Detected binary concepts: {binary_concepts}")
    
    # 1. Cluster continuous concepts
    for concept_name in continuous_concepts:
        concept_idx = int(concept_name[1:]) - 1  # C1 -> 0, C2 -> 1, ...
        bins = cluster_concept_continuous(ig_out_dir, concept_name, concept_idx, n_samples_per_bin=n_samples)
        
        # Save representatives
        concept_dir_out = cluster_dir / concept_name
        concept_dir_out.mkdir(exist_ok=True)
        
        for bin_name, frames in bins.items():
            bin_dir = concept_dir_out / bin_name
            bin_dir.mkdir(exist_ok=True)
            
            for i, frame_path in enumerate(frames):
                dest = bin_dir / f"sample_{i:03d}.png"
                shutil.copy2(frame_path, dest)
            
            print(f"  ✓ Saved {len(frames)} samples to {bin_dir}")
    
    # 2. Cluster binary concepts
    for concept_name in binary_concepts:
        states = cluster_concept_binary(ig_out_dir, concept_name, n_samples_per_state=n_samples)
        
        # Save representatives
        concept_dir_out = cluster_dir / concept_name
        concept_dir_out.mkdir(exist_ok=True)
        
        for state_name, frames in states.items():
            state_dir = concept_dir_out / state_name
            state_dir.mkdir(exist_ok=True)
            
            for i, frame_path in enumerate(frames):
                dest = state_dir / f"sample_{i:03d}.png"
                shutil.copy2(frame_path, dest)
            
            print(f"  ✓ Saved {len(frames)} {state_name} samples to {state_dir}")
    
    print(f"\n{'='*60}")
    print(f"✅ Clustering Complete!")
    print(f"Output: {cluster_dir}")
    print(f"{'='*60}\n")
    
    return cluster_dir


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Cluster concepts for representative samples")
    parser.add_argument("--ig-dir", type=str, required=True, help="Path to ig_YYYYMMDD_HHMMSS folder")
    parser.add_argument("--n-samples", type=int, default=5, help="Number of samples per cluster/bin")
    
    args = parser.parse_args()
    
    ig_dir = Path(args.ig_dir)
    if not ig_dir.exists():
        print(f"❌ Error: {ig_dir} does not exist!")
        exit(1)
    
    create_clustering_output(ig_dir, n_samples=args.n_samples)