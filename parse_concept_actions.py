"""
Parse concept-action pairs from IG output for Decision Tree training

Extracts training data from success_runs.txt and failed_runs.txt
Handles data preprocessing, balancing, and deduplication
"""

import numpy as np
import re
from pathlib import Path
from collections import defaultdict
from typing import Tuple, List, Dict
import pickle
import argparse


ACTION_NAMES = {
    0: "TURN_LEFT",
    1: "TURN_RIGHT",
    2: "MOVE_FORWARD",
    3: "PICKUP_OBJECT",
    4: "DROP_OBJECT",
    5: "TOGGLE_DOOR",
    6: "DONE"
}


def detect_repetitive_sequences(concept_vectors: List[np.ndarray], 
                                 actions: List[int], 
                                 window_size: int = 5,
                                 threshold: float = 0.8) -> Tuple[List[np.ndarray], List[int]]:
    """
    Remove repetitive state-action sequences from failed runs
    
    Args:
        concept_vectors: List of concept vectors
        actions: List of corresponding actions
        window_size: Size of sliding window to check for repetition
        threshold: Fraction of identical windows to trigger removal (0.8 = 80%)
    
    Returns:
        Filtered concept_vectors and actions
    """
    if len(concept_vectors) < window_size * 2:
        return concept_vectors, actions
    
    # Convert to numpy for easier comparison
    concepts_array = np.array(concept_vectors)
    actions_array = np.array(actions)
    
    # Track which indices to keep
    keep_indices = set(range(len(concept_vectors)))
    
    # Sliding window to detect repetitions
    seen_windows = defaultdict(list)  # (concept_hash, action) -> [indices]
    
    for i in range(len(concepts_array) - window_size + 1):
        # Create window signature: (concept_state, action)
        window_concepts = concepts_array[i:i+window_size]
        window_actions = actions_array[i:i+window_size]
        
        # Hash concept state (round to 2 decimals for fuzzy matching)
        concept_hash = tuple(np.round(window_concepts[0], 2))
        action_hash = tuple(window_actions)
        
        signature = (concept_hash, action_hash)
        seen_windows[signature].append(i)
    
    # Find repetitive patterns
    for signature, indices in seen_windows.items():
        if len(indices) >= window_size * threshold:
            # This pattern repeats too much - keep only first few occurrences
            keep_first = max(3, int(len(indices) * 0.2))  # Keep 20% or min 3
            remove_indices = indices[keep_first:]
            
            # Mark for removal
            for idx in remove_indices:
                for j in range(idx, min(idx + window_size, len(concept_vectors))):
                    keep_indices.discard(j)
    
    # Filter
    keep_indices = sorted(keep_indices)
    filtered_concepts = [concept_vectors[i] for i in keep_indices]
    filtered_actions = [actions[i] for i in keep_indices]
    
    return filtered_concepts, filtered_actions


def remove_consecutive_duplicates(concept_vectors: List[np.ndarray], 
                                   actions: List[int],
                                   tolerance: float = 0.01) -> Tuple[List[np.ndarray], List[int]]:
    """
    Remove consecutive duplicate states (agent stuck in same state)
    
    Args:
        concept_vectors: List of concept vectors
        actions: List of actions
        tolerance: Maximum difference to consider states identical
    
    Returns:
        Filtered lists with consecutive duplicates removed
    """
    if len(concept_vectors) <= 1:
        return concept_vectors, actions
    
    filtered_concepts = [concept_vectors[0]]
    filtered_actions = [actions[0]]
    
    for i in range(1, len(concept_vectors)):
        # Check if current state is different from previous
        diff = np.abs(np.array(concept_vectors[i]) - np.array(concept_vectors[i-1]))
        
        if np.max(diff) > tolerance or actions[i] != actions[i-1]:
            # State or action changed - keep it
            filtered_concepts.append(concept_vectors[i])
            filtered_actions.append(actions[i])
        # else: skip this duplicate
    
    return filtered_concepts, filtered_actions


def remove_all_duplicates(concept_vectors: List[np.ndarray], 
                          actions: List[int],
                          tolerance: float = 0.001) -> Tuple[List[np.ndarray], List[int]]:
    """
    Remove ALL duplicate (concept_vector, action) pairs from dataset
    Keeps only the first occurrence of each unique pair
    
    Args:
        concept_vectors: List of concept vectors
        actions: List of actions
        tolerance: Maximum difference to consider vectors identical
    
    Returns:
        Filtered lists with all duplicates removed
    """
    if len(concept_vectors) == 0:
        return concept_vectors, actions
    
    seen_pairs = []
    filtered_concepts = []
    filtered_actions = []
    
    for concept_vec, action in zip(concept_vectors, actions):
        # Check if this (concept, action) pair already exists
        is_duplicate = False
        concept_arr = np.array(concept_vec)
        
        for seen_concept, seen_action in seen_pairs:
            if action == seen_action:
                diff = np.abs(concept_arr - seen_concept)
                if np.max(diff) <= tolerance:
                    is_duplicate = True
                    break
        
        if not is_duplicate:
            filtered_concepts.append(concept_vec)
            filtered_actions.append(action)
            seen_pairs.append((concept_arr, action))
    
    return filtered_concepts, filtered_actions


def parse_run_file(txt_path: Path, 
                   preprocess_failed: bool = True,
                   is_failed: bool = False) -> Tuple[List[np.ndarray], List[int], Dict]:
    """
    Parse success_runs.txt or failed_runs.txt
    
    Args:
        txt_path: Path to txt file
        preprocess_failed: Whether to remove repetitive patterns (for failed runs)
        is_failed: Whether this is a failed run file
    
    Returns:
        concept_vectors: List of concept vectors [K,]
        actions: List of action indices
        metadata: Dict with environment, n_concepts, etc.
    """
    
    if not txt_path.exists():
        print(f"⚠️  File not found: {txt_path}")
        return [], [], {}
    
    print(f"\n📖 Parsing {txt_path.name}...")
    
    with open(txt_path, 'r') as f:
        content = f.read()
    
    # Extract model metadata from first episode
    metadata = {
        'n_concepts': None,  # Will auto-detect from first vector
        'concept_mode': 5,
        'n_continuous_concepts': 1,
        'environment': 'Unknown'
    }
    
    model_info_match = re.search(r'Model: (\d+) concepts, mode (\d+), (\d+) continuous', content)
    if model_info_match:
        metadata['n_concepts'] = int(model_info_match.group(1))
        metadata['concept_mode'] = int(model_info_match.group(2))
        metadata['n_continuous_concepts'] = int(model_info_match.group(3))
    
    env_match = re.search(r'Environment: ([\w-]+)', content)
    if env_match:
        metadata['environment'] = env_match.group(1)
    
    # Auto-detect n_concepts from first vector if not found in metadata
    if metadata['n_concepts'] is None:
        first_vector_match = re.search(r'Concept Vector[^\[]*\[([\d.,\s]+)\]', content)
        if first_vector_match:
            first_vector = [float(v.strip()) for v in first_vector_match.group(1).split(",")]
            metadata['n_concepts'] = len(first_vector)
            print(f"  ℹ️  Auto-detected {metadata['n_concepts']} concepts from first vector")
        else:
            # Last resort default
            metadata['n_concepts'] = 3
            print(f"  ⚠️  Could not detect n_concepts, using default: {metadata['n_concepts']}")
    
    # Parse episodes
    episode_blocks = content.split("="*70)
    
    all_concept_vectors = []
    all_actions = []
    
    episodes_processed = 0
    
    for episode_block in episode_blocks:
        if "Step" not in episode_block:
            continue
        
        # Extract all steps (allow any whitespace including newlines between parts)
        steps = re.findall(
            r"Step\s+\d+:\s+Concept Vector[^\[]*\[([\d.,\s]+)\]\s+Action:\s*(\d+)",
            episode_block,
            re.DOTALL
        )
        
        if not steps:
            # Debug: try to understand why no match
            if "Concept Vector" in episode_block and "Action:" in episode_block:
                print(f"  ⚠️  Episode block contains data but regex didn't match")
                # Show sample
                lines = episode_block.strip().split('\n')
                if len(lines) > 5:
                    print(f"  Sample lines: {lines[5:8]}")
            continue
        
        episode_concepts = []
        episode_actions = []
        
        for step in steps:
            vector_str, action_str = step
            
            # Parse concept vector
            vector = [float(v.strip()) for v in vector_str.split(",")]
            
            # Auto-update n_concepts if mismatch (flexible parsing)
            if len(vector) != metadata['n_concepts']:
                if len(episode_concepts) == 0:  # First vector in dataset
                    print(f"  ℹ️  Updating n_concepts from {metadata['n_concepts']} to {len(vector)}")
                    metadata['n_concepts'] = len(vector)
                else:  # Skip inconsistent vectors
                    continue
            
            # Parse action
            action = int(action_str)
            
            episode_concepts.append(np.array(vector))
            episode_actions.append(action)
        
        if len(episode_concepts) == 0:
            continue
        
        # Preprocess failed runs only (success runs handled via global deduplication later)
        if is_failed and preprocess_failed:
            # Failed runs: Remove both consecutive duplicates AND loops
            episode_concepts, episode_actions = remove_consecutive_duplicates(
                episode_concepts, episode_actions
            )
            episode_concepts, episode_actions = detect_repetitive_sequences(
                episode_concepts, episode_actions
            )
        
        all_concept_vectors.extend(episode_concepts)
        all_actions.extend(episode_actions)
        episodes_processed += 1
    
    print(f"  ✓ Processed {episodes_processed} episodes")
    print(f"  ✓ Extracted {len(all_actions)} samples")
    
    if is_failed and preprocess_failed:
        # Show reduction from preprocessing
        original_estimate = episodes_processed * 360  # Failed runs ~360 steps
        reduction = (1 - len(all_actions) / max(original_estimate, 1)) * 100
        print(f"  ✓ Preprocessing reduced samples by ~{reduction:.1f}%")
    
    return all_concept_vectors, all_actions, metadata


def load_concept_action_data(ig_dir: Path,
                              use_success: bool = True,
                              use_failed: bool = True,
                              failed_sample_ratio: float = 0.1,
                              preprocess_failed: bool = True,
                              preprocess_success: bool = False) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Load and combine concept-action data from IG output directory
    
    Args:
        ig_dir: Path to IG output directory (e.g., ig_20260224_124815)
        use_success: Include success_runs.txt
        use_failed: Include failed_runs.txt
        failed_sample_ratio: Downsample failed runs (e.g., 0.1 = keep 10%)
        preprocess_failed: Remove repetitive patterns from failed runs
        preprocess_success: Remove ALL duplicate (concept, action) pairs from entire dataset
                          This removes duplicates across ALL episodes, not just consecutive ones.
                          Significantly reduces data (50-70%) but improves generalization.
    
    Returns:
        X: Concept vectors [N, K]
        y: Action labels [N,]
        metadata: Dict with environment info, concept metadata, etc.
    """
    
    print(f"\n{'='*60}")
    print(f"📊 Loading Concept-Action Dataset")
    print(f"{'='*60}")
    print(f"Input: {ig_dir}")
    print(f"Use success runs: {use_success}")
    print(f"Use failed runs: {use_failed} (sample ratio: {failed_sample_ratio})")
    print(f"Preprocess failed: {preprocess_failed}")
    print(f"Preprocess success: {preprocess_success}")
    
    all_concepts = []
    all_actions = []
    metadata = {}
    
    # Load success runs
    if use_success:
        success_file = ig_dir / "success_runs.txt"
        success_concepts, success_actions, metadata = parse_run_file(
            success_file, 
            preprocess_failed=False,  # Success runs not preprocessed per-episode
            is_failed=False
        )
        all_concepts.extend(success_concepts)
        all_actions.extend(success_actions)
        
        print(f"\n✓ Success runs: {len(success_actions)} samples")
    
    # Load failed runs
    if use_failed:
        failed_file = ig_dir / "failed_runs.txt"
        failed_concepts, failed_actions, meta_failed = parse_run_file(
            failed_file,
            preprocess_failed=preprocess_failed,
            is_failed=True
        )
        
        # Update metadata if not set
        if not metadata:
            metadata = meta_failed
        
        # Downsample failed runs
        if failed_sample_ratio < 1.0 and len(failed_actions) > 0:
            n_keep = int(len(failed_actions) * failed_sample_ratio)
            indices = np.random.choice(len(failed_actions), n_keep, replace=False)
            failed_concepts = [failed_concepts[i] for i in indices]
            failed_actions = [failed_actions[i] for i in indices]
            
            print(f"\n✓ Failed runs: {len(failed_actions)} samples (downsampled from {len(failed_concepts):.0f})")
        else:
            print(f"\n✓ Failed runs: {len(failed_actions)} samples")
        
        all_concepts.extend(failed_concepts)
        all_actions.extend(failed_actions)
    
    # Check if we have any data
    if len(all_concepts) == 0:
        print(f"\n{'='*60}")
        print(f"📈 Dataset Summary")
        print(f"{'='*60}")
        print(f"⚠️  WARNING: No samples found!")
        print(f"Please check:")
        print(f"  - success_runs.txt and/or failed_runs.txt exist")
        print(f"  - Files contain valid episode data")
        print(f"  - File format matches expected pattern")
        print(f"{'='*60}\n")
        
        # Return empty arrays with proper shape
        n_concepts = metadata.get('n_concepts', 3)
        X = np.zeros((0, n_concepts))
        y = np.array([], dtype=int)
        metadata['n_samples'] = 0
        metadata['n_success_samples'] = 0
        metadata['n_failed_samples'] = 0
        metadata['action_distribution'] = {}
        return X, y, metadata
    
    # Global deduplication if requested (removes ALL duplicate concept-action pairs)
    if preprocess_success and use_success:
        print(f"\n🔄 Applying global deduplication...")
        original_count = len(all_concepts)
        all_concepts, all_actions = remove_all_duplicates(all_concepts, all_actions)
        removed_count = original_count - len(all_concepts)
        print(f"  ✓ Removed {removed_count:,} duplicate samples ({removed_count/original_count*100:.1f}%)")
        print(f"  ✓ Remaining: {len(all_concepts):,} unique (concept, action) pairs")
    
    # Convert to numpy
    X = np.array(all_concepts)
    y = np.array(all_actions)
    
    # Add dataset statistics to metadata
    metadata['n_samples'] = len(y)
    metadata['n_success_samples'] = len(success_actions) if use_success else 0
    metadata['n_failed_samples'] = len(failed_actions) if use_failed else 0
    
    # Action distribution
    unique_actions, counts = np.unique(y, return_counts=True)
    metadata['action_distribution'] = {ACTION_NAMES[a]: int(c) for a, c in zip(unique_actions, counts)}
    
    print(f"\n{'='*60}")
    print(f"📈 Dataset Summary")
    print(f"{'='*60}")
    print(f"Total samples: {len(y)}")
    print(f"Concept dimensions: {X.shape[1]}")
    print(f"Action classes: {len(unique_actions)}")
    print(f"\nAction distribution:")
    for action_idx, count in zip(unique_actions, counts):
        action_name = ACTION_NAMES.get(action_idx, f"Unknown_{action_idx}")
        percentage = (count / len(y)) * 100
        print(f"  {action_name:20s}: {count:6d} ({percentage:5.2f}%)")
    print(f"{'='*60}\n")
    
    return X, y, metadata


def save_dataset(X: np.ndarray, y: np.ndarray, metadata: Dict, output_path: Path):
    """Save processed dataset to file"""
    
    data = {
        'X': X,
        'y': y,
        'metadata': metadata,
        'action_names': ACTION_NAMES
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"✓ Saved dataset to: {output_path}")
    print(f"  - Shape: X={X.shape}, y={y.shape}")


def load_dataset(path: Path) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Load processed dataset from file"""
    
    with open(path, 'rb') as f:
        data = pickle.load(f)
    
    return data['X'], data['y'], data['metadata']


def main():
    parser = argparse.ArgumentParser(description="Parse concept-action data for DT training")
    parser.add_argument("--ig-dir", type=str, required=True,
                       help="Path to IG output directory (e.g., ig_out/.../ig_20260224_124815)")
    parser.add_argument("--output", type=str, default="dt_training_data.pkl",
                       help="Output pickle file path")
    parser.add_argument("--no-success", action="store_true",
                       help="Don't use success runs")
    parser.add_argument("--no-failed", action="store_true",
                       help="Don't use failed runs")
    parser.add_argument("--failed-ratio", type=float, default=0.1,
                       help="Sampling ratio for failed runs (default: 0.1)")
    parser.add_argument("--no-preprocess-failed", action="store_true",
                       help="Don't preprocess failed runs (remove loops/duplicates)")
    parser.add_argument("--preprocess-success", action="store_true",
                       help="Remove ALL duplicate (concept, action) pairs from entire dataset (reduces 50-70%)")
    
    args = parser.parse_args()
    
    ig_dir = Path(args.ig_dir)
    if not ig_dir.exists():
        raise FileNotFoundError(f"IG directory not found: {ig_dir}")
    
    # Load data
    X, y, metadata = load_concept_action_data(
        ig_dir=ig_dir,
        use_success=not args.no_success,
        use_failed=not args.no_failed,
        failed_sample_ratio=args.failed_ratio,
        preprocess_failed=not args.no_preprocess_failed,
        preprocess_success=args.preprocess_success
    )
    
    # Save
    output_path = Path(args.output)
    save_dataset(X, y, metadata, output_path)
    
    print(f"\n✅ Dataset ready for Decision Tree training!")
    print(f"Next step: python train_decision_tree_policy.py --data {args.output}")


if __name__ == "__main__":
    main()
