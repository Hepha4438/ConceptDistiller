"""
Automatic Concept Labeling using Gemini API
Analyzes IG output and labels concepts based on patterns

NOTE: This script uses the new google.genai SDK (not deprecated google.generativeai)
"""

try:
    from google import genai
    from google.genai import types
    SDK_VERSION = "new"  # google.genai
except ImportError:
    import google.generativeai as genai
    SDK_VERSION = "legacy"  # google.generativeai (deprecated)
    print("⚠️  Warning: Using deprecated google.generativeai. Install: pip install google-genai")

import numpy as np
import json
from pathlib import Path
from collections import defaultdict
import argparse
from typing import Dict, List, Tuple
import re
import os


def parse_success_runs(txt_path: Path) -> Tuple[Dict, Dict]:
    """Parse success_runs.txt to extract concept-action patterns and model metadata
    
    Returns:
        (summary_stats, model_metadata)
    """
    
    with open(txt_path, 'r') as f:
        content = f.read()
    
    # Extract model metadata from first episode header
    model_metadata = {
        'n_concepts': 4,  # Default fallback
        'concept_mode': 5,  # Default fallback (PPO Mode 5) or 'ste' (PostHoc STE)
        'n_continuous_concepts': 1,  # Default fallback
        'model_type': 'PPO_CONCEPT'  # Default fallback
    }
    
    # Try to parse model info from episode header
    # Handle both PPO_CONCEPT (mode is integer) and PostHoc STE (mode is string)
    
    # Try PostHoc format first: "Model: 5 concepts, mode ste, 1 continuous"
    model_info_match = re.search(r'Model: (\d+) concepts, mode (ste|gated), (\d+) continuous', content)
    if model_info_match:
        model_metadata['n_concepts'] = int(model_info_match.group(1))
        model_metadata['concept_mode'] = model_info_match.group(2)  # 'ste' or 'gated'
        model_metadata['n_continuous_concepts'] = int(model_info_match.group(3))
        model_metadata['model_type'] = 'PostHoc'
        print(f"\n📊 Model Metadata from success_runs.txt:")
        print(f"   Model Type: PostHoc {model_metadata['concept_mode'].upper()}")
        print(f"   Total concepts: {model_metadata['n_concepts']}")
        print(f"   Continuous concepts: {model_metadata['n_continuous_concepts']}")
        print(f"   STE concepts: {model_metadata['n_concepts'] - model_metadata['n_continuous_concepts']}\n")
    else:
        # Try PPO_CONCEPT format: "Model: 8 concepts, mode 5, 2 continuous"
        model_info_match = re.search(r'Model: (\d+) concepts, mode (\d+), (\d+) continuous', content)
        if model_info_match:
            model_metadata['n_concepts'] = int(model_info_match.group(1))
            model_metadata['concept_mode'] = int(model_info_match.group(2))
            model_metadata['n_continuous_concepts'] = int(model_info_match.group(3))
            model_metadata['model_type'] = 'PPO_CONCEPT'
            print(f"\n📊 Model Metadata from success_runs.txt:")
            print(f"   Model Type: PPO_CONCEPT Mode {model_metadata['concept_mode']}")
            print(f"   Total concepts: {model_metadata['n_concepts']}")
            print(f"   Continuous concepts: {model_metadata['n_continuous_concepts']}")
            print(f"   STE concepts: {model_metadata['n_concepts'] - model_metadata['n_continuous_concepts']}\n")
        else:
            print(f"\n⚠️  No model metadata found in success_runs.txt. Using defaults: {model_metadata}\n")
    
    n_concepts = model_metadata['n_concepts']
    n_continuous = model_metadata['n_continuous_concepts']
    
    # Statistics for each concept (dynamic based on n_concepts)
    concept_stats = {
        f"C{i}": {
            "values": [],
            "actions": defaultdict(list),  # action -> [values]
            "co_occurrence": defaultdict(int)  # other_concepts -> count
        }
        for i in range(1, n_concepts + 1)
    }
    
    # Parse episodes
    episodes = content.split("======================================================================")
    
    for episode in episodes:
        if "Step" not in episode:
            continue
            
        # Extract all steps
        steps = re.findall(
            r"Step\s+\d+:\s+Concept Vector.*?\[([\d.,\s]+)\]\s+Action:\s+(\d+)\s+\((\w+)\)",
            episode,
            re.DOTALL
        )
        
        for step in steps:
            vector_str, action_num, action_name = step
            vector = [float(v.strip()) for v in vector_str.split(",")]
            
            if len(vector) != n_concepts:
                continue
            
            # Process each concept (continuous vs STE based on n_continuous)
            for i, val in enumerate(vector):
                concept_name = f"C{i+1}"
                concept_stats[concept_name]["values"].append(val)
                concept_stats[concept_name]["actions"][action_name].append(val)
                
                # Co-occurrence with other active concepts (for STE only)
                # STE concepts are at indices >= n_continuous
                if i >= n_continuous and val == 1.0:  # STE concepts
                    for j, other_val in enumerate(vector):
                        if j != i and j >= n_continuous and other_val == 1.0:
                            other_name = f"C{j+1}"
                            concept_stats[concept_name]["co_occurrence"][other_name] += 1
    
    # Compute summary statistics
    summary = {}
    for concept, data in concept_stats.items():
        if not data["values"]:
            continue
            
        values = np.array(data["values"])
        concept_idx = int(concept[1:]) - 1  # C1 -> 0, C2 -> 1, ...
        is_continuous = (concept_idx < n_continuous)
        
        # Overall statistics
        summary[concept] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "activation_rate": float(np.mean(values > 0.5)),
            "is_binary": not is_continuous,  # STE concepts are binary
            "is_continuous": is_continuous,
            "total_samples": len(values),
            "action_correlations": {},
            "co_occurrence": dict(data["co_occurrence"])
        }
        
        # Action-specific statistics
        for action, action_vals in data["actions"].items():
            if len(action_vals) < 5:  # Skip rare actions
                continue
            action_vals = np.array(action_vals)
            summary[concept]["action_correlations"][action] = {
                "mean": float(np.mean(action_vals)),
                "count": len(action_vals),
                "activation_rate": float(np.mean(action_vals > 0.5))
            }
    
    return summary, model_metadata


def select_representative_images_with_clustering(ig_out_dir: Path, n_samples: int = 5) -> Dict[str, List[Path]]:
    """
    Select representative images using clustering OR directly from concept folders
    Works with both PostHoc (has C2/, C3/, etc with activated/inactive subfolders directly) 
    and PPO_CONCEPT (has episode_XXX/C1_activated/, etc.)
    
    Returns dict: {
        "C1_low": [paths], "C1_medium": [paths], "C1_high": [paths],
        "C2_activated": [paths], "C2_inactive": [paths],
        ...
    }
    """
    from cluster_concepts import create_clustering_output
    
    print("🎯 Selecting representative images from concept folders...")
    
    # First, check if concept folders already exist at ig_out_dir level (PostHoc structure)
    # PostHoc structure: C2/, C3/, C4/ with activated/ inactive/ subfolders
    concept_folders_exist = any(ig_out_dir.glob("C[0-9]"))
    
    clustered_images = {}
    
    if concept_folders_exist:
        # PostHoc structure: concept folders are at top level with activated/inactive subfolders
        print("  ✓ Found concept folders at top level (PostHoc STE structure)")
        
        # Collect images from existing folders
        for concept_dir in sorted(ig_out_dir.glob("C[0-9]*")):
            if not concept_dir.is_dir():
                continue
            concept_name = concept_dir.name  # C2, C3, etc.
            
            # Check for activated/inactive/low/medium/high subfolders
            for state_pattern in ["activated", "inactive", "low", "medium", "high"]:
                state_dir = concept_dir / state_pattern
                if state_dir.exists() and state_dir.is_dir():
                    images = list(state_dir.glob("*.png"))
                    
                    # Sample n_samples from available images
                    if len(images) > n_samples:
                        import random
                        images = random.sample(images, n_samples)
                    
                    if images:  # Only add if we found images
                        clustered_images[f"{concept_name}_{state_pattern}"] = images
                        print(f"   ✓ {concept_name} {state_pattern}: {len(images)} samples from {len(list(state_dir.glob('*.png')))} total")
    else:
        # PPO_CONCEPT structure: need to run clustering on episodes
        print("  ℹ Clustering concept patterns from episodes...")
        
        # Run clustering
        cluster_dir = create_clustering_output(ig_out_dir, n_samples=n_samples)
        
        # Collect clustered images
        # Iterate through all concept folders
        for concept_dir in sorted(cluster_dir.glob("C*")):
            concept_name = concept_dir.name
            
            # Check if it's a continuous concept (has low/medium/high bins)
            has_bins = (concept_dir / "low").exists() or (concept_dir / "medium").exists() or (concept_dir / "high").exists()
            
            if has_bins:
                # Continuous concept - collect from bins
                for bin_name in ["low", "medium", "high"]:
                    bin_dir = concept_dir / bin_name
                    if bin_dir.exists():
                        images = list(bin_dir.glob("sample_*.png"))
                        clustered_images[f"{concept_name}_{bin_name}"] = images
                        print(f"   ✓ {concept_name} {bin_name}: {len(images)} samples")
            else:
                # Binary concept - collect from activated/inactive
                for state_name in ["activated", "inactive"]:
                    state_dir = concept_dir / state_name
                    if state_dir.exists():
                        images = list(state_dir.glob("sample_*.png"))
                        clustered_images[f"{concept_name}_{state_name}"] = images
                        print(f"   ✓ {concept_name} {state_name}: {len(images)} samples")
    
    print(f"  ✓ Total images collected: {sum(len(v) for v in clustered_images.values())}")
    return clustered_images


def create_gemini_prompt(summary: Dict, env_name: str, clustered_images: Dict[str, List[Path]], model_metadata: Dict) -> str:
    """Create detailed prompt for Gemini with clustered images and model metadata"""
    
    # Extract model metadata
    n_total_concepts = model_metadata.get('n_concepts', len(summary))
    n_continuous = model_metadata.get('n_continuous_concepts', 1)
    n_ste = n_total_concepts - n_continuous
    model_type = model_metadata.get('model_type', 'PPO_CONCEPT')
    concept_mode = model_metadata.get('concept_mode', 5)
    
    # Determine model type string for display
    if model_type == 'PostHoc':
        model_type_str = f"PostHoc {concept_mode.upper()}"
    elif model_type == 'PPO_CONCEPT':
        model_type_str = f"PPO_CONCEPT Mode {concept_mode}"
    else:
        model_type_str = str(model_type)
    
    # Build concept lists
    continuous_concepts = [f"C{i}" for i in range(1, n_continuous + 1)]
    ste_concepts = [f"C{i}" for i in range(n_continuous + 1, n_total_concepts + 1)]
    
    continuous_str = ", ".join(continuous_concepts) if continuous_concepts else "None"
    ste_str = ", ".join(ste_concepts) if ste_concepts else "None"
    all_concepts_str = ", ".join(continuous_concepts + ste_concepts)
    
    prompt = f"""You are analyzing learned concepts from a Reinforcement Learning agent trained on {env_name}.

**Model Information:**
- Architecture: {model_type_str}
- Total concepts: {n_total_concepts} ({all_concepts_str})

**Environment Description:**
- Grid world: 6x6 cells
- Task: Pick up a key, open a door, reach the goal
- Actions: TURN_LEFT, TURN_RIGHT, MOVE_FORWARD, PICKUP_OBJECT, TOGGLE_DOOR
- Observation: Agent's partial field-of-view (ego-centric)

**Concept Architecture:**
The agent has {n_total_concepts} learned concepts ({all_concepts_str}):
"""
    
    if continuous_concepts:
        prompt += f"- **{continuous_str}**: Continuous sigmoid activation (range 0-1)\n"
    
    if ste_concepts:
        prompt += f"- **{ste_str}**: Binary STE (Straight-Through Estimator) - values are 0 or 1\n"
    
    prompt += "\n**Statistical Analysis from Successful Episodes:**\n\n"
    
    for concept, stats in sorted(summary.items()):
        concept_idx = int(concept[1:]) - 1  # C1 -> 0, C2 -> 1, ...
        is_continuous = stats.get('is_continuous', concept_idx < n_continuous)
        
        prompt += f"\n### {concept} Statistics:\n"
        prompt += f"- Type: {'Continuous (sigmoid)' if is_continuous else 'Binary (STE)'}\n"
        prompt += f"- Overall mean: {stats['mean']:.3f}\n"
        prompt += f"- Std dev: {stats['std']:.3f}\n"
        
        if not is_continuous:
            prompt += f"- Activation rate: {stats['activation_rate']*100:.1f}%\n"
        
        prompt += f"- Total samples: {stats['total_samples']}\n"
        
        # Action correlations
        if stats["action_correlations"]:
            prompt += f"\n**Action Correlations for {concept}:**\n"
            for action, action_stats in sorted(stats["action_correlations"].items(), 
                                              key=lambda x: x[1]["count"], reverse=True):
                prompt += f"  - {action}: mean={action_stats['mean']:.3f}, "
                prompt += f"count={action_stats['count']}, "
                if not is_continuous:
                    prompt += f"activation_rate={action_stats['activation_rate']*100:.1f}%"
                prompt += "\n"
        
        # Co-occurrence (only for STE concepts)
        if not is_continuous and stats.get("co_occurrence"):
            prompt += f"\n**Co-activation with other concepts:**\n"
            for other_concept, count in sorted(stats["co_occurrence"].items(), 
                                              key=lambda x: x[1], reverse=True):
                prompt += f"  - Often active with {other_concept}: {count} times\n"
    
    # Add information about attached images
    prompt += "\n\n**Attached Images:**\n"
    prompt += "The following clustered representative images are attached for visual analysis:\n\n"
    
    # List continuous and binary concepts dynamically
    for i in range(1, n_continuous + 1):
        concept_name = f"C{i}"
        if any(k.startswith(f"{concept_name}_") for k in clustered_images.keys()):
            prompt += f"\n**{concept_name} (Continuous Concept) - Representative samples from 3 value ranges:**\n"
            for bin_name in ["low", "medium", "high"]:
                key = f"{concept_name}_{bin_name}"
                if key in clustered_images:
                    prompt += f"  - {concept_name} {bin_name} values (0.0-0.33/0.33-0.67/0.67-1.0): {len(clustered_images[key])} samples\n"
    
    # Binary concepts (STE)
    for i in range(n_continuous + 1, n_total_concepts + 1):
        concept_name = f"C{i}"
        act_key = f"{concept_name}_activated"
        inact_key = f"{concept_name}_inactive"
        
        if act_key in clustered_images or inact_key in clustered_images:
            prompt += f"\n**{concept_name} (Binary Concept):**\n"
            if act_key in clustered_images:
                prompt += f"  - When ACTIVATED (value=1): {len(clustered_images[act_key])} samples\n"
            if inact_key in clustered_images:
                prompt += f"  - When INACTIVE (value=0): {len(clustered_images[inact_key])} samples\n"
    
    prompt += "\nEach image shows:\n"
    prompt += "  - Left: God View (full environment from above)\n"
    prompt += "  - Middle: Model Input (agent's ego-centric view)\n"
    prompt += "  - Right side: Heatmaps showing spatial attribution per concept\n"
    
    # Build guidelines dynamically
    continuous_examples = ", ".join(continuous_concepts) if continuous_concepts else "C1"
    ste_examples = ", ".join(ste_concepts) if ste_concepts else "C2, C3, ..."
    
    prompt += f"""

**Your Task:**
Based on the statistical patterns AND the attached clustered heatmap images, 
provide semantic labels for each concept.

**Guidelines:**
"""
    
    if continuous_concepts:
        prompt += f"- {continuous_examples} (sigmoid): May represent continuous features like \"confidence\", \"distance to goal\", \"obstacle proximity\"\n"
        prompt += f"  * Compare low/medium/high value samples to understand what these concepts are measuring\n"
    
    if ste_concepts:
        prompt += f"- {ste_examples} (binary): More likely discrete states like \"has-key\", \"door-open\", \"facing-goal\", \"at-wall\"\n"
        prompt += f"  * Compare activated vs inactive samples to understand when concept \"turns on\"\n"
        prompt += f"  * Look at heatmaps to see WHAT the concept is attending to\n"
    
    prompt += """- Consider action correlations (e.g., high activation with TOGGLE_DOOR → "at-door")
- Consider co-activation patterns (e.g., concepts often together)
- Use visual patterns from heatmaps to confirm or refine your hypothesis

**Confidence Scoring Guidelines (STRICT):**

For **HIGH confidence**, ALL of the following must be met:
- Statistical pattern is extremely clear and consistent (>85% correlation or very distinct value ranges)
- Visual heatmaps show OBVIOUS and CONSISTENT spatial patterns across ALL samples
- Action correlations are VERY STRONG and make logical sense (>85% correlation)
- The concept's meaning is UNAMBIGUOUS and easily explainable
- **The concept represents ONE CLEAR, DISTINCT semantic meaning** (not multiple overlapping concepts)
- **No ambiguity or overlap** with other possible interpretations

For **MEDIUM confidence**, ALL of the following must be met:
- Statistical pattern shows clear trend (70-85% correlation)
- Visual heatmaps show recognizable patterns in MOST samples (with some variation)
- Action correlations are reasonably strong (70-85% correlation)
- The concept's meaning is plausible but may have alternative interpretations
- Some minor overlap with other concepts is acceptable, but ONE interpretation dominates

For **LOW confidence**:
- Statistical pattern is weak or inconsistent (50-69% correlation)
- Visual heatmaps show mixed or unclear patterns
- Action correlations are weak or contradictory (50-69% correlation)
- Multiple competing interpretations are equally plausible
- **The concept appears to represent MULTIPLE overlapping semantic meanings** (e.g., could be "has-key" OR "near-door" OR "ready-to-act")
- **Unclear whether concept is unitary or composite**

Use **UNLABELED** when:
- No clear pattern emerges from statistics OR visuals (<50% correlation)
- Heatmaps are random or incomprehensible
- Cannot form any reasonable hypothesis about the concept's meaning
- The concept appears to be noise or an artifact
- **Severe overlap/confusion**: The concept could equally represent 3+ different semantic meanings with no way to distinguish

**CRITICAL: Concept Overlap/Ambiguity Rule**
If a concept shows patterns that could reasonably represent MULTIPLE different semantic concepts (e.g., activates both when "has-key" AND when "facing-door"), you MUST:
1. Explicitly mention this overlap in your reasoning
2. Assign LOW confidence (or UNLABELED if severely unclear)
3. State something like: "Concept shows overlap between X and Y - not a clear unitary semantic"

**Important:** When in doubt between two confidence levels, ALWAYS choose the LOWER one.
Default to LOW/UNLABELED rather than MEDIUM/HIGH if the evidence is not overwhelming.

**Examples of SUFFICIENT evidence for HIGH confidence (>85%):**
✅ "C2 activates in 100% of TOGGLE_DOOR actions (10/10), heatmaps consistently highlight door region with >0.8 intensity in all activated samples, and never activates when agent is >3 cells from door"
✅ "C3 shows distinct value progression: low (mean=0.15) when far from goal, medium (mean=0.52) when mid-distance, high (mean=0.89) when adjacent. Heatmaps show focused attention on goal object in all high-value samples."
✅ "C4 activates exclusively (100%, 15/15 samples) when agent holds the key, and never activates without key. Heatmaps show attention on key/inventory area. Clear unitary concept: 'has-key'."

**Examples of MEDIUM confidence (70-85%):**
⚠️ "C2 activates in 85% of TOGGLE_DOOR actions (17/20), with occasional false positives when near door. Heatmaps usually highlight door region but sometimes show noise."
⚠️ "C3 correlates with distance to goal (r=0.75), with clear visual patterns in 15/20 samples. Some ambiguity in mid-range states."

**Examples of LOW confidence (50-69%):**
⚠️ "Activates in 65% of PICKUP actions, but also seen in 40% of MOVE_FORWARD. Heatmaps show mixed patterns."
⚠️ **"C2 activates both when picking up key (70%, 7/10) AND when toggling door (60%, 6/10), heatmaps highlight both key and door regions"** → **LOW confidence: Overlap between 'has-key' and 'at-door' - not unitary**

**Examples of UNLABELED (<50%):**
❌ "The concept seems to relate to doors" (too vague, no specific evidence)
❌ "High activation with MOVE_FORWARD suggests movement" (circular reasoning)
❌ "Heatmaps show some patterns" (not specific enough)
❌ **"C3 could represent 'near-goal' OR 'task-complete' OR 'forward-clearance' based on different samples"** → **UNLABELED: Multiple overlapping interpretations**
❌ **"Activates in 45% of samples across ALL actions with no clear spatial pattern"** → **UNLABELED: Too diffuse, no clear semantic meaning**

**Remember: Be SKEPTICAL. Only assign HIGH confidence when evidence is overwhelming (>85% correlation). When unclear, use LOW or UNLABELED.**

**Output Format (JSON):**
{{"""
    
    # Add all concepts to output format (continuous + STE)
    all_concept_list = continuous_concepts + ste_concepts
    for i, concept in enumerate(all_concept_list):
        is_continuous = concept in continuous_concepts
        separator = "" if i == 0 else ","
        
        prompt += f"""{separator}
  "{concept}": {{
    "label": "short_semantic_label (or 'Unable to label' if cannot determine)",
    "reasoning": "DETAILED explanation citing SPECIFIC quantitative evidence from BOTH statistics AND visual patterns. Must justify confidence level with numbers (e.g., '{'95% activation rate' if not is_continuous else 'mean=0.85 in high bin'}', '{'all 10 samples show X pattern' if not is_continuous else 'clear value progression'}').",
    "confidence": "high/medium/low/unlabeled"
  }}"""
    
    prompt += "\n}\n\nProvide your analysis:\n"
    
    return prompt


def label_concepts_with_gemini(
    ig_out_dir: Path,
    api_key: str,
    n_sample_images: int = 5,
    model_name: str = "gemini-1.5-flash"
) -> Dict:
    """Main function to label concepts using Gemini API"""
    
    # Convert to Path if string, and normalize the path
    if isinstance(ig_out_dir, str):
        ig_out_dir = Path(ig_out_dir)
    
    # Resolve to absolute path and normalize (handles //, .., etc)
    ig_out_dir = ig_out_dir.resolve()
    
    print(f"🔍 Analyzing IG output: {ig_out_dir}")
    
    # Parse success_runs.txt
    success_txt = ig_out_dir / "success_runs.txt"
    if not success_txt.exists():
        raise FileNotFoundError(f"success_runs.txt not found in {ig_out_dir}")
    
    print("📊 Parsing concept patterns...")
    summary, model_metadata = parse_success_runs(success_txt)
    
    # Save summary to file
    summary_file = ig_out_dir / "concept_statistics.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved statistics to {summary_file}")
    
    # Perform clustering and select representative images
    print(f"\n🖼️  Clustering concepts and selecting {n_sample_images} samples per cluster...")
    clustered_images = select_representative_images_with_clustering(ig_out_dir, n_sample_images)
    
    # Flatten all images for sending to Gemini
    all_image_paths = []
    for category, paths in sorted(clustered_images.items()):
        all_image_paths.extend(paths)
    
    print(f"   Total images to send to Gemini: {len(all_image_paths)}")
    
    # Create prompt
    env_name = ig_out_dir.parts[-4]  # Extract env name from path
    prompt = create_gemini_prompt(summary, env_name, clustered_images, model_metadata)
    
    # Save prompt for debugging
    prompt_file = ig_out_dir / "gemini_prompt.txt"
    with open(prompt_file, 'w') as f:
        f.write(prompt)
    print(f"✓ Saved prompt to {prompt_file}")
    
    # Configure Gemini API
    print(f"\n🤖 Calling Gemini API...")
    print(f"   SDK version: {SDK_VERSION}")
    
    # Try multiple model names (fallback logic)
    # Note: Legacy SDK requires "models/" prefix
    model_names_to_try = [
        "models/gemini-2.5-flash",        # Latest fast model (2026)
        "models/gemini-2.0-flash",        # Stable fast model
        "models/gemini-flash-latest",     # Alias to latest flash
        "models/gemini-2.5-pro",          # More capable but slower
        "models/gemini-pro-latest",       # Alias to latest pro
        "models/gemini-2.0-flash-exp",    # Experimental
        "gemini-1.5-flash-latest",        # Old format (no prefix)
        "gemini-1.5-flash",
        "gemini-1.5-pro",
        "gemini-pro"
    ]
    
    response_text = None
    last_error = None
    
    if SDK_VERSION == "new":
        # New google.genai SDK
        client = genai.Client(api_key=api_key)
        
        # Upload images
        uploaded_files = []
        for img_path in all_image_paths:
            try:
                with open(img_path, 'rb') as f:
                    img_data = f.read()
                uploaded_files.append(types.Part.from_bytes(data=img_data, mime_type='image/png'))
                print(f"   ✓ Prepared: {img_path.name}")
            except Exception as e:
                print(f"   ⚠ Failed to prepare {img_path.name}: {e}")
        
        print(f"\n   📊 Summary:")
        print(f"      - Prompt length: {len(prompt)} chars")
        print(f"      - Images prepared: {len(uploaded_files)}")
        print(f"      - Total content parts: {1 + len(uploaded_files)} (1 text + {len(uploaded_files)} images)")
        
        # Try each model until one works
        for model_attempt in model_names_to_try:
            try:
                print(f"   Trying model: {model_attempt}...")
                response = client.models.generate_content(
                    model=model_attempt,
                    contents=[prompt] + uploaded_files
                )
                print(f"   ✅ Success with model: {model_attempt}")
                print(f"      - Response length: {len(response.text)} chars")
                response_text = response.text
                break
            except Exception as e:
                last_error = e
                print(f"   ⚠ Failed with {model_attempt}: {str(e)[:100]}")
                continue
        
        if response_text is None:
            print(f"❌ All models failed. Last error: {last_error}")
            raise last_error
            
    else:
        # Legacy google.generativeai SDK
        genai.configure(api_key=api_key)
        
        # Upload images first
        uploaded_files = []
        for img_path in all_image_paths:
            try:
                uploaded_file = genai.upload_file(img_path)
                uploaded_files.append(uploaded_file)
                print(f"   ✓ Uploaded: {img_path.name}")
            except Exception as e:
                print(f"   ⚠ Failed to upload {img_path.name}: {e}")
        
        print(f"\n   📊 Summary:")
        print(f"      - Prompt length: {len(prompt)} chars")
        print(f"      - Images uploaded: {len(uploaded_files)}")
        print(f"      - Total content parts: {1 + len(uploaded_files)} (1 text + {len(uploaded_files)} images)")
        
        # Try each model until one works
        for model_attempt in model_names_to_try:
            try:
                print(f"   Trying model: {model_attempt}...")
                model = genai.GenerativeModel(model_attempt)
                response = model.generate_content([prompt] + uploaded_files)
                print(f"   ✅ Success with model: {model_attempt}")
                print(f"      - Response length: {len(response.text)} chars")
                response_text = response.text
                break
            except Exception as e:
                last_error = e
                print(f"   ⚠ Failed with {model_attempt}: {str(e)[:100]}")
                continue
        
        if response_text is None:
            print(f"❌ All models failed. Last error: {last_error}")
            raise last_error
    
    print(f"\n✅ Received response from Gemini")
    
    # Parse JSON response (common for both SDKs)
    # Extract JSON (handle markdown code blocks)
    json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
    if json_match:
        response_text = json_match.group(1)
    
    try:
        labels = json.loads(response_text)
    except json.JSONDecodeError as e:
        print(f"❌ Failed to parse JSON response: {e}")
        print(f"   Raw response: {response_text[:500]}...")
        raise
        
    # Save labels
    labels_file = ig_out_dir / "concept_labels.json"
    with open(labels_file, 'w') as f:
        json.dump(labels, f, indent=2)
    print(f"✓ Saved labels to {labels_file}")
    
    # Print results
    print("\n" + "="*70)
    print("CONCEPT LABELS")
    print("="*70)
    for concept in ["C1", "C2", "C3", "C4"]:
        if concept in labels:
            print(f"\n{concept}: {labels[concept]['label']}")
            print(f"  Reasoning: {labels[concept]['reasoning']}")
            print(f"  Confidence: {labels[concept]['confidence']}")
    print("="*70)
    
    return labels


def main():
    parser = argparse.ArgumentParser(description="Label concepts using Gemini API")
    parser.add_argument("--ig-dir", type=str, required=True,
                       help="Path to IG output directory (e.g., ig_out/.../ig_20260102_080512)")
    parser.add_argument("--api-key", type=str, required=True,
                       help="Gemini API key from aistudio.google.com")
    parser.add_argument("--n-images", type=int, default=5,
                       help="Number of sample images to send to Gemini")
    parser.add_argument("--model", type=str, default="gemini-1.5-flash",
                       choices=["gemini-1.5-flash", "gemini-1.5-pro"],
                       help="Gemini model to use")
    
    args = parser.parse_args()
    
    # Normalize and resolve path
    ig_dir = Path(args.ig_dir).resolve()
    if not ig_dir.exists():
        raise FileNotFoundError(f"Directory not found: {ig_dir}")
    
    labels = label_concepts_with_gemini(
        ig_out_dir=ig_dir,
        api_key=args.api_key,
        n_sample_images=args.n_images,
        model_name=args.model
    )
    
    return labels


if __name__ == "__main__":
    main()
