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


def parse_success_runs(txt_path: Path) -> Dict:
    """Parse success_runs.txt to extract concept-action patterns"""
    
    with open(txt_path, 'r') as f:
        content = f.read()
    
    # Statistics for each concept
    concept_stats = {
        f"C{i}": {
            "values": [],
            "actions": defaultdict(list),  # action -> [values]
            "co_occurrence": defaultdict(int)  # other_concepts -> count
        }
        for i in range(1, 5)  # C1, C2, C3, C4
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
            
            if len(vector) != 4:
                continue
            
            # C1 is sigmoid (index 0), C2-C4 are STE (index 1-3)
            for i, val in enumerate(vector):
                concept_name = f"C{i+1}"
                concept_stats[concept_name]["values"].append(val)
                concept_stats[concept_name]["actions"][action_name].append(val)
                
                # Co-occurrence with other active concepts (for STE only)
                if i >= 1 and val == 1.0:  # STE concepts
                    for j, other_val in enumerate(vector):
                        if j != i and j >= 1 and other_val == 1.0:
                            other_name = f"C{j+1}"
                            concept_stats[concept_name]["co_occurrence"][other_name] += 1
    
    # Compute summary statistics
    summary = {}
    for concept, data in concept_stats.items():
        if not data["values"]:
            continue
            
        values = np.array(data["values"])
        
        # Overall statistics
        summary[concept] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "activation_rate": float(np.mean(values > 0.5)),  # For C1 (sigmoid)
            "is_binary": concept != "C1",  # C2-C4 are binary STE
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
    
    return summary


def select_representative_images_with_clustering(ig_out_dir: Path, n_samples: int = 5) -> Dict[str, List[Path]]:
    """
    Select representative images using clustering
    Returns dict: {
        "C1_low": [paths], "C1_medium": [paths], "C1_high": [paths],
        "C2_activated": [paths], "C2_inactive": [paths],
        ...
    }
    """
    from cluster_concepts import create_clustering_output
    
    print("🎯 Performing clustering on all concepts...")
    
    # Run clustering
    cluster_dir = create_clustering_output(ig_out_dir, n_samples=n_samples)
    
    # Collect clustered images
    clustered_images = {}
    
    # C1 bins
    c1_dir = cluster_dir / "C1"
    if c1_dir.exists():
        for bin_name in ["low", "medium", "high"]:
            bin_dir = c1_dir / bin_name
            if bin_dir.exists():
                images = list(bin_dir.glob("sample_*.png"))
                clustered_images[f"C1_{bin_name}"] = images
                print(f"   ✓ C1 {bin_name}: {len(images)} samples")
    
    # Binary concepts (C2, C3, C4, ...)
    for concept_dir in sorted(cluster_dir.glob("C*")):
        if concept_dir.name == "C1":
            continue
        
        concept_name = concept_dir.name
        
        for state_name in ["activated", "inactive"]:
            state_dir = concept_dir / state_name
            if state_dir.exists():
                images = list(state_dir.glob("sample_*.png"))
                clustered_images[f"{concept_name}_{state_name}"] = images
                print(f"   ✓ {concept_name} {state_name}: {len(images)} samples")
    
    return clustered_images


def create_gemini_prompt(summary: Dict, env_name: str, clustered_images: Dict[str, List[Path]]) -> str:
    """Create detailed prompt for Gemini with clustered images"""
    
    # Detect number of concepts from clustered_images
    binary_concepts = set()
    for key in clustered_images.keys():
        if "_activated" in key or "_inactive" in key:
            concept = key.split("_")[0]  # Extract C2, C3, C4, ...
            binary_concepts.add(concept)
    
    binary_concepts = sorted(binary_concepts)
    n_total_concepts = 1 + len(binary_concepts)  # C1 + binary concepts
    
    # Build concept list string
    if binary_concepts:
        binary_list = ", ".join(binary_concepts)
        all_concepts = f"C1, {binary_list}"
    else:
        all_concepts = "C1"
    
    prompt = f"""You are analyzing learned concepts from a Reinforcement Learning agent trained on {env_name}.

**Environment Description:**
- Grid world: 6x6 cells
- Task: Pick up a key, open a door, reach the goal
- Actions: TURN_LEFT, TURN_RIGHT, MOVE_FORWARD, PICKUP_OBJECT, TOGGLE_DOOR
- Observation: Agent's partial field-of-view (ego-centric)

**Concept Architecture:**
The agent has {n_total_concepts} learned concepts ({all_concepts}):
- **C1**: Continuous sigmoid activation (range 0-1)
"""
    
    if binary_concepts:
        prompt += f"- **{binary_list}**: Binary STE (Straight-Through Estimator) - values are 0 or 1\n"
    
    prompt += "\n**Statistical Analysis from Successful Episodes:**\n\n"
    
    for concept, stats in sorted(summary.items()):
        prompt += f"\n### {concept} Statistics:\n"
        prompt += f"- Type: {'Continuous (sigmoid)' if concept == 'C1' else 'Binary (STE)'}\n"
        prompt += f"- Overall mean: {stats['mean']:.3f}\n"
        prompt += f"- Std dev: {stats['std']:.3f}\n"
        
        if concept != "C1":
            prompt += f"- Activation rate: {stats['activation_rate']*100:.1f}%\n"
        
        prompt += f"- Total samples: {stats['total_samples']}\n"
        
        # Action correlations
        if stats["action_correlations"]:
            prompt += f"\n**Action Correlations for {concept}:**\n"
            for action, action_stats in sorted(stats["action_correlations"].items(), 
                                              key=lambda x: x[1]["count"], reverse=True):
                prompt += f"  - {action}: mean={action_stats['mean']:.3f}, "
                prompt += f"count={action_stats['count']}, "
                if concept != "C1":
                    prompt += f"activation_rate={action_stats['activation_rate']*100:.1f}%"
                prompt += "\n"
        
        # Co-occurrence (only for STE concepts)
        if concept != "C1" and stats.get("co_occurrence"):
            prompt += f"\n**Co-activation with other concepts:**\n"
            for other_concept, count in sorted(stats["co_occurrence"].items(), 
                                              key=lambda x: x[1], reverse=True):
                prompt += f"  - Often active with {other_concept}: {count} times\n"
    
    # Add information about attached images
    prompt += "\n\n**Attached Images:**\n"
    prompt += "The following clustered representative images are attached for visual analysis:\n\n"
    
    # C1 bins
    if any(k.startswith("C1_") for k in clustered_images.keys()):
        prompt += "**C1 (Continuous Concept) - Representative samples from 3 value ranges:**\n"
        for bin_name in ["low", "medium", "high"]:
            key = f"C1_{bin_name}"
            if key in clustered_images:
                prompt += f"  - C1 {bin_name} values (0.0-0.33/0.33-0.67/0.67-1.0): {len(clustered_images[key])} samples\n"
    
    # Binary concepts
    for i in range(2, 10):  # C2-C9
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
    binary_examples = ", ".join(binary_concepts) if binary_concepts else "C2, C3, ..."
    
    prompt += f"""

**Your Task:**
Based on the statistical patterns AND the attached clustered heatmap images, 
provide semantic labels for each concept.

**Guidelines:**
- C1 (sigmoid): May represent continuous features like "confidence", "distance to goal", "obstacle proximity"
  * Compare low/medium/high value samples to understand what C1 is measuring
- {binary_examples} (binary): More likely discrete states like "has-key", "door-open", "facing-goal", "at-wall"
  * Compare activated vs inactive samples to understand when concept "turns on"
  * Look at heatmaps to see WHAT the concept is attending to
- Consider action correlations (e.g., high activation with TOGGLE_DOOR → "at-door")
- Consider co-activation patterns (e.g., C2+C3 often together)
- Use visual patterns from heatmaps to confirm or refine your hypothesis

**Output Format (JSON):**
{{
  "C1": {{
    "label": "short_semantic_label (or 'Unable to label' if cannot determine)",
    "reasoning": "explanation based on statistics AND visual patterns from low/med/high samples",
    "confidence": "high/medium/low/unlabeled"
  }}"""
    
    # Add binary concepts to output format
    for concept in binary_concepts:
        prompt += f""",
  "{concept}": {{
    "label": "short_semantic_label (or 'Unable to label' if cannot determine)",
    "reasoning": "explanation based on statistics AND visual patterns from activated/inactive samples",
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
    
    print(f"🔍 Analyzing IG output: {ig_out_dir}")
    
    # Parse success_runs.txt
    success_txt = ig_out_dir / "success_runs.txt"
    if not success_txt.exists():
        raise FileNotFoundError(f"success_runs.txt not found in {ig_out_dir}")
    
    print("📊 Parsing concept patterns...")
    summary = parse_success_runs(success_txt)
    
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
    prompt = create_gemini_prompt(summary, env_name, clustered_images)
    
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
        
        # Try each model until one works
        for model_attempt in model_names_to_try:
            try:
                print(f"   Trying model: {model_attempt}...")
                response = client.models.generate_content(
                    model=model_attempt,
                    contents=[prompt] + uploaded_files
                )
                print(f"   ✅ Success with model: {model_attempt}")
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
        
        # Try each model until one works
        for model_attempt in model_names_to_try:
            try:
                print(f"   Trying model: {model_attempt}...")
                model = genai.GenerativeModel(model_attempt)
                response = model.generate_content([prompt] + uploaded_files)
                print(f"   ✅ Success with model: {model_attempt}")
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
    
    ig_dir = Path(args.ig_dir)
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
