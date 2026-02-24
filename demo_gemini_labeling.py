"""
Demo script - Shows what the Gemini labeling would produce
(without actually calling the API)
"""

from label_concepts_gemini import parse_success_runs
from pathlib import Path
import json

def mock_gemini_label(summary):
    """Mock Gemini response based on statistics"""
    
    labels = {}
    
    # C1 - Continuous sigmoid
    c1_stats = summary['C1']
    c1_forward_mean = c1_stats['action_correlations'].get('MOVE_FORWARD', {}).get('mean', 0)
    c1_turn_mean = c1_stats['action_correlations'].get('TURN_RIGHT', {}).get('mean', 0)
    
    labels['C1'] = {
        "label": "confidence-to-move-forward",
        "reasoning": f"C1 is continuous (sigmoid) with high activation during MOVE_FORWARD (mean={c1_forward_mean:.3f}) and low during turns (mean={c1_turn_mean:.3f}). This suggests it represents the agent's confidence or safety assessment for moving forward.",
        "confidence": "high"
    }
    
    # C2 - Binary STE
    c2_stats = summary['C2']
    c2_turn_rate = c2_stats['action_correlations'].get('TURN_RIGHT', {}).get('activation_rate', 0)
    
    labels['C2'] = {
        "label": "rotation-or-reorientation-mode",
        "reasoning": f"C2 is binary STE with very high activation during TURN_RIGHT ({c2_turn_rate*100:.1f}%) and low activation during MOVE_FORWARD. This indicates the agent is in 'rotation mode' when C2=1, likely adjusting orientation to face objects or goals.",
        "confidence": "high"
    }
    
    # C3 - Binary STE
    c3_stats = summary['C3']
    c3_forward_rate = c3_stats['action_correlations'].get('MOVE_FORWARD', {}).get('activation_rate', 0)
    c3_co_c4 = c3_stats.get('co_occurrence', {}).get('C4', 0)
    
    labels['C3'] = {
        "label": "clear-path-forward",
        "reasoning": f"C3 activates in {c3_forward_rate*100:.1f}% of MOVE_FORWARD actions but rarely during other actions. It co-occurs with C4 ({c3_co_c4} times), suggesting it represents 'clear path forward' or 'safe to advance'.",
        "confidence": "medium"
    }
    
    # C4 - Binary STE  
    c4_stats = summary['C4']
    c4_pickup_rate = c4_stats['action_correlations'].get('PICKUP_OBJECT', {}).get('activation_rate', 0)
    
    labels['C4'] = {
        "label": "at-target-object",
        "reasoning": f"C4 has perfect activation ({c4_pickup_rate*100:.0f}%) when executing PICKUP_OBJECT and low activation otherwise. This strongly indicates C4=1 means 'agent is positioned at a pickable object (key)'.",
        "confidence": "high"
    }
    
    return labels


def main():
    success_txt = Path('ig_out/MiniGrid-DoorKey-6x6-v0/ppo_concept/ppo_concept_minigrid_016/ig_20260102_093113/success_runs.txt')
    
    if not success_txt.exists():
        print(f"❌ File not found: {success_txt}")
        return
    
    print("="*70)
    print("DEMO: Automatic Concept Labeling (Mock Gemini)")
    print("="*70)
    print()
    
    # Parse statistics
    print("📊 Step 1: Parsing concept patterns...")
    summary, model_metadata = parse_success_runs(success_txt)
    print(f"✓ Parsed steps from successful episodes")
    print(f"  Model: {model_metadata['n_concepts']} concepts, {model_metadata['n_continuous_concepts']} continuous\n")
    
    # Mock Gemini labeling
    print("🤖 Step 2: Generating labels (simulated Gemini API)...")
    labels = mock_gemini_label(summary)
    print("✓ Generated semantic labels\n")
    
    # Display results
    print("="*70)
    print("CONCEPT LABELS (Gemini Mock)")
    print("="*70)
    
    for concept_id in ['C1', 'C2', 'C3', 'C4']:
        label_data = labels[concept_id]
        print(f"\n{concept_id}: {label_data['label']}")
        print(f"  Type: {'Sigmoid (continuous)' if concept_id == 'C1' else 'STE (binary)'}")
        print(f"  Confidence: {label_data['confidence']}")
        print(f"  Reasoning:")
        # Word wrap reasoning
        words = label_data['reasoning'].split()
        line = "    "
        for word in words:
            if len(line + word) > 66:
                print(line)
                line = "    " + word + " "
            else:
                line += word + " "
        if line.strip():
            print(line)
    
    print("\n" + "="*70)
    
    # Save results
    output_file = success_txt.parent / 'concept_labels_demo.json'
    with open(output_file, 'w') as f:
        json.dump(labels, f, indent=2)
    
    print(f"\n✓ Saved demo labels to: {output_file.name}")
    print(f"\n💡 To use real Gemini API:")
    print(f"   1. Get API key from https://aistudio.google.com")
    print(f"   2. Run: python label_concepts_gemini.py --ig-dir {success_txt.parent} --api-key YOUR_KEY")
    print(f"   3. Cost: ~$0.0004 per model (with 5 images)")
    

if __name__ == "__main__":
    main()
