import re
from collections import defaultdict

file_path = "/Users/HIEU/Downloads/ConceptDistiller/ig_out/MiniGrid-DoorKey-6x6-v0/ppo_concept/ppo_concept_minigrid_016/ig_20260225_225550/success_runs.txt"

# Parse concept vectors and actions
concept_action_map = defaultdict(set)
concept_counts = defaultdict(int)

with open(file_path, 'r') as f:
    lines = f.readlines()
    
for i, line in enumerate(lines):
    if "Concept Vector" in line:
        # Extract concept vector
        match_concept = re.search(r'\[([\d\., ]+)\]', line)
        if match_concept:
            concept = match_concept.group(1).strip()
            
            # Get next line for action
            if i + 1 < len(lines):
                next_line = lines[i + 1]
                match_action = re.search(r'Action: (\d+)', next_line)
                if match_action:
                    action = match_action.group(1)
                    concept_action_map[concept].add(action)
                    concept_counts[concept] += 1

# Find conflicts (same concept → different actions)
conflicts = []
for concept, actions in concept_action_map.items():
    if len(actions) > 1:
        conflicts.append((concept, actions, concept_counts[concept]))

print(f"Total unique concept vectors: {len(concept_action_map)}")
print(f"Total concept-action pairs: {sum(concept_counts.values())}")
print(f"\nConflicts found: {len(conflicts)}")

if conflicts:
    print(f"\n{'='*80}")
    print("CONFLICTS - Same concept vector → Different actions:")
    print(f"{'='*80}\n")
    for concept, actions, count in sorted(conflicts, key=lambda x: x[2], reverse=True):
        print(f"Concept: [{concept}]")
        print(f"  Actions: {sorted(actions)}")
        print(f"  Occurrences: {count}")
        print()
else:
    print("\n✅ NO CONFLICTS: Each unique concept vector always produces the same action")

# Show some statistics
print(f"\n{'='*80}")
print("Top 10 most frequent concept vectors:")
print(f"{'='*80}")
for concept, count in sorted(concept_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
    actions = concept_action_map[concept]
    print(f"[{concept}] → Action {list(actions)[0]} (occurs {count} times)")
