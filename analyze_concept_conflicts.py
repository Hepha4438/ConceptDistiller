#!/usr/bin/env python3
"""
Analyze concept vector conflicts in MLP training data
Checks if same concept vector produces different actions
"""

import re
import argparse
from collections import defaultdict
from pathlib import Path


def parse_concept_actions(file_path):
    """
    Parse concept vectors and actions from IG log file
    
    Returns:
        dict: {concept_vector_str: {action: count}}
    """
    data = defaultdict(lambda: defaultdict(int))
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines):
        if 'Concept Vector' in line and 'after STE' in line:
            # Extract concept vector
            cv_match = re.search(r'\[([\d\., ]+)\]', line)
            if cv_match and i + 1 < len(lines):
                concept = cv_match.group(1).strip()
                
                # Extract action from next line
                next_line = lines[i + 1]
                act_match = re.search(r'Action: (\d+)', next_line)
                if act_match:
                    action = act_match.group(1)
                    data[concept][action] += 1
    
    return data


def analyze_conflicts(data, output_file=None):
    """
    Analyze and report conflicts
    
    Args:
        data: {concept: {action: count}}
        output_file: Optional file to write detailed results
    """
    action_names = {
        '0': 'TURN_LEFT',
        '1': 'TURN_RIGHT', 
        '2': 'MOVE_FORWARD',
        '3': 'PICKUP',
        '4': 'DROP',
        '5': 'TOGGLE',
        '6': 'DONE'
    }
    
    # Find conflicts
    conflicts = [(c, acts) for c, acts in data.items() if len(acts) > 1]
    total_concepts = len(data)
    total_pairs = sum(sum(acts.values()) for acts in data.values())
    
    # Print summary to console
    print('='*80)
    print('CONCEPT-ACTION CONFLICT ANALYSIS')
    print('='*80)
    print(f'\n📊 Statistics:')
    print(f'  Total unique concept vectors: {total_concepts}')
    print(f'  Total (concept, action) pairs: {total_pairs:,}')
    print(f'  Conflicting concept vectors: {len(conflicts)} ({len(conflicts)/total_concepts*100:.2f}%)')
    
    if conflicts:
        print(f'\n⚠️  ANSWER: YES - MLP produces DIFFERENT actions for same concept vector!')
        print(f'     This is why Decision Tree cannot reach 100% accuracy.')
        print(f'     Maximum theoretical DT accuracy ≈ {(total_pairs - sum(min(c.values()) for c in [acts for _, acts in conflicts]))/total_pairs*100:.1f}%')
    else:
        print(f'\n✅ ANSWER: NO - MLP is deterministic (same concept → same action)')
        print(f'     Decision Tree should be able to reach 100% accuracy on training data.')
    
    print('\n' + '='*80)
    print('CONFLICT DETAILS')
    print('='*80)
    
    if conflicts:
        # Sort by total occurrences
        conflicts_sorted = sorted(conflicts, key=lambda x: sum(x[1].values()), reverse=True)
        
        for idx, (concept, actions) in enumerate(conflicts_sorted, 1):
            total = sum(actions.values())
            print(f'\nConflict #{idx}:')
            print(f'  Concept Vector: [{concept}]')
            print(f'  Total occurrences: {total}')
            print(f'  Actions:')
            
            for action, count in sorted(actions.items(), key=lambda x: x[1], reverse=True):
                pct = count / total * 100
                action_name = action_names.get(action, 'UNKNOWN')
                print(f'    • Action {action} ({action_name}): {count:4d} times ({pct:5.1f}%)')
    else:
        print('\n✅ No conflicts found - all concept vectors map to unique actions!')
    
    # Write to file if specified
    if output_file:
        with open(output_file, 'w') as f:
            f.write('DEDUPLICATED CONCEPT-ACTION PAIRS\n')
            f.write('='*80 + '\n\n')
            
            # Write all unique pairs
            for concept, actions in sorted(data.items()):
                for action, count in sorted(actions.items()):
                    f.write(f'[{concept}] -> Action {action} ({action_names.get(action, "UNKNOWN")}): {count} occurrences\n')
            
            f.write('\n' + '='*80 + '\n')
            f.write('CONFLICTS ONLY\n')
            f.write('='*80 + '\n\n')
            
            if conflicts:
                for idx, (concept, actions) in enumerate(sorted(conflicts, key=lambda x: sum(x[1].values()), reverse=True), 1):
                    total = sum(actions.values())
                    f.write(f'\nConflict #{idx}: [{concept}]\n')
                    f.write(f'  Total: {total} occurrences\n')
                    for action, count in sorted(actions.items(), key=lambda x: x[1], reverse=True):
                        pct = count / total * 100
                        f.write(f'    Action {action} ({action_names.get(action, "UNKNOWN")}): {count} times ({pct:.1f}%)\n')
            else:
                f.write('No conflicts found!\n')
        
        print(f'\n📄 Detailed results written to: {output_file}')


def main():
    parser = argparse.ArgumentParser(description='Analyze concept-action conflicts in MLP training data')
    parser.add_argument('--input', type=str, 
                       default='ig_out/MiniGrid-DoorKey-6x6-v0/ppo_concept/ppo_concept_minigrid_016/ig_20260226_144048/success_runs.txt',
                       help='Path to success_runs.txt or failed_runs.txt')
    parser.add_argument('--output', type=str, default='concept_action_analysis.txt',
                       help='Output file for detailed results')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f'❌ Error: File not found: {input_path}')
        return
    
    print(f'\n📂 Analyzing: {input_path}')
    print(f'   Please wait...\n')
    
    # Parse data
    data = parse_concept_actions(input_path)
    
    # Analyze and report
    analyze_conflicts(data, output_file=args.output)
    
    print('\n' + '='*80)
    print('DONE')
    print('='*80 + '\n')


if __name__ == '__main__':
    main()
