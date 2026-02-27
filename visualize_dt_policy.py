"""
Visualize Decision Tree Policy for Interpretability

Creates comprehensive visualizations to understand DT decision-making:
- Tree structure with concept splits
- Feature importance analysis
- Decision path tracing
- Concept-action pattern heatmaps
- Integration with Gemini labels for semantic interpretation
"""

import numpy as np
# Set matplotlib to non-interactive backend (MUST be before importing pyplot)
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for threading
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional
from sklearn.tree import plot_tree, export_text
import matplotlib.patches as mpatches

from parse_concept_actions import ACTION_NAMES


class DTVisualizer:
    """Visualizer for Decision Tree policy interpretability"""
    
    def __init__(self, dt_model_path: str, concept_labels_path: Optional[str] = None):
        """
        Args:
            dt_model_path: Path to trained DT model (.pkl)
            concept_labels_path: Optional path to concept_labels.json from Gemini
        """
        
        # Load DT model
        with open(dt_model_path, 'rb') as f:
            dt_data = pickle.load(f)
        
        self.dt_model = dt_data['model']
        self.metadata = dt_data.get('metadata', {})
        self.action_names = dt_data.get('action_names', ACTION_NAMES)
        
        # Get concept configuration
        self.n_concepts = self.metadata.get('n_concepts', 10)
        self.n_continuous = self.metadata.get('n_continuous_concepts', 1)
        
        # Load concept labels if provided
        self.concept_labels = None
        if concept_labels_path and Path(concept_labels_path).exists():
            with open(concept_labels_path, 'r', encoding='utf-8') as f:
                labels_data = json.load(f)
                self.concept_labels = {
                    f"C{i+1}": labels_data.get(f"C{i+1}", {}).get("concept_name", f"Concept {i+1}")
                    for i in range(self.n_concepts)
                }
        else:
            # Default labels
            self.concept_labels = {f"C{i+1}": f"Concept {i+1}" for i in range(self.n_concepts)}
        
        print(f"✓ Loaded DT model with {self.n_concepts} concepts")
        print(f"  Continuous: {self.n_continuous}, Binary: {self.n_concepts - self.n_continuous}")
        if concept_labels_path:
            print(f"✓ Loaded concept labels from: {concept_labels_path}")
    
    def plot_tree_structure(self, output_path: str, max_depth: Optional[int] = None, figsize=(30, 20)):
        """
        Plot full tree structure with sklearn's plot_tree
        
        Args:
            output_path: Where to save the plot
            max_depth: Maximum depth to display (None = full tree)
            figsize: Figure size (larger for detailed trees)
        """
        
        print(f"\n📊 Plotting tree structure...")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Build feature names with semantic labels
        feature_names = [
            f"{self.concept_labels.get(f'C{i+1}', f'C{i+1}')}"
            for i in range(self.n_concepts)
        ]
        
        # Build class names from action names
        class_names = [
            self.action_names.get(i, f"Action_{i}")
            for i in range(len(self.action_names))
        ]
        
        # Plot tree
        plot_tree(
            self.dt_model,
            ax=ax,
            feature_names=feature_names,
            class_names=class_names,
            filled=True,
            rounded=True,
            fontsize=10,
            max_depth=max_depth,
            impurity=True,
            proportion=True
        )
        
        ax.set_title(
            f"Decision Tree Policy Visualization\n"
            f"Environment: {self.metadata.get('environment', 'N/A')}, "
            f"Depth: {self.dt_model.get_depth()}, "
            f"Nodes: {self.dt_model.tree_.node_count}",
            fontsize=16,
            pad=20
        )
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved tree structure to: {output_path}")
    
    def plot_feature_importance_detailed(self, output_path: str, top_k: Optional[int] = None):
        """
        Detailed feature importance plot with concept labels and type indicators
        
        Args:
            output_path: Where to save the plot
            top_k: Show only top K features (None = all)
        """
        
        print(f"\n📊 Plotting detailed feature importance...")
        
        importances = self.dt_model.feature_importances_
        
        # Create concept info
        concept_info = []
        for i in range(self.n_concepts):
            concept_type = "Continuous" if i < self.n_continuous else "Binary"
            label = self.concept_labels.get(f"C{i+1}", f"C{i+1}")
            concept_info.append({
                'index': i,
                'name': f"C{i+1}",
                'label': label,
                'type': concept_type,
                'importance': importances[i]
            })
        
        # Sort by importance
        concept_info = sorted(concept_info, key=lambda x: x['importance'], reverse=True)
        
        # Filter top-k
        if top_k:
            concept_info = concept_info[:top_k]
        
        # Prepare data for plotting
        names = [c['name'] for c in concept_info]
        labels = [c['label'] for c in concept_info]
        types = [c['type'] for c in concept_info]
        values = [c['importance'] for c in concept_info]
        
        # Color by type
        colors = ['#3498db' if t == 'Continuous' else '#e74c3c' for t in types]
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, max(8, len(concept_info) * 0.4)))
        
        y_pos = np.arange(len(names))
        bars = ax.barh(y_pos, values, color=colors, alpha=0.7)
        
        # Annotate with labels
        ax.set_yticks(y_pos)
        yticklabels = [f"{name}\n{label[:30]}..." if len(label) > 30 else f"{name}\n{label}"
                       for name, label in zip(names, labels)]
        ax.set_yticklabels(yticklabels, fontsize=10)
        
        ax.set_xlabel('Importance (Gini decrease)', fontsize=12)
        ax.set_title('Feature Importance with Semantic Labels', fontsize=14, pad=15)
        
        # Add legend for concept types
        continuous_patch = mpatches.Patch(color='#3498db', alpha=0.7, label='Continuous')
        binary_patch = mpatches.Patch(color='#e74c3c', alpha=0.7, label='Binary')
        ax.legend(handles=[continuous_patch, binary_patch], loc='lower right')
        
        # Add grid
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved detailed feature importance to: {output_path}")
        
        # Print top features
        print(f"\n🏆 Top {min(5, len(concept_info))} Most Important Concepts:")
        for i, c in enumerate(concept_info[:5], 1):
            print(f"  {i}. {c['name']} ({c['type']}): {c['importance']:.4f}")
            print(f"     Label: {c['label']}")
    
    def plot_concept_action_heatmap(self, data_path: str, output_path: str):
        """
        Heatmap showing concept activation patterns for each action
        Requires training data to analyze patterns
        
        Args:
            data_path: Path to training data (.pkl from parse_concept_actions.py)
            output_path: Where to save the plot
        """
        
        print(f"\n📊 Plotting concept-action heatmap...")
        
        # Load training data
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        X = data['X']  # [N, K]
        y = data['y']  # [N,]
        
        # Compute mean activation per action
        n_actions = len(self.action_names)
        concept_patterns = np.zeros((n_actions, self.n_concepts))
        
        for action_idx in range(n_actions):
            mask = (y == action_idx)
            if mask.sum() > 0:
                concept_patterns[action_idx] = X[mask].mean(axis=0)
        
        # Plot heatmap
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Action names for y-axis
        action_labels = [self.action_names.get(i, f"A{i}") for i in range(n_actions)]
        
        # Concept names for x-axis
        concept_names = [f"C{i+1}" for i in range(self.n_concepts)]
        
        # Create heatmap
        sns.heatmap(
            concept_patterns,
            ax=ax,
            xticklabels=concept_names,
            yticklabels=action_labels,
            cmap='viridis',
            annot=True,
            fmt='.2f',
            cbar_kws={'label': 'Mean Activation'},
            linewidths=0.5,
            linecolor='white'
        )
        
        ax.set_xlabel('Concepts', fontsize=12)
        ax.set_ylabel('Actions', fontsize=12)
        ax.set_title('Concept Activation Patterns per Action', fontsize=14, pad=15)
        
        # Add type markers (continuous vs binary)
        for i in range(self.n_concepts):
            if i < self.n_continuous:
                ax.axvline(x=i+1, color='blue', linestyle='--', alpha=0.3, linewidth=1)
        
        # Add legend
        continuous_line = plt.Line2D([0], [0], color='blue', linestyle='--', alpha=0.3, linewidth=2, label='Continuous')
        ax.legend(handles=[continuous_line], loc='upper left', bbox_to_anchor=(1.15, 1))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved concept-action heatmap to: {output_path}")
    
    def trace_decision_paths(self, sample_concepts: np.ndarray, output_path: str):
        """
        Trace decision paths for sample concept vectors through the tree
        
        Args:
            sample_concepts: Array of concept vectors [M, K]
            output_path: Where to save the trace
        """
        
        print(f"\n📊 Tracing decision paths for {len(sample_concepts)} samples...")
        
        # Get decision paths
        decision_paths = self.dt_model.decision_path(sample_concepts)
        
        # Get leaf IDs
        leaf_ids = self.dt_model.apply(sample_concepts)
        
        # Get predictions
        predictions = self.dt_model.predict(sample_concepts)
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("DECISION PATH TRACES\n")
            f.write("="*80 + "\n\n")
            
            for idx, (concept_vec, leaf_id, action) in enumerate(zip(sample_concepts, leaf_ids, predictions)):
                f.write(f"\nSample {idx+1}:\n")
                f.write(f"{'-'*60}\n")
                f.write(f"Concept Vector: {concept_vec}\n")
                f.write(f"Predicted Action: {self.action_names.get(action, f'A{action}')}\n")
                f.write(f"Leaf Node ID: {leaf_id}\n\n")
                
                # Extract path
                path_nodes = decision_paths[idx].nonzero()[1]
                
                f.write("Decision Path:\n")
                for node_idx, node_id in enumerate(path_nodes):
                    if node_id == leaf_id:
                        # Leaf node
                        f.write(f"  └─ LEAF: {self.action_names.get(action, f'A{action}')}\n")
                    else:
                        # Internal node
                        feature_idx = self.dt_model.tree_.feature[node_id]
                        threshold = self.dt_model.tree_.threshold[node_id]
                        feature_value = concept_vec[feature_idx]
                        
                        concept_name = f"C{feature_idx+1}"
                        concept_label = self.concept_labels.get(concept_name, concept_name)
                        
                        direction = "<=" if feature_value <= threshold else ">"
                        f.write(f"  ├─ {concept_label} ({concept_name}) = {feature_value:.3f} {direction} {threshold:.3f}\n")
                
                f.write("\n")
        
        print(f"✓ Saved decision path traces to: {output_path}")
    
    def export_tree_rules_with_labels(self, output_path: str):
        """
        Export tree rules with semantic concept labels
        
        Args:
            output_path: Where to save the rules
        """
        
        print(f"\n📊 Exporting tree rules with semantic labels...")
        
        # Build feature names with labels
        feature_labels = [
            f"{self.concept_labels.get(f'C{i+1}', f'C{i+1}')}"
            for i in range(self.n_concepts)
        ]
        
        # Export text representation
        tree_rules = export_text(
            self.dt_model,
            feature_names=feature_labels,
            show_weights=True,
            decimals=6
        )
        
        # Write to file with header
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("DECISION TREE RULES WITH SEMANTIC LABELS\n")
            f.write("="*80 + "\n")
            f.write(f"Environment: {self.metadata.get('environment', 'N/A')}\n")
            f.write(f"Concepts: {self.n_concepts} ({self.n_continuous} continuous + {self.n_concepts - self.n_continuous} binary)\n")
            f.write(f"Tree Depth: {self.dt_model.get_depth()}\n")
            f.write(f"Number of Nodes: {self.dt_model.tree_.node_count}\n")
            f.write(f"Number of Leaves: {self.dt_model.get_n_leaves()}\n")
            f.write("="*80 + "\n\n")
            
            f.write(tree_rules)
            
            f.write("\n\n")
            f.write("="*80 + "\n")
            f.write("ACTION MAPPING\n")
            f.write("="*80 + "\n")
            for action_idx, action_name in sorted(self.action_names.items()):
                f.write(f"Class {action_idx}: {action_name}\n")
        
        print(f"✓ Saved tree rules to: {output_path}")
    
    def create_summary_report(self, output_path: str, test_results_path: Optional[str] = None):
        """
        Create comprehensive summary report
        
        Args:
            output_path: Where to save the report
            test_results_path: Optional path to test results (.pkl from test_decision_tree_agent.py)
        """
        
        print(f"\n📊 Creating summary report...")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("DECISION TREE POLICY SUMMARY REPORT\n")
            f.write("="*80 + "\n\n")
            
            # Model info
            f.write("📋 Model Information:\n")
            f.write(f"  Environment: {self.metadata.get('environment', 'N/A')}\n")
            f.write(f"  Concepts: {self.n_concepts}\n")
            f.write(f"    - Continuous: {self.n_continuous}\n")
            f.write(f"    - Binary: {self.n_concepts - self.n_continuous}\n")
            f.write(f"  Tree Statistics:\n")
            f.write(f"    - Depth: {self.dt_model.get_depth()}\n")
            f.write(f"    - Nodes: {self.dt_model.tree_.node_count}\n")
            f.write(f"    - Leaves: {self.dt_model.get_n_leaves()}\n")
            f.write("\n")
            
            # Concept labels
            f.write("🏷️  Concept Labels:\n")
            for i in range(self.n_concepts):
                concept_type = "Continuous" if i < self.n_continuous else "Binary"
                label = self.concept_labels.get(f"C{i+1}", f"C{i+1}")
                importance = self.dt_model.feature_importances_[i]
                f.write(f"  C{i+1} ({concept_type:10s}): {label:<40s} [Importance: {importance:.4f}]\n")
            f.write("\n")
            
            # Test results if provided
            if test_results_path and Path(test_results_path).exists():
                with open(test_results_path, 'rb') as tf:
                    test_results = pickle.load(tf)
                
                f.write("🎯 Test Performance:\n")
                if 'dt_results' in test_results:
                    # Comparison mode
                    dt_res = test_results['dt_results']
                    f.write(f"  DT Policy:\n")
                    f.write(f"    - Success Rate: {dt_res['success_rate']:.2f}%\n")
                    f.write(f"    - Avg Reward: {dt_res['avg_reward']:.2f} ± {dt_res['std_reward']:.2f}\n")
                    f.write(f"    - Avg Steps: {dt_res['avg_length']:.2f} ± {dt_res['std_length']:.2f}\n")
                    f.write(f"  MLP Policy:\n")
                    f.write(f"    - Success Rate: {test_results['mlp_success_rate']:.2f}%\n")
                    f.write(f"    - Avg Reward: {test_results['mlp_avg_reward']:.2f}\n")
                    f.write(f"    - Avg Steps: {test_results['mlp_avg_length']:.2f}\n")
                    f.write(f"  Performance Ratio: {test_results['performance_ratio']:.2f}%\n")
                else:
                    # Single test mode
                    f.write(f"  Success Rate: {test_results['success_rate']:.2f}%\n")
                    f.write(f"  Avg Reward: {test_results['avg_reward']:.2f} ± {test_results['std_reward']:.2f}\n")
                    f.write(f"  Avg Steps: {test_results['avg_length']:.2f} ± {test_results['std_length']:.2f}\n")
                f.write("\n")
            
            f.write("="*80 + "\n")
        
        print(f"✓ Saved summary report to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize Decision Tree Policy for Interpretability")
    parser.add_argument("--model", type=str, required=True,
                       help="Path to trained DT model (.pkl)")
    parser.add_argument("--concept-labels", type=str, default=None,
                       help="Path to concept_labels.json from Gemini (optional)")
    parser.add_argument("--training-data", type=str, default=None,
                       help="Path to training data (.pkl) for concept-action heatmap")
    parser.add_argument("--test-results", type=str, default=None,
                       help="Path to test results (.pkl) for summary report")
    parser.add_argument("--output-dir", type=str, default="dt_visualizations",
                       help="Output directory for visualizations")
    parser.add_argument("--max-tree-depth", type=int, default=None,
                       help="Max depth for tree visualization (default: full tree)")
    parser.add_argument("--top-k-features", type=int, default=None,
                       help="Show only top K features in importance plot (default: all)")
    parser.add_argument("--sample-concepts", type=str, default=None,
                       help="Path to sample concepts (.npy) for decision path tracing")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"🎨 Decision Tree Policy Visualization")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}\n")
    
    # Initialize visualizer
    visualizer = DTVisualizer(args.model, args.concept_labels)
    
    # 1. Tree structure
    visualizer.plot_tree_structure(
        output_path=str(output_dir / "tree_structure.png"),
        max_depth=args.max_tree_depth
    )
    
    # 2. Feature importance
    visualizer.plot_feature_importance_detailed(
        output_path=str(output_dir / "feature_importance_detailed.png"),
        top_k=args.top_k_features
    )
    
    # 3. Concept-action heatmap (if training data provided)
    if args.training_data:
        visualizer.plot_concept_action_heatmap(
            data_path=args.training_data,
            output_path=str(output_dir / "concept_action_heatmap.png")
        )
    
    # 4. Decision path tracing (if sample concepts provided)
    if args.sample_concepts:
        sample_concepts = np.load(args.sample_concepts)
        visualizer.trace_decision_paths(
            sample_concepts=sample_concepts,
            output_path=str(output_dir / "decision_paths.txt")
        )
    
    # 5. Export tree rules
    visualizer.export_tree_rules_with_labels(
        output_path=str(output_dir / "tree_rules_semantic.txt")
    )
    
    # 6. Summary report
    visualizer.create_summary_report(
        output_path=str(output_dir / "summary_report.txt"),
        test_results_path=args.test_results
    )
    
    print(f"\n{'='*60}")
    print(f"✅ Visualization Complete!")
    print(f"{'='*60}")
    print(f"📁 All outputs saved to: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
