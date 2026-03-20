"""
Train Decision Tree Policy from Concept-Action Dataset

Trains sklearn DecisionTreeClassifier to map concept vectors to actions
Includes hyperparameter tuning, evaluation, and model export
"""

import numpy as np
import pickle
import argparse
from pathlib import Path
from typing import Dict, Tuple

# Set matplotlib to non-interactive backend (MUST be before importing pyplot)
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for threading
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score
)

from parse_concept_actions import load_dataset, ACTION_NAMES


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray, 
                   concept_labels: list = None) -> Dict:
    """
    Comprehensive model evaluation
    
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    
    print(f"\n{'='*60}")
    print(f"📊 Model Evaluation")
    print(f"{'='*60}")
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Overall metrics
    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    weighted_f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\n🎯 Overall Performance:")
    print(f"  Accuracy:    {accuracy:.4f}")
    print(f"  Macro F1:    {macro_f1:.4f}")
    print(f"  Weighted F1: {weighted_f1:.4f}")
    
    # Per-class metrics
    print(f"\n📋 Per-Action Performance:")
    print(f"{'Action':<20s} {'Precision':>10s} {'Recall':>10s} {'F1-Score':>10s} {'Support':>10s}")
    print(f"{'-'*70}")
    
    unique_classes = np.unique(np.concatenate([y_test, y_pred]))
    
    per_action_metrics = {}
    for action_idx in unique_classes:
        action_name = ACTION_NAMES.get(action_idx, f"Unknown_{action_idx}")
        
        # Binary classification for each action
        y_true_binary = (y_test == action_idx).astype(int)
        y_pred_binary = (y_pred == action_idx).astype(int)
        
        if np.sum(y_true_binary) > 0:  # Only calculate if action exists in test set
            precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
            recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
            f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
            support = np.sum(y_true_binary)
            
            per_action_metrics[action_idx] = {
                'precision': precision,
                'recall': recall, 
                'f1': f1,
                'support': support
            }
            
            print(f"{action_name:<20s} {precision:>10.4f} {recall:>10.4f} {f1:>10.4f} {support:>10d}")
    
    # Check for imbalanced performance
    recalls = [m['recall'] for m in per_action_metrics.values()]
    if len(recalls) > 1:
        recall_std = np.std(recalls)
        if recall_std > 0.15:
            print(f"\n⚠️  WARNING: High recall variance ({recall_std:.3f}) - model may be biased!")
            print(f"    Some actions have very poor recall. Check class balance in training data.")
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        print(f"\n🔍 Feature Importance (Concept Usage):")
        importances = model.feature_importances_
        
        if concept_labels is None:
            concept_labels = [f"C{i+1}" for i in range(len(importances))]
        
        # Sort by importance
        indices = np.argsort(importances)[::-1]
        
        for idx in indices:
            if importances[idx] > 0.01:  # Only show significant features
                print(f"  {concept_labels[idx]:<10s}: {importances[idx]:.4f}")
    
    # Tree statistics
    if hasattr(model, 'tree_'):
        n_nodes = model.tree_.node_count
        max_depth = model.tree_.max_depth
        n_leaves = model.tree_.n_leaves
        
        print(f"\n🌳 Tree Structure:")
        print(f"  Total nodes:    {n_nodes}")
        print(f"  Max depth:      {max_depth}")
        print(f"  Leaf nodes:     {n_leaves}")
    
    print(f"{'='*60}\n")
    
    metrics = {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'feature_importances': model.feature_importances_ if hasattr(model, 'feature_importances_') else None,
        'n_nodes': n_nodes if hasattr(model, 'tree_') else None,
        'max_depth': max_depth if hasattr(model, 'tree_') else None,
        'n_leaves': n_leaves if hasattr(model, 'tree_') else None,
    }
    
    return metrics


def plot_confusion_matrix(cm: np.ndarray, output_path: Path):
    """Plot and save confusion matrix"""
    
    plt.figure(figsize=(10, 8))
    
    # Get action names for labels
    unique_actions = list(range(len(cm)))
    labels = [ACTION_NAMES.get(i, f"A{i}") for i in unique_actions]
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Count'})
    
    plt.xlabel('Predicted Action', fontsize=12)
    plt.ylabel('True Action', fontsize=12)
    plt.title('Decision Tree Policy - Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved confusion matrix to: {output_path}")
    plt.close()


def plot_feature_importance(importances: np.ndarray, concept_labels: list, output_path: Path):
    """Plot and save feature importance"""
    
    plt.figure(figsize=(10, 6))
    
    # Sort by importance
    indices = np.argsort(importances)[::-1]
    sorted_importances = importances[indices]
    sorted_labels = [concept_labels[i] for i in indices]
    
    colors = plt.cm.viridis(sorted_importances / sorted_importances.max())
    
    plt.barh(range(len(sorted_importances)), sorted_importances, color=colors)
    plt.yticks(range(len(sorted_importances)), sorted_labels)
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Concept', fontsize=12)
    plt.title('Decision Tree Policy - Concept Importance', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved feature importance to: {output_path}")
    plt.close()


def train_decision_tree(X_train: np.ndarray, y_train: np.ndarray,
                        X_val: np.ndarray = None, y_val: np.ndarray = None,
                        hyperparameter_tuning: bool = True,
                        concept_labels: list = None,
                        skip_validation_split: bool = False,
                        max_depth_override: int = None) -> DecisionTreeClassifier:
    """
    Train Decision Tree with optional hyperparameter tuning
    
    Args:
        X_train: Training concept vectors
        y_train: Training action labels
        X_val: Validation concept vectors (optional, will split from train if None)
        y_val: Validation action labels (optional, will split from train if None)
        hyperparameter_tuning: Whether to perform GridSearchCV
        concept_labels: List of concept names
        skip_validation_split: If True, use all training data without validation split (for deduplicated data)
        max_depth_override: If provided, force this max_depth instead of tuning (0 or None = no override)
    
    Returns:
        Trained DecisionTreeClassifier
    """
    
    # If validation set not provided, split from training set (unless skip_validation_split=True)
    if X_val is None or y_val is None:
        if skip_validation_split:
            print(f"ℹ️  Training on ALL data without validation split (deduplicated dataset)...")
            X_val = X_train
            y_val = y_train
        else:
            print(f"ℹ️  No validation set provided, splitting training set (80/20)...")
            unique_classes, counts = np.unique(y_train, return_counts=True)
            min_class_count = np.min(counts)
            
            if min_class_count < 2:
                print(f"⚠️  WARNING: Found class with < 2 samples in training set. Disabling stratification.")
                X_train, X_val, y_train, y_val = train_test_split(
                    X_train, y_train, test_size=0.2, random_state=42
                )
            else:
                X_train, X_val, y_train, y_val = train_test_split(
                    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
                )
    
    print(f"\n{'='*60}")
    print(f"🌳 Training Decision Tree Policy")
    print(f"{'='*60}")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Concept dimensions: {X_train.shape[1]}")
    print(f"Action classes: {len(np.unique(y_train))}")
    
    # Check for max_depth override
    if max_depth_override and max_depth_override > 0:
        print(f"\n⚠️  Max Depth Override: {max_depth_override}")
        print(f"   Skipping hyperparameter tuning, using fixed depth.")
        model = DecisionTreeClassifier(
            max_depth=max_depth_override,
            min_samples_split=2,
            min_samples_leaf=1,
            criterion='gini',
            class_weight='balanced',
            random_state=42
        )
        model.fit(X_train, y_train)
    elif hyperparameter_tuning:
        print(f"\n🔧 Performing Grid Search for Hyperparameters...")
        
        # Define parameter grid - ADAPTIVE BASED ON DATA SIZE
        # For small datasets (<1000 samples): allow deeper, more flexible trees
        # For large datasets (>10000 samples): use shallow trees to prevent overfitting
        n_samples = len(X_train)
        
        if n_samples < 1000:
            # Small dataset: Allow maximum flexibility to fit patterns and achieve >90% accuracy
            # Use very deep trees or unlimited depth to memorize deduplicated patterns
            param_grid = {
                'max_depth': [15, 20, 25, 30, None],  # Include None for unlimited depth
                'min_samples_split': [2, 3, 5],  # Very low thresholds
                'min_samples_leaf': [1, 2],  # Minimum leaf size
                'criterion': ['gini', 'entropy'],
                'class_weight': ['balanced', None],
            }
            print(f"   Using FLEXIBLE grid for small dataset (n={n_samples})")
            print(f"   Target: >90% training accuracy (may use unlimited depth)")
        elif n_samples < 10000:
            # Medium dataset: Balanced regularization
            param_grid = {
                'max_depth': [8, 10, 12],
                'min_samples_split': [10, 20, 50],
                'min_samples_leaf': [5, 10, 20],
                'criterion': ['gini', 'entropy'],
                'class_weight': ['balanced', None],
            }
            print(f"   Using BALANCED grid for medium dataset (n={n_samples})")
        else:
            # Large dataset: Strong regularization to prevent threshold overfitting
            param_grid = {
                'max_depth': [6, 8, 10],
                'min_samples_split': [30, 50, 100],
                'min_samples_leaf': [20, 30, 50],
                'criterion': ['gini', 'entropy'],
                'class_weight': ['balanced', None],
                'min_impurity_decrease': [0.0, 0.001, 0.005]  # Require meaningful improvement per split
            }
            print(f"   Using REGULARIZED grid for large dataset (n={n_samples})")
        
        # Base model
        dt_base = DecisionTreeClassifier(random_state=42)
        
        # Grid search with 5-fold cross-validation
        grid_search = GridSearchCV(
            dt_base,
            param_grid,
            cv=5,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"\n✓ Best parameters found:")
        for param, value in grid_search.best_params_.items():
            print(f"  {param}: {value}")
        
        print(f"\n✓ Best cross-validation F1 score: {grid_search.best_score_:.4f}")
        
        # Use best model
        model = grid_search.best_estimator_
        
    else:
        print(f"\n🔧 Training with adaptive hyperparameters (no grid search)...")
        
        # Adaptive parameters based on dataset size
        n_samples = len(X_train)
        
        if n_samples < 1000:
            # Small dataset: Allow very deep tree to achieve >90% accuracy
            print(f"   Using FLEXIBLE params for small dataset (n={n_samples})")
            print(f"   Target: >90% training accuracy (using deep tree)")
            model = DecisionTreeClassifier(
                max_depth=25,  # Very deep to memorize deduplicated patterns
                min_samples_split=2,  # Minimum to allow maximum splits
                min_samples_leaf=1,  # Minimum for finest decisions
                min_impurity_decrease=0.0,  # No constraint
                criterion='gini',
                class_weight='balanced',
                random_state=42
            )
        elif n_samples < 10000:
            # Medium dataset: Balanced approach
            print(f"   Using BALANCED params for medium dataset (n={n_samples})")
            model = DecisionTreeClassifier(
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10,
                min_impurity_decrease=0.0,
                criterion='gini',
                class_weight='balanced',
                random_state=42
            )
        else:
            # Large dataset: Strong regularization to prevent threshold overfitting
            print(f"   Using REGULARIZED params for large dataset (n={n_samples})")
            model = DecisionTreeClassifier(
                max_depth=8,  # Reduced from 10 for better generalization
                min_samples_split=50,  # Increased from 20 for robust splits
                min_samples_leaf=30,  # Increased from 10 for larger leaves
                min_impurity_decrease=0.001,  # Require meaningful improvement
                criterion='gini',
                class_weight='balanced',
                random_state=42
            )
        
        model.fit(X_train, y_train)
    
    # Validation performance
    y_val_pred = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred, average='weighted')
    
    print(f"\n✓ Validation Performance:")
    print(f"  Accuracy: {val_accuracy:.4f}")
    print(f"  Weighted F1: {val_f1:.4f}")
    
    print(f"{'='*60}\n")
    
    return model


def train_random_forest(X_train: np.ndarray, y_train: np.ndarray,
                        X_val: np.ndarray = None, y_val: np.ndarray = None,
                        hyperparameter_tuning: bool = True,
                        skip_validation_split: bool = False,
                        max_depth_override: int = None) -> RandomForestClassifier:
    """
    Train RandomForest classifier (often achieves better accuracy than single tree)
    
    Args:
        X_train: Training concept vectors
        y_train: Training action labels
        X_val: Validation concept vectors (optional)
        y_val: Validation action labels (optional)
        hyperparameter_tuning: Whether to perform GridSearchCV
        skip_validation_split: If True, use all training data without validation split
        max_depth_override: If provided, force this max_depth
    
    Returns:
        Trained RandomForestClassifier
    """
    
    # Handle validation split
    if X_val is None or y_val is None:
        if skip_validation_split:
            print(f"ℹ️  Training RandomForest on ALL data without validation split...")
            X_val = X_train
            y_val = y_train
        else:
            print(f"ℹ️  Splitting training set for validation (80/20)...")
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
            )
    
    print(f"\n{'='*60}")
    print(f"🌲 Training RandomForest Policy")
    print(f"{'='*60}")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    n_samples = len(X_train)
    
    # Determine max_depth
    if max_depth_override and max_depth_override > 0:
        max_depth = max_depth_override
        print(f"Using max_depth override: {max_depth}")
    elif n_samples < 1000:
        max_depth = 20
        print(f"Using deep trees for small dataset: max_depth={max_depth}")
    else:
        max_depth = 15
        print(f"Using moderate depth for larger dataset: max_depth={max_depth}")
    
    if hyperparameter_tuning and (not max_depth_override or max_depth_override == 0):
        print(f"\n🔧 Grid Search for RandomForest...")
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [15, 20, 25, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'class_weight': ['balanced', None]
        }
        
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='f1_weighted', n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)
        
        print(f"\n✓ Best parameters:")
        for param, value in grid_search.best_params_.items():
            print(f"  {param}: {value}")
        
        model = grid_search.best_estimator_
    else:
        print(f"\n🔧 Training with default parameters...")
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=max_depth,
            min_samples_split=2,
            min_samples_leaf=1,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
    
    # Validation performance
    y_val_pred = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred, average='weighted')
    
    print(f"\n✓ Validation Performance:")
    print(f"  Accuracy: {val_accuracy:.4f}")
    print(f"  Weighted F1: {val_f1:.4f}")
    print(f"  Number of Trees: {model.n_estimators}")
    print(f"{'='*60}\n")
    
    return model


def export_tree_rules(model: DecisionTreeClassifier, concept_labels: list, output_path: Path):
    """Export decision tree rules to text file"""
    
    # Use action names for classes that actually appear in the model
    action_names = [ACTION_NAMES.get(i, f"Action_{i}") for i in model.classes_]
    
    tree_rules = export_text(
        model,
        feature_names=concept_labels,
        class_names=action_names,
        decimals=6
    )
    
    with open(output_path, 'w') as f:
        f.write("DECISION TREE POLICY RULES\n")
        f.write("="*60 + "\n\n")
        f.write(tree_rules)
        f.write("\n\n" + "="*60 + "\n")
        f.write("Legend:\n")
        f.write("- Concept values: [0.0 - 1.0] for continuous, {0.0, 1.0} for binary\n")
        f.write("- class: Predicted action\n")
        f.write("- value: Sample distribution across action classes\n")
    
    print(f"✓ Saved tree rules to: {output_path}")


def save_model(model: DecisionTreeClassifier, metadata: Dict, output_path: Path,
               X_train: np.ndarray = None, y_train: np.ndarray = None):
    """Save trained model with metadata and training statistics"""
    
    # Calculate training data statistics if provided
    training_stats = {}
    if X_train is not None and y_train is not None:
        # Unique concept states in training
        unique_train_states = len(np.unique(X_train, axis=0))
        total_train_states = len(X_train)
        
        # Action distribution in training
        unique_actions, action_counts = np.unique(y_train, return_counts=True)
        action_distribution = {int(a): int(c) for a, c in zip(unique_actions, action_counts)}
        
        training_stats = {
            'unique_states': unique_train_states,
            'total_states': total_train_states,
            'state_diversity': unique_train_states / total_train_states,
            'action_distribution': action_distribution,
            'concept_ranges': {
                'min': X_train.min(axis=0).tolist(),
                'max': X_train.max(axis=0).tolist(),
                'mean': X_train.mean(axis=0).tolist(),
                'std': X_train.std(axis=0).tolist()
            }
        }
        
        print(f"\n📊 Training Data Statistics:")
        print(f"  Total states: {total_train_states}")
        print(f"  Unique states: {unique_train_states}")
        print(f"  Diversity: {training_stats['state_diversity']*100:.1f}%")
        print(f"\n  Action distribution in training:")
        for action_idx, count in sorted(action_distribution.items()):
            pct = count / total_train_states * 100
            print(f"    {ACTION_NAMES.get(action_idx, f'A{action_idx}'):<20s}: {count:>5d} ({pct:>5.1f}%)")
        
        # Check for severe class imbalance
        action_percentages = [c / total_train_states for c in action_distribution.values()]
        if len(action_percentages) > 1:
            max_pct = max(action_percentages)
            min_pct = min(action_percentages)
            if max_pct / min_pct > 10:
                print(f"\n⚠️  WARNING: Severe class imbalance detected!")
                print(f"    Max action: {max_pct*100:.1f}%, Min action: {min_pct*100:.1f}%")
                print(f"    Ratio: {max_pct/min_pct:.1f}x - Model may be biased toward frequent actions!")
    
    model_data = {
        'model': model,
        'metadata': metadata,
        'action_names': ACTION_NAMES,
        'training_stats': training_stats
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\n✓ Saved trained model to: {output_path}")



def main():
    parser = argparse.ArgumentParser(description="Train Decision Tree Policy")
    parser.add_argument("--data", type=str, required=True,
                       help="Path to training data pickle file")
    parser.add_argument("--output", type=str, default="dt_policy.pkl",
                       help="Output model file path")
    parser.add_argument("--output-dir", type=str, default="dt_output",
                       help="Output directory for visualizations")
    parser.add_argument("--test-split", type=float, default=0.2,
                       help="Test set split ratio (default: 0.2)")
    parser.add_argument("--no-tuning", action="store_true",
                       help="Skip hyperparameter tuning")
    parser.add_argument("--random-seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.random_seed)
    
    # Load dataset
    print(f"📂 Loading dataset from: {args.data}")
    X, y, metadata = load_dataset(Path(args.data))
    
    # Create concept labels
    n_concepts = metadata.get('n_concepts', X.shape[1])
    n_continuous = metadata.get('n_continuous_concepts', 1)
    
    concept_labels = []
    for i in range(n_concepts):
        label = f"C{i+1}"
        if i < n_continuous:
            label += " (continuous)"
        else:
            label += " (binary)"
        concept_labels.append(label)
    
    # Train/test split (if test_split=0, train and test on all data)
    if args.test_split == 0 or args.test_split < 0.01:
        print(f"\n⚠️  test_split={args.test_split}: Training and testing on ALL data (no split)")
        print(f"   This shows training accuracy on deduplicated dataset.")
        X_train = X_test = X
        y_train = y_test = y
        skip_val_split = True
    else:
        # Check for classes with only 1 sample, which breaks stratify
        unique_classes, counts = np.unique(y, return_counts=True)
        min_class_count = np.min(counts)
        
        if min_class_count < 2:
            print(f"\n⚠️  WARNING: Found class with < 2 samples. Disabling stratification in train_test_split.")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=args.test_split, random_state=args.random_seed
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=args.test_split, random_state=args.random_seed, stratify=y
            )
        skip_val_split = False
    
    print(f"\n✓ Train/Test Split:")
    print(f"  Training: {len(X_train)} samples")
    print(f"  Testing: {len(X_test)} samples")
    
    # Train model (will auto-split train into train/val internally unless skip_val_split=True)
    model = train_decision_tree(
        X_train, y_train,
        hyperparameter_tuning=not args.no_tuning,
        concept_labels=concept_labels,
        skip_validation_split=skip_val_split
    )
    
    # Evaluate
    metrics = evaluate_model(model, X_test, y_test, concept_labels)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Save model with training statistics
    output_path = Path(args.output)
    save_model(model, metadata, output_path, X_train, y_train)
    
    # Export tree rules
    rules_path = output_dir / "tree_rules.txt"
    export_tree_rules(model, concept_labels, rules_path)
    
    # Plot confusion matrix
    cm_path = output_dir / "confusion_matrix.png"
    plot_confusion_matrix(metrics['confusion_matrix'], cm_path)
    
    # Plot feature importance
    if metrics['feature_importances'] is not None:
        fi_path = output_dir / "feature_importance.png"
        plot_feature_importance(metrics['feature_importances'], concept_labels, fi_path)
    
    # Save metrics
    metrics_path = output_dir / "evaluation_metrics.pkl"
    with open(metrics_path, 'wb') as f:
        pickle.dump(metrics, f)
    print(f"✓ Saved metrics to: {metrics_path}")
    
    print(f"\n{'='*60}")
    print(f"✅ Training Complete!")
    print(f"{'='*60}")
    print(f"Model saved to: {output_path}")
    print(f"Outputs saved to: {output_dir}/")
    print(f"\nNext step: python test_decision_tree_agent.py --model {args.output}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
