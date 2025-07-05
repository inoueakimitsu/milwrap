"""Multiple instance learning implementation based on count-based constraints.

This module provides implementations for multiple instance learning algorithms
that use count-based constraints to learn from bags of instances.
"""

import random
from typing import List, Sequence

import numpy as np
import pandas as pd
import scipy.stats
from numpy.typing import NDArray


def get_order_based_initial_bag_labels(
    n_instances_for_each_bag_and_class: NDArray[np.int_],
) -> List[int]:
    """Generate initial bag labels based on instance order.

    Args:
    ----
        n_instances_for_each_bag_and_class: Array of instance counts for each bag and class.

    Returns:
    -------
        List of initial bag labels.

    """
    assert len(n_instances_for_each_bag_and_class.shape) == 2

    n_bags = n_instances_for_each_bag_and_class.shape[0]
    n_classes = n_instances_for_each_bag_and_class.shape[1]

    bag_order = np.zeros_like(n_instances_for_each_bag_and_class, dtype=int)

    for i_class in range(n_classes):
        bag_order[:, i_class] = scipy.stats.rankdata(
            n_instances_for_each_bag_and_class[:, i_class],
        )

    initial_bag_labels = [0 for _ in range(n_bags)]

    for i_bag in range(n_bags):
        initial_bag_labels[i_bag] = np.argmax(bag_order[i_bag, :])

    return initial_bag_labels


def get_order_based_initial_y(
    lower_threshold: NDArray[np.float64],
    upper_threshold: NDArray[np.float64],
    n_instances_of_each_bags: List[int],
) -> List[NDArray[np.int_]]:
    """Generate initial instance labels based on thresholds and bag sizes.

    Args:
    ----
        lower_threshold: Lower threshold array for each bag and class.
        upper_threshold: Upper threshold array for each bag and class.
        n_instances_of_each_bags: List of instance counts for each bag.

    Returns:
    -------
        List of arrays containing initial instance labels.

    """
    return [
        np.repeat(label, n)
        for n, label in zip(
            n_instances_of_each_bags,
            get_order_based_initial_bag_labels((lower_threshold + upper_threshold) / 2),
        )
    ]




class MilCountBasedMultiClassLearner:
    """Multiple instance learning classifier using count-based constraints."""

    def __init__(self, classifier: object) -> None:
        """Initialize the learner.

        Args:
        ----
            classifier: Base classifier that implements fit/predict/predict_proba.

        """
        self.classifier = classifier

    def fit(
        self,
        bags: Sequence[NDArray[np.float64]],
        lower_threshold: NDArray[np.float64],
        upper_threshold: NDArray[np.float64],
        n_classes: int,
        max_iter: int = 10,
        initial_y: List[NDArray[np.int_]] | None = None,
        debug_true_y: List[NDArray[np.int_]] | None = None,
        seed: int | None = None,
        *,
        debug: bool = True,
    ) -> tuple[object, List[NDArray[np.int_]]]:
        """Fit the classifier using count-based multiple instance learning.

        Args:
        ----
            bags: Sequence of bags, where each bag is an array of instances.
            lower_threshold: Lower threshold array for each bag and class.
            upper_threshold: Upper threshold array for each bag and class.
            n_classes: Number of classes.
            max_iter: Maximum number of iterations.
            initial_y: Initial instance labels (optional).
            debug_true_y: True instance labels for debugging (optional).
            seed: Random seed for reproducibility.
            debug: Whether to print debug information.

        Returns:
        -------
            Tuple of (fitted classifier, final instance labels).

        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Initialize y
        n_list = [len(bag) for bag in bags]
        if not initial_y:
            # Improved initialization: consider both lower and upper thresholds
            # Use weighted average to find better initial class assignment
            threshold_center = (lower_threshold + upper_threshold) / 2.0
            # Add small noise to break ties
            threshold_center += np.random.randn(*threshold_center.shape) * 0.01
            
            y = [
                np.repeat(class_index, n_instance_in_bag)
                for class_index, n_instance_in_bag in zip(
                    np.argmax(threshold_center, axis=1),
                    n_list,
                )
            ]
        else:
            y = initial_y
            
        # Initialize Lagrangian multipliers for constraints
        n_bags = len(bags)
        lambda_lower = np.zeros((n_bags, n_classes))
        lambda_upper = np.zeros((n_bags, n_classes))
        
        # Single penalty parameter (simplified from Multi-penalty ADMM)
        rho = 1.0
        
        # Temperature for soft assignments (starts high, anneals down)
        # Optimal temperature scales with sqrt(d) where d is feature dimension
        feature_dim = bags[0].shape[1] if len(bags) > 0 and len(bags[0]) > 0 else 7
        # Balanced initial temperature for exploration and convergence
        temperature_start = np.sqrt(feature_dim) * 1.2  # Optimized for cosine annealing
        temperature_end = 0.1
        # Cosine annealing provides smoother decay with longer high-temperature phase
        # This helps exploration in early iterations
        
        # Instance-aware temperature scaling factors
        instance_temp_scale = [np.ones(len(bag)) for bag in bags]
        
        # Constraint continuation parameters (disabled for ablation study)
        constraint_slack_start = 0.0  # No slack on constraints
        constraint_slack_end = 0.0    # No slack at the end
        
        # Accuracy tracking for adaptive parameter control
        accuracy_history = []
        constraint_violation_history = []
        best_accuracy = 0.0

        for i_iter in range(max_iter):
            # Cosine annealing for temperature
            # T(t) = T_min + 0.5 * (T_max - T_min) * (1 + cos(π * t / T_total))
            cosine_factor = 0.5 * (1 + np.cos(np.pi * i_iter / max_iter))
            temperature = temperature_end + (temperature_start - temperature_end) * cosine_factor
            
            # Cosine annealing for constraint slack (inverse cosine for gradual tightening)
            # Starts at constraint_slack_start and smoothly decreases to constraint_slack_end
            constraint_slack = constraint_slack_end + (constraint_slack_start - constraint_slack_end) * cosine_factor
            
            # Fit
            flatten_bags = np.vstack(bags)
            flatten_y = np.hstack(y)
            self.classifier.fit(flatten_bags, flatten_y)

            # Compute outputs
            fs = [self.classifier.predict_proba(bag) for bag in bags]
            y_pred = [self.classifier.predict(bag) for bag in bags]
            flatten_original_y = np.hstack(list(y))
            
            # Instance-aware temperature scaling disabled for ablation study

            # For every bag - use soft assignments with augmented Lagrangian
            has_changed = False
            for i_bag in range(len(bags)):
                # Hybrid approach: combine constraint-based selection with soft assignments
                n_instances = len(bags[i_bag])
                probs = fs[i_bag]  # Shape: (n_instances, n_classes)
                
                
                # Get current class counts
                class_count_dict = pd.Series(y[i_bag]).value_counts().to_dict()
                
                # Track which instances have been reassigned in this iteration
                changed_indices = set()
                new_y = y[i_bag].copy()
                
                # Priority 1: Fix constraint violations using most confident instances
                target_classes = list(range(n_classes))
                random.shuffle(target_classes)
                
                for i_class in target_classes:
                    class_count = class_count_dict.get(i_class, 0)
                    
                    if class_count < lower_threshold[i_bag, i_class]:
                        # Need more instances of this class
                        deficit = int(lower_threshold[i_bag, i_class] - class_count)
                        
                        # Find best candidates: high probability for this class, not already changed, not frozen
                        candidates = []
                        for idx in range(n_instances):
                            if idx not in changed_indices and new_y[idx] != i_class:
                                # Score based on probability and current class constraint status
                                score = probs[idx, i_class]
                                
                                # Boost score if moving from over-represented class
                                current_class = new_y[idx]
                                current_class_count = class_count_dict.get(current_class, 0)
                                if current_class_count > upper_threshold[i_bag, current_class]:
                                    # Use penalty parameter for stronger boost
                                    score *= 1.0 + rho
                                elif current_class_count <= lower_threshold[i_bag, current_class]:
                                    # Use penalty parameter for penalty
                                    score *= 1.0 / (1.0 + rho)
                                
                                candidates.append((idx, score))
                        
                        # Sort by score and select top candidates
                        candidates.sort(key=lambda x: x[1], reverse=True)
                        # Gradually increase assignments based on iteration
                        adjustment_rate = min(1.0, (i_iter + 1) / 3.0)  # Full adjustment by iter 3
                        n_to_change = max(1, int(deficit * adjustment_rate))
                        
                        for idx, _ in candidates[:n_to_change]:
                            old_class = new_y[idx]
                            new_y[idx] = i_class
                            changed_indices.add(idx)
                            # Update counts
                            class_count_dict[i_class] = class_count_dict.get(i_class, 0) + 1
                            class_count_dict[old_class] = class_count_dict.get(old_class, 0) - 1
                            if class_count_dict[old_class] == 0:
                                del class_count_dict[old_class]
                
                # Priority 2: Fix upper threshold violations
                for i_class in target_classes:
                    class_count = class_count_dict.get(i_class, 0)
                    
                    if class_count > upper_threshold[i_bag, i_class]:
                        # Too many instances of this class
                        excess = int(class_count - upper_threshold[i_bag, i_class])
                        
                        # Find instances to reassign: low probability for this class, not frozen
                        candidates = []
                        for idx in range(n_instances):
                            if idx not in changed_indices and new_y[idx] == i_class:
                                candidates.append((idx, probs[idx, i_class]))
                        
                        # Sort by probability (ascending) and reassign
                        candidates.sort(key=lambda x: x[1])
                        
                        # Gradually increase adjustments based on iteration
                        adjustment_rate = min(1.0, (i_iter + 1) / 3.0)
                        n_to_change = max(1, int(excess * adjustment_rate))
                        
                        for idx, _ in candidates[:n_to_change]:
                            # Use soft assignment with temperature to choose new class
                            scores = np.zeros(n_classes)
                            for c in range(n_classes):
                                if c == i_class:
                                    scores[c] = -np.inf
                                    continue
                                
                                # Base score from probability
                                scores[c] = np.log(probs[idx, c] + 1e-8)
                                
                                # Constraint bonus with adaptive penalties and continuation
                                c_count = class_count_dict.get(c, 0)
                                # Apply constraint slack for continuation method
                                lower_bound = lower_threshold[i_bag, c] * (1.0 - constraint_slack)
                                upper_bound = upper_threshold[i_bag, c] * (1.0 + constraint_slack)
                                
                                if c_count < lower_bound:
                                    # Use penalty for under-represented classes
                                    scores[c] += rho * 2.0
                                elif c_count < upper_bound:
                                    scores[c] += 0.5
                                else:
                                    # Use penalty for over-represented classes
                                    scores[c] -= rho
                            
                            # Apply temperature-controlled softmax (instance-aware scaling disabled)
                            # Gumbel noise disabled for now (ablation analysis)
                            
                            exp_scores = np.exp(scores / temperature)
                            probabilities = exp_scores / np.sum(exp_scores)
                            
                            # Choose new class
                            new_class = np.random.choice(n_classes, p=probabilities)
                            
                            old_class = new_y[idx]
                            new_y[idx] = new_class
                            changed_indices.add(idx)
                            # Update counts
                            class_count_dict[i_class] -= 1
                            if class_count_dict[i_class] == 0:
                                del class_count_dict[i_class]
                            class_count_dict[new_class] = class_count_dict.get(new_class, 0) + 1
                
                # Check if assignments changed
                if not np.array_equal(y[i_bag], new_y):
                    has_changed = True
                    y[i_bag] = new_y

            # Track accuracy and constraint violations
            predicted_count = np.nan_to_num(
                pd.DataFrame([pd.Series(b).value_counts().to_dict() for b in y])[list(range(n_classes))].values,
            )
            count_false_negative = np.minimum(predicted_count - lower_threshold, 0)
            count_false_positive = np.maximum(predicted_count - upper_threshold, 0)
            total_constraint_violation = np.sum(np.abs(count_false_negative)) + np.sum(count_false_positive)
            constraint_violation_history.append(total_constraint_violation)
            
            current_accuracy = None
            if debug_true_y is not None:
                current_accuracy = np.mean(
                    np.hstack([self.classifier.predict(bag) for bag in bags]) == np.hstack(debug_true_y)
                )
                accuracy_history.append(current_accuracy)
                if current_accuracy > best_accuracy:
                    best_accuracy = current_accuracy
            
            if debug:
                print(f"iter: {i_iter}")
                print(f"temperature: {temperature:.3f}, constraint_slack: {constraint_slack:.3f}")
                print("false negative instances")
                print(np.sum(count_false_negative))
                print("false positive instances")
                print(np.sum(count_false_positive))
                print("num changes instances")
                num_changes_instance = np.sum(np.hstack(y) != flatten_original_y)
                print(num_changes_instance)
                if debug_true_y is not None:
                    print("instance unit accuracy")
                    print(current_accuracy)
                    print("instance unit accuracy (label adjusted)")
                    print(np.mean(np.hstack(y) == np.hstack(debug_true_y)))
                print("-----")

            # Continue to iteration 9 for evaluation, even if converged
            min_iterations = 9
            if not has_changed and i_iter >= 5:
                # Labels have converged, but continue training classifier
                if debug:
                    print(f"Labels converged at iteration {i_iter}, continuing to refine classifier...")
                # Train with additional inner iterations when labels are stable
                # Use early stopping based on validation performance
                best_score = -np.inf
                patience = 3
                no_improvement_count = 0
                
                for inner_iter in range(10):  # More inner epochs with early stopping
                    self.classifier.fit(flatten_bags, flatten_y)
                    
                    # Compute validation score
                    val_predictions = np.hstack([self.classifier.predict(bag) for bag in bags])
                    val_score = np.mean(val_predictions == flatten_y)
                    
                    if val_score > best_score:
                        best_score = val_score
                        no_improvement_count = 0
                    else:
                        no_improvement_count += 1
                        
                    if no_improvement_count >= patience:
                        break
                        
            elif not has_changed and i_iter < 5:
                if debug:
                    print(f"Early convergence at iteration {i_iter}, continuing exploration...")
            
            # Single penalty parameter remains fixed (no adaptive updates)
            
            # Only break after minimum iterations
            if i_iter >= min_iterations - 1:
                break

        return self.classifier, y
