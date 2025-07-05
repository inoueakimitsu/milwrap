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
            y = [
                np.repeat(class_index, n_instance_in_bag)
                for class_index, n_instance_in_bag in zip(
                    np.argmax(lower_threshold, axis=1),
                    n_list,
                )
            ]
        else:
            y = initial_y
            
        # Initialize Lagrangian multipliers for constraints (Deep-ALM approach)
        n_bags = len(bags)
        lambda_lower = np.zeros((n_bags, n_classes))
        lambda_upper = np.zeros((n_bags, n_classes))
        rho = 0.1  # Start with moderate penalty parameter
        rho_increase_rate = 1.5  # Increase rho more gradually
        
        # Temperature for soft assignments with cosine annealing
        temperature_max = 2.0
        temperature_min = 0.1  # Keep original minimum
        restart_interval = max_iter  # No warm restarts for now
        
        # MIL-Dropout parameters
        mil_dropout_rate = 0.01  # Drop top 1% of instances

        for i_iter in range(max_iter):
            # Cosine annealing with warm restarts
            restart_cycle = i_iter % restart_interval
            temperature = temperature_min + 0.5 * (temperature_max - temperature_min) * (
                1 + np.cos(np.pi * restart_cycle / restart_interval)
            )
            
            # Prepare training data with MIL-Dropout
            training_bags = []
            training_y = []
            
            for i_bag in range(len(bags)):
                bag_instances = bags[i_bag]
                bag_labels = y[i_bag]
                n_instances = len(bag_instances)
                
                if n_instances > 1 and i_iter > 0:  # Apply MIL-Dropout after first iteration
                    # Calculate instance scores based on current predictions
                    if hasattr(self.classifier, 'predict_proba'):
                        try:
                            instance_probs = self.classifier.predict_proba(bag_instances)
                            # Score = max probability for predicted class
                            instance_scores = np.max(instance_probs, axis=1)
                            
                            # Determine number to drop
                            n_drop = max(1, int(n_instances * mil_dropout_rate))
                            n_keep = n_instances - n_drop
                            
                            # Find top-k instances and randomly drop some
                            top_k_indices = np.argsort(instance_scores)[-n_drop*2:]  # Get top candidates
                            drop_indices = np.random.choice(top_k_indices, size=min(n_drop, len(top_k_indices)), replace=False)
                            keep_indices = np.setdiff1d(np.arange(n_instances), drop_indices)
                            
                            training_bags.append(bag_instances[keep_indices])
                            training_y.append(bag_labels[keep_indices])
                        except:
                            # Fallback if prediction fails
                            training_bags.append(bag_instances)
                            training_y.append(bag_labels)
                    else:
                        training_bags.append(bag_instances)
                        training_y.append(bag_labels)
                else:
                    training_bags.append(bag_instances)
                    training_y.append(bag_labels)
            
            # Fit classifier with potentially dropped instances
            flatten_bags = np.vstack(training_bags)
            flatten_y = np.hstack(training_y)
            self.classifier.fit(flatten_bags, flatten_y)

            # Compute outputs
            fs = [self.classifier.predict_proba(bag) for bag in bags]
            y_pred = [self.classifier.predict(bag) for bag in bags]
            flatten_original_y = np.hstack(list(y))

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
                        
                        # Find best candidates: high probability for this class, not already changed
                        candidates = []
                        for idx in range(n_instances):
                            if idx not in changed_indices and new_y[idx] != i_class:
                                # Deep-ALM scoring for candidate selection
                                score = probs[idx, i_class]
                                
                                # Add Lagrangian-based scoring
                                current_class = new_y[idx]
                                current_class_count = class_count_dict.get(current_class, 0)
                                
                                # Reward moving to under-represented class
                                v_lower_target = max(0, lower_threshold[i_bag, i_class] - class_count)
                                score += lambda_lower[i_bag, i_class] * v_lower_target
                                
                                # Reward moving from over-represented class
                                v_upper_current = max(0, current_class_count - upper_threshold[i_bag, current_class])
                                if v_upper_current > 0:
                                    score += lambda_upper[i_bag, current_class] * v_upper_current
                                
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
                        
                        # Find instances to reassign: low probability for this class
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
                                
                                # Deep-ALM: Use Lagrangian multipliers for constraint scoring
                                c_count = class_count_dict.get(c, 0)
                                
                                # Lower constraint violation: v_lower = max(0, lower_threshold - count)
                                v_lower = max(0, lower_threshold[i_bag, c] - c_count)
                                # Upper constraint violation: v_upper = max(0, count - upper_threshold)
                                v_upper = max(0, c_count - upper_threshold[i_bag, c])
                                
                                # Hybrid scoring: Deep-ALM + heuristic bonuses
                                # Augmented Lagrangian scoring
                                lagrangian_weight = 0.15  # Minimal Lagrangian influence
                                scores[c] += lagrangian_weight * (lambda_lower[i_bag, c] * v_lower - lambda_upper[i_bag, c] * v_upper)
                                scores[c] += lagrangian_weight * rho * (v_lower - v_upper)
                                
                                # Add heuristic bonuses (original values)
                                if c_count < lower_threshold[i_bag, c]:
                                    scores[c] += 2.0
                                elif c_count < upper_threshold[i_bag, c]:
                                    scores[c] += 0.5
                                else:
                                    scores[c] -= 1.0
                            
                            # Apply temperature-controlled softmax
                            exp_scores = np.exp(scores / temperature)
                            probabilities = exp_scores / np.sum(exp_scores)
                            
                            # AEM: Add entropy bonus to encourage exploration
                            entropy = -np.sum(probabilities * np.log(probabilities + 1e-8))
                            entropy_weight = 0.05  # As suggested by o3
                            # Adjust probabilities slightly towards uniform distribution
                            uniform_prob = 1.0 / n_classes
                            probabilities = (1 - entropy_weight) * probabilities + entropy_weight * uniform_prob
                            probabilities = probabilities / np.sum(probabilities)  # Renormalize
                            
                            # Choose new class
                            new_class = np.random.choice(n_classes, p=probabilities)
                            
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

            # Deep-ALM: Update Lagrangian multipliers and penalty parameter
            total_violations = 0
            for i_bag in range(len(bags)):
                # Calculate final class counts for this bag
                class_count_dict = pd.Series(y[i_bag]).value_counts().to_dict()
                
                for i_class in range(n_classes):
                    class_count = class_count_dict.get(i_class, 0)
                    
                    # Lower constraint violation
                    v_lower = max(0, lower_threshold[i_bag, i_class] - class_count)
                    if v_lower > 0:
                        # Update multiplier (gradient ascent)
                        lambda_lower[i_bag, i_class] += rho * v_lower
                        # Clip to prevent unbounded growth
                        lambda_lower[i_bag, i_class] = min(lambda_lower[i_bag, i_class], 5.0)
                        total_violations += v_lower
                    
                    # Upper constraint violation
                    v_upper = max(0, class_count - upper_threshold[i_bag, i_class])
                    if v_upper > 0:
                        # Update multiplier (gradient ascent)
                        lambda_upper[i_bag, i_class] += rho * v_upper
                        # Clip to prevent unbounded growth
                        lambda_upper[i_bag, i_class] = min(lambda_upper[i_bag, i_class], 5.0)
                        total_violations += v_upper
            
            # Adaptively increase rho if violations persist
            if total_violations > 0 and i_iter > 0 and i_iter % 2 == 0:
                rho = min(rho * rho_increase_rate, 1.0)  # Cap at 1.0

            if debug:
                print("iter:", i_iter)
                predicted_count = np.nan_to_num(
                    pd.DataFrame([pd.Series(b).value_counts().to_dict() for b in y])[list(range(n_classes))].values,
                )
                print("false negative instances")
                count_false_negative = np.minimum(predicted_count - lower_threshold, 0)
                print(np.sum(count_false_negative))
                print("false positive instances")
                count_false_positive = np.maximum(predicted_count - upper_threshold, 0)
                print(np.sum(count_false_positive))
                print("num changes instances")
                num_changes_instance = np.sum(np.hstack(y) != flatten_original_y)
                print(num_changes_instance)
                if debug_true_y is not None:
                    print("instance unit accuracy")
                    print(
                        np.mean(
                            np.hstack([self.classifier.predict(bag) for bag in bags]) == np.hstack(debug_true_y),
                        ),
                    )
                    print("instance unit accuracy (label adjusted)")
                    print(np.mean(np.hstack(y) == np.hstack(debug_true_y)))
                print("-----")

            if not has_changed:
                pass  # Continue to max_iter for evaluation

        return self.classifier, y
