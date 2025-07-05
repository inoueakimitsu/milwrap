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
        rho = 1.0  # Penalty parameter for augmented Lagrangian
        
        # Temperature for soft assignments (starts high, anneals down)
        temperature_start = 2.0  # Keep original value that worked well
        temperature_end = 0.1
        temperature_decay = (temperature_end / temperature_start) ** (1.0 / max_iter)

        for i_iter in range(max_iter):
            # Temperature annealing
            temperature = temperature_start * (temperature_decay ** i_iter)
            
            # Fit
            flatten_bags = np.vstack(bags)
            flatten_y = np.hstack(y)
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
                    
                    # Apply minimal slack only in first iteration for robustness
                    effective_lower = lower_threshold[i_bag, i_class]
                    if i_iter == 0:  # Only first iteration
                        effective_lower = effective_lower * 0.95  # 5% slack
                    
                    if class_count < effective_lower:
                        # Need more instances of this class
                        deficit = int(effective_lower - class_count)
                        
                        # Find best candidates: high probability for this class, not already changed
                        candidates = []
                        for idx in range(n_instances):
                            if idx not in changed_indices and new_y[idx] != i_class:
                                # Score based on probability and current class constraint status
                                score = probs[idx, i_class]
                                
                                # Boost score if moving from over-represented class
                                current_class = new_y[idx]
                                current_class_count = class_count_dict.get(current_class, 0)
                                if current_class_count > upper_threshold[i_bag, current_class]:
                                    score *= 2.0  # Stronger boost for constraint satisfaction
                                elif current_class_count <= lower_threshold[i_bag, current_class]:
                                    score *= 0.5  # Penalty for taking from under-represented class
                                
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
                    
                    # Apply minimal slack only in first iteration for robustness
                    effective_upper = upper_threshold[i_bag, i_class]
                    if i_iter == 0:  # Only first iteration
                        effective_upper = effective_upper * 1.05  # 5% slack
                    
                    if class_count > effective_upper:
                        # Too many instances of this class
                        excess = int(class_count - effective_upper)
                        
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
                                
                                # Constraint bonus
                                c_count = class_count_dict.get(c, 0)
                                if c_count < lower_threshold[i_bag, c]:
                                    scores[c] += 2.0
                                elif c_count < upper_threshold[i_bag, c]:
                                    scores[c] += 0.5
                                else:
                                    scores[c] -= 1.0
                            
                            # Apply temperature-controlled softmax
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

            # Continue to iteration 9 for evaluation, even if converged
            min_iterations = 9
            if not has_changed and i_iter >= 4:
                # Labels have converged, but continue training classifier
                if debug:
                    print(f"Labels converged at iteration {i_iter}, continuing to refine classifier...")
                # Train with additional inner iterations when labels are stable
                for inner_iter in range(5):  # 5 inner epochs after convergence
                    self.classifier.fit(flatten_bags, flatten_y)
            elif not has_changed and i_iter < 4:
                if debug:
                    print(f"Early convergence at iteration {i_iter}, continuing exploration...")
            
            # Only break after minimum iterations
            if i_iter >= min_iterations - 1:
                break

        return self.classifier, y
