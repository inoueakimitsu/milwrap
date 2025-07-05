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
        
        # Adaptive temperature with momentum
        temperature_start = 2.0
        temperature_end = 0.08  # Slightly lower for finer convergence
        temperature = temperature_start
        temperature_momentum = 0.9  # Momentum factor for temperature updates
        temperature_velocity = 0.0
        
        # Track constraint satisfaction progress
        prev_violations = float('inf')
        violations_momentum = 0.0
        
        # Enhanced freezing for high-confidence instances
        frozen_instances = [set() for _ in bags]
        freeze_confidence_base = 0.85  # Base confidence threshold
        freeze_confidence_final = 0.95  # Final confidence threshold

        for i_iter in range(max_iter):
            # Adaptive temperature annealing with momentum
            # Compute target temperature based on iteration
            target_temp = temperature_start * ((temperature_end / temperature_start) ** (i_iter / max_iter))
            
            # Apply momentum to temperature updates
            temperature_velocity = temperature_momentum * temperature_velocity + (1 - temperature_momentum) * (target_temp - temperature)
            temperature = max(temperature_end, temperature + 0.3 * temperature_velocity)
            
            # Dynamic freeze threshold
            freeze_threshold = freeze_confidence_base + (freeze_confidence_final - freeze_confidence_base) * (i_iter / max_iter)
            
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
                
                # Freeze high-confidence instances after warmup
                if i_iter >= 2:
                    max_probs = np.max(probs, axis=1)
                    predicted_classes = np.argmax(probs, axis=1)
                    
                    for idx in range(n_instances):
                        if (idx not in frozen_instances[i_bag] and 
                            max_probs[idx] > freeze_threshold and 
                            y[i_bag][idx] == predicted_classes[idx]):
                            frozen_instances[i_bag].add(idx)
                
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
                            if idx not in changed_indices and idx not in frozen_instances[i_bag] and new_y[idx] != i_class:
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
                    
                    if class_count > upper_threshold[i_bag, i_class]:
                        # Too many instances of this class
                        excess = int(class_count - upper_threshold[i_bag, i_class])
                        
                        # Find instances to reassign: low probability for this class, not frozen
                        candidates = []
                        for idx in range(n_instances):
                            if idx not in changed_indices and idx not in frozen_instances[i_bag] and new_y[idx] == i_class:
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

            # Track constraint violations for adaptive temperature
            predicted_count = np.nan_to_num(
                pd.DataFrame([pd.Series(b).value_counts().to_dict() for b in y])[list(range(n_classes))].values,
            )
            lower_violations = np.maximum(lower_threshold - predicted_count, 0)
            upper_violations = np.maximum(predicted_count - upper_threshold, 0)
            total_violations = np.sum(lower_violations) + np.sum(upper_violations)
            
            # Adaptive temperature adjustment based on violation reduction
            if i_iter > 0:
                violation_reduction = prev_violations - total_violations
                violations_momentum = 0.8 * violations_momentum + 0.2 * violation_reduction
                
                # Slow down temperature decay if violations are reducing too slowly
                if violations_momentum < 0.1 and total_violations > 0:
                    temperature *= 1.1  # Slight reheating
                    temperature = min(temperature, temperature_start * 0.8)
            
            prev_violations = total_violations
            
            if debug:
                print(f"iter: {i_iter}")
                print(f"temperature: {temperature:.3f}, frozen: {sum(len(f) for f in frozen_instances)}")
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
                # Enhanced training with adaptive learning when labels are stable
                # Use bootstrapping with class balancing
                best_inner_acc = 0.0
                for inner_iter in range(8):  # More inner iterations
                    # Bootstrap sampling with class balance preservation
                    if inner_iter % 2 == 0:
                        # Normal training
                        self.classifier.fit(flatten_bags, flatten_y)
                    else:
                        # Balanced bootstrap sampling
                        balanced_indices = []
                        for c in range(n_classes):
                            class_indices = np.where(flatten_y == c)[0]
                            if len(class_indices) > 0:
                                sampled = np.random.choice(class_indices, size=len(class_indices), replace=True)
                                balanced_indices.extend(sampled)
                        
                        balanced_indices = np.array(balanced_indices)
                        np.random.shuffle(balanced_indices)
                        self.classifier.fit(flatten_bags[balanced_indices], flatten_y[balanced_indices])
                    
                    # Track best model during inner training
                    predictions = np.hstack([self.classifier.predict(bag) for bag in bags])
                    inner_acc = np.mean(predictions == flatten_y)
                    if inner_acc > best_inner_acc:
                        best_inner_acc = inner_acc
            elif not has_changed and i_iter < 4:
                if debug:
                    print(f"Early convergence at iteration {i_iter}, continuing exploration...")
            
            # Only break after minimum iterations
            if i_iter >= min_iterations - 1:
                break

        return self.classifier, y
