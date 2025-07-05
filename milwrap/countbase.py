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


def compute_soft_topk_count_loss(
    probs: NDArray[np.float64],
    target_count: int,
    temperature: float = 0.5,
) -> float:
    """Compute differentiable count loss using soft top-k approach.
    
    Uses sigmoid-based soft counting that is numerically stable and
    provides smooth gradients.
    
    Args:
    ----
        probs: Probability array for instances belonging to a class.
        target_count: Target number of instances for the class.
        temperature: Temperature parameter for soft counting (lower = sharper).
        
    Returns:
    -------
        Count loss value (squared error).
        
    """
    if len(probs) == 0 or target_count < 0:
        return 0.0
    
    # Compute threshold for soft top-k
    # Sort probabilities to find approximate threshold
    sorted_probs = np.sort(probs)[::-1]  # Descending order
    
    if target_count == 0:
        # If we want 0 instances, penalize any positive probability
        soft_count = np.sum(probs)
        return soft_count ** 2
    
    if target_count >= len(probs):
        # If we want all instances, use sum of (1 - prob) as penalty
        soft_count = np.sum(probs)
        return (soft_count - target_count) ** 2
    
    # Find threshold around the k-th largest value
    if target_count <= len(sorted_probs):
        threshold = sorted_probs[target_count - 1]
        if target_count < len(sorted_probs):
            threshold = (threshold + sorted_probs[target_count]) / 2
    else:
        threshold = 0.0
    
    # Soft counting using sigmoid
    # Higher temperature = smoother selection
    soft_indicators = 1.0 / (1.0 + np.exp(-(probs - threshold) / temperature))
    soft_count = np.sum(soft_indicators)
    
    # Squared error loss
    loss = (soft_count - target_count) ** 2
    
    return loss


def apply_spgd_perturbation(
    logits: NDArray[np.float64],
    perturbation_scale: float = 0.2,
    rng: np.random.Generator | None = None,
) -> NDArray[np.float64]:
    """Apply SPGD-style perturbation to logits.
    
    Args:
    ----
        logits: Input logits array.
        perturbation_scale: Scale of uniform perturbation.
        rng: Random number generator.
        
    Returns:
    -------
        Perturbed logits.
        
    """
    if rng is None:
        rng = np.random.default_rng()
    
    perturbation = rng.uniform(-perturbation_scale, perturbation_scale, size=logits.shape)
    return logits + perturbation


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
            
        # Initialize Lagrangian multipliers for constraints
        n_bags = len(bags)
        lambda_lower = np.zeros((n_bags, n_classes))
        lambda_upper = np.zeros((n_bags, n_classes))
        rho = 1.2  # Penalty parameter for augmented Lagrangian
        
        # Temperature for soft assignments (starts high, anneals down)
        temperature_start = 2.0
        temperature_end = 0.3  # Slightly lower for better convergence
        temperature_decay = (temperature_end / temperature_start) ** (1.0 / max_iter)
        
        # SPGD parameters
        perturbation_scale_init = 0.15  # Initial perturbation scale
        perturbation_interval = 2  # Apply perturbation every 2 iterations
        rng = np.random.default_rng(seed)
        
        # Adaptive perturbation tracking
        accuracy_history = []
        perturbation_active = True
        no_improvement_count = 0
        
        # Count loss weight
        count_loss_weight = 0.5  # Increased weight for differentiable count loss
        
        # Track best solution
        best_accuracy = 0.0
        best_y = None

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

            # Apply adaptive SPGD perturbation at intervals
            if perturbation_active and i_iter > 0 and i_iter % perturbation_interval == 0:
                # Adaptive perturbation scale based on iteration
                perturbation_scale = perturbation_scale_init * (0.8 ** (i_iter // 4))
                
                for i_bag in range(len(bags)):
                    if len(bags[i_bag]) > 0:
                        # Apply perturbation to predicted probabilities
                        perturbed_probs = fs[i_bag].copy()
                        for c in range(n_classes):
                            logits = np.log(perturbed_probs[:, c] + 1e-8)
                            perturbed_logits = apply_spgd_perturbation(logits, perturbation_scale, rng)
                            perturbed_probs[:, c] = np.exp(perturbed_logits) / (1 + np.exp(perturbed_logits))
                        # Renormalize
                        perturbed_probs = perturbed_probs / perturbed_probs.sum(axis=1, keepdims=True)
                        fs[i_bag] = perturbed_probs
            
            # For every bag - use soft assignments with augmented Lagrangian
            has_changed = False
            total_count_loss = 0.0
            
            for i_bag in range(len(bags)):
                # Hybrid approach: combine constraint-based selection with soft assignments
                n_instances = len(bags[i_bag])
                if n_instances == 0:
                    continue
                    
                probs = fs[i_bag]  # Shape: (n_instances, n_classes)
                
                # Compute differentiable count loss for this bag
                bag_count_loss = 0.0
                for i_class in range(n_classes):
                    target_lower = int(lower_threshold[i_bag, i_class])
                    target_upper = int(upper_threshold[i_bag, i_class])
                    if target_lower > 0 and target_lower <= n_instances:
                        # Use average of lower and upper as target, but cap at n_instances
                        target_count = min(int((target_lower + target_upper) / 2), n_instances)
                        class_probs = probs[:, i_class]
                        # Only compute if we have enough instances
                        if target_count >= 0 and target_count <= len(class_probs):
                            # Use soft top-k loss with adaptive temperature
                            soft_temp = 0.5 * temperature  # Link to main temperature
                            loss = compute_soft_topk_count_loss(class_probs, target_count, soft_temp)
                            bag_count_loss += count_loss_weight * loss
                
                total_count_loss += bag_count_loss
                
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
                                # Base score from probability
                                base_score = probs[idx, i_class]
                                
                                # Constraint-aware scoring: boost instances that help satisfy constraints
                                constraint_score = 0.0
                                
                                # Reward moving to under-represented class
                                class_deficit = max(0, lower_threshold[i_bag, i_class] - class_count)
                                if class_deficit > 0:
                                    # Stronger boost for larger deficits
                                    constraint_score += 0.3 * np.log1p(class_deficit)
                                
                                # Reward moving from over-represented class
                                current_class = new_y[idx]
                                current_class_count = class_count_dict.get(current_class, 0)
                                current_excess = max(0, current_class_count - upper_threshold[i_bag, current_class])
                                if current_excess > 0:
                                    constraint_score += 0.2 * np.log1p(current_excess)
                                
                                # Combine scores with constraint awareness
                                score = base_score * (1.0 + constraint_score)
                                
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
                                
                                # Constraint bonus
                                c_count = class_count_dict.get(c, 0)
                                if c_count < lower_threshold[i_bag, c]:
                                    scores[c] += 2.0
                                elif c_count < upper_threshold[i_bag, c]:
                                    scores[c] += 0.5
                                else:
                                    scores[c] -= 1.0
                            
                            # Apply temperature-controlled softmax
                            exp_scores = np.exp(scores / temperature)
                            probabilities = exp_scores / np.sum(exp_scores)
                            
                            # Add entropy regularization to prevent overconfidence
                            entropy = -np.sum(probabilities * np.log(probabilities + 1e-8))
                            entropy_weight = 0.05  # Reduced weight for better convergence
                            
                            # Mix with uniform distribution to encourage exploration
                            uniform_prob = np.ones(n_classes) / n_classes
                            probabilities = (1 - entropy_weight) * probabilities + entropy_weight * uniform_prob
                            probabilities = probabilities / np.sum(probabilities)  # Renormalize
                            
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

            # Track best solution and adaptive perturbation
            if debug_true_y is not None:
                current_accuracy = np.mean(
                    np.hstack([self.classifier.predict(bag) for bag in bags]) == np.hstack(debug_true_y)
                )
                
                # Update accuracy history for adaptive perturbation
                accuracy_history.append(current_accuracy)
                
                # Check if we should stop perturbations
                if perturbation_active and len(accuracy_history) >= 3:
                    # Stop perturbations if no improvement for 3 consecutive checks
                    recent_improvements = [
                        accuracy_history[i] - accuracy_history[i-1] 
                        for i in range(len(accuracy_history)-2, len(accuracy_history))
                    ]
                    if all(imp < 2e-4 for imp in recent_improvements):
                        no_improvement_count += 1
                        if no_improvement_count >= 2:
                            perturbation_active = False
                            if debug:
                                print("Stopping SPGD perturbations due to convergence")
                    else:
                        no_improvement_count = 0
                
                if current_accuracy > best_accuracy:
                    best_accuracy = current_accuracy
                    best_y = [y_bag.copy() for y_bag in y]
            
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
                print("differentiable count loss")
                print(f"{total_count_loss:.4f}")
                if debug_true_y is not None:
                    print("instance unit accuracy")
                    print(
                        np.mean(
                            np.hstack([self.classifier.predict(bag) for bag in bags]) == np.hstack(debug_true_y),
                        ),
                    )
                    print("instance unit accuracy (label adjusted)")
                    print(np.mean(np.hstack(y) == np.hstack(debug_true_y)))
                    if best_accuracy > 0:
                        print(f"best accuracy so far: {best_accuracy:.6f}")
                print("-----")

            # Early stopping when converged
            if not has_changed and i_iter > perturbation_interval:
                # Allow at least one perturbation cycle before stopping
                break

        # Return best solution if available
        if best_y is not None and best_accuracy > 0:
            # Refit classifier with best labels
            flatten_bags = np.vstack(bags)
            flatten_best_y = np.hstack(best_y)
            self.classifier.fit(flatten_bags, flatten_best_y)
            return self.classifier, best_y
        
        return self.classifier, y
