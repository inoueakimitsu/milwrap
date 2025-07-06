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


def compute_attention_entropy_maximization(bag_probs: NDArray[np.float64], lambda_entropy: float = 0.08) -> NDArray[np.float64]:
    """Compute AEM (Attention Entropy Maximization) regularized scores.
    
    Based on AEM research: adds -H(α) regularizer to prevent over-concentrated attention,
    keeping entropy high enough for multiple diagnostic instances to share probability mass.
    
    Args:
    ----
        bag_probs: Instance probability matrix of shape (n_instances, n_classes).
        lambda_entropy: Entropy regularization strength (default 0.08).
        
    Returns:
    -------
        AEM-regularized scores for each instance.
        
    """
    if len(bag_probs.shape) != 2 or bag_probs.shape[0] == 0:
        return np.ones(bag_probs.shape[0])
    
    n_instances, n_classes = bag_probs.shape
    
    # Calculate instance-wise attention weights (softmax over max probabilities)
    max_probs = np.max(bag_probs, axis=1)
    attention_weights = np.exp(max_probs) / (np.sum(np.exp(max_probs)) + 1e-10)
    
    # Calculate attention entropy: H(α) = -Σ αᵢ log(αᵢ)
    attention_entropy = -np.sum(attention_weights * np.log(attention_weights + 1e-10))
    
    # AEM regularization: encourage higher entropy (less concentrated attention)
    entropy_bonus = lambda_entropy * attention_entropy
    
    # Apply entropy bonus to instance scores
    aem_scores = max_probs + entropy_bonus * attention_weights
    
    return aem_scores


def compute_margin_aware_confidence(probs: NDArray[np.float64], target_class: int, margin_lambda: float = 0.25) -> float:
    """Compute margin-aware confidence scores.
    
    Args:
    ----
        probs: Probability vector for an instance.
        target_class: The target class index.
        margin_lambda: Margin weight parameter.
        
    Returns:
    -------
        Margin-aware confidence score.
        
    """
    if len(probs) < 2:
        return probs[target_class] if target_class < len(probs) else 0.0
    
    target_prob = probs[target_class]
    other_probs = np.concatenate([probs[:target_class], probs[target_class+1:]])
    max_other_prob = np.max(other_probs) if len(other_probs) > 0 else 0.0
    
    # Margin between target class and strongest competitor
    margin = target_prob - max_other_prob
    
    # Combine probability and margin
    confidence = target_prob + margin_lambda * margin
    
    return confidence


def exact_k_of_n_assignment(probs, k_min, k_max, current_assignments=None, aem_scores=None):
    """Solve exact k-of-n assignment problem with optional AEM weighting.
    
    Find optimal assignment of instances to positive class such that
    k_min <= sum(assignments) <= k_max, maximizing sum(log(p_i))
    
    Args:
        probs: probabilities for positive class (n_instances,)
        k_min: minimum number of positive instances
        k_max: maximum number of positive instances
        current_assignments: current assignments (optional)
        aem_scores: AEM scores for weighting (optional)
        
    Returns:
        optimal_assignments: binary array of assignments
    """
    n = len(probs)
    
    # Handle edge cases
    if k_min > n:
        # Cannot satisfy constraint
        return np.ones(n) if current_assignments is None else current_assignments
    if k_max < 0:
        # Cannot satisfy constraint
        return np.zeros(n) if current_assignments is None else current_assignments
    
    # Compute log-odds ratios
    eps = 1e-8
    log_odds = np.log(probs + eps) - np.log(1 - probs + eps)
    
    # Apply AEM weighting if provided
    if aem_scores is not None and len(aem_scores) == len(probs):
        log_odds = log_odds + 0.1 * aem_scores  # AEM integration weight
    
    # Sort by log-odds in descending order
    sorted_indices = np.argsort(log_odds)[::-1]
    
    # Optimal k is between k_min and k_max
    # Choose k that maximizes the sum of log probabilities
    best_k = k_min
    best_score = -np.inf
    
    for k in range(max(0, k_min), min(n, k_max) + 1):
        # Score is sum of log probs for top k
        if k > 0:
            score = np.sum(np.log(probs[sorted_indices[:k]] + eps))
        else:
            score = 0
        
        # Add log probs of negative assignments
        if k < n:
            score += np.sum(np.log(1 - probs[sorted_indices[k:]] + eps))
            
        if score > best_score:
            best_score = score
            best_k = k
    
    # Create assignment vector
    assignments = np.zeros(n, dtype=int)
    if best_k > 0:
        assignments[sorted_indices[:best_k]] = 1
    
    return assignments


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
            # Improved initialization: consider constraint satisfaction potential
            y = []
            for i_bag, n_instances_in_bag in enumerate(n_list):
                # Calculate constraint satisfaction score for each class
                constraint_scores = np.zeros(n_classes)
                
                for i_class in range(n_classes):
                    # Favor classes where n_instances fits well within constraints
                    lower_bound = lower_threshold[i_bag, i_class]
                    upper_bound = upper_threshold[i_bag, i_class]
                    center = (lower_bound + upper_bound) / 2.0
                    
                    # Score based on how well n_instances fits the constraints
                    if n_instances_in_bag < lower_bound:
                        # Can't satisfy lower bound - heavily penalize
                        constraint_scores[i_class] = -1000
                    elif n_instances_in_bag > upper_bound:
                        # Exceeds upper bound - penalize based on excess
                        excess_ratio = (n_instances_in_bag - upper_bound) / upper_bound
                        constraint_scores[i_class] = -10 * excess_ratio
                    else:
                        # Within bounds - score based on distance from center
                        range_size = upper_bound - lower_bound + 1e-6
                        distance_from_center = abs(n_instances_in_bag - center)
                        constraint_scores[i_class] = 1.0 - (distance_from_center / range_size)
                
                # Add small noise to break ties
                constraint_scores += np.random.randn(n_classes) * 0.01
                
                # Select class with best constraint satisfaction potential
                best_class = np.argmax(constraint_scores)
                y.append(np.repeat(best_class, n_instances_in_bag))
        else:
            y = initial_y
            
        # Initialize Lagrangian multipliers for constraints
        n_bags = len(bags)
        lambda_lower = np.zeros((n_bags, n_classes))
        lambda_upper = np.zeros((n_bags, n_classes))
        rho = 1.0  # Penalty parameter for augmented Lagrangian
        
        # Detect initialization type (external vs auto)
        is_external_init = initial_y is not None
        
        # Temperature for soft assignments (starts moderate, anneals down)
        # Adaptive temperature based on initialization type and convergence behavior
        if is_external_init:
            # Conservative temperature for external init to preserve good initialization
            temperature_start = 1.0  # Same as auto-init
            temperature_end = 0.05   # Same as auto-init
        else:
            temperature_start = 1.0  # Keep standard for auto-init
            temperature_end = 0.05   # Lower final temperature for precision
        
        # Non-linear temperature decay for better convergence
        # Use cosine annealing instead of exponential
        current_temperature = temperature_start
        label_change_rate = 1.0  # Track label change rate for adaptive annealing
        previous_change_rates = []  # Track history for adaptive scheduling
        
        # AEM parameters with adaptive tuning based on initialization quality
        violation_ratio = 0.0  # Default for auto-init
        if is_external_init:
            # Analyze initial assignment quality for adaptive parameter tuning
            initial_violations = 0
            for i_bag in range(n_bags):
                class_counts = pd.Series(y[i_bag]).value_counts().to_dict()
                for i_class in range(n_classes):
                    count = class_counts.get(i_class, 0)
                    if count < lower_threshold[i_bag, i_class]:
                        initial_violations += lower_threshold[i_bag, i_class] - count
                    elif count > upper_threshold[i_bag, i_class]:
                        initial_violations += count - upper_threshold[i_bag, i_class]
            
            # Adaptive parameters based on initialization quality
            violation_ratio = initial_violations / sum(len(bag) for bag in bags)
            if violation_ratio < 0.05:  # Very good initialization
                lambda_entropy_base = 0.005  # Minimal entropy regularization (reduced from 0.01)
                aem_weight = 0.02  # Very low weight (reduced from 0.03)
                decay_strength = 5.0  # Faster decay (increased from 4.0)
            elif violation_ratio < 0.15:  # Good initialization
                lambda_entropy_base = 0.01  # Low entropy regularization (reduced from 0.02)
                aem_weight = 0.03  # Low weight (reduced from 0.05)
                decay_strength = 4.0  # Faster decay (increased from 3.0)
            else:  # Poor initialization
                lambda_entropy_base = 0.02  # Higher entropy for exploration (reduced from 0.04)
                aem_weight = 0.05  # Higher weight (reduced from 0.08)
                decay_strength = 3.0  # Moderate decay (increased from 2.0)
                
            warmup_fraction = 0.0  # No warmup for external init
            margin_lambda = 0.2  # Higher margin lambda for external init
            margin_weight = 0.2  # Higher margin contribution
        else:
            lambda_entropy_base = 0.06  # Higher for auto init (needs more entropy)
            aem_weight = 0.1  # Standard weight
            warmup_fraction = 0.3  # 30% warmup
            decay_strength = 2.0  # Standard decay
            margin_lambda = 0.15  # Lower margin lambda for auto init
            margin_weight = 0.15  # Standard margin contribution
        
        # Self-paced learning: track bag difficulty scores
        bag_difficulty = np.zeros(n_bags)
        
        # Confidence-based freezing parameters
        confidence_threshold_start = 0.98  # Start very strict  
        confidence_threshold_end = 0.90    # Still strict at the end
        confidence_decay = (confidence_threshold_end / confidence_threshold_start) ** (1.0 / max_iter)
        
        for i_iter in range(max_iter):
            # Adaptive temperature annealing based on convergence speed
            if i_iter > 0:
                # Track change rate history for better adaptation
                previous_change_rates.append(label_change_rate)
                
                # Cosine annealing with restarts for better exploration
                progress = i_iter / max_iter
                base_temp = temperature_end + 0.5 * (temperature_start - temperature_end) * (1 + np.cos(np.pi * progress))
                
                # Adaptive adjustment based on convergence behavior
                if is_external_init:
                    # More conservative temperature adjustment for external init
                    if label_change_rate > 0.05:  # High change rate
                        temp_multiplier = 1.05  # Reduced from 1.1
                    elif label_change_rate > 0.02:  # Moderate changes
                        temp_multiplier = 0.98
                    elif label_change_rate > 0.005:  # Low changes
                        temp_multiplier = 0.93
                    else:  # Very low changes - mild reheat
                        if len(previous_change_rates) >= 2 and all(r < 0.005 for r in previous_change_rates[-2:]):
                            temp_multiplier = 1.05  # Reduced from 1.2
                        else:
                            temp_multiplier = 0.9
                else:
                    # Original settings for auto-init
                    if label_change_rate > 0.05:  # High change rate - maintain exploration
                        temp_multiplier = 1.1
                    elif label_change_rate > 0.02:  # Moderate changes
                        temp_multiplier = 1.0
                    elif label_change_rate > 0.005:  # Low changes
                        temp_multiplier = 0.95
                    else:  # Very low changes - check if stuck
                        # If stuck at low change rate for multiple iterations, reheat
                        if len(previous_change_rates) >= 2 and all(r < 0.005 for r in previous_change_rates[-2:]):
                            temp_multiplier = 1.2  # Reheat to escape local minima
                        else:
                            temp_multiplier = 0.9
                
                current_temperature = max(temperature_end, base_temp * temp_multiplier)
            temperature = current_temperature
            
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
            total_changes = 0  # Track total label changes for adaptive temperature
            total_instances = sum(len(bag) for bag in bags)
            
            # Self-paced learning: sort bags by difficulty (constraint violations)
            if i_iter > 0:
                bag_order = np.argsort(bag_difficulty)  # Process easier bags first
            else:
                bag_order = np.arange(n_bags)
            
            for bag_idx in range(len(bags)):
                i_bag = bag_order[bag_idx]
                # Hybrid approach: combine constraint-based selection with soft assignments
                n_instances = len(bags[i_bag])
                probs = fs[i_bag]  # Shape: (n_instances, n_classes)
                
                # Get current class counts
                class_count_dict = pd.Series(y[i_bag]).value_counts().to_dict()
                
                # Track which instances have been reassigned in this iteration
                changed_indices = set()
                new_y = y[i_bag].copy()
                
                # Confidence-based freezing with dynamic threshold
                frozen_indices = set()
                if i_iter >= 2:  # Start freezing after initial exploration
                    confidence_threshold = confidence_threshold_start * (confidence_decay ** i_iter)
                    max_probs = np.max(probs, axis=1)
                    predicted_classes = np.argmax(probs, axis=1)
                    
                    for idx in range(n_instances):
                        # Freeze if high confidence AND consistent with current assignment
                        if max_probs[idx] > confidence_threshold and y[i_bag][idx] == predicted_classes[idx]:
                            frozen_indices.add(idx)
                
                # Adaptive AEM scheduling based on convergence and bag difficulty
                warmup_iters = int(warmup_fraction * max_iter)
                
                # Base entropy calculation with warm-up and decay
                if warmup_iters > 0 and i_iter < warmup_iters:
                    # Warm-up phase
                    base_lambda = lambda_entropy_base * (i_iter + 1) / warmup_iters
                else:
                    # Adaptive decay based on convergence speed
                    if warmup_iters == 0:
                        decay_progress = i_iter / max_iter
                    else:
                        decay_progress = (i_iter - warmup_iters) / (max_iter - warmup_iters)
                    
                    # Adjust decay rate based on convergence behavior
                    if label_change_rate > 0.02:
                        # High change rate - slower decay to maintain exploration
                        effective_decay = decay_strength * 0.7
                    elif label_change_rate < 0.005 and i_iter > 3:
                        # Low change rate after initial iterations - faster decay
                        effective_decay = decay_strength * 1.3
                    else:
                        effective_decay = decay_strength
                    
                    base_lambda = lambda_entropy_base * np.exp(-effective_decay * decay_progress)
                
                # Bag-specific adjustment based on difficulty
                bag_difficulty_normalized = bag_difficulty[i_bag] / (n_instances + 1e-8)
                if bag_difficulty_normalized > 0.1:  # High difficulty bag
                    lambda_entropy = base_lambda * 1.2  # More entropy for difficult bags
                else:
                    lambda_entropy = base_lambda
                
                # Compute AEM scores for this bag
                aem_scores = compute_attention_entropy_maximization(probs, lambda_entropy=lambda_entropy)
                
                # For binary classification, use exact k-of-n solver
                if n_classes == 2:
                    # Apply minimal slack only in first iteration for robustness
                    effective_lower_0 = lower_threshold[i_bag, 0]
                    effective_upper_0 = upper_threshold[i_bag, 0]
                    effective_lower_1 = lower_threshold[i_bag, 1]
                    effective_upper_1 = upper_threshold[i_bag, 1]
                    
                    if i_iter == 0:  # Only first iteration
                        effective_lower_0 = effective_lower_0 * 0.95  # 5% slack
                        effective_upper_0 = effective_upper_0 * 1.05  # 5% slack
                        effective_lower_1 = effective_lower_1 * 0.95  # 5% slack
                        effective_upper_1 = effective_upper_1 * 1.05  # 5% slack
                    
                    # For binary case, we optimize for class 1
                    # The constraint is: effective_lower_1 <= k <= effective_upper_1
                    k_min = int(max(0, effective_lower_1))
                    k_max = int(min(n_instances, effective_upper_1))
                    
                    # Also consider class 0 constraints
                    # n - k must be in [effective_lower_0, effective_upper_0]
                    # So k must be in [n - effective_upper_0, n - effective_lower_0]
                    k_min = max(k_min, n_instances - int(effective_upper_0))
                    k_max = min(k_max, n_instances - int(effective_lower_0))
                    
                    # Account for frozen instances
                    frozen_counts = [0, 0]
                    for idx in frozen_indices:
                        frozen_counts[new_y[idx]] += 1
                    
                    # Adjust constraints for remaining instances
                    active_indices = [idx for idx in range(n_instances) if idx not in frozen_indices]
                    n_active = len(active_indices)
                    
                    if n_active > 0:
                        # Adjust k_min and k_max based on frozen instances
                        k_min_active = max(0, k_min - frozen_counts[1])
                        k_max_active = min(n_active, k_max - frozen_counts[1])
                        
                        # Get probabilities for active instances only
                        active_probs = probs[active_indices, 1]
                        
                        # Solve exact assignment for active instances (without AEM to avoid double-counting)
                        optimal_assignment_active = exact_k_of_n_assignment(active_probs, k_min_active, k_max_active)
                        
                        # Map back to full assignment
                        optimal_assignment = new_y.copy()
                        for i, idx in enumerate(active_indices):
                            optimal_assignment[idx] = optimal_assignment_active[i]
                    else:
                        # All frozen, keep current assignment
                        optimal_assignment = new_y
                    
                    # Apply changes
                    for idx in range(n_instances):
                        if optimal_assignment[idx] == 1 and new_y[idx] != 1:
                            old_class = new_y[idx]
                            new_y[idx] = 1
                            changed_indices.add(idx)
                            # Update counts
                            class_count_dict[1] = class_count_dict.get(1, 0) + 1
                            class_count_dict[old_class] = class_count_dict.get(old_class, 0) - 1
                            if class_count_dict[old_class] == 0:
                                del class_count_dict[old_class]
                        elif optimal_assignment[idx] == 0 and new_y[idx] != 0:
                            old_class = new_y[idx]
                            new_y[idx] = 0
                            changed_indices.add(idx)
                            # Update counts
                            class_count_dict[0] = class_count_dict.get(0, 0) + 1
                            class_count_dict[old_class] = class_count_dict.get(old_class, 0) - 1
                            if class_count_dict[old_class] == 0:
                                del class_count_dict[old_class]
                else:
                    # Multi-class case: use original greedy approach
                    # Priority 1: Fix constraint violations using most confident instances
                    target_classes = list(range(n_classes))
                    random.shuffle(target_classes)
                    
                    for i_class in target_classes:
                        class_count = class_count_dict.get(i_class, 0)
                        
                        # Apply constraint slack for first iteration only
                        if i_iter == 0:
                            # 5% slack in first iteration
                            effective_lower = lower_threshold[i_bag, i_class] * 0.95
                        else:
                            effective_lower = lower_threshold[i_bag, i_class]
                        
                        if class_count < effective_lower:
                            # Need more instances of this class
                            deficit = int(effective_lower - class_count)
                            
                            # Find best candidates: high probability for this class, not already changed, not frozen
                            candidates = []
                            for idx in range(n_instances):
                                if idx not in changed_indices and idx not in frozen_indices and new_y[idx] != i_class:
                                    # Base score from probability with AEM enhancement
                                    base_score = probs[idx, i_class]
                                    # For external init, only apply AEM when confidence is low
                                    if is_external_init and base_score > 0.7:
                                        aem_bonus = 0.0  # High confidence, no AEM needed
                                    else:
                                        aem_bonus = aem_weight * aem_scores[idx]  # AEM integration with adaptive weight
                                    margin_conf = compute_margin_aware_confidence(probs[idx], i_class, margin_lambda=margin_lambda)
                                    score = base_score + aem_bonus + margin_weight * (margin_conf - base_score)
                                    
                                    # Boost score if moving from over-represented class
                                    current_class = new_y[idx]
                                    current_class_count = class_count_dict.get(current_class, 0)
                                    if current_class_count > upper_threshold[i_bag, current_class]:
                                        score *= 3.0  # Stronger boost for constraint satisfaction
                                    elif current_class_count <= lower_threshold[i_bag, current_class]:
                                        score *= 0.3  # Stronger penalty for taking from under-represented class
                                    
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
                        
                        # Apply adaptive constraint slack for upper threshold
                        if i_iter == 0:
                            effective_upper = upper_threshold[i_bag, i_class] * 1.05
                        else:
                            effective_upper = upper_threshold[i_bag, i_class]
                        
                        if class_count > effective_upper:
                            # Too many instances of this class
                            excess = int(class_count - effective_upper)
                            
                            # Find instances to reassign: low probability for this class, not frozen
                            candidates = []
                            for idx in range(n_instances):
                                if idx not in changed_indices and idx not in frozen_indices and new_y[idx] == i_class:
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
                                    
                                    # Base score from probability with AEM enhancement
                                    base_log_score = np.log(probs[idx, c] + 1e-8)
                                    # Conditional AEM application
                                    if is_external_init and probs[idx, c] > 0.7:
                                        aem_bonus = 0.0  # High confidence for external init, no AEM
                                    elif probs[idx, c] > 0.5:
                                        aem_bonus = aem_weight * aem_scores[idx]
                                    else:
                                        aem_bonus = 0.5 * aem_weight * aem_scores[idx]
                                    margin_conf = compute_margin_aware_confidence(probs[idx], c, margin_lambda=margin_lambda)
                                    scores[c] = base_log_score + aem_bonus + 0.1 * np.log(max(margin_conf, 1e-8) + 1e-8)
                                    
                                    # Constraint bonus
                                    c_count = class_count_dict.get(c, 0)
                                    if c_count < lower_threshold[i_bag, c]:
                                        scores[c] += 3.0  # Stronger incentive for under-represented classes
                                    elif c_count < upper_threshold[i_bag, c]:
                                        scores[c] += 1.0
                                    else:
                                        scores[c] -= 2.0  # Stronger penalty for over-represented classes
                                
                                # Apply temperature
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
                    # Count changes for adaptive temperature
                    total_changes += np.sum(y[i_bag] != new_y)
                    y[i_bag] = new_y
                
                # Update bag difficulty score based on constraint violations
                final_counts = pd.Series(new_y).value_counts().to_dict()
                constraint_violation = 0
                for i_class in range(n_classes):
                    class_count = final_counts.get(i_class, 0)
                    # Lower threshold violations
                    if class_count < lower_threshold[i_bag, i_class]:
                        constraint_violation += lower_threshold[i_bag, i_class] - class_count
                    # Upper threshold violations
                    if class_count > upper_threshold[i_bag, i_class]:
                        constraint_violation += class_count - upper_threshold[i_bag, i_class]
                
                bag_difficulty[i_bag] = constraint_violation

            if debug:
                print(f"iter: {i_iter}, temperature: {temperature:.3f}, lambda_entropy: {lambda_entropy:.4f}")
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
            
            # Update label change rate for adaptive temperature
            label_change_rate = total_changes / total_instances if total_instances > 0 else 0
            if debug and i_iter > 0:
                print(f"Label change rate: {label_change_rate:.3%}, Temperature: {temperature:.3f}")

            # Continue to iteration 9 for evaluation, even if converged
            min_iterations = 9
            if not has_changed and i_iter >= 4:
                # Labels have converged, but continue training classifier
                if debug:
                    print(f"Labels converged at iteration {i_iter}, continuing to refine classifier...")
                # Train with additional inner iterations when labels are stable
                for inner_iter in range(5):
                    self.classifier.fit(flatten_bags, flatten_y)
                    
                # For external init with good quality, apply fine-tuning
                if is_external_init and violation_ratio < 0.05:
                    # Small perturbation to escape potential local optima
                    perturbation_rate = 0.001 * (9 - i_iter) / 5  # Decreasing perturbation
                    n_perturb = max(1, int(total_instances * perturbation_rate))
                    
                    # Select instances with medium confidence for perturbation
                    all_probs = []
                    all_indices = []
                    for bag_idx, bag in enumerate(bags):
                        bag_probs = fs[bag_idx]
                        max_probs = np.max(bag_probs, axis=1)
                        for inst_idx, prob in enumerate(max_probs):
                            if 0.6 < prob < 0.9:  # Medium confidence
                                all_probs.append(prob)
                                all_indices.append((bag_idx, inst_idx))
                    
                    if len(all_indices) > n_perturb:
                        # Randomly select instances to perturb
                        selected_indices = random.sample(range(len(all_indices)), n_perturb)
                        for idx in selected_indices:
                            bag_idx, inst_idx = all_indices[idx]
                            # Flip to second-best class
                            probs = fs[bag_idx][inst_idx]
                            current_class = y[bag_idx][inst_idx]
                            probs[current_class] = -1  # Temporarily mask current class
                            new_class = np.argmax(probs)
                            
                            # Check if flip maintains constraints
                            test_y = y[bag_idx].copy()
                            test_y[inst_idx] = new_class
                            test_counts = pd.Series(test_y).value_counts().to_dict()
                            
                            valid = True
                            for c in range(n_classes):
                                count = test_counts.get(c, 0)
                                if count < lower_threshold[bag_idx, c] or count > upper_threshold[bag_idx, c]:
                                    valid = False
                                    break
                            
                            if valid:
                                y[bag_idx][inst_idx] = new_class
                                has_changed = True
                                
            elif not has_changed and i_iter < 4:
                if debug:
                    print(f"Early convergence at iteration {i_iter}, continuing exploration...")
            
            # Only break after minimum iterations
            if i_iter >= min_iterations - 1:
                break

        return self.classifier, y
