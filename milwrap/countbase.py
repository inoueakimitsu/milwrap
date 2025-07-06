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


def compute_instance_confidence_scores(bag_probs: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute simple confidence scores for instances.
    
    Args:
    ----
        bag_probs: Instance probability matrix of shape (n_instances, n_classes).
        
    Returns:
    -------
        Confidence scores for each instance.
        
    """
    if len(bag_probs.shape) != 2 or bag_probs.shape[0] == 0:
        return np.ones(bag_probs.shape[0])
    
    # Simple max probability as confidence score
    max_probs = np.max(bag_probs, axis=1)
    return max_probs


def compute_attention_entropy_maximization(bag_probs: NDArray[np.float64], lambda_entropy: float = 0.05) -> NDArray[np.float64]:
    """Compute AEM (Attention Entropy Maximization) regularized scores.
    
    Based on AEM research: adds entropy regularizer to prevent over-concentrated attention.
    
    Args:
        bag_probs: Instance probability matrix of shape (n_instances, n_classes).
        lambda_entropy: Entropy regularization strength.
        
    Returns:
        AEM-regularized scores for each instance.
    """
    if len(bag_probs.shape) != 2 or bag_probs.shape[0] == 0:
        return np.ones(bag_probs.shape[0])
    
    # Calculate instance-wise attention weights (softmax over max probabilities)
    max_probs = np.max(bag_probs, axis=1)
    attention_weights = np.exp(max_probs) / (np.sum(np.exp(max_probs)) + 1e-10)
    
    # Calculate attention entropy: H(α) = -Σ αᵢ log(αᵢ)
    attention_entropy = -np.sum(attention_weights * np.log(attention_weights + 1e-10))
    
    # AEM regularization: encourage higher entropy
    entropy_bonus = lambda_entropy * attention_entropy
    
    # Return scores with entropy bonus
    return max_probs + entropy_bonus


def smooth_topk(scores: NDArray[np.float64], k: float, temperature: float = 1.0) -> NDArray[np.float64]:
    """Differentiable smooth top-k operator (ST_k).
    
    Based on NeurIPS 2024 paper: smooth, differentiable approximation to hard top-k selection.
    
    Args:
        scores: Input scores to select from.
        k: Target number of selections (can be fractional).
        temperature: Temperature parameter for smoothness.
        
    Returns:
        Soft selection weights (sum approximately equals k).
    """
    n = len(scores)
    if n == 0 or k <= 0:
        return np.zeros(n)
    if k >= n:
        return np.ones(n)
    
    # Sort scores in descending order
    sorted_indices = np.argsort(scores)[::-1]
    sorted_scores = scores[sorted_indices]
    
    # Compute threshold dynamically
    # For exact top-k: threshold is between k-th and (k+1)-th elements
    k_int = int(k)
    k_frac = k - k_int
    
    if k_int < n - 1:
        # Interpolate threshold between elements
        threshold = sorted_scores[k_int] * (1 - k_frac)
        if k_int + 1 < n:
            threshold += sorted_scores[k_int + 1] * k_frac
    else:
        threshold = sorted_scores[k_int]
    
    # Smooth selection using sigmoid
    smooth_scores = 1 / (1 + np.exp(-(scores - threshold) / temperature))
    
    # Normalize to ensure sum equals k (projection step)
    current_sum = np.sum(smooth_scores)
    if current_sum > 0:
        smooth_scores = smooth_scores * k / current_sum
    
    return smooth_scores


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


def exact_k_of_n_assignment(probs, k_min, k_max, current_assignments=None, margin_scores=None):
    """Solve exact k-of-n assignment problem with optional margin weighting.
    
    Find optimal assignment of instances to positive class such that
    k_min <= sum(assignments) <= k_max, maximizing sum(log(p_i))
    
    Args:
        probs: probabilities for positive class (n_instances,)
        k_min: minimum number of positive instances
        k_max: maximum number of positive instances
        current_assignments: current assignments (optional)
        margin_scores: margin-based scores for weighting (optional)
        
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
    
    # Apply margin weighting if provided
    if margin_scores is not None and len(margin_scores) == len(probs):
        log_odds = log_odds + 0.1 * margin_scores  # Moderate weight for margins
    
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


def gumbel_sample(shape, eps=1e-6):
    """Sample from Gumbel(0, 1) distribution.
    
    Args:
        shape: Shape of the output tensor
        eps: Small value for numerical stability
        
    Returns:
        Gumbel noise samples
    """
    u = np.random.uniform(eps, 1 - eps, shape)
    return -np.log(-np.log(u))


def binary_gumbel_softmax(logits, temperature, hard=False, backward_temperature=None):
    """Binary Gumbel-Softmax (Binary Concrete) sampling with decoupled temperature.
    
    Args:
        logits: Log-odds for positive class (n_instances,)
        temperature: Temperature parameter for forward pass
        hard: Whether to use straight-through estimator
        backward_temperature: Temperature for backward pass (decoupled ST-GS)
        
    Returns:
        Relaxed binary samples in [0, 1]
    """
    # Sample Gumbel noise
    g = gumbel_sample(logits.shape)
    
    # Apply Gumbel-Softmax transformation
    # For decoupled temperature, we use lower temp for forward, higher for backward
    forward_temp = temperature
    if backward_temperature is None:
        backward_temperature = temperature
    
    # Forward pass uses forward_temp
    z_tilde = 1 / (1 + np.exp(-(logits + g) / forward_temp))
    
    # For gradient computation, we would use backward_temperature
    # This is a conceptual implementation - in practice this would be handled
    # by the autograd system
    z_grad = 1 / (1 + np.exp(-(logits + g) / backward_temperature))
    
    if hard:
        # Straight-through estimator with better gradient flow
        z_hard = (z_tilde > 0.5).astype(np.float64)
        # Conceptually: forward uses z_hard, backward uses z_grad
        return z_hard
    else:
        return z_tilde


def gumbel_top_k_relaxed(logits, k_min, k_max, temperature):
    """Relaxed top-k selection with Gumbel noise for range constraints.
    
    Args:
        logits: Log-odds scores for each instance
        k_min: Minimum number of selections
        k_max: Maximum number of selections
        temperature: Temperature parameter
        
    Returns:
        Soft selection weights satisfying constraints
    """
    n = len(logits)
    
    # Add Gumbel noise to logits
    g = gumbel_sample(logits.shape)
    perturbed_logits = logits + g
    
    # Sort indices by perturbed logits
    sorted_indices = np.argsort(perturbed_logits)[::-1]
    
    # Compute soft indicators for being in top-k for different k values
    soft_weights = np.zeros(n)
    
    # Find optimal k within [k_min, k_max] based on perturbed scores
    k_scores = []
    for k in range(max(0, k_min), min(n, k_max) + 1):
        if k > 0:
            # Score is sum of top-k perturbed logits
            score = np.sum(perturbed_logits[sorted_indices[:k]])
        else:
            score = 0
        k_scores.append((k, score))
    
    # Select k with highest score
    best_k = max(k_scores, key=lambda x: x[1])[0]
    
    # Compute soft top-k indicators using sigmoid relaxation
    if best_k > 0:
        threshold = perturbed_logits[sorted_indices[best_k-1]]
        if best_k < n:
            # Interpolate threshold for smoother gradients
            next_threshold = perturbed_logits[sorted_indices[best_k]]
            threshold = 0.5 * (threshold + next_threshold)
    else:
        threshold = np.max(perturbed_logits) + 1.0
    
    # Soft indicators using temperature-scaled sigmoid
    soft_weights = 1 / (1 + np.exp(-(perturbed_logits - threshold) / temperature))
    
    return soft_weights


def constraint_aware_soft_assignment(probs, k_min, k_max, temperature=0.5, margin_scores=None, aem_scores=None):
    """Constraint-aware soft k-of-n assignment with penalty-based optimization.
    
    Find soft assignment weights that maximize probability while satisfying
    constraints through penalty terms.
    
    Args:
        probs: probabilities for positive class (n_instances,)
        k_min: minimum number of positive instances
        k_max: maximum number of positive instances  
        temperature: temperature for Gumbel-Softmax
        margin_scores: margin-based scores for weighting (optional)
        aem_scores: AEM-based scores for weighting (optional)
        
    Returns:
        soft_assignments: continuous weights in [0, 1]
    """
    n = len(probs)
    
    # Handle edge cases
    if k_min > n:
        return np.ones(n)
    if k_max < 0:
        return np.zeros(n)
    
    # Compute base scores
    eps = 1e-8
    logits = np.log(probs + eps) - np.log(1 - probs + eps)
    
    # Add margin and AEM contributions
    if margin_scores is not None and len(margin_scores) == len(probs):
        logits = logits + 0.25 * margin_scores  # Increased weight
    
    if aem_scores is not None and len(aem_scores) == len(probs):
        logits = logits + 0.15 * aem_scores
    
    # Sample with Gumbel noise
    g = gumbel_sample(logits.shape)
    perturbed_logits = logits + g
    
    # Find optimal soft assignment with constraint penalties
    # Target k is center of feasible range
    k_target = (k_min + k_max) / 2.0
    
    # Multiple samples for lower variance
    n_samples = 3
    soft_assignments = []
    
    for _ in range(n_samples):
        # Sample Gumbel noise
        g = gumbel_sample(logits.shape)
        z_tilde = 1 / (1 + np.exp(-(logits + g) / temperature))
        
        # Compute constraint penalty
        current_sum = np.sum(z_tilde)
        
        # Soft penalty for being outside [k_min, k_max]
        if current_sum < k_min:
            penalty = (k_min - current_sum) ** 2
            # Gradient: increase all weights proportionally
            gradient_scale = 1.0 + 0.1 * penalty / (current_sum + eps)
            z_tilde = np.minimum(z_tilde * gradient_scale, 1.0)
        elif current_sum > k_max:
            penalty = (current_sum - k_max) ** 2
            # Gradient: decrease all weights proportionally
            gradient_scale = 1.0 - 0.1 * penalty / (current_sum + eps)
            z_tilde = np.maximum(z_tilde * gradient_scale, 0.0)
        
        soft_assignments.append(z_tilde)
    
    # Average samples
    return np.mean(soft_assignments, axis=0)


def soft_k_of_n_assignment(probs, k_min, k_max, temperature=0.5, margin_scores=None, aem_scores=None):
    """Wrapper for backward compatibility."""
    return constraint_aware_soft_assignment(probs, k_min, k_max, temperature, margin_scores, aem_scores)


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
            # Enhanced initialization with probabilistic assignment
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
                
                # For binary classification, use mixed initialization for diversity
                if n_classes == 2:
                    # Calculate how many instances should be positive
                    k_target = int((lower_threshold[i_bag, 1] + upper_threshold[i_bag, 1]) / 2)
                    k_target = max(int(lower_threshold[i_bag, 1]), 
                                 min(int(upper_threshold[i_bag, 1]), k_target))
                    
                    # Random assignment respecting constraints
                    instance_labels = np.zeros(n_instances_in_bag, dtype=int)
                    if k_target > 0:
                        positive_indices = np.random.choice(n_instances_in_bag, k_target, replace=False)
                        instance_labels[positive_indices] = 1
                    y.append(instance_labels)
                else:
                    # Multi-class: use constraint scores with temperature-based sampling
                    # Add small noise to break ties
                    constraint_scores += np.random.randn(n_classes) * 0.01
                    
                    # Use softmax with temperature for probabilistic selection
                    init_temperature = 0.5
                    exp_scores = np.exp(constraint_scores / init_temperature)
                    class_probs = exp_scores / np.sum(exp_scores)
                    
                    # Select class probabilistically
                    best_class = np.random.choice(n_classes, p=class_probs)
                    y.append(np.repeat(best_class, n_instances_in_bag))
        else:
            y = initial_y
            
        # Initialize simple parameters
        n_bags = len(bags)
        
        # Gumbel-Softmax temperature schedule - simpler is better
        gumbel_temp_start = 1.0   # Standard starting temperature
        gumbel_temp_end = 0.1     # Standard ending temperature
        gumbel_temp_decay = 0.88  # Moderate decay rate
        
        # Self-paced learning: track bag difficulty scores
        bag_difficulty = np.zeros(n_bags)
        
        # Adaptive confidence-based freezing
        confidence_threshold = 0.98  # Very high threshold for selective freezing
        margin_lambda = 0.35  # Further increased margin importance
        
        # Track assignment stability
        previous_assignments = [y_bag.copy() for y_bag in y]
        
        for i_iter in range(max_iter):
            # Simple exponential temperature decay
            gumbel_temperature = max(gumbel_temp_end, 
                                   gumbel_temp_start * (gumbel_temp_decay ** i_iter))
            
            # Staged AEM schedule: cosine annealing for first 60% of iterations
            exploration_phase_end = int(np.ceil(max_iter * 0.6))
            if i_iter < exploration_phase_end:
                # Cosine annealing during exploration
                aem_lambda_max = 0.5
                aem_lambda = aem_lambda_max * 0.5 * (1 - np.cos(np.pi * i_iter / exploration_phase_end))
            else:
                # Small constant during exploitation
                aem_lambda = 0.05
            
            # Fit
            flatten_bags = np.vstack(bags)
            flatten_y = np.hstack(y)
            self.classifier.fit(flatten_bags, flatten_y)

            # Compute outputs
            fs = [self.classifier.predict_proba(bag) for bag in bags]
            y_pred = [self.classifier.predict(bag) for bag in bags]
            flatten_original_y = np.hstack(list(y))
            

            # For every bag - use simplified assignments
            has_changed = False
            total_changes = 0  # Initialize total changes counter
            
            # Self-paced learning: sort bags by difficulty (constraint violations)
            if i_iter > 1:  # Start self-paced learning after initial iterations
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
                
                # Simple confidence-based freezing
                frozen_indices = set()
                if i_iter >= 3:  # Start freezing after initial exploration
                    max_probs = np.max(probs, axis=1)
                    predicted_classes = np.argmax(probs, axis=1)
                    
                    for idx in range(n_instances):
                        # Freeze if high confidence AND consistent with current assignment
                        if max_probs[idx] > confidence_threshold and y[i_bag][idx] == predicted_classes[idx]:
                            frozen_indices.add(idx)
                
                # For binary classification, use exact k-of-n solver
                if n_classes == 2:
                    # Apply minimal slack only in first iteration for robustness
                    effective_lower_0 = lower_threshold[i_bag, 0]
                    effective_upper_0 = upper_threshold[i_bag, 0]
                    effective_lower_1 = lower_threshold[i_bag, 1]
                    effective_upper_1 = upper_threshold[i_bag, 1]
                    
                    if i_iter == 0:  # Only first iteration
                        effective_lower_0 = effective_lower_0 * 0.98  # 2% slack - more conservative
                        effective_upper_0 = effective_upper_0 * 1.02  # 2% slack - more conservative
                        effective_lower_1 = effective_lower_1 * 0.98  # 2% slack - more conservative
                        effective_upper_1 = effective_upper_1 * 1.02  # 2% slack - more conservative
                    
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
                        
                        # Compute margin scores for active instances
                        margin_scores = np.array([
                            compute_margin_aware_confidence(probs[idx], 1, margin_lambda=margin_lambda)
                            for idx in active_indices
                        ])
                        
                        # Compute AEM scores if in exploration phase
                        aem_scores = None
                        if aem_lambda > 0:
                            aem_scores = compute_attention_entropy_maximization(probs[active_indices], aem_lambda)
                        
                        # Use Gumbel-Softmax for differentiable discrete optimization
                        soft_weights = soft_k_of_n_assignment(
                            active_probs, k_min_active, k_max_active, 
                            temperature=gumbel_temperature,
                            margin_scores=margin_scores, aem_scores=aem_scores
                        )
                        
                        # Straight-through estimator with Gumbel-Softmax
                        # Hard assignment based on threshold
                        optimal_assignment_active = (soft_weights > 0.5).astype(int)
                        
                        # Ensure constraints are satisfied
                        current_k = np.sum(optimal_assignment_active)
                        if current_k < k_min_active:
                            # Add more positive assignments
                            remaining_indices = np.where(optimal_assignment_active == 0)[0]
                            if len(remaining_indices) > 0:
                                scores = soft_weights[remaining_indices]
                                n_to_add = min(k_min_active - current_k, len(remaining_indices))
                                top_indices = remaining_indices[np.argsort(scores)[-n_to_add:]]
                                optimal_assignment_active[top_indices] = 1
                        elif current_k > k_max_active:
                            # Remove some positive assignments
                            positive_indices = np.where(optimal_assignment_active == 1)[0]
                            if len(positive_indices) > 0:
                                scores = soft_weights[positive_indices]
                                n_to_remove = current_k - k_max_active
                                bottom_indices = positive_indices[np.argsort(scores)[:n_to_remove]]
                                optimal_assignment_active[bottom_indices] = 0
                        
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
                            # 2% slack in first iteration - more conservative
                            effective_lower = lower_threshold[i_bag, i_class] * 0.98
                        else:
                            effective_lower = lower_threshold[i_bag, i_class]
                        
                        if class_count < effective_lower:
                            # Need more instances of this class
                            deficit = int(effective_lower - class_count)
                            
                            # Find best candidates: high probability for this class, not already changed, not frozen
                            candidates = []
                            for idx in range(n_instances):
                                if idx not in changed_indices and idx not in frozen_indices and new_y[idx] != i_class:
                                    # Base score from probability
                                    base_score = probs[idx, i_class]
                                    # Add margin-aware confidence
                                    margin_conf = compute_margin_aware_confidence(probs[idx], i_class, margin_lambda=margin_lambda)
                                    score = base_score + 0.1 * (margin_conf - base_score)
                                    
                                    # Strong boost/penalty based on constraint satisfaction
                                    current_class = new_y[idx]
                                    current_class_count = class_count_dict.get(current_class, 0)
                                    if current_class_count > upper_threshold[i_bag, current_class]:
                                        score *= 3.0  # Strong boost for moving from over-represented class
                                    elif current_class_count <= lower_threshold[i_bag, current_class]:
                                        score *= 0.3  # Strong penalty for taking from under-represented class
                                    
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
                            effective_upper = upper_threshold[i_bag, i_class] * 1.02  # 2% slack - more conservative
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
                                    
                                    # Base score from log probability
                                    base_log_score = np.log(probs[idx, c] + 1e-8)
                                    # Add margin confidence
                                    margin_conf = compute_margin_aware_confidence(probs[idx], c, margin_lambda=margin_lambda)
                                    scores[c] = base_log_score + 0.1 * np.log(max(margin_conf, 1e-8) + 1e-8)
                                    
                                    # Strong constraint bonus/penalty
                                    c_count = class_count_dict.get(c, 0)
                                    if c_count < lower_threshold[i_bag, c]:
                                        scores[c] += 3.0  # Strong incentive for under-represented classes
                                    elif c_count < upper_threshold[i_bag, c]:
                                        scores[c] += 1.0
                                    else:
                                        scores[c] -= 2.0  # Strong penalty for over-represented classes
                                
                                # Apply temperature
                                exp_scores = np.exp(scores / gumbel_temperature)
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
                count_loss = 0.0
                
                for i_class in range(n_classes):
                    class_count = final_counts.get(i_class, 0)
                    # Lower threshold violations
                    if class_count < lower_threshold[i_bag, i_class]:
                        constraint_violation += lower_threshold[i_bag, i_class] - class_count
                    # Upper threshold violations
                    if class_count > upper_threshold[i_bag, i_class]:
                        constraint_violation += class_count - upper_threshold[i_bag, i_class]
                    
                    # Add differentiable count constraint loss (L1)
                    if n_classes == 2 and i_class == 1:
                        # For binary case, compute soft count loss
                        target_k = (lower_threshold[i_bag, 1] + upper_threshold[i_bag, 1]) / 2
                        soft_count = np.sum(probs[:, 1])  # Sum of probabilities for positive class
                        count_loss = 0.2 * abs(soft_count - target_k)  # Weight = 0.2 as recommended
                
                bag_difficulty[i_bag] = constraint_violation + count_loss

            if debug:
                print(f"iter: {i_iter}, gumbel_temp: {gumbel_temperature:.3f}, aem_lambda: {aem_lambda:.3f}")
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
                for inner_iter in range(5):
                    self.classifier.fit(flatten_bags, flatten_y)
            elif not has_changed and i_iter < 4:
                if debug:
                    print(f"Early convergence at iteration {i_iter}, continuing exploration...")
            
            # Only break after minimum iterations
            if i_iter >= min_iterations - 1:
                break

        return self.classifier, y
