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


def compute_attention_entropy_maximization(bag_probs: NDArray[np.float64], lambda_entropy: float = 0.08, normalize_by_size: bool = True) -> NDArray[np.float64]:
    """Compute AEM (Attention Entropy Maximization) regularized scores.
    
    Enhanced version with probability-space processing for better initialization robustness.
    
    Args:
        bag_probs: Instance probability matrix of shape (n_instances, n_classes).
        lambda_entropy: Entropy regularization strength (default 0.08).
        normalize_by_size: Whether to normalize entropy by bag size.
        
    Returns:
        AEM-regularized scores for each instance.
    """
    if len(bag_probs.shape) != 2 or bag_probs.shape[0] == 0:
        return np.ones(bag_probs.shape[0])
    
    n_instances = bag_probs.shape[0]
    n_classes = bag_probs.shape[1]
    eps = 1e-10
    
    # Compute instance-level uncertainty (entropy across classes)
    instance_entropy = -np.sum(bag_probs * np.log(bag_probs + eps), axis=1)
    max_entropy_per_instance = np.log(n_classes)
    normalized_uncertainty = instance_entropy / (max_entropy_per_instance + eps)
    
    # Transform to probability space for stable attention computation
    max_probs = np.max(bag_probs, axis=1)
    # Clip to avoid numerical issues
    max_probs_clipped = np.clip(max_probs, eps, 1 - eps)
    
    # Temperature-scaled attention weights with uncertainty awareness
    # Higher uncertainty -> more uniform attention (prevents overconfidence)
    temperature = 1.0 + 0.5 * np.mean(normalized_uncertainty)
    attention_logits = np.log(max_probs_clipped) / temperature
    
    # Stable softmax computation
    attention_logits_shifted = attention_logits - np.max(attention_logits)
    attention_weights = np.exp(attention_logits_shifted) / (np.sum(np.exp(attention_logits_shifted)) + eps)
    
    # Calculate attention entropy: H(α) = -Σ αᵢ log(αᵢ)
    attention_entropy = -np.sum(attention_weights * np.log(attention_weights + eps))
    
    # Normalize by bag size to make entropy scale-invariant
    if normalize_by_size and n_instances > 1:
        max_entropy = np.log(n_instances)
        attention_entropy = attention_entropy / max_entropy
    
    # Enhanced AEM regularization with instance-specific modulation
    # Instances with high uncertainty get stronger entropy bonus
    instance_specific_bonus = lambda_entropy * attention_entropy * (1 + 0.3 * normalized_uncertainty)
    
    # Combine base scores with entropy bonus
    # Use log-space for numerical stability
    log_max_probs = np.log(max_probs_clipped)
    aem_scores = log_max_probs + instance_specific_bonus * attention_weights
    
    # Add diversity term to prevent mode collapse
    diversity_bonus = 0.02 * np.std(aem_scores)
    aem_scores = aem_scores + diversity_bonus * (aem_scores - np.mean(aem_scores))
    
    return aem_scores


def gumbel_sample(shape: tuple, eps: float = 1e-6) -> NDArray[np.float64]:
    """Sample from Gumbel(0, 1) distribution.
    
    Args:
        shape: Shape of the output tensor
        eps: Small value for numerical stability
        
    Returns:
        Gumbel noise samples
    """
    u = np.random.uniform(eps, 1 - eps, shape)
    return -np.log(-np.log(u))


def gumbel_softmax_topk(scores: NDArray[np.float64], k: float, temperature: float = 1.0, 
                       hard: bool = False, backward_temperature: float | None = None) -> NDArray[np.float64]:
    """Gumbel-Softmax based differentiable top-k selection.
    
    Combines Gumbel-Softmax with top-k constraint for robust discrete optimization.
    
    Args:
        scores: Input scores to select from (log-odds)
        k: Target number of selections (can be fractional)
        temperature: Temperature parameter for forward pass
        hard: Whether to use straight-through estimator
        backward_temperature: Temperature for backward pass (decoupled ST-GS)
        
    Returns:
        Soft selection weights (sum approximately equals k)
    """
    n = len(scores)
    if n == 0 or k <= 0:
        return np.zeros(n)
    if k >= n:
        return np.ones(n)
    
    # Add Gumbel noise for stochastic selection
    gumbel_noise = gumbel_sample(scores.shape)
    perturbed_scores = scores + gumbel_noise
    
    # Set backward temperature (higher for better gradient flow)
    if backward_temperature is None:
        backward_temperature = temperature * 2.0  # Empirically better for MIL
    
    # Sort perturbed scores for threshold computation
    sorted_indices = np.argsort(perturbed_scores)[::-1]
    sorted_scores = perturbed_scores[sorted_indices]
    
    # Dynamic threshold with fractional k support
    k_int = int(k)
    k_frac = k - k_int
    
    if k_int < n - 1:
        threshold = sorted_scores[k_int] * (1 - k_frac)
        if k_int + 1 < n:
            threshold += sorted_scores[k_int + 1] * k_frac
    else:
        threshold = sorted_scores[k_int]
    
    # Gumbel-Softmax transformation
    z_soft = 1 / (1 + np.exp(-(perturbed_scores - threshold) / temperature))
    
    if hard:
        # Straight-through estimator: hard selection forward, soft gradients backward
        # Get top-k indices based on soft weights
        top_k_indices = np.argsort(z_soft)[-int(np.round(k)):] 
        z_hard = np.zeros_like(z_soft)
        z_hard[top_k_indices] = 1.0
        
        # Normalize hard selections to sum to k
        z_hard = z_hard * k / np.sum(z_hard)
        return z_hard
    else:
        # Normalize soft scores to sum to k
        current_sum = np.sum(z_soft)
        if current_sum > 0:
            z_soft = z_soft * k / current_sum
        return z_soft


def smooth_topk(scores: NDArray[np.float64], k: float, temperature: float = 1.0) -> NDArray[np.float64]:
    """Legacy wrapper for backward compatibility."""
    return gumbel_softmax_topk(scores, k, temperature, hard=False)


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


def soft_k_of_n_assignment(probs, k_min, k_max, temperature=0.5, margin_scores=None, aem_scores=None, constraint_weight=1.0):
    """Solve soft k-of-n assignment using Gumbel-Softmax based selection.
    
    Find soft assignment weights using probabilistic relaxation for better
    gradient flow and initialization robustness.
    
    Args:
        probs: probabilities for positive class (n_instances,)
        k_min: minimum number of positive instances
        k_max: maximum number of positive instances  
        temperature: temperature for Gumbel-Softmax
        margin_scores: margin-based scores for weighting (optional)
        aem_scores: AEM-based scores for weighting (optional)
        constraint_weight: weight for constraint enforcement
        
    Returns:
        soft_assignments: continuous weights in [0, 1]
    """
    n = len(probs)
    
    # Handle edge cases
    if k_min > n:
        return np.ones(n)
    if k_max < 0:
        return np.zeros(n)
    
    # Transform to probability space for better stability
    eps = 1e-8
    # Clip probabilities to avoid extreme log-odds
    probs_clipped = np.clip(probs, eps, 1 - eps)
    log_odds = np.log(probs_clipped / (1 - probs_clipped))
    
    # Combine with margin scores in probability space
    if margin_scores is not None and len(margin_scores) == len(probs):
        # Scale margin scores based on probability magnitudes
        prob_scale = np.mean(np.abs(log_odds))
        log_odds = log_odds + 0.12 * margin_scores / (1 + prob_scale)
    
    # Enhanced AEM integration with probability-aware weighting
    if aem_scores is not None and len(aem_scores) == len(probs):
        # Adaptive AEM weight based on exploration/exploitation trade-off
        exploration_factor = np.exp(-constraint_weight / 2.0)  # Exponential decay
        # Stronger AEM influence for better initialization robustness
        aem_weight = 0.25 * exploration_factor
        
        # Scale AEM scores by instance uncertainty (entropy)
        instance_entropy = -probs * np.log(probs + eps) - (1 - probs) * np.log(1 - probs + eps)
        # Increased uncertainty weighting for better exploration
        uncertainty_weight = 1 + 0.8 * instance_entropy
        
        # Additional diversity term based on probability variance
        prob_variance = np.var(probs)
        diversity_factor = 1 + 0.3 * np.sqrt(prob_variance)
        
        log_odds = log_odds + aem_weight * aem_scores * uncertainty_weight * diversity_factor
    
    # Find optimal k using probabilistic search
    best_k = k_min
    best_score = -np.inf
    
    # Use decoupled temperature for better gradient flow
    forward_temp = temperature
    backward_temp = temperature * 1.5  # Higher for gradients
    
    for k in range(max(0, k_min), min(n, k_max) + 1):
        # Use Gumbel-Softmax for probabilistic selection
        soft_weights = gumbel_softmax_topk(log_odds, k, forward_temp, 
                                          hard=False, backward_temperature=backward_temp)
        
        # Score in probability space with entropy regularization
        score = np.sum(soft_weights * np.log(probs_clipped))
        
        # Add selection entropy bonus for diversity (prevents mode collapse)
        selection_entropy = -np.sum(soft_weights * np.log(soft_weights + eps))
        entropy_bonus = 0.05 * selection_entropy / np.log(n + 1)  # Normalized by max entropy
        score += entropy_bonus
        
        if score > best_score:
            best_score = score
            best_k = k
    
    # Final assignment with adaptive temperature and hard selection option
    # Use hard selection in late stages for sharper decisions
    use_hard = constraint_weight > 1.5  # Hard selection in exploitation phase
    
    # Adaptive temperature based on constraint satisfaction
    final_temp = forward_temp * max(0.3, 1.0 - 0.4 * min(1.0, constraint_weight / 2.5))
    
    return gumbel_softmax_topk(log_odds, best_k, final_temp, 
                              hard=use_hard, backward_temperature=backward_temp)


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
            
        # Initialize simple parameters
        n_bags = len(bags)
        
        # Structured initialization detection for external-init robustness
        is_external_init = initial_y is not None
        
        # Enhanced temperature schedule for Gumbel-Softmax
        # Higher initial temperature for better exploration with stochastic selection
        temperature_start = 6.0   # Increased for Gumbel noise exploration
        temperature_end = 0.25    # Lower end for sharper decisions
        temperature_decay = 0.92  # Slower decay for smoother transition
        
        # Adaptive AEM-temperature coupling with initialization awareness
        # External init needs different coupling due to potentially biased start
        aem_coupling_factor = 0.4 if is_external_init else 0.35
        
        # Constraint penalty schedule with initialization-aware start
        # External init gets gentler constraint enforcement initially
        constraint_penalty_start = 0.05 if is_external_init else 0.1
        constraint_penalty_end = 2.5  # Higher end for stronger convergence
        
        # Self-paced learning: track bag difficulty scores
        bag_difficulty = np.zeros(n_bags)
        
        # Adaptive confidence-based freezing with probability-aware thresholds
        confidence_threshold_start = 0.995  # Even higher for Gumbel noise
        confidence_threshold_end = 0.97     # Slightly higher to account for stochasticity
        margin_lambda = 0.4  # Further increased for probability-space robustness
        
        # Set warmup epochs based on initialization type
        warmup_epochs = 4 if is_external_init else 2  # Extended warmup for both cases
        
        # Gumbel-specific parameters
        gumbel_noise_scale = 1.0  # Can be adapted over iterations
        use_decoupled_temperature = True  # Better gradient flow
        
        for i_iter in range(max_iter):
            # Exponential temperature decay: T(t) = max(Tmin, T0 * α^t)
            temperature = max(temperature_end, 
                            temperature_start * (temperature_decay ** i_iter))
            
            # Adaptive AEM weight with non-linear coupling for Gumbel-Softmax
            # Use sqrt coupling for smoother transition
            aem_lambda = aem_coupling_factor * np.sqrt(temperature)
            
            # Constraint penalty increases over time: μ(t)
            progress = i_iter / max(1, max_iter - 1)
            constraint_penalty = constraint_penalty_start + \
                               (constraint_penalty_end - constraint_penalty_start) * progress
            
            # Adaptive confidence threshold for freezing
            confidence_threshold = confidence_threshold_start - \
                                 (confidence_threshold_start - confidence_threshold_end) * progress
            
            # Skip constraint enforcement during warmup for external-init
            enforce_constraints = (i_iter >= warmup_epochs) or not is_external_init
            
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
                
                # Adaptive confidence-based freezing with warmup consideration
                frozen_indices = set()
                if i_iter >= warmup_epochs:  # Respect warmup period
                    max_probs = np.max(probs, axis=1)
                    predicted_classes = np.argmax(probs, axis=1)
                    
                    for idx in range(n_instances):
                        # Consider margin in addition to max probability
                        margin = compute_margin_aware_confidence(probs[idx], predicted_classes[idx], margin_lambda)
                        
                        # Freeze if high confidence AND consistent assignment AND good margin
                        if (max_probs[idx] > confidence_threshold and 
                            y[i_bag][idx] == predicted_classes[idx] and
                            margin > confidence_threshold * 0.9):
                            frozen_indices.add(idx)
                
                # For binary classification, use exact k-of-n solver
                if n_classes == 2:
                    # Apply constraint relaxation during warmup or high temperature
                    effective_lower_0 = lower_threshold[i_bag, 0]
                    effective_upper_0 = upper_threshold[i_bag, 0]
                    effective_lower_1 = lower_threshold[i_bag, 1]
                    effective_upper_1 = upper_threshold[i_bag, 1]
                    
                    if not enforce_constraints or temperature > 2.0:
                        # Softer constraints during exploration with smoother transition
                        temp_ratio = min(1.0, temperature / temperature_start)
                        # Use quadratic easing for smoother constraint relaxation
                        slack_factor = 1 + 0.02 * (temp_ratio ** 2)
                        effective_lower_0 = effective_lower_0 * (2 - slack_factor)
                        effective_upper_0 = effective_upper_0 * slack_factor
                        effective_lower_1 = effective_lower_1 * (2 - slack_factor)
                        effective_upper_1 = effective_upper_1 * slack_factor
                    
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
                        
                        # Always compute AEM scores with adaptive weight
                        # Enhanced AEM for external init robustness
                        aem_lambda_adjusted = aem_lambda * (1.2 if is_external_init else 1.0)
                        aem_scores = compute_attention_entropy_maximization(
                            probs[active_indices], 
                            lambda_entropy=aem_lambda_adjusted,
                            normalize_by_size=True
                        )
                        
                        # Always use soft assignment with ST_k for differentiability
                        # Adjust temperature for external init (higher for more exploration)
                        effective_temperature = temperature * (1.15 if is_external_init and i_iter < warmup_epochs else 1.0)
                        soft_weights = soft_k_of_n_assignment(
                            active_probs, k_min_active, k_max_active, 
                            temperature=effective_temperature,
                            margin_scores=margin_scores, aem_scores=aem_scores,
                            constraint_weight=constraint_penalty
                        )
                        
                        # Straight-through estimator: hard assignment for forward, soft for backward
                        # Get top-k indices based on soft weights
                        k_target = int(np.round(np.sum(soft_weights)))
                        k_target = max(k_min_active, min(k_max_active, k_target))
                        
                        if k_target > 0:
                            top_k_indices = np.argsort(soft_weights)[-k_target:]
                            optimal_assignment_active = np.zeros_like(soft_weights, dtype=int)
                            optimal_assignment_active[top_k_indices] = 1
                        else:
                            optimal_assignment_active = np.zeros(len(active_probs), dtype=int)
                        
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
                        
                        # Apply adaptive constraint relaxation
                        if not enforce_constraints:
                            # Softer constraints during warmup
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
                        
                        # Apply adaptive constraint relaxation for upper threshold
                        if not enforce_constraints:
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
                count_loss = 0.0
                
                for i_class in range(n_classes):
                    class_count = final_counts.get(i_class, 0)
                    # Lower threshold violations
                    if class_count < lower_threshold[i_bag, i_class]:
                        constraint_violation += lower_threshold[i_bag, i_class] - class_count
                    # Upper threshold violations
                    if class_count > upper_threshold[i_bag, i_class]:
                        constraint_violation += class_count - upper_threshold[i_bag, i_class]
                    
                    # Add differentiable count constraint loss with adaptive weight
                    if n_classes == 2 and i_class == 1:
                        # For binary case, compute soft count loss
                        target_k = (lower_threshold[i_bag, 1] + upper_threshold[i_bag, 1]) / 2
                        soft_count = np.sum(probs[:, 1])  # Sum of probabilities for positive class
                        # Use constraint penalty weight from schedule
                        count_loss = constraint_penalty * 0.15 * abs(soft_count - target_k)
                
                bag_difficulty[i_bag] = constraint_violation + count_loss

            if debug:
                print(f"iter: {i_iter}, temp: {temperature:.3f}, aem_λ: {aem_lambda:.3f}, constraint_μ: {constraint_penalty:.3f}")
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
                # Adaptive inner iterations based on convergence stage
                if is_external_init:
                    # More aggressive training for external init
                    inner_iterations = min(10, 6 + (i_iter - 4))
                else:
                    inner_iterations = min(7, 4 + (i_iter - 4) // 2)
                    
                for inner_iter in range(inner_iterations):
                    self.classifier.fit(flatten_bags, flatten_y)
            elif not has_changed and i_iter < 4:
                if debug:
                    print(f"Early convergence at iteration {i_iter}, continuing exploration...")
            
            # Only break after minimum iterations
            if i_iter >= min_iterations - 1:
                break

        return self.classifier, y
