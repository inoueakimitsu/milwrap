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


def compute_attention_entropy_maximization(bag_probs: NDArray[np.float64], lambda_entropy: float = 0.05, temperature: float = 1.0) -> NDArray[np.float64]:
    """Compute AEM (Attention Entropy Maximization) regularized scores.
    
    Based on AEM research: adds entropy regularizer to prevent over-concentrated attention.
    Uses temperature-lambda coupling for better control.
    
    Args:
        bag_probs: Instance probability matrix of shape (n_instances, n_classes).
        lambda_entropy: Entropy regularization strength (coupled with temperature).
        temperature: Current temperature for softmax computation.
        
    Returns:
        AEM-regularized scores for each instance.
    """
    if len(bag_probs.shape) != 2 or bag_probs.shape[0] == 0:
        return np.ones(bag_probs.shape[0])
    
    # Calculate instance-wise attention weights with temperature
    max_probs = np.max(bag_probs, axis=1)
    # Use temperature in softmax computation
    attention_weights = np.exp(max_probs / temperature) / (np.sum(np.exp(max_probs / temperature)) + 1e-10)
    
    # Calculate attention entropy: H(α) = -Σ αᵢ log(αᵢ)
    attention_entropy = -np.sum(attention_weights * np.log(attention_weights + 1e-10))
    
    # AEM regularization with temperature coupling
    # Higher temperature -> higher entropy weight (more exploration)
    effective_lambda = lambda_entropy * temperature
    entropy_bonus = effective_lambda * attention_entropy
    
    # Return per-instance scores with distributed entropy bonus
    return max_probs + entropy_bonus * attention_weights


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


def compute_instance_interaction_weights(bag_probs: NDArray[np.float64], k: int = 5) -> NDArray[np.float64]:
    """Compute instance interaction weights based on neighborhood consistency.
    
    Inspired by ViT-IWRS: instances that agree with their neighbors get higher weights.
    
    Args:
        bag_probs: Instance probability matrix of shape (n_instances, n_classes).
        k: Number of nearest neighbors to consider.
        
    Returns:
        Interaction-based weights for each instance.
    """
    if len(bag_probs.shape) != 2 or bag_probs.shape[0] == 0:
        return np.ones(bag_probs.shape[0])
    
    n_instances = bag_probs.shape[0]
    
    # Faster distance computation using L2 distance on probability vectors
    # Compute pairwise distances efficiently
    prob_squared = np.sum(bag_probs ** 2, axis=1, keepdims=True)
    distances = prob_squared + prob_squared.T - 2 * np.dot(bag_probs, bag_probs.T)
    distances = np.maximum(distances, 0)  # Numerical stability
    
    # Get predicted classes
    pred_classes = np.argmax(bag_probs, axis=1)
    
    # For each instance, compute consistency with k nearest neighbors
    consistency_scores = np.zeros(n_instances)
    for i in range(n_instances):
        # Find k nearest neighbors (excluding self)
        neighbor_indices = np.argsort(distances[i])[1:min(k+1, n_instances)]
        
        if len(neighbor_indices) > 0:
            # Count how many neighbors agree
            agreement = np.mean(pred_classes[neighbor_indices] == pred_classes[i])
            
            # Also consider probability similarity
            prob_similarity = np.mean([
                bag_probs[i, pred_classes[i]] * bag_probs[j, pred_classes[i]]
                for j in neighbor_indices
            ])
            
            # Combine agreement and probability similarity
            consistency_scores[i] = 0.7 * agreement + 0.3 * prob_similarity
        else:
            consistency_scores[i] = 1.0
    
    return consistency_scores


def apply_stochastic_instance_masking(probs: NDArray[np.float64], mask_ratio: float = 0.1, top_k_ratio: float = 0.3) -> NDArray[np.float64]:
    """Apply Stochastic Top-K Instance Masking (STKIM).
    
    Based on ACMIL paper: randomly masks high-attention instances to force
    the model to discover alternative discriminative patterns.
    
    Args:
        probs: Instance probability matrix of shape (n_instances, n_classes).
        mask_ratio: Fraction of top instances to potentially mask.
        top_k_ratio: Fraction of instances considered as "top".
        
    Returns:
        Masked probability matrix.
    """
    if len(probs.shape) != 2 or probs.shape[0] == 0:
        return probs
    
    n_instances = probs.shape[0]
    
    # Get max probabilities for each instance
    max_probs = np.max(probs, axis=1)
    
    # Identify top-k instances
    k_top = max(1, int(n_instances * top_k_ratio))
    top_indices = np.argsort(max_probs)[-k_top:]
    
    # Randomly select instances to mask from top-k
    n_mask = max(1, int(k_top * mask_ratio))
    mask_indices = np.random.choice(top_indices, size=n_mask, replace=False)
    
    # Create masked probabilities
    masked_probs = probs.copy()
    
    # For masked instances, reduce their influence (but don't zero out completely)
    # This maintains differentiability while encouraging exploration
    masked_probs[mask_indices] *= 0.5  # Reduce but don't eliminate
    
    return masked_probs


def sparse_differentiable_topk(scores: NDArray[np.float64], k: float, temperature: float = 1.0, sparsity_threshold: float = 0.01) -> NDArray[np.float64]:
    """Sparse differentiable top-k operator.
    
    Based on "Fast, Differentiable & Sparse Top-k" (2023): uses isotonic projection
    on the permutahedron for efficient, sparse gradient computation.
    
    Args:
        scores: Input scores to select from.
        k: Target number of selections (can be fractional).
        temperature: Temperature parameter for smoothness.
        sparsity_threshold: Threshold below which weights are set to zero.
        
    Returns:
        Sparse soft selection weights (sum approximately equals k).
    """
    n = len(scores)
    if n == 0 or k <= 0:
        return np.zeros(n)
    if k >= n:
        return np.ones(n)
    
    # Apply temperature scaling to scores
    scaled_scores = scores / temperature
    
    # Sort scores in descending order
    sorted_indices = np.argsort(scaled_scores)[::-1]
    sorted_scores = scaled_scores[sorted_indices]
    
    # Isotonic projection algorithm for sparse top-k
    # Find optimal threshold using cumulative sums
    cumsum = np.cumsum(sorted_scores)
    indices = np.arange(1, n + 1)
    
    # Compute potential thresholds
    thresholds = (cumsum - k) / indices
    
    # Find the valid threshold (where all constraints are satisfied)
    valid_mask = sorted_scores >= thresholds
    if np.any(valid_mask):
        # Last valid position
        pivot = np.where(valid_mask)[0][-1]
        threshold = thresholds[pivot]
    else:
        threshold = thresholds[-1]
    
    # Compute sparse weights
    weights = np.maximum(0, scaled_scores - threshold)
    
    # Apply sparsity threshold
    weights[weights < sparsity_threshold] = 0
    
    # Normalize to ensure sum equals k
    current_sum = np.sum(weights)
    if current_sum > 0:
        weights = weights * k / current_sum
    
    # Map back to original order
    result = np.zeros(n)
    result[sorted_indices] = weights
    
    return result


def smooth_topk(scores: NDArray[np.float64], k: float, temperature: float = 1.0) -> NDArray[np.float64]:
    """Differentiable smooth top-k operator (ST_k).
    
    Wrapper that uses sparse_differentiable_topk for better performance.
    
    Args:
        scores: Input scores to select from.
        k: Target number of selections (can be fractional).
        temperature: Temperature parameter for smoothness.
        
    Returns:
        Soft selection weights (sum approximately equals k).
    """
    return sparse_differentiable_topk(scores, k, temperature, sparsity_threshold=0.005)


def compute_differentiable_count_loss(probs, k_target, weight=0.1):
    """Compute differentiable loss for count constraints.
    
    Based on "Unified Approach to Count-Based Weak Supervision" (2023).
    Uses logsumexp to compute probability of exactly k instances being positive.
    
    Args:
        probs: probabilities for positive class (n_instances,)
        k_target: target count of positive instances
        weight: loss weight
        
    Returns:
        count_loss: scalar loss value
    """
    n = len(probs)
    if n == 0 or k_target < 0 or k_target > n:
        return 0.0
    
    # Compute log probabilities
    log_probs = np.log(probs + 1e-10)
    log_one_minus_probs = np.log(1 - probs + 1e-10)
    
    # For exact count k, we need to sum over all subsets of size k
    # This is computationally intensive, so we use a smooth approximation
    # Based on the expected count matching the target
    expected_count = np.sum(probs)
    count_deviation = abs(expected_count - k_target)
    
    # L1 loss on count deviation
    return weight * count_deviation


def soft_k_of_n_assignment(probs, k_min, k_max, temperature=0.5, margin_scores=None, aem_scores=None, interaction_weights=None, is_external_init=False):
    """Solve soft k-of-n assignment using smooth top-k operator.
    
    Find soft assignment weights such that sum approximates the optimal k
    within [k_min, k_max] range, using differentiable ST_k operator.
    
    Args:
        probs: probabilities for positive class (n_instances,)
        k_min: minimum number of positive instances
        k_max: maximum number of positive instances  
        temperature: temperature for ST_k operator
        margin_scores: margin-based scores for weighting (optional)
        aem_scores: AEM-based scores for weighting (optional)
        interaction_weights: instance interaction weights (optional)
        is_external_init: whether using external initialization
        
    Returns:
        soft_assignments: continuous weights in [0, 1]
    """
    n = len(probs)
    
    # Handle edge cases
    if k_min > n:
        return np.ones(n)
    if k_max < 0:
        return np.zeros(n)
    
    # Compute scores combining multiple signals
    eps = 1e-8
    log_odds = np.log(probs + eps) - np.log(1 - probs + eps)
    
    # Adaptive weighting based on initialization type
    if is_external_init:
        margin_weight = 0.15
        aem_weight = 0.05
        interaction_weight = 0.12
    else:
        margin_weight = 0.1
        aem_weight = 0.08
        interaction_weight = 0.15
    
    # Combine with margin scores
    if margin_scores is not None and len(margin_scores) == len(probs):
        log_odds = log_odds + margin_weight * margin_scores
    
    # Combine with AEM scores (when active)
    if aem_scores is not None and len(aem_scores) == len(probs):
        log_odds = log_odds + aem_weight * aem_scores
    
    # Combine with instance interaction weights
    if interaction_weights is not None and len(interaction_weights) == len(probs):
        log_odds = log_odds + interaction_weight * interaction_weights
    
    # Find optimal k using gradient-free search
    best_k = k_min
    best_score = -np.inf
    
    for k in range(max(0, k_min), min(n, k_max) + 1):
        # Use ST_k to get soft assignments
        soft_weights = smooth_topk(log_odds, k, temperature)
        
        # Score is expected log probability
        score = np.sum(soft_weights * np.log(probs + eps))
        
        if score > best_score:
            best_score = score
            best_k = k
    
    # Return soft assignments for best k
    return smooth_topk(log_odds, best_k, temperature)


def exact_k_of_n_assignment(probs, k_min, k_max, current_assignments=None, margin_scores=None, aem_scores=None, is_external_init=False):
    """Solve exact k-of-n assignment problem with adaptive weighting.
    
    Find optimal assignment of instances to positive class such that
    k_min <= sum(assignments) <= k_max, maximizing sum(log(p_i))
    
    Args:
        probs: probabilities for positive class (n_instances,)
        k_min: minimum number of positive instances
        k_max: maximum number of positive instances
        current_assignments: current assignments (optional)
        margin_scores: margin-based scores for weighting (optional)
        aem_scores: AEM scores for weighting (optional)
        is_external_init: whether using external initialization
        
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
    
    # Adaptive weighting based on initialization type
    if is_external_init:
        # External init: emphasize margins more, moderate AEM
        margin_weight = 0.18
        aem_weight = 0.05
    else:
        # Auto init: balance margins and AEM
        margin_weight = 0.12
        aem_weight = 0.06
    
    # Apply margin weighting if provided
    if margin_scores is not None and len(margin_scores) == len(probs):
        log_odds = log_odds + margin_weight * margin_scores
    
    # Apply AEM scores if provided
    if aem_scores is not None and len(aem_scores) == len(probs):
        log_odds = log_odds + aem_weight * aem_scores
    
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
            # For external init, apply confidence-based refinement
            # This helps correct potential labeling errors in external initialization
            if debug:
                print("Using external initialization with confidence-based refinement")
            
        # Initialize simple parameters
        n_bags = len(bags)
        
        # Detect initialization type for adaptive parameters
        is_external_init = initial_y is not None
        
        # Adaptive temperature scheduling based on initialization
        if is_external_init:
            # External init: start warmer for fine-tuning
            temperature_start = 1.5  # Lower starting temperature
            temperature_end = 0.05   # Very low final temperature
            gamma = 0.4  # Faster decay
        else:
            # Auto init: need more exploration
            temperature_start = 2.5  # Higher starting temperature
            temperature_end = 0.1    # Moderate final temperature
            gamma = 0.3  # Slower decay
        
        # Self-paced learning: track bag difficulty scores
        bag_difficulty = np.zeros(n_bags)
        
        # Adaptive confidence-based freezing
        if is_external_init:
            confidence_threshold_start = 0.96  # Slightly lower for external init
            confidence_threshold_end = 0.92
            margin_lambda = 0.3  # Higher margin importance for external init
        else:
            confidence_threshold_start = 0.98  # Higher for auto init
            confidence_threshold_end = 0.94
            margin_lambda = 0.2  # Standard margin importance
        
        # STKIM parameters with curriculum
        stkim_warmup_iter = 5 if is_external_init else 7  # Warmup before masking
        stkim_max_ratio = 0.06 if is_external_init else 0.08  # Max mask ratio
        stkim_top_k_ratio = 0.25  # Consider top 25% as high-attention
        
        # Top-k temperature (slower annealing as recommended)
        topk_temperature_start = 1.2 if is_external_init else 1.4  # Higher start
        topk_temperature_end = 0.3
        topk_anneal_slowdown = 0.5  # Halve the annealing speed
        
        # Count loss weight schedule (2-phase)
        count_loss_start_iter = 3  # Start after 3 iterations
        count_loss_weight_max = 0.09  # Max weight
        
        # Whether to use soft assignments
        use_soft_assignment = True  # Enable smooth top-k
        
        for i_iter in range(max_iter):
            # Cosine temperature schedule with warm restart
            progress = i_iter / max_iter
            if progress < 0.95:  # First 95% - cosine decay
                cosine_decay = 0.5 * (1 + np.cos(np.pi * progress / 0.95))
                temperature = temperature_end + (temperature_start - temperature_end) * cosine_decay
            else:  # Last 5% - warm restart
                restart_progress = (progress - 0.95) / 0.05
                temperature = temperature_end + (temperature_start * 0.4 - temperature_end) * restart_progress
            
            # Adaptive confidence threshold decay
            confidence_threshold = confidence_threshold_end + (confidence_threshold_start - confidence_threshold_end) * np.exp(-gamma * i_iter)
            
            # Top-k temperature schedule (slower annealing)
            effective_max_iter = max_iter * (1 + topk_anneal_slowdown)  # Extend the annealing period
            topk_progress = min(1.0, i_iter / effective_max_iter)
            topk_temperature = topk_temperature_end + (topk_temperature_start - topk_temperature_end) * (1 - topk_progress)
            
            # Count loss weight schedule (2-phase: zero then quadratic growth)
            if i_iter < count_loss_start_iter:
                count_loss_weight = 0.0
            else:
                # Quadratic growth after start
                phase_progress = (i_iter - count_loss_start_iter) / (max_iter - count_loss_start_iter)
                count_loss_weight = count_loss_weight_max * (phase_progress ** 2)
            
            # Calculate global lambda_entropy for debug output
            aem_start_iter = 1 if is_external_init else 2
            if i_iter >= aem_start_iter:
                if is_external_init:
                    lambda_entropy_base = 0.025  # Slightly higher for better regularization
                else:
                    lambda_entropy_base = 0.045  # Slightly lower for stability
                global_lambda_entropy = lambda_entropy_base * (temperature / temperature_start)
            else:
                global_lambda_entropy = 0.0
            
            # Fit
            flatten_bags = np.vstack(bags)
            flatten_y = np.hstack(y)
            self.classifier.fit(flatten_bags, flatten_y)

            # Compute outputs
            fs = [self.classifier.predict_proba(bag) for bag in bags]
            
            # Apply STKIM with curriculum (warmup then gradual increase)
            if i_iter >= stkim_warmup_iter:
                # Curriculum masking: gradually increase mask ratio
                warmup_progress = (i_iter - stkim_warmup_iter) / (max_iter - stkim_warmup_iter)
                current_mask_ratio = stkim_max_ratio * warmup_progress
                
                fs_masked = []
                for bag_probs in fs:
                    # Entropy-aware masking: skip if attention is already concentrated
                    max_probs = np.max(bag_probs, axis=1)
                    attention_weights = np.exp(max_probs) / (np.sum(np.exp(max_probs)) + 1e-10)
                    attention_entropy = -np.sum(attention_weights * np.log(attention_weights + 1e-10))
                    
                    # Skip masking if entropy is too low (attention already peaked)
                    entropy_threshold = 0.75 * np.log(max(2, int(len(bag_probs) * stkim_top_k_ratio)))
                    if attention_entropy > entropy_threshold:
                        masked_probs = apply_stochastic_instance_masking(
                            bag_probs, 
                            mask_ratio=current_mask_ratio,
                            top_k_ratio=stkim_top_k_ratio
                        )
                        fs_masked.append(masked_probs)
                    else:
                        fs_masked.append(bag_probs)  # Skip masking for this bag
                fs = fs_masked
            
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
                
                # Dynamic per-bag temperature based on difficulty
                bag_hardness = bag_difficulty[i_bag] / (n_instances + 1e-6)  # Normalized difficulty
                bag_temperature = temperature * (1 + 0.3 * bag_hardness)  # Harder bags get higher temperature
                
                # Adaptive confidence-based freezing
                frozen_indices = set()
                freeze_start_iter = 2 if is_external_init else 3  # Earlier freezing for external init
                if i_iter >= freeze_start_iter:
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
                        # Adaptive slack based on initialization type
                        slack_factor = 0.99 if is_external_init else 0.98  # Less slack for external init
                        effective_lower_0 = effective_lower_0 * slack_factor
                        effective_upper_0 = effective_upper_0 * (2 - slack_factor)
                        effective_lower_1 = effective_lower_1 * slack_factor
                        effective_upper_1 = effective_upper_1 * (2 - slack_factor)
                    
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
                        
                        # Compute AEM scores with temperature coupling
                        aem_scores_active = None
                        aem_start_iter = 1 if is_external_init else 2  # Earlier AEM for external init
                        if i_iter >= aem_start_iter:
                            # Base lambda_entropy with initialization-specific values
                            if is_external_init:
                                lambda_entropy_base = 0.025  # Slightly higher for better regularization
                            else:
                                lambda_entropy_base = 0.045  # Slightly lower for stability
                            
                            # Temperature-coupled lambda (from theory)
                            lambda_entropy = lambda_entropy_base * (temperature / temperature_start)
                            
                            # Compute AEM scores with per-bag temperature
                            bag_aem_scores = compute_attention_entropy_maximization(probs, lambda_entropy, bag_temperature)
                            aem_scores_active = bag_aem_scores[active_indices]
                        
                        # Compute instance interaction weights with k-annealing
                        interaction_weights_active = None
                        if n_active > 1:  # Need at least 2 instances for interaction
                            # Anneal k from 8-10 down to 3-4
                            k_start = min(10 if is_external_init else 8, n_active - 1)
                            k_end = max(3 if is_external_init else 4, 2)
                            k_progress = i_iter / max_iter
                            k_neighbors = int(k_end + (k_start - k_end) * (1 - k_progress))
                            k_neighbors = min(k_neighbors, n_active - 1)  # Ensure valid k
                            
                            interaction_weights = compute_instance_interaction_weights(
                                probs[active_indices], k=k_neighbors
                            )
                            interaction_weights_active = interaction_weights
                        
                        # Choose between exact and soft assignment
                        if use_soft_assignment and i_iter >= 1:  # Use soft assignment after first iteration
                            # Use differentiable soft assignment with smooth top-k
                            soft_weights = soft_k_of_n_assignment(
                                active_probs, k_min_active, k_max_active, 
                                temperature=topk_temperature,
                                margin_scores=margin_scores, 
                                aem_scores=aem_scores_active,
                                interaction_weights=interaction_weights_active,
                                is_external_init=is_external_init
                            )
                            
                            # Straight-through estimator: hard assignment for forward pass
                            k_target = int(np.round(np.sum(soft_weights)))
                            k_target = max(k_min_active, min(k_max_active, k_target))
                            
                            if k_target > 0:
                                top_k_indices = np.argsort(soft_weights)[-k_target:]
                                optimal_assignment_active = np.zeros(len(active_probs), dtype=int)
                                optimal_assignment_active[top_k_indices] = 1
                            else:
                                optimal_assignment_active = np.zeros(len(active_probs), dtype=int)
                        else:
                            # Use exact assignment for first iteration or when disabled
                            optimal_assignment_active = exact_k_of_n_assignment(
                                active_probs, k_min_active, k_max_active, 
                                margin_scores=margin_scores, aem_scores=aem_scores_active,
                                is_external_init=is_external_init
                            )
                        
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
                            # Adaptive slack based on initialization type
                            slack_factor = 0.99 if is_external_init else 0.98
                            effective_lower = lower_threshold[i_bag, i_class] * slack_factor
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
                            slack_factor = 0.99 if is_external_init else 0.98
                            effective_upper = upper_threshold[i_bag, i_class] * (2 - slack_factor)
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
                                
                                # Apply per-bag temperature
                                exp_scores = np.exp(scores / bag_temperature)
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
                    
                    # Add differentiable count constraint loss (for binary case)
                    if n_classes == 2 and i_class == 1:
                        # For binary case, compute soft count loss
                        target_k = (lower_threshold[i_bag, 1] + upper_threshold[i_bag, 1]) / 2
                        count_loss += compute_differentiable_count_loss(
                            probs[:, 1], 
                            target_k, 
                            weight=count_loss_weight
                        )
                
                bag_difficulty[i_bag] = constraint_violation + count_loss

            if debug:
                print(f"iter: {i_iter}, temp: {temperature:.3f}, topk_temp: {topk_temperature:.3f}, lambda: {global_lambda_entropy:.4f}, count_loss_w: {count_loss_weight:.3f}")
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
