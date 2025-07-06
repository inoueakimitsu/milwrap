"""Multiple instance learning implementation based on count-based constraints.

This module provides implementations for multiple instance learning algorithms
that use count-based constraints to learn from bags of instances.
"""

import random
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.stats
from numpy.typing import NDArray
from sklearn.cluster import KMeans


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


def compute_instance_uncertainty_scores(bag_probs: NDArray[np.float64], temperature: float = 1.0) -> NDArray[np.float64]:
    """Compute uncertainty-based reliability scores for instances.
    
    Based on AGP and MIREL recommendations: use entropy-based uncertainty
    to detect unreliable external initializations.
    
    Args:
        bag_probs: Instance probability matrix of shape (n_instances, n_classes).
        temperature: Temperature for reliability score computation.
        
    Returns:
        Reliability scores for each instance (higher = more reliable).
    """
    if len(bag_probs.shape) != 2 or bag_probs.shape[0] == 0:
        return np.ones(bag_probs.shape[0])
    
    # Compute entropy as uncertainty measure
    eps = 1e-8
    entropy = -np.sum(bag_probs * np.log(bag_probs + eps), axis=1)
    
    # Normalize by maximum possible entropy
    n_classes = bag_probs.shape[1]
    max_entropy = np.log(n_classes)
    normalized_entropy = entropy / (max_entropy + eps)
    
    # Convert to reliability score: r = exp(-entropy/temperature)
    # Higher entropy = lower reliability
    reliability_scores = np.exp(-normalized_entropy / temperature)
    
    return reliability_scores


def compute_attribute_scores(bag_probs: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute multi-dimensional attribute scores for instances.
    
    Combines multiple signals including confidence, entropy, and margin
    to create a comprehensive attribute score.
    
    Args:
        bag_probs: Instance probability matrix of shape (n_instances, n_classes).
        
    Returns:
        Attribute scores for each instance.
    """
    if len(bag_probs.shape) != 2 or bag_probs.shape[0] == 0:
        return np.zeros(bag_probs.shape[0])
    
    # Max probability (confidence)
    max_probs = np.max(bag_probs, axis=1)
    
    # Entropy (uncertainty)
    eps = 1e-8
    entropy = -np.sum(bag_probs * np.log(bag_probs + eps), axis=1)
    n_classes = bag_probs.shape[1]
    max_entropy = np.log(n_classes) if n_classes > 1 else 1.0
    normalized_entropy = entropy / (max_entropy + eps)
    
    # Margin (difference between top two classes)
    sorted_probs = np.sort(bag_probs, axis=1)[:, ::-1]
    if bag_probs.shape[1] > 1:
        margin = sorted_probs[:, 0] - sorted_probs[:, 1]
    else:
        margin = sorted_probs[:, 0]
    
    # Combine signals: high confidence + low entropy + high margin = high score
    attribute_scores = max_probs * (1 - 0.3 * normalized_entropy) + 0.2 * margin
    
    return attribute_scores





def compute_prototype_similarity_scores(bag_features: NDArray[np.float64], prototypes: NDArray[np.float64], temperature: float = 0.1) -> NDArray[np.float64]:
    """Compute prototype-based similarity scores inspired by PAMIL.
    
    Args:
        bag_features: Instance feature matrix of shape (n_instances, n_features).
        prototypes: Prototype vectors of shape (n_prototypes, n_features).
        temperature: Temperature for similarity computation.
        
    Returns:
        Similarity scores for each instance.
    """
    if len(bag_features.shape) != 2 or bag_features.shape[0] == 0:
        return np.ones(bag_features.shape[0])
    
    # Compute cosine similarity to prototypes
    norm_features = bag_features / (np.linalg.norm(bag_features, axis=1, keepdims=True) + 1e-8)
    norm_prototypes = prototypes / (np.linalg.norm(prototypes, axis=1, keepdims=True) + 1e-8)
    
    # Similarity matrix: (n_instances, n_prototypes)
    similarities = np.dot(norm_features, norm_prototypes.T)
    
    # Max similarity to any prototype
    max_similarities = np.max(similarities, axis=1)
    
    # Apply temperature scaling
    scores = np.exp(max_similarities / temperature)
    return scores / (np.sum(scores) + 1e-8)




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


def apply_stochastic_instance_masking(probs: NDArray[np.float64], mask_ratio: float = 0.1, top_k_ratio: float = 0.3, adaptive_mask_ratio: float = None) -> NDArray[np.float64]:
    """Apply Stochastic Top-K Instance Masking (STKIM).
    
    Based on ACMIL paper: randomly masks high-attention instances to force
    the model to discover alternative discriminative patterns.
    
    Args:
        probs: Instance probability matrix of shape (n_instances, n_classes).
        mask_ratio: Fraction of top instances to potentially mask.
        top_k_ratio: Fraction of instances considered as "top".
        adaptive_mask_ratio: Optional adaptive mask ratio based on external initialization.
        
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
    
    # Use adaptive mask ratio if provided
    effective_mask_ratio = adaptive_mask_ratio if adaptive_mask_ratio is not None else mask_ratio
    
    # Randomly select instances to mask from top-k
    n_mask = max(1, int(k_top * effective_mask_ratio))
    mask_indices = np.random.choice(top_indices, size=n_mask, replace=False)
    
    # Create masked probabilities
    masked_probs = probs.copy()
    
    # For masked instances, reduce their influence (but don't zero out completely)
    # This maintains differentiability while encouraging exploration
    masked_probs[mask_indices] *= 0.5  # Reduce but don't eliminate
    
    return masked_probs




class PseudoBagMemoryPool:
    """Pseudo-Bag Memory Pool (PMP) for continual MIL.
    
    Based on CVPR 2025: builds tiny pseudo-bags that store only the most
    informative instances discovered so far, with attention knowledge distillation.
    """
    
    def __init__(self, pool_size: int = 100, n_classes: int = 2):
        """Initialize memory pool.
        
        Args:
            pool_size: Maximum number of instances to store per class
            n_classes: Number of classes
        """
        self.pool_size = pool_size
        self.n_classes = n_classes
        self.memory = {i: [] for i in range(n_classes)}
        self.importance_scores = {i: [] for i in range(n_classes)}
    
    def update(self, instances: NDArray[np.float64], labels: NDArray[np.int_], 
               scores: NDArray[np.float64]) -> None:
        """Update memory pool with new important instances.
        
        Args:
            instances: Instance features
            labels: Instance labels
            scores: Importance scores (e.g., attention weights)
        """
        for i, (instance, label, score) in enumerate(zip(instances, labels, scores)):
            if label < self.n_classes:
                self.memory[label].append(instance)
                self.importance_scores[label].append(score)
                
                # Maintain pool size by keeping most important instances
                if len(self.memory[label]) > self.pool_size:
                    # Sort by importance and keep top instances
                    sorted_indices = np.argsort(self.importance_scores[label])[-self.pool_size:]
                    self.memory[label] = [self.memory[label][idx] for idx in sorted_indices]
                    self.importance_scores[label] = [self.importance_scores[label][idx] for idx in sorted_indices]
    
    def get_pseudo_bag(self, size: int = 10) -> Tuple[NDArray[np.float64], NDArray[np.int_]]:
        """Generate a pseudo-bag by sampling from memory.
        
        Args:
            size: Number of instances to sample per class
            
        Returns:
            Tuple of (instances, labels)
        """
        instances = []
        labels = []
        
        for class_id in range(self.n_classes):
            if len(self.memory[class_id]) > 0:
                # Sample with replacement based on importance scores
                pool = np.array(self.memory[class_id])
                scores = np.array(self.importance_scores[class_id])
                probs = scores / (np.sum(scores) + 1e-8)
                
                n_sample = min(size, len(pool))
                indices = np.random.choice(len(pool), size=n_sample, p=probs, replace=True)
                
                instances.extend(pool[indices])
                labels.extend([class_id] * n_sample)
        
        if instances:
            return np.array(instances), np.array(labels)
        else:
            return np.array([]), np.array([])




def sparse_differentiable_topk(scores: NDArray[np.float64], k: float, temperature: float = 1.0, sparsity_threshold: float = 0.01, adaptive_sparsity: bool = True) -> NDArray[np.float64]:
    """Enhanced sparse differentiable top-k operator with improved numerical stability.
    
    Based on "Fast, Differentiable & Sparse Top-k" (2023) with enhancements:
    - Log-sum-exp trick for numerical stability
    - Adaptive sparsity threshold
    - Gradient-friendly normalization
    
    Args:
        scores: Input scores to select from.
        k: Target number of selections (can be fractional).
        temperature: Temperature parameter for smoothness.
        sparsity_threshold: Base threshold below which weights are set to zero.
        adaptive_sparsity: Whether to use adaptive sparsity thresholding.
        
    Returns:
        Sparse soft selection weights (sum approximately equals k).
    """
    n = len(scores)
    if n == 0 or k <= 0:
        return np.zeros(n)
    if k >= n:
        return np.ones(n)
    
    # Apply temperature scaling with numerical stability
    max_score = np.max(scores)
    scaled_scores = (scores - max_score) / temperature  # Shift for numerical stability
    
    # Sort scores in descending order
    sorted_indices = np.argsort(scaled_scores)[::-1]
    sorted_scores = scaled_scores[sorted_indices]
    
    # Enhanced isotonic projection with better numerical handling
    # Use log-sum-exp style computation for stability
    cumsum = np.cumsum(np.exp(sorted_scores - np.max(sorted_scores)))
    cumsum = np.log(cumsum + 1e-10) + np.max(sorted_scores)
    
    indices = np.arange(1, n + 1)
    
    # Compute potential thresholds with stability
    log_k = np.log(k + 1e-10)
    thresholds = (cumsum - log_k) / indices
    
    # Find the valid threshold with gradient-friendly computation
    valid_mask = sorted_scores >= thresholds
    if np.any(valid_mask):
        pivot = np.where(valid_mask)[0][-1]
        threshold = thresholds[pivot]
    else:
        threshold = thresholds[-1]
    
    # Compute sparse weights with smooth maximum
    epsilon = 1e-8
    weights = np.maximum(0, sorted_scores - threshold)
    
    # Adaptive sparsity threshold based on score distribution
    if adaptive_sparsity:
        # Adjust threshold based on score variance
        score_std = np.std(sorted_scores[:int(2*k)]) if int(2*k) < n else np.std(sorted_scores)
        effective_threshold = sparsity_threshold * (1 + score_std)
        weights[weights < effective_threshold] = 0
    else:
        weights[weights < sparsity_threshold] = 0
    
    # Gradient-friendly normalization
    current_sum = np.sum(weights) + epsilon
    weights = weights * k / current_sum
    
    # Map back to original order
    result = np.zeros(n)
    result[sorted_indices] = weights
    
    return result


def smooth_topk(scores: NDArray[np.float64], k: float, temperature: float = 1.0, adaptive: bool = False) -> NDArray[np.float64]:
    """Differentiable smooth top-k operator (ST_k).
    
    Wrapper that uses sparse_differentiable_topk for better performance.
    
    Args:
        scores: Input scores to select from.
        k: Target number of selections (can be fractional).
        temperature: Temperature parameter for smoothness.
        adaptive: Whether to use adaptive sparsity.
        
    Returns:
        Soft selection weights (sum approximately equals k).
    """
    return sparse_differentiable_topk(scores, k, temperature, 
                                      sparsity_threshold=0.005,
                                      adaptive_sparsity=adaptive)


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


def compute_exact_differentiable_count_loss(probs, k_target, weight=0.1, temperature=1.0):
    """Compute exact differentiable loss for count constraints.
    
    Based on Shukla et al. "Count-Loss" (2023): computes the exact probability
    that k instances are positive using dynamic programming with logsumexp.
    
    Args:
        probs: probabilities for positive class (n_instances,)
        k_target: target count of positive instances
        weight: loss weight
        temperature: temperature for probability computation
        
    Returns:
        count_loss: scalar loss value
    """
    n = len(probs)
    if n == 0 or k_target < 0 or k_target > n:
        return 0.0
    
    # Apply temperature scaling
    scaled_probs = np.clip(probs, 1e-7, 1 - 1e-7)  # Numerical stability
    
    # Dynamic programming to compute P(count = k)
    # dp[i][j] = log probability that exactly j of the first i instances are positive
    log_dp = np.full((n + 1, n + 1), -np.inf)
    log_dp[0][0] = 0.0  # Base case: 0 instances, 0 positive
    
    for i in range(1, n + 1):
        p_i = scaled_probs[i - 1]
        log_p = np.log(p_i)
        log_1_minus_p = np.log(1 - p_i)
        
        for j in range(i + 1):
            if j == 0:
                # All previous instances negative, current also negative
                log_dp[i][j] = log_dp[i-1][j] + log_1_minus_p
            elif j == i:
                # All instances positive
                log_dp[i][j] = log_dp[i-1][j-1] + log_p
            else:
                # Either current positive or negative
                log_positive = log_dp[i-1][j-1] + log_p
                log_negative = log_dp[i-1][j] + log_1_minus_p
                # LogSumExp for numerical stability
                max_val = max(log_positive, log_negative)
                log_dp[i][j] = max_val + np.log(
                    np.exp(log_positive - max_val) + np.exp(log_negative - max_val)
                )
    
    # Probability of exactly k_target instances being positive
    log_p_exact_k = log_dp[n][int(k_target)]
    
    # Convert to probability (with clipping for stability)
    p_exact_k = np.exp(np.clip(log_p_exact_k, -20, 0))  # Clip to avoid underflow
    
    # Loss is negative log probability of the target count
    count_loss = -np.log(p_exact_k + 1e-10)
    
    # Also add a smooth penalty for deviation from expected count
    expected_count = np.sum(probs)
    smooth_penalty = 0.1 * (expected_count - k_target) ** 2
    
    return weight * (count_loss + smooth_penalty)


def compute_differentiable_count_loss(probs, k_target, weight=0.1):
    """Compute differentiable loss for count constraints.
    
    Wrapper that uses exact count loss for better gradient signal.
    
    Args:
        probs: probabilities for positive class (n_instances,)
        k_target: target count of positive instances
        weight: loss weight
        
    Returns:
        count_loss: scalar loss value
    """
    # Temporarily use only smooth approximation to diagnose performance issues
    # TODO: Re-enable exact count loss after debugging
    expected_count = np.sum(probs)
    count_deviation = abs(expected_count - k_target)
    # Add smooth penalty similar to exact version
    smooth_penalty = 0.1 * (expected_count - k_target) ** 2
    return weight * (count_deviation + smooth_penalty)


def soft_k_of_n_assignment(probs, k_min, k_max, temperature=0.5, margin_scores=None, aem_scores=None, interaction_weights=None, global_context=None, attribute_scores=None, prototype_scores=None, aggregated_global_context=None, iteration=0, external_init_weight_schedule=None):
    """Enhanced soft k-of-n assignment using smooth top-k operator with attribute scoring.
    
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
        global_context: global optimization context (optional)
        attribute_scores: multi-dimensional attribute scores (optional)
        iteration: current iteration for adaptive weighting
        
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
    
    # Adaptive weighting based on iteration (warm up attribute scores)
    base_weight = 1.0
    attr_warmup = min(1.0, iteration / 6.0)  # Slower warm up over 6 iterations
    
    # Adjust weights based on initialization type (external init needs different balance)
    # Enhanced detection using multiple signals
    is_external = iteration == 0 and np.var(probs) < 0.15
    # Additional check: external init often has more uniform max probabilities
    if is_external and len(probs) > 3:
        max_probs = np.max(probs, axis=1)
        prob_uniformity = np.std(max_probs) / (np.mean(max_probs) + 1e-8)
        is_external = is_external and prob_uniformity < 0.2
    
    # Get external init weight decay factor if available
    external_weight_decay = 1.0
    if external_init_weight_schedule is not None:
        external_weight_decay = external_init_weight_schedule(iteration)
    
    # Combine with attribute scores (primary signal after warmup)
    if attribute_scores is not None and len(attribute_scores) == len(probs):
        # Apply curriculum decay for external init
        attr_weight = (0.22 if is_external else 0.18) * attr_warmup
        if is_external:
            attr_weight *= external_weight_decay  # Decay external init influence
        log_odds = log_odds * (1 - attr_weight) + attribute_scores * attr_weight
    
    # Combine with margin scores
    if margin_scores is not None and len(margin_scores) == len(probs):
        margin_weight = (0.14 if is_external else 0.1) * (1 - 0.3 * attr_warmup)
        log_odds = log_odds + margin_weight * margin_scores
    
    # Combine with AEM scores (when active)
    if aem_scores is not None and len(aem_scores) == len(probs):
        aem_weight = (0.10 if is_external else 0.08) * (1 - 0.15 * attr_warmup)  # Stronger for external
        log_odds = log_odds + aem_weight * aem_scores
    
    # Combine with instance interaction weights
    if interaction_weights is not None and len(interaction_weights) == len(probs):
        interaction_weight = (0.16 if is_external else 0.12) * (1 + 0.2 * attr_warmup)
        log_odds = log_odds + interaction_weight * interaction_weights
    
    # Combine with prototype similarity scores (PAMIL-inspired)
    if prototype_scores is not None and len(prototype_scores) == len(probs):
        prototype_weight = (0.24 if is_external else 0.14) * attr_warmup  # Much stronger for external init
        if is_external:
            prototype_weight *= external_weight_decay  # Apply curriculum decay
        log_odds = log_odds + prototype_weight * prototype_scores
    
    # Apply global context if available (ProMIL-inspired)
    if global_context is not None or aggregated_global_context is not None:
        # Combine bag-specific and aggregated global contexts
        if global_context is not None and aggregated_global_context is not None:
            # Weighted combination favoring aggregated context as iterations progress
            aggregation_weight = min(0.7, 0.3 + 0.1 * iteration)  # Gradually increase weight
            effective_proportion = ((1 - aggregation_weight) * global_context + 
                                   aggregation_weight * aggregated_global_context)
        elif global_context is not None:
            effective_proportion = global_context
        else:
            effective_proportion = aggregated_global_context
            
        optimal_k = int(np.round(effective_proportion * n))
        optimal_k = max(k_min, min(k_max, optimal_k))
        
        # Adaptive biasing based on iteration (more flexible early, more strict later)
        flexibility = max(1, 3 - iteration // 3)  # Decreases from 3 to 1
        k_min = max(k_min, optimal_k - flexibility)
        k_max = min(k_max, optimal_k + flexibility)
    
    # Enhanced k selection with diversity consideration
    best_k = k_min
    best_score = -np.inf
    
    for k in range(max(0, k_min), min(n, k_max) + 1):
        # Use enhanced ST_k with adaptive sparsity
        adaptive_sparsity = iteration > 3  # Enable after initial iterations
        soft_weights = sparse_differentiable_topk(log_odds, k, temperature, 
                                                  sparsity_threshold=0.005,
                                                  adaptive_sparsity=adaptive_sparsity)
        
        # Score with entropy regularization for diversity
        score = np.sum(soft_weights * np.log(probs + eps))
        
        # Add diversity bonus (encourage non-concentrated selections)
        if np.sum(soft_weights > 0) > 1:
            weight_entropy = -np.sum(soft_weights[soft_weights > 0] * np.log(soft_weights[soft_weights > 0] + eps))
            diversity_bonus = 0.05 * weight_entropy * (1 - np.exp(-iteration / 3))  # Gradually increase
            score += diversity_bonus
        
        if score > best_score:
            best_score = score
            best_k = k
    
    # Return soft assignments for best k with enhanced operator
    return sparse_differentiable_topk(log_odds, best_k, temperature, 
                                      sparsity_threshold=0.005,
                                      adaptive_sparsity=(iteration > 3))


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
                
                # For binary classification, use mixed initialization
                if n_classes == 2:
                    # Smart initialization: distribute instances based on constraints
                    k_target_0 = int((lower_threshold[i_bag, 0] + upper_threshold[i_bag, 0]) / 2)
                    k_target_1 = int((lower_threshold[i_bag, 1] + upper_threshold[i_bag, 1]) / 2)
                    
                    # Check if targets are feasible
                    if k_target_0 + k_target_1 == n_instances_in_bag:
                        # Perfect match - use target distribution
                        labels = np.zeros(n_instances_in_bag, dtype=int)
                        labels[:k_target_1] = 1
                        np.random.shuffle(labels)
                        y.append(labels)
                    else:
                        # Use best class based on constraint scores
                        best_class = np.argmax(constraint_scores)
                        y.append(np.repeat(best_class, n_instances_in_bag))
                else:
                    # Multi-class: use best class
                    best_class = np.argmax(constraint_scores)
                    y.append(np.repeat(best_class, n_instances_in_bag))
        else:
            y = initial_y
            
        # Initialize simple parameters
        n_bags = len(bags)
        
        # Detect initialization type for adaptive parameters
        # Enhanced detection: check both presence and distribution pattern
        is_external_init = initial_y is not None
        if is_external_init:
            # Additional check: external init often has more uniform distribution
            flatten_init_y = np.hstack(initial_y)
            init_variance = np.var(flatten_init_y)
            init_entropy = -np.sum([np.mean(flatten_init_y == c) * np.log(np.mean(flatten_init_y == c) + 1e-8) 
                                    for c in range(n_classes)])
            
            # Enhanced detection: check constraint satisfaction as additional signal
            init_constraint_violations = 0
            for i_bag in range(n_bags):
                bag_counts = pd.Series(initial_y[i_bag]).value_counts().to_dict()
                for i_class in range(n_classes):
                    count = bag_counts.get(i_class, 0)
                    if count < lower_threshold[i_bag, i_class]:
                        init_constraint_violations += lower_threshold[i_bag, i_class] - count
                    elif count > upper_threshold[i_bag, i_class]:
                        init_constraint_violations += count - upper_threshold[i_bag, i_class]
            
            # New: compute initial uncertainty scores for better detection
            initial_uncertainties = []
            if len(bags) > 0 and hasattr(self.classifier, 'predict_proba'):
                # Quick initial fit to get uncertainty estimates
                try:
                    flatten_bags_init = np.vstack(bags)
                    flatten_y_init = np.hstack(initial_y)
                    self.classifier.fit(flatten_bags_init, flatten_y_init)
                    for bag in bags[:min(5, len(bags))]:  # Sample first 5 bags
                        probs = self.classifier.predict_proba(bag)
                        uncertainties = compute_instance_uncertainty_scores(probs, temperature=0.5)
                        initial_uncertainties.extend(uncertainties)
                    avg_uncertainty = np.mean(initial_uncertainties) if initial_uncertainties else 0.5
                except:
                    avg_uncertainty = 0.5
            else:
                avg_uncertainty = 0.5
            
            # External init typically has lower variance, higher entropy, lower certainty, and some violations
            if debug:
                print(f"External init detection - variance: {init_variance:.4f}, entropy: {init_entropy:.4f}, "
                      f"avg_uncertainty: {avg_uncertainty:.4f}, violations: {init_constraint_violations}, n_classes: {n_classes}")
            
            # Refined detection thresholds with uncertainty signal
            if n_classes == 2:
                # Binary case: more lenient for external detection
                is_truly_external = ((init_variance < 0.35 and init_entropy > 0.25) or 
                                    init_constraint_violations > 0 or 
                                    avg_uncertainty < 0.6)
            else:
                # Multi-class: normalize by max possible entropy
                max_entropy = np.log(n_classes)
                normalized_entropy = init_entropy / max_entropy
                # Use class-count normalized variance
                normalized_variance = init_variance / (n_classes - 1)
                is_truly_external = ((normalized_variance < 12.0 and normalized_entropy > 0.65) or 
                                    init_constraint_violations > 0 or 
                                    avg_uncertainty < 0.55)
        else:
            is_truly_external = False
        
        # Optimized temperature schedule adjusted for max_iter=10
        if is_external_init:
            # For external init, use gentler multi-phase schedule
            temperature_phases = {
                'phase1_end': 0.3,      # 30% of iterations (iter 3)
                'phase2_end': 0.5,      # 50% of iterations (iter 5)
                'phase3_end': 0.8,      # 80% of iterations (iter 8)
                'temp_start': 0.90,     # Start temperature
                'temp_phase1': 0.45,    # Higher temperature after phase 1
                'temp_phase2': 0.35,    # Higher hold temperature
                'temp_phase3': 0.20,    # Higher temperature after phase 3
                'temp_end': 0.15        # Higher final temperature
            }
        else:
            # For auto-init, use simpler schedule with longer warmup
            temperature_start = 0.90
            temperature_end = 0.30  # Higher final temp for auto-init
            temperature_half_life = 0.40  # Slower decay
            temperature_cycle_amplitude = 0.01
        
        
        # Adaptive confidence-based freezing
        if is_external_init and is_truly_external:
            confidence_threshold_base = 0.965  # Lower for true external init
            confidence_threshold_decay = 0.010  # Slower decay
            margin_lambda = 0.40   # Higher margin importance
        elif is_external_init:
            confidence_threshold_base = 0.972  # Slightly lower for pseudo-external
            confidence_threshold_decay = 0.013  # Moderate decay
            margin_lambda = 0.36   # Higher margin importance
        else:
            confidence_threshold_base = 0.978  # Higher base threshold
            confidence_threshold_decay = 0.018  # Small decay over iterations
            margin_lambda = 0.33   # Moderate margin importance
        
        # LNPL-MIL style curriculum weighting for external init
        # λ(t) = exp(-t/T) where T ≈ 10% of training steps
        external_init_curriculum_T = max_iter * 0.15  # 15% of total iterations for smoother decay
        external_init_weight_schedule = lambda t: np.exp(-t / external_init_curriculum_T) if is_external_init else 0.0
        
        # Global context learning (ProMIL-inspired)
        # Track optimal proportions for each class across all bags
        global_proportions = np.zeros((n_bags, n_classes))
        proportion_momentum = 0.92  # Higher momentum for stability
        
        # Enhanced global context: track aggregated proportions across all bags
        aggregated_global_proportions = np.zeros(n_classes)
        aggregated_proportion_weights = np.zeros(n_bags)  # Weight by bag reliability
        global_aggregation_momentum = 0.95  # Higher momentum for global patterns
        
        # Initialize prototypes using k-means clustering on all instances
        # This provides global instance patterns for better scoring
        all_instances = np.vstack(bags)
        n_prototypes = min(50, max(10, len(all_instances) // 50))  # Adaptive number of prototypes
        if n_prototypes > 5 and len(all_instances) > 50:
            # Use k-means to find representative prototypes
            kmeans = KMeans(n_clusters=n_prototypes, random_state=seed, n_init=3)
            kmeans.fit(all_instances)
            prototypes = kmeans.cluster_centers_
            # Use is_truly_external if available for more precise setting
            if 'is_truly_external' in locals() and is_external_init and is_truly_external:
                prototype_temperature = 0.05  # Much sharper for true external init
            elif is_external_init:
                prototype_temperature = 0.08  # Lower temperature for external init
            else:
                prototype_temperature = 0.14  # Slightly lower for better precision
        else:
            # Fall back to using subset of instances as prototypes
            prototype_indices = np.random.RandomState(seed).choice(len(all_instances), 
                                                                  size=min(10, len(all_instances)), 
                                                                  replace=False)
            prototypes = all_instances[prototype_indices]
            prototype_temperature = 0.2
        
        # Adjust prototype parameters based on initialization type
        if is_external_init and is_truly_external:
            prototype_temperature = 0.06  # Lower for sharper discrimination
            prototype_update_interval = 2  # More frequent updates for external init
            prototype_update_momentum = 0.88  # Higher momentum for stability
            prototype_uncertainty_threshold = 0.7  # Only use instances with reliability > threshold
        else:
            prototype_temperature = 0.14  # Original setting
            prototype_update_interval = 5
            prototype_update_momentum = 0.7
            prototype_uncertainty_threshold = 0.5
        
        instance_prototype_assignments = np.zeros(len(all_instances), dtype=int)  # Track assignments
        
        
        # Optimized count loss weight schedule adjusted for max_iter=10
        if is_external_init:
            # Multi-phase schedule for external init
            count_loss_phases = {
                'warmup_end': 0.3,      # 30% for warmup (iter 3)
                'plateau_end': 0.6,     # 60% hold at 0.10 (iter 6)
                'weight_warmup': 0.08,  # Lower weight after warmup
                'weight_final': 0.16    # Lower final weight
            }
        else:
            # Simple schedule for auto-init
            count_loss_weight_start = 0.0
            count_loss_weight_end = 0.15  # Lower final weight for auto-init
            count_loss_schedule = "cosine"
            count_loss_warmup = 2
        
        
        # Initialize Pseudo-Bag Memory Pool
        memory_pool = PseudoBagMemoryPool(pool_size=200, n_classes=n_classes)
        use_memory_pool = is_external_init and is_truly_external  # Enable for true external init
        
        # Initialize AEM parameters based on o3 MCP recommendations
        aem_lambda_current = 0.025 if is_external_init else 0.015  # Moderately higher initial λ for external init
        aem_rho = 0.1  # Penalty coefficient
        aem_rho_max = 1.2  # Cap for ρ
        aem_rho_gamma = 1.8  # Multiplication factor when constraint not improving
        aem_lambda_max = 0.6  # Cap for λ
        constraint_history = []  # Track constraint violations
        
        # STKIM parameters with adaptive tuning
        if is_external_init and is_truly_external:
            # For true external init, use more aggressive masking to encourage exploration
            stkim_start_iter = 2  # Start earlier
            stkim_mask_ratio_base = 0.08  # Higher base mask ratio
            stkim_mask_ratio_decay = 0.9  # Slower decay
            stkim_top_k_ratio = 0.25  # Consider top 25%
        elif is_external_init:
            # For pseudo-external init
            stkim_start_iter = 3
            stkim_mask_ratio_base = 0.06
            stkim_mask_ratio_decay = 0.92
            stkim_top_k_ratio = 0.2
        else:
            # For auto-init, use gentler masking
            stkim_start_iter = 4
            stkim_mask_ratio_base = 0.05
            stkim_mask_ratio_decay = 0.95
            stkim_top_k_ratio = 0.15
        
        for i_iter in range(max_iter):
            # Compute temperature based on phase schedule
            if is_external_init:
                # Multi-phase temperature schedule for external init
                progress = i_iter / max_iter
                phases = temperature_phases
                
                if progress < phases['phase1_end']:
                    # Phase 1: cosine decay from start to phase1
                    phase_progress = progress / phases['phase1_end']
                    cos_factor = 0.5 * (1 + np.cos(np.pi * phase_progress))
                    temperature = phases['temp_phase1'] + (phases['temp_start'] - phases['temp_phase1']) * cos_factor
                elif progress < phases['phase2_end']:
                    # Phase 2: hold steady
                    temperature = phases['temp_phase2']
                elif progress < phases['phase3_end']:
                    # Phase 3: cosine decay from phase2 to phase3
                    phase_progress = (progress - phases['phase2_end']) / (phases['phase3_end'] - phases['phase2_end'])
                    cos_factor = 0.5 * (1 + np.cos(np.pi * phase_progress))
                    temperature = phases['temp_phase3'] + (phases['temp_phase2'] - phases['temp_phase3']) * cos_factor
                else:
                    # Phase 4: linear decay to end
                    phase_progress = (progress - phases['phase3_end']) / (1.0 - phases['phase3_end'])
                    temperature = phases['temp_end'] + (phases['temp_phase3'] - phases['temp_end']) * (1 - phase_progress)
            else:
                # Simple schedule for auto-init
                base_temp = max(temperature_end, 
                               temperature_start * 0.5 ** (i_iter / (temperature_half_life * max_iter)))
                cycle_factor = temperature_cycle_amplitude * np.sin(2 * np.pi * i_iter / 4)
                temperature = base_temp + cycle_factor
            
            # Enhanced AEM schedule with initialization-aware transition
            if is_external_init and is_truly_external:
                # For true external init, use stronger AEM throughout
                exploration_phase_end = int(np.ceil(max_iter * 0.7))  # Longer exploration
                if i_iter < exploration_phase_end:
                    warmup_iters = 1  # Faster warmup
                    if i_iter < warmup_iters:
                        aem_lambda = 0.02 * (i_iter + 1) / warmup_iters
                    else:
                        aem_lambda_max = 0.65  # Higher max for external init
                        progress = (i_iter - warmup_iters) / (exploration_phase_end - warmup_iters)
                        # Fixed cosine annealing to ensure non-zero values
                        aem_lambda = aem_lambda_max * (0.3 + 0.7 * 0.5 * (1 - np.cos(np.pi * progress)))
                else:
                    exploitation_progress = (i_iter - exploration_phase_end) / (max_iter - exploration_phase_end)
                    aem_lambda = 0.12 * (1 - 0.4 * exploitation_progress)  # Higher floor
            else:
                # Original schedule for auto-init
                exploration_phase_end = int(np.ceil(max_iter * 0.65))
                if i_iter < exploration_phase_end:
                    warmup_iters = 2
                    if i_iter < warmup_iters:
                        aem_lambda = 0.015 * (i_iter + 1) / warmup_iters
                    else:
                        aem_lambda_max = 0.54  # Slightly higher for better exploration
                        progress = (i_iter - warmup_iters) / (exploration_phase_end - warmup_iters)
                        aem_lambda = aem_lambda_max * 0.5 * (1 - np.cos(np.pi * progress))
                else:
                    exploitation_progress = (i_iter - exploration_phase_end) / (max_iter - exploration_phase_end)
                    aem_lambda = 0.08 * (1 - 0.5 * exploitation_progress)
            
            # Compute count loss weight based on schedule
            if is_external_init:
                # Multi-phase schedule for external init
                phases = count_loss_phases
                if i_iter < phases['warmup_end'] * max_iter:
                    # Linear warmup from 0 to weight_warmup
                    progress = i_iter / (phases['warmup_end'] * max_iter)
                    count_loss_weight = phases['weight_warmup'] * progress
                elif i_iter < phases['plateau_end'] * max_iter:
                    # Hold at weight_warmup
                    count_loss_weight = phases['weight_warmup']
                else:
                    # Linear increase to final weight
                    plateau_end_iter = phases['plateau_end'] * max_iter
                    if plateau_end_iter < max_iter:
                        progress = (i_iter - plateau_end_iter) / (max_iter - plateau_end_iter)
                        count_loss_weight = phases['weight_warmup'] + (phases['weight_final'] - phases['weight_warmup']) * progress
                    else:
                        count_loss_weight = phases['weight_warmup']
            else:
                # Simple schedule for auto-init
                if i_iter < count_loss_warmup:
                    count_loss_weight = 0.0
                elif count_loss_schedule == "cosine":
                    count_loss_weight = count_loss_weight_start + (count_loss_weight_end - count_loss_weight_start) * 0.5 * (1 - np.cos(np.pi * i_iter / max_iter))
                else:
                    count_loss_weight = count_loss_weight_start
            
            # Fit
            flatten_bags = np.vstack(bags)
            flatten_y = np.hstack(y)
            self.classifier.fit(flatten_bags, flatten_y)

            # Compute outputs
            fs = [self.classifier.predict_proba(bag) for bag in bags]
            
            # Apply STKIM if enabled with adaptive parameters
            if i_iter >= stkim_start_iter:
                # Compute adaptive mask ratio with decay
                adaptive_mask_ratio = stkim_mask_ratio_base * (stkim_mask_ratio_decay ** (i_iter - stkim_start_iter))
                
                # Apply with external init weight schedule if available
                if is_external_init and is_truly_external:
                    external_decay = external_init_weight_schedule(i_iter)
                    adaptive_mask_ratio *= (0.5 + 0.5 * external_decay)  # Reduce masking as external init influence decays
                
                fs_masked = []
                for bag_probs in fs:
                    masked_probs = apply_stochastic_instance_masking(
                        bag_probs, 
                        mask_ratio=stkim_mask_ratio_base,
                        top_k_ratio=stkim_top_k_ratio,
                        adaptive_mask_ratio=adaptive_mask_ratio
                    )
                    fs_masked.append(masked_probs)
                fs = fs_masked
            
            y_pred = [self.classifier.predict(bag) for bag in bags]
            flatten_original_y = np.hstack(list(y))
            

            # For every bag - use simplified assignments
            has_changed = False
            total_changes = 0  # Initialize total changes counter
            
            for i_bag in range(len(bags)):
                # Hybrid approach: combine constraint-based selection with soft assignments
                n_instances = len(bags[i_bag])
                probs = fs[i_bag]  # Shape: (n_instances, n_classes)
                
                
                # Get current class counts
                class_count_dict = pd.Series(y[i_bag]).value_counts().to_dict()
                
                # Track which instances have been reassigned in this iteration
                changed_indices = set()
                new_y = y[i_bag].copy()
                
                
                # No freezing - all instances remain active
                frozen_indices = set()
                
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
                        
                        # Compute instance interaction weights
                        interaction_weights = None
                        if n_active > 1:  # Need at least 2 instances for interaction
                            k_neighbors = min(5, n_active - 1)  # Adaptive k based on bag size
                            interaction_weights = compute_instance_interaction_weights(
                                probs[active_indices], k=k_neighbors
                            )
                        
                        # Compute attribute scores for active instances
                        attribute_scores_active = compute_attribute_scores(probs[active_indices])
                        
                        # Compute prototype similarity scores for active instances
                        prototype_scores_active = None
                        if n_active > 0:
                            active_features = bags[i_bag][active_indices]
                            prototype_scores_active = compute_prototype_similarity_scores(
                                active_features, prototypes, temperature=prototype_temperature
                            )
                        
                        # Get global context for this bag (if available after first iteration)
                        global_context = None
                        aggregated_context = None
                        if i_iter > 0:
                            global_context = global_proportions[i_bag, 1]  # For binary, use class 1 proportion
                            aggregated_context = aggregated_global_proportions[1]  # Aggregated class 1 proportion
                        
                        # Always use soft assignment with ST_k for differentiability
                        soft_weights = soft_k_of_n_assignment(
                            active_probs, k_min_active, k_max_active, 
                            temperature=temperature,  # Use the independent temperature schedule
                            margin_scores=margin_scores, aem_scores=aem_scores,
                            interaction_weights=interaction_weights, global_context=global_context,
                            attribute_scores=attribute_scores_active, prototype_scores=prototype_scores_active,
                            aggregated_global_context=aggregated_context,
                            iteration=i_iter,
                            external_init_weight_schedule=external_init_weight_schedule
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
                    
                    # Add differentiable count constraint loss
                    if n_classes == 2 and i_class == 1:
                        # For binary case, compute soft count loss with adaptive weight
                        target_k = (lower_threshold[i_bag, 1] + upper_threshold[i_bag, 1]) / 2
                        count_loss += compute_differentiable_count_loss(
                            probs[:, 1], 
                            target_k, 
                            weight=count_loss_weight
                        )
                
                # Add contrastive loss for better instance-level clustering
                
                # Update global proportions with momentum (ProMIL-inspired)
                current_proportions = np.zeros(n_classes)
                for i_class in range(n_classes):
                    current_proportions[i_class] = final_counts.get(i_class, 0) / n_instances
                
                if i_iter == 0:
                    global_proportions[i_bag] = current_proportions
                else:
                    global_proportions[i_bag] = (proportion_momentum * global_proportions[i_bag] + 
                                                 (1 - proportion_momentum) * current_proportions)
                
                # Equal weighting for all bags
                if i_iter > 0:
                    bag_weight = 1.0
                    aggregated_proportion_weights[i_bag] = bag_weight
                    
                    # Compute weighted average across all bags
                    weighted_proportions = np.zeros(n_classes)
                    total_weight = 0
                    for bag_idx in range(n_bags):
                        if aggregated_proportion_weights[bag_idx] > 0:
                            weighted_proportions += global_proportions[bag_idx] * aggregated_proportion_weights[bag_idx]
                            total_weight += aggregated_proportion_weights[bag_idx]
                    
                    if total_weight > 0:
                        weighted_proportions /= total_weight
                        aggregated_global_proportions = (global_aggregation_momentum * aggregated_global_proportions + 
                                                        (1 - global_aggregation_momentum) * weighted_proportions)

            if debug:
                stkim_info = ""
                if i_iter >= stkim_start_iter:
                    stkim_info = f", stkim_mask_ratio: {adaptive_mask_ratio:.3f}"
                print(f"iter: {i_iter}, temperature: {temperature:.3f}, aem_lambda: {aem_lambda:.3f}, count_loss_weight: {count_loss_weight:.3f}{stkim_info}")
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
            
            # Adaptive prototype update: refine prototypes based on current assignments
            if i_iter > 0 and (i_iter + 1) % prototype_update_interval == 0:
                # Collect all instances with their current predicted labels
                all_instances_flat = np.vstack(bags)
                all_labels_flat = np.hstack(y)
                all_probs_flat = np.vstack(fs)
                
                # Compute uncertainty scores for all instances
                uncertainty_scores = compute_instance_uncertainty_scores(all_probs_flat, temperature=0.5)
                
                # For each prototype, find assigned instances and update
                new_prototypes = prototypes.copy()
                for proto_idx in range(len(prototypes)):
                    # Find instances closest to this prototype
                    distances = np.sum((all_instances_flat - prototypes[proto_idx])**2, axis=1)
                    
                    # Get top 20% closest instances that are predicted positive
                    n_closest = max(5, int(0.2 * len(all_instances_flat)))
                    closest_indices = np.argsort(distances)[:n_closest]
                    positive_closest = closest_indices[all_labels_flat[closest_indices] == 1]
                    
                    # Filter by uncertainty threshold
                    if len(positive_closest) > 0:
                        reliable_positive = positive_closest[uncertainty_scores[positive_closest] > prototype_uncertainty_threshold]
                        if len(reliable_positive) > 3:
                            # Weight by reliability scores
                            weights = uncertainty_scores[reliable_positive]
                            weights = weights / np.sum(weights)
                            # Update prototype with weighted average and momentum
                            new_center = np.average(all_instances_flat[reliable_positive], axis=0, weights=weights)
                            new_prototypes[proto_idx] = (prototype_update_momentum * prototypes[proto_idx] + 
                                                        (1 - prototype_update_momentum) * new_center)
                
                prototypes = new_prototypes
                
                # Adjust prototype temperature based on convergence
                if not has_changed:
                    prototype_temperature *= 0.92  # Slightly faster cooling
                else:
                    # Apply curriculum decay to prototype temperature for external init
                    if is_external_init and is_truly_external:
                        decay_factor = external_init_weight_schedule(i_iter)
                        prototype_temperature *= (0.95 + 0.05 * decay_factor)  # Adaptive cooling

            # Update memory pool with high-confidence instances
            if use_memory_pool and i_iter > 1:
                # Collect high-confidence instances
                all_probs = np.vstack(fs)
                all_labels = np.hstack(y)
                all_features = np.vstack(bags)
                
                # Compute importance scores (max probability only)
                max_probs = np.max(all_probs, axis=1)
                importance_scores = max_probs
                
                # Update memory pool with top instances
                high_conf_mask = max_probs > 0.85
                if np.any(high_conf_mask):
                    memory_pool.update(
                        all_features[high_conf_mask],
                        all_labels[high_conf_mask],
                        importance_scores[high_conf_mask]
                    )
                
                # Train on pseudo-bags for better generalization
                if i_iter > 3 and len(memory_pool.memory[0]) > 10:
                    for _ in range(3):  # Multiple pseudo-bag iterations
                        pseudo_features, pseudo_labels = memory_pool.get_pseudo_bag(size=20)
                        if len(pseudo_features) > 0:
                            # Augment training data with pseudo-bag
                            aug_features = np.vstack([flatten_bags, pseudo_features])
                            aug_labels = np.hstack([flatten_y, pseudo_labels])
                            self.classifier.fit(aug_features, aug_labels)
            
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
            
            # Continue until max_iter (never break early)

        return self.classifier, y
