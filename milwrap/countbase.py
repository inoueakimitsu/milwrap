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


def compute_bag_embedding_loss(bag_features: List[NDArray[np.float64]], bag_labels: List[int], 
                               temperature: float = 0.1, use_hard_negative: bool = True, margin: float = 0.2) -> float:
    """Compute Bag-Embedding Loss (BEL) for contrastive learning at bag level.
    
    Based on BEL (2023): contrastive loss in bag-embedding space to ensure
    bags of same class are close while different classes are separated.
    Enhanced with hard negative mining for better discrimination.
    
    Args:
        bag_features: List of bag feature matrices (each bag's instance features)
        bag_labels: List of bag labels
        temperature: Temperature parameter for contrastive loss
        use_hard_negative: Whether to use hard negative mining
        margin: Margin for hard negative mining
        
    Returns:
        BEL loss value
    """
    if len(bag_features) < 2:
        return 0.0
    
    n_bags = len(bag_features)
    
    # Compute bag embeddings as mean of instance features
    bag_embeddings = []
    valid_indices = []
    for i, features in enumerate(bag_features):
        if len(features) > 0:
            bag_embedding = np.mean(features, axis=0)
            # L2 normalize
            bag_embedding = bag_embedding / (np.linalg.norm(bag_embedding) + 1e-8)
            bag_embeddings.append(bag_embedding)
            valid_indices.append(i)
    
    if len(bag_embeddings) < 2:
        return 0.0
    
    bag_embeddings = np.stack(bag_embeddings)
    valid_labels = [bag_labels[i] for i in valid_indices]
    n_valid = len(valid_labels)
    
    # Compute pairwise cosine similarities
    similarities = np.dot(bag_embeddings, bag_embeddings.T)
    
    # Create label mask (1 if same class, 0 otherwise)
    label_mask = np.array([[1 if valid_labels[i] == valid_labels[j] else 0 
                           for j in range(n_valid)] for i in range(n_valid)])
    
    # Remove diagonal
    mask = 1 - np.eye(n_valid)
    similarities = similarities * mask
    label_mask = label_mask * mask
    
    # Compute contrastive loss with optional hard negative mining
    losses = []
    for i in range(n_valid):
        # Positive pairs (same class)
        pos_mask = label_mask[i] > 0
        if np.sum(pos_mask) > 0:
            # Average positive similarity
            pos_sim = similarities[i] * pos_mask
            avg_pos_sim = np.sum(pos_sim) / np.sum(pos_mask)
            
            if use_hard_negative:
                # Hard negative mining: find hardest negative (highest similarity)
                neg_mask = (1 - label_mask[i]) * mask[i]
                if np.sum(neg_mask) > 0:
                    neg_sims = similarities[i] * neg_mask - (1 - neg_mask) * 1e6  # Mask out non-negatives
                    hardest_neg_idx = np.argmax(neg_sims)
                    hardest_neg_sim = similarities[i, hardest_neg_idx]
                    
                    # Apply margin between positive and hard negative
                    margin_loss = np.maximum(0, hardest_neg_sim - avg_pos_sim + margin)
                    
                    # Combine with standard contrastive loss
                    exp_sim = np.exp(similarities[i] / temperature)
                    pos_exp = np.sum(exp_sim * pos_mask) / np.sum(pos_mask)
                    all_exp = np.sum(exp_sim * mask[i])
                    
                    if all_exp > 0:
                        contrastive_loss = -np.log(pos_exp / all_exp + 1e-8)
                        # Weighted combination
                        loss = contrastive_loss + 0.5 * margin_loss
                        losses.append(loss)
            else:
                # Standard contrastive loss
                exp_sim = np.exp(similarities[i] / temperature)
                pos_exp = np.sum(exp_sim * pos_mask) / np.sum(pos_mask)
                all_exp = np.sum(exp_sim * mask[i])
                
                if all_exp > 0:
                    loss = -np.log(pos_exp / all_exp + 1e-8)
                    losses.append(loss)
    
    return np.mean(losses) if losses else 0.0


def compute_attention_entropy_maximization(bag_probs: NDArray[np.float64], lambda_entropy: float = 0.1) -> NDArray[np.float64]:
    """Compute AEM (Attention Entropy Maximization) regularized scores.
    
    Based on AEM research: adds entropy regularizer to prevent over-concentrated attention.
    Uses simpler global entropy bonus for more stable gradients.
    
    Args:
        bag_probs: Instance probability matrix of shape (n_instances, n_classes).
        lambda_entropy: Entropy regularization strength.
        
    Returns:
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
    
    # Normalize entropy by bag size for consistent regularization strength
    normalized_entropy = attention_entropy / np.log(n_instances + 1e-10)
    
    # AEM regularization: simple global entropy bonus (more stable than per-instance weighting)
    entropy_bonus = lambda_entropy * normalized_entropy
    
    # Return scores with uniform entropy bonus for all instances
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


def gumbel_topk(scores: NDArray[np.float64], k: int, temperature: float = 1.0) -> NDArray[np.float64]:
    """Gumbel-Top-k operator for stochastic differentiable top-k selection.
    
    Based on Kool et al. 2019 "Stochastic Beams": adds Gumbel noise to enable
    stochastic sampling while maintaining differentiability.
    
    Args:
        scores: Input scores to select from.
        k: Number of elements to select.
        temperature: Temperature parameter for Gumbel-Softmax.
        
    Returns:
        Soft selection weights (probabilistic k-hot vector).
    """
    n = len(scores)
    if n == 0 or k <= 0:
        return np.zeros(n)
    if k >= n:
        return np.ones(n)
    
    # Generate Gumbel noise
    eps = 1e-8
    uniform = np.random.uniform(0, 1, size=n)
    gumbel_noise = -np.log(-np.log(uniform + eps) + eps)
    
    # Add Gumbel noise to scores and apply temperature
    perturbed_scores = (scores + gumbel_noise) / temperature
    
    # Get top-k indices
    top_k_indices = np.argpartition(perturbed_scores, -k)[-k:]
    
    # Compute softmax over the top-k elements only
    top_k_scores = perturbed_scores[top_k_indices]
    exp_scores = np.exp(top_k_scores - np.max(top_k_scores))  # Numerical stability
    softmax_probs = exp_scores / (np.sum(exp_scores) + eps)
    
    # Create sparse mask
    mask = np.zeros(n)
    mask[top_k_indices] = softmax_probs
    
    return mask


def apply_mil_dropout(probs: NDArray[np.float64], dropout_k: int = 1, dropout_rate: float = 0.5) -> NDArray[np.float64]:
    """Apply MIL-Dropout by stochastically removing top-k scoring instances.
    
    Based on arxiv.org/abs/2504.14783: forces the model to find alternative evidence
    by masking out the current top-scoring instances during training.
    
    Args:
        probs: Instance probability matrix of shape (n_instances, n_classes)
        dropout_k: Number of top instances to potentially drop
        dropout_rate: Probability of dropping each top instance
        
    Returns:
        Modified probabilities with dropout applied
    """
    if len(probs.shape) != 2 or probs.shape[0] <= dropout_k:
        return probs
    
    # Calculate max probability for each instance
    max_probs = np.max(probs, axis=1)
    
    # Get indices of top-k instances
    top_k_indices = np.argpartition(max_probs, -dropout_k)[-dropout_k:]
    
    # Create dropout mask
    dropout_mask = np.ones(probs.shape[0], dtype=bool)
    
    # Stochastically drop top-k instances
    for idx in top_k_indices:
        if np.random.random() < dropout_rate:
            dropout_mask[idx] = False
    
    # Apply mask (set dropped instances to uniform distribution)
    probs_dropout = probs.copy()
    n_classes = probs.shape[1]
    uniform_prob = 1.0 / n_classes
    
    for idx in range(probs.shape[0]):
        if not dropout_mask[idx]:
            probs_dropout[idx] = uniform_prob
    
    return probs_dropout


def logsumexp_smooth_cardinality(scores: NDArray[np.float64], kappa: float = 25.0) -> float:
    """LogSumExp smooth approximation to cardinality (count of positive assignments).
    
    Args:
        scores: Assignment scores or probabilities
        kappa: Temperature parameter (higher = tighter approximation to max)
        
    Returns:
        Smooth approximation to the count of positive elements
    """
    # For binary assignments, use sigmoid-like transformation
    # LogSumExp approximation: (1/kappa) * log(sum(exp(kappa * scores)))
    # This approximates sum(scores > 0.5) for appropriate kappa
    
    # Numerical stability: subtract max before exp
    max_score = np.max(scores)
    exp_scores = np.exp(kappa * (scores - max_score))
    lse = max_score + (1.0 / kappa) * np.log(np.sum(exp_scores) + 1e-8)
    
    # Scale to approximate count
    return lse * len(scores) / (1.0 + lse)


def compute_instance_early_stopping_mask(
    probs_history: List[NDArray[np.float64]], 
    y_history: List[NDArray[np.int_]], 
    convergence_threshold: float = 0.01,
    min_history: int = 3
) -> NDArray[np.bool_]:
    """Compute Instance-dependent Early Stopping (IES) mask.
    
    Based on arxiv.org/abs/2502.07547: stops updating instances whose predictions
    have stabilized, freeing capacity to learn harder instances.
    
    Args:
        probs_history: History of probability predictions for each instance
        y_history: History of label assignments for each instance
        convergence_threshold: Threshold for considering an instance as converged
        min_history: Minimum history length required to compute convergence
        
    Returns:
        Boolean mask where True means the instance should be frozen (stop updating)
    """
    if len(probs_history) < min_history:
        # Not enough history, don't freeze any instances
        n_instances = len(probs_history[0]) if probs_history else 0
        return np.zeros(n_instances, dtype=bool)
    
    n_instances = len(probs_history[0])
    freeze_mask = np.zeros(n_instances, dtype=bool)
    
    # Check convergence for each instance
    for i in range(n_instances):
        # Check if predictions have been stable
        recent_probs = [probs[i] for probs in probs_history[-min_history:]]
        recent_labels = [y[i] for y in y_history[-min_history:]]
        
        # Check label stability
        if len(set(recent_labels)) == 1:  # Same label for recent history
            # Check probability stability (low variance)
            prob_std = np.std([np.max(p) for p in recent_probs])
            if prob_std < convergence_threshold:
                # Also check if confidence is high
                avg_confidence = np.mean([np.max(p) for p in recent_probs])
                if avg_confidence > 0.9:  # High confidence threshold
                    freeze_mask[i] = True
    
    return freeze_mask


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


def soft_k_of_n_assignment(probs, k_min, k_max, temperature=0.5, margin_scores=None, aem_scores=None, use_gumbel=False):
    """Solve soft k-of-n assignment using smooth top-k or Gumbel-Top-k operator.
    
    Find soft assignment weights such that sum approximates the optimal k
    within [k_min, k_max] range, using differentiable operators.
    
    Args:
        probs: probabilities for positive class (n_instances,)
        k_min: minimum number of positive instances
        k_max: maximum number of positive instances  
        temperature: temperature for ST_k/GT_k operator
        margin_scores: margin-based scores for weighting (optional)
        aem_scores: AEM-based scores for weighting (optional)
        use_gumbel: whether to use Gumbel-Top-k (True) or Smooth Top-k (False)
        
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
    
    # Combine with margin scores
    if margin_scores is not None and len(margin_scores) == len(probs):
        log_odds = log_odds + 0.1 * margin_scores
    
    # Combine with AEM scores (when active)
    if aem_scores is not None and len(aem_scores) == len(probs):
        log_odds = log_odds + 0.1 * aem_scores  # Moderate weight for AEM
    
    # Find optimal k using gradient-free search
    best_k = k_min
    best_score = -np.inf
    
    for k in range(max(0, k_min), min(n, k_max) + 1):
        # Choose operator based on phase
        if use_gumbel:
            # Use Gumbel-Top-k for exploration phase
            soft_weights = gumbel_topk(log_odds, k, temperature)
        else:
            # Use Smooth Top-k for exploitation phase
            soft_weights = smooth_topk(log_odds, k, temperature)
        
        # Score is expected log probability
        score = np.sum(soft_weights * np.log(probs + eps))
        
        if score > best_score:
            best_score = score
            best_k = k
    
    # Return soft assignments for best k using appropriate operator
    if use_gumbel:
        return gumbel_topk(log_odds, best_k, temperature)
    else:
        return smooth_topk(log_odds, best_k, temperature)


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
        n_bags = len(bags)
        
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
            # For external initialization, apply sophisticated refinement
            y = []
            bag_init_quality = np.zeros(n_bags)  # Track initialization quality per bag
            instance_init_confidence = []  # Track per-instance init confidence
            
            for i_bag, (init_labels, n_instances_in_bag) in enumerate(zip(initial_y, n_list)):
                # Check if external init satisfies constraints
                init_class_counts = pd.Series(init_labels).value_counts().to_dict()
                
                # Calculate detailed constraint violations
                total_violation = 0
                violation_details = {}
                for i_class in range(n_classes):
                    count = init_class_counts.get(i_class, 0)
                    lower = lower_threshold[i_bag, i_class]
                    upper = upper_threshold[i_bag, i_class]
                    
                    if count < lower:
                        violation = lower - count
                        total_violation += violation
                        violation_details[i_class] = -violation  # Negative means deficit
                    elif count > upper:
                        violation = count - upper
                        total_violation += violation
                        violation_details[i_class] = violation  # Positive means excess
                
                # Calculate initialization quality score (0-1, higher is better)
                violation_ratio = total_violation / (n_instances_in_bag + 1e-8)
                bag_init_quality[i_bag] = np.exp(-2.0 * violation_ratio)  # Exponential penalty
                
                # Instance-level confidence based on class distribution
                instance_confidence = np.ones(n_instances_in_bag)
                for idx in range(n_instances_in_bag):
                    class_label = init_labels[idx]
                    count = init_class_counts.get(class_label, 0)
                    lower = lower_threshold[i_bag, class_label]
                    upper = upper_threshold[i_bag, class_label]
                    
                    # Lower confidence for instances in violated classes
                    if count < lower:
                        # Under-represented class: low confidence for changing away
                        instance_confidence[idx] = 0.3 + 0.4 * (count / lower)
                    elif count > upper:
                        # Over-represented class: low confidence for keeping
                        excess_ratio = (count - upper) / upper
                        instance_confidence[idx] = max(0.2, 0.8 - excess_ratio)
                    else:
                        # Within bounds: high confidence
                        instance_confidence[idx] = 0.9
                
                instance_init_confidence.append(instance_confidence)
                
                # Adaptive correction based on violation severity
                if violation_ratio > 0.2:  # More than 20% violation (less strict than before)
                    # Smart correction: prioritize based on violation details
                    corrected_labels = init_labels.copy()
                    
                    # Calculate desired changes per class
                    changes_needed = {}
                    for i_class, violation in violation_details.items():
                        changes_needed[i_class] = -violation  # Negative violation means we need more
                    
                    # Sort instances by confidence (ascending) for potential changes
                    change_candidates = []
                    for idx in range(n_instances_in_bag):
                        if instance_confidence[idx] < 0.7:  # Low confidence instances
                            change_candidates.append((idx, instance_confidence[idx]))
                    
                    change_candidates.sort(key=lambda x: x[1])
                    
                    # Apply targeted corrections
                    n_corrections = min(len(change_candidates), int(total_violation * 0.4))
                    
                    for i in range(n_corrections):
                        idx, _ = change_candidates[i]
                        current_class = corrected_labels[idx]
                        
                        # Find best target class based on needs
                        best_target = None
                        best_score = -np.inf
                        
                        for target_class in range(n_classes):
                            if target_class == current_class:
                                continue
                            
                            # Score based on how much this change helps
                            score = 0
                            if current_class in changes_needed:
                                score += changes_needed[current_class]  # Benefit of removing from current
                            if target_class in changes_needed:
                                score -= changes_needed[target_class]  # Benefit of adding to target
                            
                            if score > best_score:
                                best_score = score
                                best_target = target_class
                        
                        if best_target is not None:
                            corrected_labels[idx] = best_target
                            # Update needs
                            if current_class in changes_needed:
                                changes_needed[current_class] += 1
                            if best_target in changes_needed:
                                changes_needed[best_target] -= 1
                            # Update counts
                            init_class_counts[current_class] -= 1
                            if init_class_counts[current_class] == 0:
                                del init_class_counts[current_class]
                            init_class_counts[best_target] = init_class_counts.get(best_target, 0) + 1
                    
                    y.append(corrected_labels)
                else:
                    # Keep external init if constraints are mostly satisfied
                    y.append(init_labels.copy())
            
        # Initialize simple parameters
        # Check if using external initialization
        is_external_init = initial_y is not None
        
        # Adaptive temperature schedule based on initialization type
        if is_external_init:
            # For external init: conservative exploration (1.3x auto-init temperature)
            # External init already provides good starting point, so less exploration needed
            smooth_temp_start = 1.3   # Conservative increase for stability
            smooth_temp_end = 0.1     # Lower final temperature for precision
            gumbel_temp = 1.2         # Slightly reduced for external init
            # Use cosine schedule for smoother transition
            temp_decay_rate = None    # Use cosine schedule like auto-init
        else:
            # Original temperature schedule for auto-init
            smooth_temp_start = 1.0
            smooth_temp_end = 0.2
            gumbel_temp = 1.0
            temp_decay_rate = None    # Use cosine schedule
        
        # Temperature scheduling parameters
        temperature_cosine_period = 0.6  # Cosine annealing for first 60% of iterations
        
        # Instance-specific temperature scaling based on init confidence
        if is_external_init:
            # Use init confidence to scale temperature per instance
            instance_temp_scale = []
            for i_bag in range(n_bags):
                # Scale inversely with confidence: low confidence -> high temperature
                # Range: [0.8, 2.0] based on confidence [0.9, 0.2]
                bag_scale = 2.0 - 1.2 * instance_init_confidence[i_bag]
                # Further scale by bag quality
                bag_scale = bag_scale * (2.0 - bag_init_quality[i_bag])
                instance_temp_scale.append(bag_scale)
        else:
            # Default scaling for auto-init
            instance_temp_scale = [np.ones(len(bag)) for bag in bags]
        
        # Self-paced learning: track bag difficulty scores and convergence
        bag_difficulty = np.zeros(n_bags)
        bag_convergence = np.zeros(n_bags)  # Track how stable each bag's assignments are
        
        # Adaptive confidence-based freezing
        if is_external_init:
            confidence_threshold = 0.97  # Higher threshold for external init to allow more changes
            margin_lambda = 0.5   # Higher margin importance for external init
        else:
            confidence_threshold = 0.95  # Original for auto-init
            margin_lambda = 0.4   # Original margin importance
        
        # Fixed penalty parameters for constraints (more stable than adaptive)
        # Based on staged-AEM approach: consistent pressure throughout training
        constraint_boost_penalty = 3.0  # Penalty for moving from over-represented class
        constraint_deficit_bonus = 3.0  # Bonus for moving to under-represented class
        
        # Initialize MIL-Dropout parameters
        mil_dropout_k = 1  # Number of top instances to potentially drop
        mil_dropout_rate = 0.5  # Probability of dropping
        # Apply MIL-Dropout longer for external init to prevent early convergence
        mil_dropout_schedule = 0.4 if is_external_init else 0.2  # 40% for external, 20% for auto
        
        # Initialize IES (Instance-dependent Early Stopping) tracking
        ies_history = []  # Track probability and label history per bag
        for _ in range(n_bags):
            ies_history.append({
                'probs': [],
                'labels': []
            })
        
        for i_iter in range(max_iter):
            # Conservative Gumbel exploration to preserve good initial assignments
            if is_external_init:
                use_gumbel = (i_iter == 0)  # Only first iteration for external init
            else:
                use_gumbel = (i_iter == 0)  # First iteration only for auto-init
            
            # Temperature schedule based on initialization type
            if is_external_init and temp_decay_rate is not None:
                # Exponential decay for external init: τₜ = max(τ_min, τ₀ * γ^t)
                temperature = max(smooth_temp_end, smooth_temp_start * (temp_decay_rate ** i_iter))
            else:
                # Original cosine annealing for auto-init
                if i_iter < int(max_iter * temperature_cosine_period):
                    # Cosine annealing for exploration phase
                    cosine_progress = i_iter / (max_iter * temperature_cosine_period)
                    temperature = smooth_temp_end + 0.5 * (smooth_temp_start - smooth_temp_end) * (1 + np.cos(np.pi * cosine_progress))
                else:
                    # Linear decay for exploitation phase
                    linear_progress = (i_iter - int(max_iter * temperature_cosine_period)) / (max_iter - int(max_iter * temperature_cosine_period))
                    temperature = smooth_temp_end + (smooth_temp_start * 0.3 - smooth_temp_end) * (1 - linear_progress)
            
            # Override temperature for Gumbel case
            if use_gumbel:
                temperature = gumbel_temp
            
            # AEM schedule: adapt based on initialization type
            if is_external_init:
                # For external init: conservative AEM to preserve good initial assignments
                # Start with λ = 0.1 and gentle cosine decay
                if i_iter < int(max_iter * 0.7):  # First 70% of iterations
                    # Gentle cosine annealing from 0.1 to 0
                    aem_progress = i_iter / (max_iter * 0.7)
                    aem_lambda = 0.1 * 0.5 * (1 + np.cos(np.pi * aem_progress))
                else:
                    aem_lambda = 0.0  # No entropy regularization in final iterations
            else:
                # Original AEM schedule for auto-init
                if i_iter == 0:
                    aem_lambda = 0.05  # Low at start to establish baseline
                elif i_iter < int(max_iter * 0.7):
                    # Cosine schedule peaking at iteration 3-4
                    aem_progress = min(1.0, i_iter / 4.0)
                    aem_lambda = 0.4 * 0.5 * (1 - np.cos(np.pi * aem_progress))
                else:
                    # Gradual reduction in final iterations
                    aem_lambda = 0.1 * (1 - (i_iter - int(max_iter * 0.7)) / (max_iter - int(max_iter * 0.7)))
            
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
                
                # Apply MIL-Dropout during early iterations
                if i_iter < int(max_iter * mil_dropout_schedule):
                    probs = apply_mil_dropout(probs, dropout_k=mil_dropout_k, dropout_rate=mil_dropout_rate)
                
                # Update IES history
                ies_history[i_bag]['probs'].append(probs.copy())
                ies_history[i_bag]['labels'].append(y[i_bag].copy())
                
                # Get current class counts
                class_count_dict = pd.Series(y[i_bag]).value_counts().to_dict()
                
                # Track which instances have been reassigned in this iteration
                changed_indices = set()
                new_y = y[i_bag].copy()
                
                # Compute IES mask (instances that have converged)
                ies_mask = compute_instance_early_stopping_mask(
                    ies_history[i_bag]['probs'],
                    ies_history[i_bag]['labels'],
                    convergence_threshold=0.01,
                    min_history=3
                )
                
                # Adaptive confidence-based freezing with convergence tracking
                frozen_indices = set()
                # Delay freezing for external init to allow more exploration
                freeze_start_iter = 6 if is_external_init else 2  # Delay freezing to iteration 6 for external init
                if i_iter >= freeze_start_iter:
                    max_probs = np.max(probs, axis=1)
                    predicted_classes = np.argmax(probs, axis=1)
                    
                    # Adjust threshold based on bag convergence
                    adaptive_threshold = confidence_threshold - 0.02 * min(1.0, bag_convergence[i_bag] / 3.0)
                    
                    for idx in range(n_instances):
                        # Freeze if high confidence AND consistent with current assignment OR IES converged
                        if (max_probs[idx] > adaptive_threshold and y[i_bag][idx] == predicted_classes[idx]) or ies_mask[idx]:
                            frozen_indices.add(idx)
                
                # For binary classification, use exact k-of-n solver
                if n_classes == 2:
                    # Apply constraint slack based on initialization type
                    effective_lower_0 = lower_threshold[i_bag, 0]
                    effective_upper_0 = upper_threshold[i_bag, 0]
                    effective_lower_1 = lower_threshold[i_bag, 1]
                    effective_upper_1 = upper_threshold[i_bag, 1]
                    
                    # More slack for external init to allow exploration
                    if is_external_init and i_iter < 4:
                        slack_ratio = 0.95 if i_iter < 2 else 0.97  # 5% slack first 2 iters, then 3%
                        effective_lower_0 = effective_lower_0 * slack_ratio
                        effective_upper_0 = effective_upper_0 * (2.0 - slack_ratio)
                        effective_lower_1 = effective_lower_1 * slack_ratio
                        effective_upper_1 = effective_upper_1 * (2.0 - slack_ratio)
                    elif i_iter == 0:  # Auto-init: only first iteration
                        effective_lower_0 = effective_lower_0 * 0.98  # 2% slack
                        effective_upper_0 = effective_upper_0 * 1.02  # 2% slack
                        effective_lower_1 = effective_lower_1 * 0.98  # 2% slack
                        effective_upper_1 = effective_upper_1 * 1.02  # 2% slack
                    
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
                        
                        # Use Gumbel-Top-k only during first iteration
                        # use_gumbel already set above
                        
                        # Calculate instance-specific temperatures
                        active_temperatures = []
                        for idx in active_indices:
                            inst_temp = temperature * instance_temp_scale[i_bag][idx]
                            active_temperatures.append(inst_temp)
                        
                        # Use mean temperature for soft assignment (weighted by instance confidence)
                        if is_external_init:
                            # Weight by init confidence for better adaptation
                            active_confidences = instance_init_confidence[i_bag][active_indices]
                            weighted_temp = np.sum(active_temperatures * active_confidences) / (np.sum(active_confidences) + 1e-8)
                        else:
                            weighted_temp = np.mean(active_temperatures)
                        
                        # Always use soft assignment for differentiability
                        soft_weights = soft_k_of_n_assignment(
                            active_probs, k_min_active, k_max_active, 
                            temperature=weighted_temp,  # Use weighted temperature
                            margin_scores=margin_scores, aem_scores=aem_scores,
                            use_gumbel=use_gumbel
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
                        
                        # Apply constraint slack based on initialization type
                        if is_external_init and i_iter < 4:
                            # More slack for external init
                            slack_ratio = 0.95 if i_iter < 2 else 0.97
                            effective_lower = lower_threshold[i_bag, i_class] * slack_ratio
                        elif i_iter == 0:
                            # 2% slack in first iteration for auto-init
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
                                    
                                    # Fixed boost/penalty based on constraint satisfaction
                                    current_class = new_y[idx]
                                    current_class_count = class_count_dict.get(current_class, 0)
                                    if current_class_count > upper_threshold[i_bag, current_class]:
                                        # Fixed boost for moving away from over-represented class
                                        score *= constraint_boost_penalty
                                    elif current_class_count <= lower_threshold[i_bag, current_class]:
                                        # Fixed penalty for moving away from under-represented class
                                        score *= (1.0 / constraint_boost_penalty)
                                    
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
                        if is_external_init and i_iter < 4:
                            # More slack for external init
                            slack_ratio = 0.95 if i_iter < 2 else 0.97
                            effective_upper = upper_threshold[i_bag, i_class] * (2.0 - slack_ratio)
                        elif i_iter == 0:
                            effective_upper = upper_threshold[i_bag, i_class] * 1.02  # 2% slack for auto-init
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
                                    
                                    # Fixed constraint bonus/penalty
                                    c_count = class_count_dict.get(c, 0)
                                    if c_count < lower_threshold[i_bag, c]:
                                        # Fixed bonus for under-represented class
                                        scores[c] += constraint_deficit_bonus
                                    elif c_count < upper_threshold[i_bag, c]:
                                        scores[c] += 1.0
                                    else:
                                        # Fixed penalty for over-represented class
                                        scores[c] -= constraint_boost_penalty
                                
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
                    bag_convergence[i_bag] = 0  # Reset convergence counter
                else:
                    bag_convergence[i_bag] += 1  # Increment convergence counter
                
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
                    
                    # Add differentiable count constraint loss using LogSumExp
                    if n_classes == 2 and i_class == 1:
                        # For binary case, compute soft count loss using LogSumExp
                        target_k = (lower_threshold[i_bag, 1] + upper_threshold[i_bag, 1]) / 2
                        # Use LogSumExp for smoother gradient
                        lse_count = logsumexp_smooth_cardinality(probs[:, 1], kappa=25.0)
                        count_loss = 0.3 * abs(lse_count - target_k)  # Slightly higher weight for external init
                
                bag_difficulty[i_bag] = constraint_violation + count_loss

            if debug:
                # Check which operator is being used
                if use_gumbel:
                    operator = "Gumbel-Top-k"
                else:
                    operator = "Smooth-Top-k"
                print(f"iter: {i_iter}, operator: {operator}, temperature: {temperature:.3f}, aem_lambda: {aem_lambda:.3f}")
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
