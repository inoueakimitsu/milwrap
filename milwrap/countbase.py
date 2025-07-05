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


def compute_instance_uncertainty(bag_probs: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute uncertainty scores for instances based on prediction entropy.
    
    Args:
        bag_probs: Instance probability matrix of shape (n_instances, n_classes).
        
    Returns:
        Uncertainty scores for each instance (higher = more uncertain).
    """
    if len(bag_probs.shape) != 2 or bag_probs.shape[0] == 0:
        return np.zeros(bag_probs.shape[0])
    
    # Compute entropy of predictions as uncertainty measure
    # H(p) = -Σ p_i * log(p_i)
    eps = 1e-10
    entropy = -np.sum(bag_probs * np.log(bag_probs + eps), axis=1)
    
    # Normalize by log(n_classes) to get values in [0, 1]
    n_classes = bag_probs.shape[1]
    if n_classes > 1:
        normalized_entropy = entropy / np.log(n_classes)
    else:
        normalized_entropy = np.zeros_like(entropy)
    
    return normalized_entropy


def compute_soft_count_loss(predicted_counts: NDArray[np.float64], lower_threshold: NDArray[np.float64], 
                          upper_threshold: NDArray[np.float64], temperature: float = 0.25) -> float:
    """Compute differentiable soft count loss for constraint violations.
    
    Args:
        predicted_counts: Predicted soft counts for each class.
        lower_threshold: Lower threshold for each class.
        upper_threshold: Upper threshold for each class.
        temperature: Temperature for soft counting.
        
    Returns:
        Soft count loss value.
    """
    # Lower bound violations: ReLU(lower - count)²
    lower_violations = np.maximum(0, lower_threshold - predicted_counts)
    lower_loss = np.sum(lower_violations ** 2)
    
    # Upper bound violations: ReLU(count - upper)²
    upper_violations = np.maximum(0, predicted_counts - upper_threshold)
    upper_loss = np.sum(upper_violations ** 2)
    
    return (lower_loss + upper_loss) * temperature


def compute_diversity_score(selected_instances: NDArray[np.float64], embeddings: NDArray[np.float64]) -> float:
    """Compute diversity score for selected instances using cosine distance.
    
    Args:
        selected_instances: Binary array indicating selected instances.
        embeddings: Feature embeddings or probabilities for instances.
        
    Returns:
        Diversity score (higher = more diverse).
    """
    selected_idx = np.where(selected_instances > 0)[0]
    if len(selected_idx) < 2:
        return 1.0  # Maximum diversity for single/no selection
    
    # Use probability distributions as embeddings
    selected_embeddings = embeddings[selected_idx]
    
    # Compute pairwise cosine similarities
    norms = np.linalg.norm(selected_embeddings, axis=1, keepdims=True) + 1e-10
    normalized = selected_embeddings / norms
    similarities = np.dot(normalized, normalized.T)
    
    # Average off-diagonal similarities (lower = more diverse)
    n = len(selected_idx)
    mask = 1 - np.eye(n)
    avg_similarity = np.sum(similarities * mask) / (n * (n - 1) + 1e-10)
    
    # Convert to diversity score
    diversity = 1.0 - avg_similarity
    return diversity


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


def exact_k_of_n_assignment(probs, k_min, k_max, current_assignments=None, margin_scores=None, aem_scores=None, uncertainty_scores=None, embeddings=None, diversity_weight=0.1):
    """Solve exact k-of-n assignment problem with optional margin weighting.
    
    Find optimal assignment of instances to positive class such that
    k_min <= sum(assignments) <= k_max, maximizing sum(log(p_i))
    
    Args:
        probs: probabilities for positive class (n_instances,)
        k_min: minimum number of positive instances
        k_max: maximum number of positive instances
        current_assignments: current assignments (optional)
        margin_scores: margin-based scores for weighting (optional)
        aem_scores: AEM scores for weighting (optional)
        uncertainty_scores: uncertainty scores for weighting (optional)
        embeddings: instance embeddings for diversity computation (optional)
        diversity_weight: weight for diversity bonus (optional)
        
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
    
    # Apply AEM scores if provided
    if aem_scores is not None and len(aem_scores) == len(probs):
        log_odds = log_odds + 0.05 * aem_scores  # Small weight for AEM
    
    # Apply uncertainty penalty if provided (prefer certain predictions)
    if uncertainty_scores is not None and len(uncertainty_scores) == len(probs):
        # Adaptive uncertainty penalty: start small to avoid early bias
        uncertainty_weight = 0.05 + 0.15 * min(1.0, len(uncertainty_scores) / 100.0)
        log_odds = log_odds - uncertainty_weight * uncertainty_scores
    
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
            
        # Initialize simple parameters
        n_bags = len(bags)
        
        # Adaptive temperature decay
        temperature_start = 2.0  # Higher starting temperature for exploration
        temperature_end = 0.1    # Final temperature for precise assignments
        temperature_current = temperature_start
        
        # Self-paced learning: track bag difficulty scores
        bag_difficulty = np.zeros(n_bags)
        
        # Track convergence metrics
        prev_total_changes = n_bags * 10  # Initialize with high value
        stagnation_counter = 0
        
        # Simple confidence-based freezing (fixed threshold)
        confidence_threshold = 0.97  # Higher threshold for more selective freezing
        margin_lambda = 0.25  # Standard margin importance
        
        for i_iter in range(max_iter):
            # Initialize total changes counter for this iteration
            total_changes = 0
            
            # Adaptive temperature decay based on convergence
            if i_iter > 0 and 'prev_iter_changes' in locals():
                # Check convergence status
                change_ratio = prev_iter_changes / (prev_total_changes + 1e-6)
                
                if change_ratio > 0.9:  # Many changes, slow down cooling
                    temperature_decay_rate = 0.95
                    stagnation_counter = 0
                elif change_ratio < 0.1:  # Few changes, might be stuck
                    stagnation_counter += 1
                    if stagnation_counter >= 2:  # Reheat if stuck
                        temperature_current = min(temperature_current * 1.2, temperature_start * 0.5)
                        stagnation_counter = 0
                    else:
                        temperature_decay_rate = 0.85  # Faster cooling
                else:  # Normal convergence
                    temperature_decay_rate = 0.9
                    stagnation_counter = 0
                
                # Apply decay
                temperature_current = max(temperature_end, temperature_current * temperature_decay_rate)
                prev_total_changes = total_changes
            
            temperature = temperature_current
            
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
                
                # Uncertainty-aware confidence-based freezing
                frozen_indices = set()
                if i_iter >= 3:  # Start freezing after initial exploration
                    max_probs = np.max(probs, axis=1)
                    predicted_classes = np.argmax(probs, axis=1)
                    uncertainty_scores = compute_instance_uncertainty(probs)
                    
                    for idx in range(n_instances):
                        # Freeze if high confidence AND low uncertainty AND consistent with current assignment
                        if (max_probs[idx] > confidence_threshold and 
                            uncertainty_scores[idx] < 0.15 and  # Low uncertainty threshold
                            y[i_bag][idx] == predicted_classes[idx]):
                            frozen_indices.add(idx)
                
                # For binary classification, use exact k-of-n solver
                if n_classes == 2:
                    # Progressive constraint tightening (curriculum learning)
                    constraint_progress = min(1.0, i_iter / 5.0)  # Full constraints by iter 5
                    slack_factor = 1.0 + 0.05 * (1.0 - constraint_progress)  # 5% to 0% slack
                    
                    effective_lower_0 = lower_threshold[i_bag, 0] * (2.0 - slack_factor)
                    effective_upper_0 = upper_threshold[i_bag, 0] * slack_factor
                    effective_lower_1 = lower_threshold[i_bag, 1] * (2.0 - slack_factor)
                    effective_upper_1 = upper_threshold[i_bag, 1] * slack_factor
                    
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
                        
                        # Compute AEM scores for the bag (only after initial iterations)
                        aem_scores_active = None
                        if i_iter >= 2:  # Start using AEM after warm-up
                            # Adaptive lambda_entropy based on multiple factors
                            base_lambda = 0.08 if initial_y is None else 0.04  # Slightly higher base
                            
                            # Factor 1: Constraint violations (higher violations = more entropy needed)
                            constraint_penalty = bag_difficulty[i_bag] / (n_instances + 1e-6)
                            constraint_factor = 1.0 + 0.5 * min(constraint_penalty, 1.0)
                            
                            # Factor 2: Convergence status (stagnation = more entropy)
                            convergence_factor = 1.0
                            if stagnation_counter > 0:
                                convergence_factor = 1.2  # Boost entropy when stuck
                            
                            # Factor 3: Bag size (larger bags need more entropy)
                            size_factor = 1.0 + 0.2 * np.log(max(1, n_instances / 20))
                            
                            # Combine factors with time decay
                            time_decay = 0.5 + 0.5 * (1 - i_iter / max_iter)  # Slower decay
                            lambda_entropy = base_lambda * constraint_factor * convergence_factor * size_factor * time_decay
                            
                            # Compute AEM scores for all instances in the bag
                            bag_aem_scores = compute_attention_entropy_maximization(probs, lambda_entropy)
                            aem_scores_active = bag_aem_scores[active_indices]
                        
                        # Compute uncertainty scores for active instances
                        uncertainty_scores = compute_instance_uncertainty(probs)
                        uncertainty_scores_active = uncertainty_scores[active_indices]
                        
                        # Solve exact assignment for active instances with all scores
                        optimal_assignment_active = exact_k_of_n_assignment(
                            active_probs, k_min_active, k_max_active, 
                            margin_scores=margin_scores, 
                            aem_scores=aem_scores_active,
                            uncertainty_scores=uncertainty_scores_active
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
                        
                        # Progressive constraint tightening
                        constraint_progress = min(1.0, i_iter / 5.0)
                        slack_factor = 1.0 + 0.05 * (1.0 - constraint_progress)
                        effective_lower = lower_threshold[i_bag, i_class] * (2.0 - slack_factor)
                        
                        if class_count < effective_lower:
                            # Need more instances of this class
                            deficit = int(effective_lower - class_count)
                            
                            # Find best candidates: high probability for this class, not already changed, not frozen
                            candidates = []
                            uncertainty_scores = compute_instance_uncertainty(probs)
                            
                            for idx in range(n_instances):
                                if idx not in changed_indices and idx not in frozen_indices and new_y[idx] != i_class:
                                    # Base score from probability
                                    base_score = probs[idx, i_class]
                                    # Add margin-aware confidence
                                    margin_conf = compute_margin_aware_confidence(probs[idx], i_class, margin_lambda=margin_lambda)
                                    # Incorporate uncertainty (prefer certain predictions)
                                    uncertainty_penalty = uncertainty_scores[idx] * 0.2
                                    
                                    # Add diversity bonus if this instance is different from already selected ones
                                    diversity_bonus = 0.0
                                    if len(changed_indices) > 0 and i_iter >= 3:
                                        # Simple diversity: average distance to changed instances
                                        changed_probs = [probs[changed_idx] for changed_idx in changed_indices if changed_idx < n_instances]
                                        if changed_probs:
                                            avg_similarity = np.mean([np.dot(probs[idx], cp) / (np.linalg.norm(probs[idx]) * np.linalg.norm(cp) + 1e-10) 
                                                                    for cp in changed_probs])
                                            diversity_bonus = 0.1 * (1.0 - avg_similarity)
                                    
                                    score = base_score + 0.1 * (margin_conf - base_score) - uncertainty_penalty + diversity_bonus
                                    
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
                        
                        # Progressive constraint tightening for upper threshold
                        effective_upper = upper_threshold[i_bag, i_class] * slack_factor
                        
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
                print(f"iter: {i_iter}, temperature: {temperature:.3f}")
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

            # Track changes for adaptive temperature
            prev_iter_changes = total_changes
            
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
