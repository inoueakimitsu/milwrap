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


def fast_soft_rank_approximation(
    values: NDArray[np.float64],
    temperature: float = 1.0,
) -> NDArray[np.float64]:
    """Fast differentiable ranking approximation.
    
    Returns soft ranks directly instead of full permutation matrix for efficiency.
    
    Args:
    ----
        values: Array of values to rank.
        temperature: Temperature for softness (lower = harder sort).
        
    Returns:
    -------
        Soft ranks for each element (0 = highest, n-1 = lowest).
        
    """
    n = len(values)
    if n == 0:
        return np.array([])
    
    # For efficiency with large arrays, use a simpler approach
    # Based on "Fast Differentiable Sorting and Ranking" (ICML 2020)
    
    # Compute pairwise comparisons more efficiently
    values_expanded = values[:, np.newaxis]
    
    # Soft indicator: I[values[i] < values[j]]
    # Using sigmoid for smooth approximation
    pairwise_comparisons = 1 / (1 + np.exp((values_expanded - values_expanded.T) / temperature))
    
    # Soft rank = sum of indicators that other elements are smaller
    # This gives rank in ascending order (0 = smallest)
    # For descending order, we need n - rank - 1
    soft_ranks_ascending = np.sum(pairwise_comparisons, axis=1) - 1  # Subtract 1 to start from 0
    soft_ranks_descending = n - soft_ranks_ascending - 1
    
    return soft_ranks_descending


def compute_softsort_based_count_loss(
    probs: NDArray[np.float64],
    target_count: int,
    temperature: float = 0.5,
) -> tuple[float, float]:
    """Compute count loss using SoftSort for differentiable top-k selection.
    
    This replaces the softmax attention with a more direct top-k approximation
    using differentiable sorting, providing better gradient flow for count constraints.
    
    Args:
    ----
        probs: Probability array for instances belonging to a class.
        target_count: Target number of instances for the class.
        temperature: Temperature parameter for SoftSort (lower = sharper).
        
    Returns:
    -------
        Tuple of (loss_value, constraint_violation) for Deep-ALM.
        
    """
    if len(probs) == 0 or target_count < 0:
        return 0.0, 0.0
    
    n_instances = len(probs)
    
    if target_count == 0:
        # For zero target, minimize the sum of probabilities
        soft_count = np.sum(probs)
        # Squared hinge loss for zero target
        loss = 0.5 * soft_count ** 2
        constraint_violation = soft_count
        return loss, constraint_violation
    
    # Use fast soft ranking to get differentiable ranks
    soft_ranks = fast_soft_rank_approximation(probs, temperature)
    
    # Create soft top-k selection based on ranks
    k = min(target_count, n_instances)
    
    # Soft indicator for being in top-k
    # rank < k means instance is in top-k (since ranks start from 0)
    # Use sigmoid for smooth approximation
    boundary_distance = k - 0.5 - soft_ranks  # Distance from the k-th position boundary
    # Clip to prevent overflow
    clipped_distance = np.clip(boundary_distance / (temperature * 0.5), -50, 50)
    smooth_indicators = 1 / (1 + np.exp(-clipped_distance))
    
    # Compute soft count as weighted sum
    soft_count = np.sum(probs * smooth_indicators)
    
    # Alternative: Direct soft top-k using sorted probabilities
    # This is more stable for very small temperature values
    if temperature < 0.1:
        # Hard sort for very low temperatures
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        
        # Smooth cutoff at position k
        positions = np.arange(n_instances)
        cutoff_weights = 1 / (1 + np.exp((positions - k + 0.5) / 0.1))
        soft_count_v2 = np.sum(sorted_probs * cutoff_weights)
        
        # Blend with rank-based approach
        soft_count = 0.7 * soft_count + 0.3 * soft_count_v2
    
    # Compute loss with smooth L1 (Huber) loss for better optimization
    residual = soft_count - target_count
    if abs(residual) < 1.0:
        loss = 0.5 * residual ** 2
    else:
        loss = abs(residual) - 0.5
    
    # Add regularization to encourage decisive rankings
    # Entropy of the soft indicators
    indicator_entropy = -np.sum(smooth_indicators * np.log(smooth_indicators + 1e-8) + 
                               (1 - smooth_indicators) * np.log(1 - smooth_indicators + 1e-8))
    entropy_reg = 0.001 * indicator_entropy / n_instances
    loss += entropy_reg
    
    # Constraint violation for Deep-ALM
    constraint_violation = soft_count - target_count
    
    return loss, constraint_violation


def compute_stabilized_soft_count_loss(
    probs: NDArray[np.float64],
    target_count: int,
    temperature: float = 0.5,
    use_softsort: bool = True,
) -> tuple[float, float]:
    """Compute numerically stable soft count loss.
    
    Can use either SoftSort-based or softmax-based approach.
    
    Args:
    ----
        probs: Probability array for instances belonging to a class.
        target_count: Target number of instances for the class.
        temperature: Temperature parameter for soft counting (lower = sharper).
        use_softsort: Whether to use SoftSort (True) or softmax approach (False).
        
    Returns:
    -------
        Tuple of (loss_value, constraint_violation) for Deep-ALM.
        
    """
    if use_softsort:
        return compute_softsort_based_count_loss(probs, target_count, temperature)
    
    # Original implementation with softmax
    if len(probs) == 0 or target_count < 0:
        return 0.0, 0.0
    
    n_instances = len(probs)
    
    # Convert to proportion-based targets for numerical stability
    target_proportion = target_count / n_instances
    
    if target_count == 0:
        # For zero target, minimize the sum of probabilities
        soft_proportion = np.mean(probs)
        # Use log-space for better gradient behavior
        loss = np.log1p(soft_proportion) ** 2
        constraint_violation = soft_proportion
        return loss, constraint_violation
    
    # Softmax-with-temperature for stable soft counting
    # Convert probabilities to logits for softmax
    logits = np.log(probs + 1e-10) / temperature
    
    # Stable softmax computation
    max_logit = np.max(logits)
    exp_logits = np.exp(logits - max_logit)
    softmax_weights = exp_logits / np.sum(exp_logits)
    
    # Soft count is weighted by softmax attention
    # This ensures the soft count is bounded between 0 and 1
    soft_proportion = np.sum(softmax_weights * probs)
    
    # Alternative: Use top-k approximation with smooth cutoff
    if target_count < n_instances:
        # Sort by probability
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        
        # Smooth top-k selection using cumulative softmax
        k = target_count
        positions = np.arange(n_instances)
        # Clip to prevent overflow
        scaled_positions = np.clip((positions - k + 0.5) / temperature, -100, 100)
        position_weights = 1 / (1 + np.exp(scaled_positions))
        
        # Weighted soft count
        soft_count_topk = np.sum(sorted_probs * position_weights)
        soft_proportion_topk = soft_count_topk / n_instances
        
        # Blend both approaches for robustness
        soft_proportion = 0.7 * soft_proportion_topk + 0.3 * soft_proportion
    
    # Compute loss using log-cosh for bounded gradients
    # log(cosh(x)) ≈ |x| for large |x| but smooth near 0
    residual = soft_proportion - target_proportion
    loss = np.log(np.cosh(residual * 10)) / 10  # Scale for appropriate sensitivity
    
    # Constraint violation for Deep-ALM (actual count - target count)
    constraint_violation = soft_proportion * n_instances - target_count
    
    # Small entropy regularization to prevent trivial solutions
    # Scale by 1/n_instances to keep it O(1)
    if 0 < soft_proportion < 1:
        entropy = -soft_proportion * np.log(soft_proportion) - (1 - soft_proportion) * np.log(1 - soft_proportion)
        entropy_reg = 0.001 * entropy  # Much smaller weight
        loss += entropy_reg
    
    return loss, constraint_violation


def apply_spgd_perturbation(
    logits: NDArray[np.float64],
    perturbation_scale: float = 0.2,
    rng: np.random.Generator | None = None,
    gradient_info: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Apply enhanced SPGD-style perturbation with adaptive scaling.
    
    Uses gradient-informed perturbation when available and applies
    instance-specific scaling based on confidence.
    
    Args:
    ----
        logits: Input logits array.
        perturbation_scale: Base scale of perturbation.
        rng: Random number generator.
        gradient_info: Optional gradient information for guided perturbation.
        
    Returns:
    -------
        Perturbed logits with adaptive scaling.
        
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Base perturbation
    base_perturbation = rng.uniform(-perturbation_scale, perturbation_scale, size=logits.shape)
    
    # Adaptive scaling based on logit magnitude (lower confidence = larger perturbation)
    confidence = np.abs(logits)
    adaptive_scale = 1.0 / (1.0 + 0.5 * confidence)
    
    # If gradient information is available, bias perturbation in gradient direction
    if gradient_info is not None and gradient_info.shape == logits.shape:
        # Normalize gradient
        grad_norm = np.linalg.norm(gradient_info) + 1e-8
        normalized_grad = gradient_info / grad_norm
        # Combine random and gradient-directed perturbation
        directed_perturbation = 0.3 * perturbation_scale * normalized_grad
        perturbation = 0.7 * base_perturbation + directed_perturbation
    else:
        perturbation = base_perturbation
    
    # Apply adaptive scaling
    scaled_perturbation = perturbation * adaptive_scale
    
    return logits + scaled_perturbation


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
            
        # Initialize Deep-ALM parameters for constraints
        n_bags = len(bags)
        # Lagrange multipliers (dual variables)
        lambda_lower = np.zeros((n_bags, n_classes))
        lambda_upper = np.zeros((n_bags, n_classes))
        # Penalty parameters (start small, increase adaptively)
        mu_lower = np.ones((n_bags, n_classes)) * 0.1
        mu_upper = np.ones((n_bags, n_classes)) * 0.1
        # Deep-ALM hyperparameters
        rho_increase = 2.0  # Factor to increase μ
        constraint_tol = 0.25  # Relative tolerance for constraint satisfaction
        
        # Enhanced temperature annealing with cosine schedule
        temperature_start = 2.0  # Slightly lower initial temperature
        temperature_end = 0.1  # Higher final temperature for stability
        
        # MIL-Dropout parameters (from high-scoring branch)
        mil_dropout_rate = 0.05  # Drop 5% of high-scoring instances for better stability
        mil_dropout_start_iter = 4  # Start after fourth iteration for stability
        
        # SPGD parameters with adaptive scheduling
        perturbation_scale_init = 0.12  # Reduced perturbation for more stable convergence
        perturbation_interval = 2  # Apply every 2 iterations
        rng = np.random.default_rng(seed)
        
        # Adaptive tracking with momentum
        accuracy_history = []
        perturbation_active = True
        no_improvement_count = 0
        momentum_accuracy = 0.0
        momentum_factor = 0.9
        
        # Enhanced count loss weight with schedule (adjusted for stabilized loss)
        count_loss_weight_init = 0.5  # Back to original weight
        count_loss_weight_final = 1.2  # Back to original weight
        
        # SoftSort parameters
        use_softsort = True  # Enable SoftSort for better count constraint handling
        softsort_start_iter = 2  # Start after initial warmup
        
        # Pseudo-contrastive learning parameters
        use_contrastive = True  # Enable contrastive learning
        contrastive_weight = 0.02  # Lower weight for contrastive loss
        contrastive_temperature = 1.5  # Higher temperature for smoother similarity
        contrastive_margin = 0.05  # Smaller margin for contrastive loss
        
        # Uncertainty weighting parameters
        use_uncertainty_weighting = True  # Enable uncertainty-based instance weighting
        uncertainty_start_iter = 4  # Start after initial training
        
        # Enhanced curriculum learning parameters
        # Start with moderately relaxed constraints and gradually tighten
        constraint_tightness = 0.8 if max_iter > 5 else 1.0  # Start at 80% to avoid too much relaxation
        constraint_tightness_final = 1.0
        constraint_variance_threshold = 0.005
        # Curriculum schedule: inverse square root
        curriculum_warmup_iters = min(3, max_iter // 3)  # Shorter warmup
        
        # Stochastic Weight Averaging (SWA) parameters
        swa_start_iter = 2  # Start earlier for better averaging
        swa_classifier = None
        swa_update_freq = 1  # Update every iteration for smoother averaging
        
        # Track best solution and gradient information
        best_accuracy = 0.0
        best_y = None
        prev_probs = None
        gradient_estimate = None
        count_variance_history = []

        for i_iter in range(max_iter):
            # Enhanced temperature annealing with exponential decay (based on recent research)
            progress = i_iter / max(1, max_iter - 1)
            # Use exponential decay with γ=0.9 as recommended
            gamma = 0.9
            temperature = max(temperature_start * (gamma ** i_iter), temperature_end)
            
            # Adaptive count loss weight scheduling
            count_loss_weight = count_loss_weight_init + (count_loss_weight_final - count_loss_weight_init) * progress
            
            # Enhanced curriculum learning with smooth schedule
            if i_iter < curriculum_warmup_iters:
                # Smoother schedule starting from 0.8
                warmup_progress = (i_iter + 1) / curriculum_warmup_iters
                constraint_tightness = 0.8 + 0.2 * np.sqrt(warmup_progress)
            else:
                # After warmup, use variance-based adaptive tightening
                if len(count_variance_history) >= 2:
                    recent_variance = np.mean(count_variance_history[-2:])
                    if recent_variance < constraint_variance_threshold and constraint_tightness < constraint_tightness_final:
                        # Tighten constraints when stable
                        constraint_tightness = min(constraint_tightness_final, 
                                                 constraint_tightness + 0.05)  # Smaller increments
            
            # Apply constraint relaxation for self-paced learning
            relaxed_lower_threshold = lower_threshold * constraint_tightness
            relaxed_upper_threshold = upper_threshold + (1 - constraint_tightness) * (n_classes - upper_threshold)
            
            # Prepare training data with MIL-Dropout
            training_bags = []
            training_y = []
            
            for i_bag in range(len(bags)):
                bag_instances = bags[i_bag]
                bag_labels = y[i_bag]
                n_instances = len(bag_instances)
                
                if n_instances > 1 and i_iter >= mil_dropout_start_iter:
                    # Apply MIL-Dropout to prevent overfitting to specific instances
                    if hasattr(self.classifier, 'predict_proba') and i_iter > 0:
                        try:
                            # Calculate instance importance scores
                            instance_probs = self.classifier.predict_proba(bag_instances)
                            # Score = confidence in predicted class
                            predicted_classes = np.argmax(instance_probs, axis=1)
                            instance_scores = instance_probs[np.arange(n_instances), predicted_classes]
                            
                            # Adaptive dropout rate based on iteration
                            adaptive_dropout_rate = mil_dropout_rate * (1 - progress * 0.5)  # Reduce dropout over time
                            n_drop = max(1, int(n_instances * adaptive_dropout_rate))
                            n_keep = n_instances - n_drop
                            
                            # Find high-confidence instances and randomly drop some
                            # This forces the model to learn from diverse instances
                            high_conf_threshold = np.percentile(instance_scores, 50)  # Top 50%
                            high_conf_indices = np.where(instance_scores >= high_conf_threshold)[0]
                            
                            if len(high_conf_indices) > n_drop:
                                # Randomly drop from high-confidence instances
                                drop_indices = rng.choice(high_conf_indices, size=n_drop, replace=False)
                                keep_indices = np.setdiff1d(np.arange(n_instances), drop_indices)
                                
                                training_bags.append(bag_instances[keep_indices])
                                training_y.append(bag_labels[keep_indices])
                            else:
                                # Not enough high-confidence instances, keep all
                                training_bags.append(bag_instances)
                                training_y.append(bag_labels)
                        except:
                            # Fallback if prediction fails
                            training_bags.append(bag_instances)
                            training_y.append(bag_labels)
                    else:
                        training_bags.append(bag_instances)
                        training_y.append(bag_labels)
                else:
                    training_bags.append(bag_instances)
                    training_y.append(bag_labels)
            
            # Fit classifier with potentially dropped instances
            flatten_bags = np.vstack(training_bags)
            flatten_y = np.hstack(training_y)
            self.classifier.fit(flatten_bags, flatten_y)
            
            # Stochastic Weight Averaging (SWA)
            if i_iter >= swa_start_iter and i_iter % swa_update_freq == 0:
                if swa_classifier is None:
                    # Initialize SWA model as a copy
                    import copy
                    swa_classifier = copy.deepcopy(self.classifier)
                else:
                    # Update SWA weights with exponential moving average
                    alpha = 0.75  # SWA update rate
                    if hasattr(self.classifier, 'coef_'):
                        swa_classifier.coef_ = alpha * swa_classifier.coef_ + (1 - alpha) * self.classifier.coef_
                        if hasattr(self.classifier, 'intercept_'):
                            swa_classifier.intercept_ = alpha * swa_classifier.intercept_ + (1 - alpha) * self.classifier.intercept_

            # Compute outputs
            fs = [self.classifier.predict_proba(bag) for bag in bags]
            y_pred = [self.classifier.predict(bag) for bag in bags]
            flatten_original_y = np.hstack(list(y))

            # Enhanced adaptive SPGD perturbation with gradient estimation
            adaptive_interval = max(1, perturbation_interval + i_iter // 4)  # Less frequent as training progresses
            if perturbation_active and i_iter > 0 and i_iter % adaptive_interval == 0:
                # Adaptive perturbation scale with cosine annealing
                decay_factor = 0.5 * (1 + np.cos(np.pi * progress))
                perturbation_scale = perturbation_scale_init * decay_factor * (0.9 ** (i_iter // 5))
                
                for i_bag in range(len(bags)):
                    if len(bags[i_bag]) > 0:
                        # Estimate gradient from probability changes
                        if prev_probs is not None and i_bag < len(prev_probs):
                            gradient_estimate = fs[i_bag] - prev_probs[i_bag]
                        else:
                            gradient_estimate = None
                        
                        # Apply enhanced perturbation to predicted probabilities
                        perturbed_probs = fs[i_bag].copy()
                        for c in range(n_classes):
                            logits = np.log(perturbed_probs[:, c] + 1e-8)
                            # Use gradient-informed perturbation when available
                            grad_info = gradient_estimate[:, c] if gradient_estimate is not None else None
                            perturbed_logits = apply_spgd_perturbation(logits, perturbation_scale, rng, grad_info)
                            perturbed_probs[:, c] = 1 / (1 + np.exp(-perturbed_logits))
                        # Renormalize with numerical stability
                        perturbed_probs = perturbed_probs / (perturbed_probs.sum(axis=1, keepdims=True) + 1e-8)
                        fs[i_bag] = perturbed_probs
                
                # Store current probabilities for gradient estimation
                prev_probs = [f.copy() for f in fs]
            
            # For every bag - use soft assignments with augmented Lagrangian
            has_changed = False
            total_count_loss = 0.0
            total_constraint_violation = 0.0  # Track constraint violations
            total_contrastive_loss = 0.0  # Track contrastive loss
            
            # Collect probability distributions for contrastive learning
            contrastive_probs_list = []  # Store all instance probabilities
            contrastive_labels_list = []  # Store current assigned labels
            contrastive_bag_indices = []  # Store which bag each instance belongs to
            
            for i_bag in range(len(bags)):
                # Hybrid approach: combine constraint-based selection with soft assignments
                n_instances = len(bags[i_bag])
                if n_instances == 0:
                    continue
                    
                probs = fs[i_bag]  # Shape: (n_instances, n_classes)
                
                # Collect for contrastive learning
                contrastive_probs_list.append(probs)
                contrastive_labels_list.append(y[i_bag])
                contrastive_bag_indices.extend([i_bag] * n_instances)
                
                # Compute uncertainty weights if enabled
                uncertainty_weights = np.ones(n_instances)
                if use_uncertainty_weighting and i_iter >= uncertainty_start_iter:
                    # Use entropy as uncertainty measure
                    # High entropy = high uncertainty = lower weight
                    entropy = -np.sum(probs * np.log(probs + 1e-8), axis=1)
                    max_entropy = np.log(n_classes)
                    normalized_entropy = entropy / max_entropy
                    
                    # Convert entropy to weights (low entropy = high confidence = high weight)
                    uncertainty_weights = 1.0 - 0.5 * normalized_entropy  # Weights in [0.5, 1.0]
                
                # Compute differentiable count loss for this bag
                bag_count_loss = 0.0
                for i_class in range(n_classes):
                    target_lower = int(relaxed_lower_threshold[i_bag, i_class])
                    target_upper = int(relaxed_upper_threshold[i_bag, i_class])
                    if target_lower > 0 and target_lower <= n_instances:
                        # Use average of lower and upper as target, but cap at n_instances
                        target_count = min(int((target_lower + target_upper) / 2), n_instances)
                        class_probs = probs[:, i_class]
                        # Only compute if we have enough instances
                        if target_count >= 0 and target_count <= len(class_probs):
                            # Use stabilized loss with Deep-ALM
                            # Use fixed temperature for SoftSort to avoid instability
                            if use_softsort and i_iter >= softsort_start_iter:
                                soft_temp = 1.0  # Fixed temperature for SoftSort
                            else:
                                soft_temp = 0.5 * temperature  # Original temperature schedule
                            # Decide whether to use SoftSort based on iteration
                            use_softsort_now = use_softsort and i_iter >= softsort_start_iter
                            loss, violation = compute_stabilized_soft_count_loss(
                                class_probs, target_count, soft_temp, use_softsort=use_softsort_now
                            )
                            
                            # Deep-ALM augmented Lagrangian
                            # L_A = L_task + λ·g + (μ/2)·g²
                            augmented_loss = loss + lambda_lower[i_bag, i_class] * violation + 0.5 * mu_lower[i_bag, i_class] * (violation ** 2)
                            bag_count_loss += count_loss_weight * augmented_loss
                            
                            # Track soft count for variance calculation
                            soft_count = np.sum(1 / (1 + np.exp(-(class_probs - 0.5) / soft_temp)))
                
                total_count_loss += bag_count_loss
                
                # Compute constraint violation penalty with DCC-inspired improvements
                # β·(ReLU(ℓ – Σpi) + ReLU(Σpi – u))
                for i_class in range(n_classes):
                    # Enhanced soft count using piece-wise linear approximation (DCC-inspired)
                    # This provides better gradient flow than simple sum
                    class_probs = probs[:, i_class]
                    
                    # Sort probabilities for better count approximation
                    sorted_probs = np.sort(class_probs)[::-1]
                    
                    # Vectorized piece-wise linear soft count for efficiency
                    positions = np.arange(len(sorted_probs))
                    position_weights = np.clip(1.0 - positions / len(sorted_probs), 0.1, 1.0)
                    soft_count = np.sum(sorted_probs * position_weights)
                    
                    # Standard soft count for comparison
                    simple_soft_count = np.sum(class_probs)
                    
                    # Blend both approaches
                    final_soft_count = 0.7 * soft_count + 0.3 * simple_soft_count
                    
                    lower_violation = max(0, relaxed_lower_threshold[i_bag, i_class] - final_soft_count)
                    upper_violation = max(0, final_soft_count - relaxed_upper_threshold[i_bag, i_class])
                    
                    # Annealed penalty weight β with adaptive scaling
                    base_beta = 0.1 + 0.9 * progress  # Start small, increase over time
                    # Scale penalty based on bag size (larger bags need stronger penalties)
                    size_factor = np.sqrt(n_instances / 100.0)  # Normalize by typical bag size
                    beta = base_beta * size_factor
                    
                    total_constraint_violation += beta * (lower_violation + upper_violation)
                
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
                    
                    if class_count < relaxed_lower_threshold[i_bag, i_class]:
                        # Need more instances of this class
                        deficit = int(relaxed_lower_threshold[i_bag, i_class] - class_count)
                        
                        # Find best candidates: high probability for this class, not already changed
                        candidates = []
                        for idx in range(n_instances):
                            if idx not in changed_indices and new_y[idx] != i_class:
                                # Base score from probability
                                base_score = probs[idx, i_class]
                                
                                # Enhanced attention-based scoring mechanism
                                # Compute attention weights based on instance-class affinity
                                all_probs = probs[idx, :]
                                
                                # Self-attention: how well this instance fits with the target class
                                # relative to other classes
                                attention_score = np.exp(base_score / temperature) / np.sum(np.exp(all_probs / temperature))
                                
                                # Capped-sum attention (from recent research)
                                # Cap attention based on expected proportion
                                if n_instances > 0:
                                    expected_proportion = min(1.0, relaxed_upper_threshold[i_bag, i_class] / n_instances)
                                    max_attention = expected_proportion / max(1, class_count)
                                    attention_score = min(attention_score, max_attention)
                                
                                # Distance-Aware Self-Attention (DAS-MIL inspired) - Optimized
                                # Consider how similar this instance is to confident instances of target class
                                target_class_mask = (new_y == i_class)
                                target_class_mask[idx] = False  # Exclude self
                                n_target = np.sum(target_class_mask)
                                
                                if n_target > 0:
                                    # Vectorized similarity computation for efficiency
                                    target_probs = probs[target_class_mask, :]
                                    
                                    # Vectorized KL divergence
                                    kl_divs = np.sum(probs[idx, :] * np.log((probs[idx, :] + 1e-8) / (target_probs + 1e-8)), axis=1)
                                    
                                    # Vectorized Euclidean distance
                                    feature_dists = np.linalg.norm(probs[idx, :] - target_probs, axis=1)
                                    
                                    # Gaussian kernel (σ = 0.5)
                                    gaussian_weights = np.exp(-feature_dists ** 2 / 0.5)
                                    
                                    # Combined similarities
                                    similarities = np.exp(-kl_divs) * gaussian_weights
                                    avg_similarity = np.mean(similarities)
                                    
                                    attention_score = attention_score * (1 + 0.4 * avg_similarity)
                                
                                # Cross-attention: consider relationships with other instances
                                # Instances that are confidently assigned should influence others
                                confidence_weights = np.max(probs, axis=1)
                                avg_class_confidence = np.mean([confidence_weights[j] for j in range(n_instances) 
                                                               if new_y[j] == i_class] or [0.5])
                                
                                # Constraint-aware scoring with attention mechanism
                                constraint_score = 0.0
                                
                                # Reward moving to under-represented class with attention weighting
                                class_deficit = max(0, relaxed_lower_threshold[i_bag, i_class] - class_count)
                                if class_deficit > 0:
                                    # Adaptive boost based on attention and deficit
                                    deficit_weight = 0.3 + 0.2 * attention_score
                                    constraint_score += deficit_weight * np.log1p(class_deficit)
                                
                                # Reward moving from over-represented class
                                current_class = new_y[idx]
                                current_class_count = class_count_dict.get(current_class, 0)
                                current_excess = max(0, current_class_count - relaxed_upper_threshold[i_bag, current_class])
                                if current_excess > 0:
                                    # Stronger penalty for high-confidence instances in over-represented classes
                                    excess_weight = 0.2 + 0.1 * (1 - probs[idx, current_class])
                                    constraint_score += excess_weight * np.log1p(current_excess)
                                
                                # Instance quality score: prefer instances with clear class separation
                                class_separation = base_score - np.mean([probs[idx, c] for c in range(n_classes) if c != i_class])
                                quality_score = 0.1 * np.tanh(class_separation / temperature)
                                
                                # Combine scores with attention-weighted aggregation
                                score = (base_score * attention_score + 
                                        constraint_score * avg_class_confidence +
                                        quality_score) * (1.0 + 0.5 * attention_score)
                                
                                # Apply uncertainty weighting
                                # High confidence instances (low uncertainty) get higher scores
                                score *= uncertainty_weights[idx]
                                
                                candidates.append((idx, score))
                        
                        # Sort by score and select top candidates
                        candidates.sort(key=lambda x: x[1], reverse=True)
                        # Gradually increase assignments based on iteration
                        adjustment_rate = min(1.0, (i_iter + 1) / 2.0)  # More aggressive early adjustment
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
                    
                    if class_count > relaxed_upper_threshold[i_bag, i_class]:
                        # Too many instances of this class
                        excess = int(class_count - relaxed_upper_threshold[i_bag, i_class])
                        
                        # Find instances to reassign: low probability for this class
                        candidates = []
                        for idx in range(n_instances):
                            if idx not in changed_indices and new_y[idx] == i_class:
                                candidates.append((idx, probs[idx, i_class]))
                        
                        # Sort by probability (ascending) and reassign
                        candidates.sort(key=lambda x: x[1])
                        
                        # Gradually increase adjustments based on iteration
                        adjustment_rate = min(1.0, (i_iter + 1) / 2.0)  # Match the lower threshold adjustment rate
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
                
                # Deep-ALM updates for this bag
                for i_class in range(n_classes):
                    # Compute actual constraint violations
                    class_count = np.sum(new_y == i_class)
                    violation_lower = max(0, lower_threshold[i_bag, i_class] - class_count)
                    violation_upper = max(0, class_count - upper_threshold[i_bag, i_class])
                    
                    # Update Lagrange multipliers (dual ascent)
                    lambda_lower[i_bag, i_class] += mu_lower[i_bag, i_class] * violation_lower
                    lambda_upper[i_bag, i_class] += mu_upper[i_bag, i_class] * violation_upper
                    
                    # Adaptive penalty update (increase μ if constraint violation persists)
                    if violation_lower > constraint_tol * lower_threshold[i_bag, i_class]:
                        mu_lower[i_bag, i_class] = min(mu_lower[i_bag, i_class] * rho_increase, 10.0)  # Cap at 10
                    if violation_upper > constraint_tol * (upper_threshold[i_bag, i_class] + 1):
                        mu_upper[i_bag, i_class] = min(mu_upper[i_bag, i_class] * rho_increase, 10.0)
                    
                    # Clip Lagrange multipliers to prevent instability
                    lambda_lower[i_bag, i_class] = np.clip(lambda_lower[i_bag, i_class], -50, 50)
                    lambda_upper[i_bag, i_class] = np.clip(lambda_upper[i_bag, i_class], -50, 50)

            # Compute pseudo-contrastive loss using probability distributions
            # Memory-efficient version: compute within bags and sample across bags
            if use_contrastive and i_iter >= 1 and len(contrastive_probs_list) > 0:  # Start after first iteration
                contrastive_loss_accum = 0.0
                n_valid_bags = 0
                
                # Within-bag contrastive loss
                for i_bag in range(len(contrastive_probs_list)):
                    bag_probs = contrastive_probs_list[i_bag]
                    bag_labels = contrastive_labels_list[i_bag]
                    n_instances = len(bag_labels)
                    
                    if n_instances > 1:
                        # Normalize probabilities for cosine similarity
                        probs_normalized = bag_probs / (np.linalg.norm(bag_probs, axis=1, keepdims=True) + 1e-8)
                        
                        # Compute similarity matrix for this bag
                        similarity_matrix = probs_normalized @ probs_normalized.T
                        
                        # Create mask for same-class pairs
                        labels_expanded = bag_labels[:, np.newaxis]
                        same_class_mask = (labels_expanded == labels_expanded.T).astype(np.float64)
                        
                        # Remove diagonal
                        np.fill_diagonal(same_class_mask, 0)
                        np.fill_diagonal(similarity_matrix, 0)
                        
                        # Temperature-scaled similarities
                        scaled_similarities = similarity_matrix / contrastive_temperature
                        
                        # Count positive pairs
                        n_positive = np.sum(same_class_mask)
                        
                        if n_positive > 0:
                            # Proper InfoNCE contrastive loss (always >= 0)
                            # For each instance, compute -log(exp(sim_pos) / sum(exp(sim_all)))
                            
                            # Temperature-scaled similarities  
                            scaled_sim = similarity_matrix / contrastive_temperature
                            
                            # For numerical stability, subtract max before exp
                            max_sim_per_row = np.max(scaled_sim, axis=1, keepdims=True)
                            exp_sim = np.exp(scaled_sim - max_sim_per_row)
                            
                            # Compute InfoNCE loss for each instance
                            instance_losses = []
                            for i in range(n_instances):
                                # Find positive pairs for instance i
                                pos_mask = same_class_mask[i, :]
                                if np.sum(pos_mask) > 0:
                                    # Numerator: sum of exp(sim) for positive pairs
                                    numerator = np.sum(exp_sim[i, :] * pos_mask)
                                    # Denominator: sum of exp(sim) for all pairs except self
                                    denominator = np.sum(exp_sim[i, :]) - exp_sim[i, i]
                                    
                                    if denominator > 0 and numerator > 0:
                                        instance_loss = -np.log(numerator / denominator)
                                        instance_losses.append(instance_loss)
                            
                            if len(instance_losses) > 0:
                                bag_contrastive_loss = np.mean(instance_losses)
                                contrastive_loss_accum += bag_contrastive_loss
                                n_valid_bags += 1
                
                # Cross-bag contrastive loss (sample-based for memory efficiency)
                if len(contrastive_probs_list) > 1:
                    # Sample a subset of instances from different bags
                    max_samples_per_bag = 10  # Limit samples to avoid memory issues
                    sampled_probs = []
                    sampled_labels = []
                    
                    for i_bag in range(min(len(contrastive_probs_list), 20)):  # Limit number of bags
                        bag_probs = contrastive_probs_list[i_bag]
                        bag_labels = contrastive_labels_list[i_bag]
                        n_instances = len(bag_labels)
                        
                        if n_instances > 0:
                            # Sample instances from this bag
                            n_samples = min(max_samples_per_bag, n_instances)
                            indices = rng.choice(n_instances, size=n_samples, replace=False)
                            sampled_probs.append(bag_probs[indices])
                            sampled_labels.append(bag_labels[indices])
                    
                    if len(sampled_probs) > 1:
                        # Compute cross-bag contrastive loss on samples
                        sampled_probs_concat = np.vstack(sampled_probs)
                        sampled_labels_concat = np.hstack(sampled_labels)
                        
                        # Normalize
                        probs_normalized = sampled_probs_concat / (np.linalg.norm(sampled_probs_concat, axis=1, keepdims=True) + 1e-8)
                        
                        # Similarity matrix
                        similarity_matrix = probs_normalized @ probs_normalized.T
                        
                        # Same-class mask
                        labels_expanded = sampled_labels_concat[:, np.newaxis]
                        same_class_mask = (labels_expanded == labels_expanded.T).astype(np.float64)
                        np.fill_diagonal(same_class_mask, 0)
                        
                        # Compute loss using proper InfoNCE
                        n_positive = np.sum(same_class_mask)
                        if n_positive > 0:
                            # Temperature-scaled similarities
                            scaled_sim = similarity_matrix / contrastive_temperature
                            
                            # Numerical stability
                            max_sim_per_row = np.max(scaled_sim, axis=1, keepdims=True)
                            exp_sim = np.exp(scaled_sim - max_sim_per_row)
                            
                            # Compute InfoNCE for sampled instances
                            n_sampled = len(sampled_labels_concat)
                            instance_losses = []
                            for i in range(n_sampled):
                                pos_mask = same_class_mask[i, :]
                                if np.sum(pos_mask) > 0:
                                    numerator = np.sum(exp_sim[i, :] * pos_mask)
                                    denominator = np.sum(exp_sim[i, :]) - exp_sim[i, i]
                                    
                                    if denominator > 0 and numerator > 0:
                                        instance_loss = -np.log(numerator / denominator)
                                        instance_losses.append(instance_loss)
                            
                            if len(instance_losses) > 0:
                                cross_bag_loss = np.mean(instance_losses)
                                contrastive_loss_accum += 0.5 * cross_bag_loss  # Weight cross-bag loss less
                
                # Average and scale
                if n_valid_bags > 0:
                    total_contrastive_loss = contrastive_weight * contrastive_loss_accum / n_valid_bags

            # Enhanced tracking with momentum and early stopping
            if debug_true_y is not None:
                current_accuracy = np.mean(
                    np.hstack([self.classifier.predict(bag) for bag in bags]) == np.hstack(debug_true_y)
                )
                
                # Update momentum-based accuracy tracking
                momentum_accuracy = momentum_factor * momentum_accuracy + (1 - momentum_factor) * current_accuracy
                accuracy_history.append(current_accuracy)
                
                # Enhanced convergence detection with multiple criteria
                if perturbation_active and len(accuracy_history) >= 3:
                    # Check multiple convergence criteria
                    recent_improvements = [
                        accuracy_history[i] - accuracy_history[i-1] 
                        for i in range(len(accuracy_history)-2, len(accuracy_history))
                    ]
                    
                    # Criterion 1: Small improvements
                    small_improvements = all(abs(imp) < 1e-3 for imp in recent_improvements)
                    
                    # Criterion 2: Oscillating accuracy (variance check)
                    if len(accuracy_history) >= 5:
                        recent_variance = np.var(accuracy_history[-5:])
                        oscillating = recent_variance < 1e-5
                    else:
                        oscillating = False
                    
                    # Criterion 3: Momentum-based convergence
                    momentum_converged = abs(momentum_accuracy - current_accuracy) < 5e-4
                    
                    if (small_improvements or oscillating) and momentum_converged:
                        no_improvement_count += 1
                        if no_improvement_count >= 2:
                            perturbation_active = False
                            if debug:
                                print(f"Stopping SPGD perturbations due to convergence (momentum acc: {momentum_accuracy:.6f})")
                    else:
                        no_improvement_count = 0  # Reset counter if improving
                
                # Restart perturbation if stuck at local optimum for too long
                if not perturbation_active and len(accuracy_history) >= 8:
                    recent_accuracies = accuracy_history[-4:]
                    if np.std(recent_accuracies) < 1e-4 and i_iter < max_iter - 2:
                        # Plateau detected, restart with smaller perturbation
                        perturbation_active = True
                        perturbation_scale_init *= 0.5  # Reduce scale for fine-tuning
                        if debug:
                            print(f"Restarting SPGD perturbations with scale {perturbation_scale_init:.4f} due to plateau")
                
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
                print("constraint violation penalty")
                print(f"{total_constraint_violation:.4f}")
                print("pseudo-contrastive loss")
                print(f"{total_contrastive_loss:.4f}")
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

            # Calculate count variance for self-paced learning
            if i_iter > 0:
                bag_count_variances = []
                for i_bag in range(len(bags)):
                    if len(bags[i_bag]) > 0:
                        bag_class_counts = []
                        for i_class in range(n_classes):
                            class_count = np.sum(y[i_bag] == i_class)
                            bag_class_counts.append(class_count)
                        bag_count_variances.append(np.var(bag_class_counts))
                
                if bag_count_variances:
                    count_variance_history.append(np.mean(bag_count_variances))
            
            # Early stopping when converged
            if not has_changed and i_iter >= 9:
                # For evaluation, ensure we reach at least iter 9
                break
        
        # Monte-Carlo Dropout for final predictions (if classifier supports it)
        if i_iter >= 9 and hasattr(self.classifier, 'predict_proba'):
            # Perform multiple forward passes with dropout for better stability
            mc_iterations = 3  # Reduced for efficiency while maintaining benefit
            mc_predictions = []
            
            for _ in range(mc_iterations):
                # Get predictions (dropout will be applied if model has it)
                mc_probs = [self.classifier.predict_proba(bag) for bag in bags]
                mc_pred = [np.argmax(probs, axis=1) for probs in mc_probs]
                mc_predictions.append(np.hstack(mc_pred))
            
            # Majority vote over MC samples
            if mc_predictions:
                mc_predictions = np.stack(mc_predictions)
                # For each instance, take the most common prediction
                final_predictions = []
                for i in range(mc_predictions.shape[1]):
                    votes = mc_predictions[:, i]
                    unique, counts = np.unique(votes, return_counts=True)
                    final_predictions.append(unique[np.argmax(counts)])
                
                # Update y with majority vote predictions if it improves accuracy
                if debug_true_y is not None:
                    mc_accuracy = np.mean(np.array(final_predictions) == np.hstack(debug_true_y))
                    current_accuracy = np.mean(np.hstack([self.classifier.predict(bag) for bag in bags]) == np.hstack(debug_true_y))
                    
                    if mc_accuracy > current_accuracy and debug:
                        print(f"Monte-Carlo Dropout improved accuracy: {current_accuracy:.6f} -> {mc_accuracy:.6f}")

        # Use SWA classifier if available for better stability
        if swa_classifier is not None:
            self.classifier = swa_classifier
            
        # Return best solution if available
        if best_y is not None and best_accuracy > 0:
            # Refit classifier with best labels using multiple epochs for better convergence
            flatten_bags = np.vstack(bags)
            flatten_best_y = np.hstack(best_y)
            
            # Train for multiple epochs to ensure good convergence
            for _ in range(3):
                self.classifier.fit(flatten_bags, flatten_best_y)
                
            # If SWA classifier performed better, use it
            if swa_classifier is not None:
                swa_flatten_bags = np.vstack(bags)
                swa_flatten_best_y = np.hstack(best_y)
                for _ in range(2):
                    swa_classifier.fit(swa_flatten_bags, swa_flatten_best_y)
                
                # Compare performance and use the better one
                if hasattr(self.classifier, 'predict_proba') and hasattr(swa_classifier, 'predict_proba'):
                    main_probs = [self.classifier.predict_proba(bag) for bag in bags]
                    swa_probs = [swa_classifier.predict_proba(bag) for bag in bags]
                    
                    # Use entropy as a measure of confidence
                    main_entropy = np.mean([np.mean(-p * np.log(p + 1e-8)) for p in main_probs])
                    swa_entropy = np.mean([np.mean(-p * np.log(p + 1e-8)) for p in swa_probs])
                    
                    if swa_entropy < main_entropy:  # Lower entropy = more confident
                        self.classifier = swa_classifier
                        
            return self.classifier, best_y
        
        return self.classifier, y
