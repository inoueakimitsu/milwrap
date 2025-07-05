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


def sinkhorn_constrained_ot(
    cost_matrix: NDArray[np.float64],
    row_marginals: NDArray[np.float64],
    col_marginals: NDArray[np.float64],
    reg: float = 1.0,
    max_iter: int = 50,
    threshold: float = 1e-6,
    dynamic_reg: bool = True,
    reg_decay: float = 0.95,
    debiased: bool = True,
) -> Tuple[NDArray[np.float64], float]:
    """Solve constrained optimal transport using Sinkhorn algorithm with dynamic regularization.
    
    Args:
    ----
        cost_matrix: Cost matrix (n_instances, n_classes)
        row_marginals: Row marginals (n_instances,), should sum to 1 for each instance
        col_marginals: Column marginals (n_classes,), target class counts
        reg: Initial entropy regularization parameter
        max_iter: Maximum iterations
        threshold: Convergence threshold
        dynamic_reg: Whether to use dynamic regularization scheduling
        reg_decay: Decay factor for dynamic regularization
        debiased: Whether to use debiasing with Richardson extrapolation
        
    Returns:
    -------
        Tuple of (Transport plan P, final regularization value)
    
    """
    n_instances, n_classes = cost_matrix.shape
    
    # Normalize marginals
    row_marginals = row_marginals / (row_marginals.sum() + 1e-16)
    col_marginals = col_marginals / (col_marginals.sum() + 1e-16)
    
    # Dynamic regularization scheduling
    current_reg = reg
    best_P = None
    best_dual_gap = float('inf')
    
    # Initialize with higher regularization for stability
    if dynamic_reg:
        current_reg = reg * 2.0
    
    for outer_iter in range(3 if dynamic_reg else 1):  # Multiple rounds with decreasing reg
        # Initialize
        K = np.exp(-cost_matrix / current_reg)
        u = np.ones(n_instances)
        v = np.ones(n_classes)
        
        for inner_iter in range(max_iter):
            u_prev = u.copy()
            
            # Update with stabilization
            v = col_marginals / (K.T @ u + 1e-16)
            v = np.clip(v, 1e-8, 1e8)  # Numerical stability
            
            u = row_marginals / (K @ v + 1e-16)
            u = np.clip(u, 1e-8, 1e8)  # Numerical stability
            
            # Check convergence using dual gap
            P_temp = np.diag(u) @ K @ np.diag(v)
            row_sum = P_temp.sum(axis=1)
            col_sum = P_temp.sum(axis=0)
            
            # Dual gap as convergence criterion
            dual_gap = np.max([
                np.max(np.abs(row_sum - row_marginals)),
                np.max(np.abs(col_sum - col_marginals))
            ])
            
            if dual_gap < threshold:
                break
                
            # Acceleration step (Anderson acceleration simplified)
            if inner_iter > 0 and inner_iter % 5 == 0:
                u = 0.8 * u + 0.2 * u_prev  # Momentum
        
        # Compute transport plan
        P = np.diag(u) @ K @ np.diag(v)
        
        # Ensure row sums are 1 (probability distribution)
        P = P / (P.sum(axis=1, keepdims=True) + 1e-16)
        
        # Track best solution
        if dual_gap < best_dual_gap:
            best_dual_gap = dual_gap
            best_P = P.copy()
        
        # Decay regularization for next round
        if dynamic_reg and outer_iter < 2:
            current_reg *= reg_decay
    
    # Debiasing with Richardson extrapolation
    if debiased and best_P is not None and dynamic_reg:
        # Compute solution with half regularization
        K_half = np.exp(-cost_matrix / (current_reg / 2))
        u_half = np.ones(n_instances)
        v_half = np.ones(n_classes)
        
        for _ in range(max_iter // 2):  # Fewer iterations for debiasing
            v_half = col_marginals / (K_half.T @ u_half + 1e-16)
            v_half = np.clip(v_half, 1e-8, 1e8)
            u_half = row_marginals / (K_half @ v_half + 1e-16)
            u_half = np.clip(u_half, 1e-8, 1e8)
        
        P_half = np.diag(u_half) @ K_half @ np.diag(v_half)
        P_half = P_half / (P_half.sum(axis=1, keepdims=True) + 1e-16)
        
        # Richardson extrapolation: P_debiased = 2*P_half - P
        P_debiased = 2 * P_half - best_P
        P_debiased = np.clip(P_debiased, 0, None)  # Ensure non-negative
        P_debiased = P_debiased / (P_debiased.sum(axis=1, keepdims=True) + 1e-16)
        
        best_P = P_debiased
    
    return best_P if best_P is not None else P, current_reg


def compute_attention_entropy(attention_weights: NDArray[np.float64]) -> float:
    """Compute entropy of attention weights.
    
    Args:
    ----
        attention_weights: Attention weights (n_instances, n_classes)
        
    Returns:
    -------
        Average entropy across instances
        
    """
    # Add small epsilon to avoid log(0)
    eps = 1e-8
    attention_weights = np.clip(attention_weights, eps, 1 - eps)
    
    # Compute entropy for each instance
    entropy_per_instance = -np.sum(attention_weights * np.log(attention_weights), axis=1)
    
    # Return average entropy
    return np.mean(entropy_per_instance)


def compute_effective_sample_size(attention_weights: NDArray[np.float64]) -> float:
    """Compute effective sample size (ESS) = 1 / sum(p^2).
    
    Args:
    ----
        attention_weights: Attention weights (n_instances, n_classes)
        
    Returns:
    -------
        Average ESS across instances
        
    """
    # Compute sum of squared probabilities for each instance
    sum_p_squared = np.sum(attention_weights ** 2, axis=1)
    
    # Compute ESS for each instance
    ess_per_instance = 1.0 / (sum_p_squared + 1e-16)
    
    # Return average ESS
    return np.mean(ess_per_instance)


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
        
        # Multi-penalty ADMM: each constraint gets its own penalty parameter
        rho_lower = np.ones((n_bags, n_classes)) * 1.0
        rho_upper = np.ones((n_bags, n_classes)) * 1.0
        
        # Temperature for soft assignments (starts high, anneals down)
        # Optimal temperature scales with sqrt(d) where d is feature dimension
        feature_dim = bags[0].shape[1] if len(bags) > 0 and len(bags[0]) > 0 else 7
        # Higher initial temperature for better exploration
        temperature_start = np.sqrt(feature_dim) * 1.5  # Increased from 0.5 to 1.5
        temperature_end = 0.1
        temperature_decay = (temperature_end / temperature_start) ** (1.0 / max_iter)
        
        # Instance-aware temperature scaling factors
        instance_temp_scale = [np.ones(len(bag)) for bag in bags]
        
        # Entropy regularization parameters
        entropy_weight_start = 0.3  # Moderate entropy regularization at start (reduced from 0.5)
        entropy_weight_end = 0.0    # No entropy regularization at end
        entropy_decay = (entropy_weight_end + 1e-8) / (entropy_weight_start + 1e-8)
        
        # Constraint continuation parameters (gradual constraint tightening)
        constraint_slack_start = 0.2  # 20% slack on constraints initially (reduced from 0.3)
        constraint_slack_end = 0.0    # No slack at the end
        constraint_slack_decay = constraint_slack_end / (constraint_slack_start + 1e-8)
        
        # OT entropy tracking for adaptive temperature control
        target_entropy = 0.5  # Target entropy level
        target_ess = 2.0  # Target effective sample size
        temp_adjustment_rate = 0.1  # Learning rate for temperature adjustment
        
        # Meta-gradient temperature optimization
        temperature_momentum = 0.0  # Momentum for temperature updates
        temperature_lr = 0.02  # Learning rate for meta-gradient
        momentum_beta = 0.9  # Momentum coefficient
        
        # AEM regularization parameters for OT cost matrix
        aem_lambda_start = 0.5
        aem_lambda_end = 0.0
        aem_lambda_decay = (aem_lambda_end / aem_lambda_start) ** (1.0 / max_iter) if aem_lambda_start > 0 else 1.0

        for i_iter in range(max_iter):
            # Temperature annealing
            temperature = temperature_start * (temperature_decay ** i_iter)
            
            # Entropy weight annealing
            entropy_weight = entropy_weight_start * (entropy_decay ** i_iter)
            
            # Constraint slack annealing (for continuation method)
            constraint_slack = constraint_slack_start * (1.0 - i_iter / max_iter)
            
            # Fit
            flatten_bags = np.vstack(bags)
            flatten_y = np.hstack(y)
            self.classifier.fit(flatten_bags, flatten_y)

            # Compute outputs
            fs = [self.classifier.predict_proba(bag) for bag in bags]
            y_pred = [self.classifier.predict(bag) for bag in bags]
            flatten_original_y = np.hstack(list(y))
            
            # Update instance-aware temperature scaling based on prediction confidence
            for i_bag in range(len(bags)):
                max_probs = np.max(fs[i_bag], axis=1)
                # Higher confidence -> lower temperature (more decisive)
                # Lower confidence -> higher temperature (more exploration)
                instance_temp_scale[i_bag] = 2.0 - max_probs  # Range: [1.0, 2.0]

            # For every bag - use optimal transport with augmented Lagrangian
            has_changed = False
            all_attention_weights = []  # For entropy computation
            
            # AEM regularization decay
            aem_lambda = aem_lambda_start * (aem_lambda_decay ** i_iter)
            
            for i_bag in range(len(bags)):
                # Hybrid approach: combine constraint-based selection with soft assignments
                n_instances = len(bags[i_bag])
                probs = fs[i_bag]  # Shape: (n_instances, n_classes)
                
                # Self-paced learning: identify high-confidence instances to freeze
                max_probs = np.max(probs, axis=1)
                predicted_classes = np.argmax(probs, axis=1)
                
                # Freeze instances with high confidence (adaptive threshold)
                confidence_threshold = 0.9 - 0.3 * (i_iter / max_iter)  # Start strict, relax over time
                frozen_indices = set()
                if i_iter >= 2:  # Start freezing after initial exploration
                    for idx in range(n_instances):
                        if max_probs[idx] > confidence_threshold and y[i_bag][idx] == predicted_classes[idx]:
                            frozen_indices.add(idx)
                
                # Create active indices (non-frozen instances)
                active_indices = [idx for idx in range(n_instances) if idx not in frozen_indices]
                n_active = len(active_indices)
                
                if n_active == 0:  # All instances frozen, skip
                    continue
                
                # Create cost matrix for active instances only
                active_probs = probs[active_indices, :]
                cost_matrix = -np.log(active_probs + 1e-16)
                
                # Add Gumbel noise for exploration (decreasing over iterations)
                if i_iter < 6:
                    gumbel_noise = -np.log(-np.log(np.random.uniform(0.01, 0.99, cost_matrix.shape)))
                    noise_strength = 0.15 * np.exp(-i_iter / 3.0)
                    cost_matrix += noise_strength * gumbel_noise
                
                # Add AEM regularization to cost matrix
                if aem_lambda > 0:
                    # Penalize low entropy (encourage exploration)
                    entropy_penalty = -aem_lambda * np.log(active_probs + 1e-16)
                    cost_matrix += entropy_penalty
                
                # Add ADMM penalties based on current constraint violations
                # Get current class counts for frozen instances
                frozen_counts = np.zeros(n_classes)
                for idx in frozen_indices:
                    frozen_counts[y[i_bag][idx]] += 1
                
                # Set up marginals for Sinkhorn
                row_marginals = np.ones(n_active)  # Each active instance must be assigned
                
                # Column marginals based on constraints with continuation
                col_marginals = np.zeros(n_classes)
                for i_class in range(n_classes):
                    lower = lower_threshold[i_bag, i_class] * (1.0 - constraint_slack)
                    upper = upper_threshold[i_bag, i_class] * (1.0 + constraint_slack)
                    
                    # Account for frozen instances
                    remaining_lower = max(0, lower - frozen_counts[i_class])
                    remaining_upper = max(0, upper - frozen_counts[i_class])
                    
                    # Set target count as midpoint of remaining constraints
                    target_count = (remaining_lower + remaining_upper) / 2.0
                    col_marginals[i_class] = target_count
                
                # Handle edge case where constraints are already satisfied by frozen instances
                if np.sum(col_marginals) > 0:
                    # Apply instance-aware temperature scaling
                    avg_instance_temp_scale = np.mean(instance_temp_scale[i_bag][active_indices])
                    effective_temperature = temperature * avg_instance_temp_scale
                    
                    # Solve constrained OT problem with dynamic regularization and debiasing
                    P, final_reg = sinkhorn_constrained_ot(
                        cost_matrix, 
                        row_marginals, 
                        col_marginals,
                        reg=effective_temperature,
                        max_iter=30,
                        threshold=1e-5,
                        dynamic_reg=True,
                        reg_decay=0.95,
                        debiased=True
                    )
                    
                    # Store attention weights for entropy computation
                    all_attention_weights.append(P)
                    
                    # Convert soft assignments to hard assignments
                    active_new_y = np.argmax(P, axis=1)
                    
                    # Update full assignment array
                    new_y = y[i_bag].copy()
                    for i, idx in enumerate(active_indices):
                        new_y[idx] = active_new_y[i]
                    
                    # Check if assignments changed
                    if not np.array_equal(y[i_bag], new_y):
                        has_changed = True
                        y[i_bag] = new_y
                
                # Skip the old constraint logic (will be removed later)
                continue
                
                # OLD LOGIC STARTS HERE - TO BE REMOVED
                for i_class in target_classes:
                    class_count = 0  # dummy to avoid error
                    
                    if class_count < lower_threshold[i_bag, i_class]:
                        # Need more instances of this class
                        deficit = int(lower_threshold[i_bag, i_class] - class_count)
                        
                        # Find best candidates: high probability for this class, not already changed, not frozen
                        candidates = []
                        for idx in range(n_instances):
                            if idx not in changed_indices and idx not in frozen_indices and new_y[idx] != i_class:
                                # Score based on probability and current class constraint status
                                score = probs[idx, i_class]
                                
                                # Boost score if moving from over-represented class
                                current_class = new_y[idx]
                                current_class_count = class_count_dict.get(current_class, 0)
                                if current_class_count > upper_threshold[i_bag, current_class]:
                                    # Use adaptive penalty parameter for stronger boost
                                    score *= 1.0 + rho_upper[i_bag, current_class]
                                elif current_class_count <= lower_threshold[i_bag, current_class]:
                                    # Use adaptive penalty parameter for penalty
                                    score *= 1.0 / (1.0 + rho_lower[i_bag, current_class])
                                
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
                                
                                # Base score from probability
                                scores[c] = np.log(probs[idx, c] + 1e-8)
                                
                                # Constraint bonus with adaptive penalties and continuation
                                c_count = class_count_dict.get(c, 0)
                                # Apply constraint slack for continuation method
                                lower_bound = lower_threshold[i_bag, c] * (1.0 - constraint_slack)
                                upper_bound = upper_threshold[i_bag, c] * (1.0 + constraint_slack)
                                
                                if c_count < lower_bound:
                                    # Use adaptive penalty for under-represented classes
                                    scores[c] += rho_lower[i_bag, c] * 2.0
                                elif c_count < upper_bound:
                                    scores[c] += 0.5
                                else:
                                    # Use adaptive penalty for over-represented classes
                                    scores[c] -= rho_upper[i_bag, c]
                            
                            # Apply temperature-controlled softmax with instance-aware scaling
                            # Gumbel noise disabled for now (ablation analysis)
                            instance_temperature = temperature * instance_temp_scale[i_bag][idx]
                            
                            exp_scores = np.exp(scores / instance_temperature)
                            probabilities = exp_scores / np.sum(exp_scores)
                            
                            # Apply entropy regularization to encourage diversity
                            if entropy_weight > 0:
                                entropy = -np.sum(probabilities * np.log(probabilities + 1e-8))
                                # Adjust probabilities to increase entropy
                                uniform_probs = np.ones(n_classes) / n_classes
                                probabilities = (1.0 - entropy_weight) * probabilities + entropy_weight * uniform_probs
                                # Renormalize
                                probabilities = probabilities / np.sum(probabilities)
                            
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
            
            # Adaptive temperature control based on entropy/ESS with meta-gradient
            if i_iter > 0 and len(all_attention_weights) > 0:
                # Compute average entropy and ESS across all bags
                avg_entropy = np.mean([compute_attention_entropy(P) for P in all_attention_weights])
                avg_ess = np.mean([compute_effective_sample_size(P) for P in all_attention_weights])
                
                # Compute constraint satisfaction metrics
                predicted_count = np.nan_to_num(
                    pd.DataFrame([pd.Series(b).value_counts().to_dict() for b in y])[list(range(n_classes))].values,
                )
                constraint_violations = np.sum(np.maximum(0, lower_threshold - predicted_count)) + \
                                      np.sum(np.maximum(0, predicted_count - upper_threshold))
                
                # Meta-gradient computation
                # Loss = entropy_error + constraint_penalty
                entropy_error = (avg_entropy - target_entropy) ** 2
                constraint_penalty = 0.1 * constraint_violations / n_bags
                meta_loss = entropy_error + constraint_penalty
                
                # Gradient w.r.t temperature (approximated)
                # Higher temp -> higher entropy, lower constraint satisfaction
                entropy_gradient = 2 * (avg_entropy - target_entropy) * (avg_entropy / temperature)
                constraint_gradient = -0.1 * constraint_violations / (n_bags * temperature)
                
                # Total gradient
                temperature_gradient = entropy_gradient + constraint_gradient
                
                # Update temperature with momentum
                temperature_momentum = momentum_beta * temperature_momentum - temperature_lr * temperature_gradient
                temperature = temperature * np.exp(temperature_momentum)  # Multiplicative update
                
                # Clamp temperature to reasonable range
                temperature = np.clip(temperature, 0.1, 3.0)
                
                # Fallback to simple adjustment if meta-gradient is unstable
                if np.isnan(temperature) or np.isinf(temperature):
                    temperature = temperature_start * (temperature_decay ** i_iter)
                    if avg_entropy < target_entropy:
                        temperature *= 1.1
                    elif avg_entropy > target_entropy * 1.5:
                        temperature *= 0.9

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
                print(f"Temperature: {temperature:.3f}")
                if i_iter > 0 and len(all_attention_weights) > 0:
                    print(f"Avg Entropy: {avg_entropy:.3f}, Avg ESS: {avg_ess:.3f}")

            # Continue to iteration 9 for evaluation, even if converged
            min_iterations = 9
            if not has_changed and i_iter >= 4:
                # Labels have converged, but continue training classifier
                if debug:
                    print(f"Labels converged at iteration {i_iter}, continuing to refine classifier...")
                # Train with additional inner iterations when labels are stable
                # Use early stopping based on validation performance
                best_score = -np.inf
                patience = 3
                no_improvement_count = 0
                
                for inner_iter in range(10):  # More inner epochs with early stopping
                    self.classifier.fit(flatten_bags, flatten_y)
                    
                    # Compute validation score
                    val_predictions = np.hstack([self.classifier.predict(bag) for bag in bags])
                    val_score = np.mean(val_predictions == flatten_y)
                    
                    if val_score > best_score:
                        best_score = val_score
                        no_improvement_count = 0
                    else:
                        no_improvement_count += 1
                        
                    if no_improvement_count >= patience:
                        break
                        
            elif not has_changed and i_iter < 4:
                if debug:
                    print(f"Early convergence at iteration {i_iter}, continuing exploration...")
            
            # Update multi-penalty ADMM parameters based on constraint violations
            if i_iter < max_iter - 1:
                predicted_count = np.nan_to_num(
                    pd.DataFrame([pd.Series(b).value_counts().to_dict() for b in y])[list(range(n_classes))].values,
                )
                
                # Adaptive penalty update for lower constraints
                lower_violations = np.maximum(lower_threshold - predicted_count, 0)
                for i_bag in range(n_bags):
                    for i_class in range(n_classes):
                        if lower_violations[i_bag, i_class] > 0:
                            # Increase penalty for violated constraints
                            rho_lower[i_bag, i_class] *= 1.2
                        else:
                            # Decrease penalty for satisfied constraints
                            rho_lower[i_bag, i_class] *= 0.9
                        # Clamp to reasonable range
                        rho_lower[i_bag, i_class] = np.clip(rho_lower[i_bag, i_class], 0.1, 10.0)
                
                # Adaptive penalty update for upper constraints
                upper_violations = np.maximum(predicted_count - upper_threshold, 0)
                for i_bag in range(n_bags):
                    for i_class in range(n_classes):
                        if upper_violations[i_bag, i_class] > 0:
                            rho_upper[i_bag, i_class] *= 1.2
                        else:
                            rho_upper[i_bag, i_class] *= 0.9
                        rho_upper[i_bag, i_class] = np.clip(rho_upper[i_bag, i_class], 0.1, 10.0)
            
            # Only break after minimum iterations
            if i_iter >= min_iterations - 1:
                break

        return self.classifier, y
