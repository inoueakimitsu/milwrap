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
) -> NDArray[np.float64]:
    """Solve constrained optimal transport using Sinkhorn algorithm.
    
    Args:
    ----
        cost_matrix: Cost matrix (n_instances, n_classes)
        row_marginals: Row marginals (n_instances,), should sum to 1 for each instance
        col_marginals: Column marginals (n_classes,), target class counts
        reg: Entropy regularization parameter
        max_iter: Maximum iterations
        threshold: Convergence threshold
        
    Returns:
    -------
        Transport plan P (n_instances, n_classes)
    
    """
    n_instances, n_classes = cost_matrix.shape
    
    # Initialize
    K = np.exp(-cost_matrix / reg)
    u = np.ones(n_instances) / n_instances
    v = np.ones(n_classes) / n_classes
    
    # Normalize marginals
    row_marginals = row_marginals / row_marginals.sum()
    col_marginals = col_marginals / col_marginals.sum()
    
    for _ in range(max_iter):
        u_prev = u.copy()
        
        # Update u (row scaling)
        v = col_marginals / (K.T @ u + 1e-16)
        # Update v (column scaling)  
        u = row_marginals / (K @ v + 1e-16)
        
        # Check convergence
        if np.max(np.abs(u - u_prev)) < threshold:
            break
    
    # Compute transport plan
    P = np.diag(u) @ K @ np.diag(v)
    
    # Ensure row sums are 1 (probability distribution)
    P = P / P.sum(axis=1, keepdims=True)
    
    return P


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
            # Simple initialization based on threshold centers with small noise
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
        rho = 1.0  # Penalty parameter for augmented Lagrangian
        
        # Temperature control parameters
        temperature = 1.0  # Initial temperature
        target_entropy = 0.5  # Target entropy level (0 = deterministic, log(n_classes) = uniform)
        target_ess = 2.0  # Target effective sample size
        temp_adjustment_rate = 0.1  # Learning rate for temperature adjustment
        
        # AEM regularization parameters
        aem_lambda_start = 0.5
        aem_lambda_end = 0.0
        aem_lambda_decay = (aem_lambda_end / aem_lambda_start) ** (1.0 / max_iter) if aem_lambda_start > 0 else 1.0

        for i_iter in range(max_iter):
            # AEM regularization decay
            aem_lambda = aem_lambda_start * (aem_lambda_decay ** i_iter)
            
            # Fit
            flatten_bags = np.vstack(bags)
            flatten_y = np.hstack(y)
            self.classifier.fit(flatten_bags, flatten_y)

            # Compute outputs
            fs = [self.classifier.predict_proba(bag) for bag in bags]
            y_pred = [self.classifier.predict(bag) for bag in bags]
            flatten_original_y = np.hstack(list(y))

            # For every bag - use constrained optimal transport
            has_changed = False
            all_attention_weights = []  # For entropy computation
            
            for i_bag in range(len(bags)):
                n_instances = len(bags[i_bag])
                probs = fs[i_bag]  # Shape: (n_instances, n_classes)
                
                # Create cost matrix from negative log probabilities
                cost_matrix = -np.log(probs + 1e-16)
                
                # Add Gumbel noise for exploration (decreasing over iterations)
                if i_iter < 6:
                    gumbel_noise = -np.log(-np.log(np.random.uniform(0.01, 0.99, probs.shape)))
                    noise_strength = 0.15 * np.exp(-i_iter / 3.0)
                    cost_matrix += noise_strength * gumbel_noise
                
                # Add AEM regularization to cost matrix
                if aem_lambda > 0:
                    # Penalize low entropy (encourage exploration)
                    entropy_penalty = -aem_lambda * np.log(probs + 1e-16)
                    cost_matrix += entropy_penalty
                
                # Set up marginals for Sinkhorn
                row_marginals = np.ones(n_instances)  # Each instance must be assigned
                
                # Column marginals based on constraints with gradual tightening
                col_marginals = np.zeros(n_classes)
                slack_factor = max(0.0, 1.0 - (i_iter / 4.0))  # Reduce slack over first 4 iterations
                
                for i_class in range(n_classes):
                    lower = lower_threshold[i_bag, i_class]
                    upper = upper_threshold[i_bag, i_class]
                    
                    # Apply slack in early iterations
                    if slack_factor > 0:
                        lower = lower * (1.0 - 0.05 * slack_factor)
                        upper = upper * (1.0 + 0.05 * slack_factor)
                    
                    # Set target count as midpoint of constraints
                    target_count = (lower + upper) / 2.0
                    col_marginals[i_class] = target_count
                
                # Solve constrained OT problem
                P = sinkhorn_constrained_ot(
                    cost_matrix, 
                    row_marginals, 
                    col_marginals,
                    reg=temperature,
                    max_iter=30,
                    threshold=1e-5
                )
                
                # Store attention weights for entropy computation
                all_attention_weights.append(P)
                
                # Convert soft assignments to hard assignments
                new_y = np.argmax(P, axis=1)
                
                # Check if assignments changed
                if not np.array_equal(y[i_bag], new_y):
                    has_changed = True
                    y[i_bag] = new_y
            
            # Adaptive temperature control based on entropy/ESS
            if i_iter > 0 and len(all_attention_weights) > 0:
                # Compute average entropy and ESS across all bags
                avg_entropy = np.mean([compute_attention_entropy(P) for P in all_attention_weights])
                avg_ess = np.mean([compute_effective_sample_size(P) for P in all_attention_weights])
                
                # Adjust temperature based on entropy/ESS
                if avg_entropy < target_entropy:
                    # Increase temperature to encourage exploration
                    temperature *= (1 + temp_adjustment_rate)
                elif avg_entropy > target_entropy * 1.5:
                    # Decrease temperature to encourage exploitation
                    temperature *= (1 - temp_adjustment_rate)
                
                # Clamp temperature to reasonable range
                temperature = np.clip(temperature, 0.1, 3.0)
            
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
                # Use instance confidence weighting in later iterations
                for inner_iter in range(3):  # 3 inner epochs with confidence weighting
                    # Compute instance confidence based on prediction probabilities
                    all_probs = []
                    for bag in bags:
                        bag_probs = self.classifier.predict_proba(bag)
                        all_probs.append(bag_probs)
                    
                    # Create sample weights based on prediction confidence
                    sample_weights = []
                    for i_bag, bag_probs in enumerate(all_probs):
                        # Weight by max probability (confidence)
                        max_probs = np.max(bag_probs, axis=1)
                        # Use more aggressive weighting: high confidence gets higher weight
                        weights = np.power(max_probs, 2)  # Square to emphasize confident predictions
                        # Apply smoothing to avoid zero weights
                        weights = 0.3 + 0.7 * weights
                        sample_weights.extend(weights)
                    
                    sample_weights = np.array(sample_weights)
                    
                    # Re-fit with weighted samples
                    if hasattr(self.classifier, 'fit') and 'sample_weight' in self.classifier.fit.__code__.co_varnames:
                        self.classifier.fit(flatten_bags, flatten_y, sample_weight=sample_weights)
                    else:
                        # If classifier doesn't support sample weights, use standard fit
                        self.classifier.fit(flatten_bags, flatten_y)
            elif not has_changed and i_iter < 4:
                if debug:
                    print(f"Early convergence at iteration {i_iter}, continuing exploration...")
            
            # Only break after minimum iterations
            if i_iter >= min_iterations - 1:
                break

        return self.classifier, y
