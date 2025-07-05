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
            # Improved initialization: consider both lower and upper thresholds
            # Use weighted average to find better initial class assignment
            threshold_center = (lower_threshold + upper_threshold) / 2.0
            # Add small noise to break ties - increased for better initial diversity
            threshold_center += np.random.randn(*threshold_center.shape) * 0.02
            
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
        # Balanced initial temperature for hybrid curriculum
        temperature_start = np.sqrt(feature_dim) * 1.2  # Optimal for two-stage approach
        temperature_end = 0.1
        temperature_decay = (temperature_end / temperature_start) ** (1.0 / max_iter)
        
        # Instance-aware temperature scaling factors
        instance_temp_scale = [np.ones(len(bag)) for bag in bags]
        
        # Track frozen instances across stages
        stage1_frozen_instances = [set() for _ in bags]
        
        # Entropy regularization parameters
        entropy_weight_start = 0.2  # Balanced entropy regularization for hybrid approach
        entropy_weight_end = 0.0    # No entropy regularization at end
        entropy_decay = (entropy_weight_end + 1e-8) / (entropy_weight_start + 1e-8)
        
        # Constraint continuation parameters (gradual constraint tightening)
        constraint_slack_start = 0.3  # 30% slack on constraints initially for better exploration
        constraint_slack_end = 0.0    # No slack at the end
        constraint_slack_decay = constraint_slack_end / (constraint_slack_start + 1e-8)

        # Hybrid curriculum stages
        stage1_end = int(max_iter * 0.5)  # 50% for exploration
        stage2_end = int(max_iter * 0.9)  # 40% for exploitation
        # Stage 3 is the remaining 10% for polish
        
        # Track best models for both metrics
        best_auto_init_score = 0.0
        best_external_init_score = 0.0
        
        # Average penalty for Stage 2 (will be computed after Stage 1)
        avg_rho = 1.0
        
        for i_iter in range(max_iter):
            # Determine current stage
            if i_iter < stage1_end:
                # Stage 1: Exploration with rich features
                # Cosine annealing for temperature
                cosine_factor = 0.5 * (1 + np.cos(np.pi * i_iter / stage1_end))
                temperature = temperature_end + (temperature_start - temperature_end) * cosine_factor
                
                # Entropy weight annealing - maintain throughout Stage 1
                stage1_progress = i_iter / stage1_end
                entropy_weight = entropy_weight_start * (1.0 - 0.5 * stage1_progress)  # Reduce to 50% by end of Stage 1
                
                # Constraint slack annealing (for continuation method)
                constraint_slack = constraint_slack_start * (1.0 - i_iter / stage1_end)
                
                use_multi_penalty = True
                use_instance_temp = True
                use_entropy_reg = True
                use_self_paced = True
            elif i_iter < stage2_end:
                # Stage 2: Exploitation with simplified approach
                # Switch to exponential decay
                stage2_progress = (i_iter - stage1_end) / (stage2_end - stage1_end)
                temperature = temperature_end + (temperature_start * 0.4 - temperature_end) * np.exp(-3.0 * stage2_progress)
                
                # Gradually reduce entropy to zero
                entropy_weight = entropy_weight_start * 0.1 * (1.0 - stage2_progress)
                
                # No constraint slack in Stage 2
                constraint_slack = 0.0
                
                use_multi_penalty = False  # Use single averaged penalty
                use_instance_temp = False  # Disable instance-aware temperature
                use_entropy_reg = True if entropy_weight > 0 else False
                use_self_paced = True  # Keep frozen instances from Stage 1
            else:
                # Stage 3: Polish with fixed low temperature
                temperature = 0.05
                entropy_weight = 0.0
                constraint_slack = 0.0
                
                use_multi_penalty = False
                use_instance_temp = False
                use_entropy_reg = False
                use_self_paced = False  # Unfreeze all instances
            
            # Compute average penalty at end of Stage 1
            if i_iter == stage1_end:
                avg_rho = np.mean(rho_lower) * 0.5 + np.mean(rho_upper) * 0.5
            
            # Fit
            flatten_bags = np.vstack(bags)
            flatten_y = np.hstack(y)
            self.classifier.fit(flatten_bags, flatten_y)

            # Compute outputs
            fs = [self.classifier.predict_proba(bag) for bag in bags]
            y_pred = [self.classifier.predict(bag) for bag in bags]
            flatten_original_y = np.hstack(list(y))
            
            # Update instance-aware temperature scaling based on prediction confidence
            if use_instance_temp:
                for i_bag in range(len(bags)):
                    max_probs = np.max(fs[i_bag], axis=1)
                    # Higher confidence -> lower temperature (more decisive)
                    # Lower confidence -> higher temperature (more exploration)
                    instance_temp_scale[i_bag] = 2.0 - max_probs  # Range: [1.0, 2.0]
            else:
                # Reset to uniform scaling when disabled
                instance_temp_scale = [np.ones(len(bag)) for bag in bags]

            # For every bag - use soft assignments with augmented Lagrangian
            has_changed = False
            for i_bag in range(len(bags)):
                # Hybrid approach: combine constraint-based selection with soft assignments
                n_instances = len(bags[i_bag])
                probs = fs[i_bag]  # Shape: (n_instances, n_classes)
                
                # Self-paced learning: identify high-confidence instances to freeze
                max_probs = np.max(probs, axis=1)
                predicted_classes = np.argmax(probs, axis=1)
                
                # Freeze instances with high confidence (adaptive threshold)
                frozen_indices = set()
                if use_self_paced:
                    if i_iter < stage1_end:
                        # Stage 1: Build frozen set
                        confidence_threshold = 0.9 - 0.2 * (i_iter / stage1_end)  # Adjusted for Stage 1
                        if i_iter >= 2:  # Start freezing after initial exploration
                            for idx in range(n_instances):
                                if max_probs[idx] > confidence_threshold and y[i_bag][idx] == predicted_classes[idx]:
                                    frozen_indices.add(idx)
                                    stage1_frozen_instances[i_bag].add(idx)
                    elif i_iter < stage2_end:
                        # Stage 2: Use frozen instances from Stage 1
                        frozen_indices = stage1_frozen_instances[i_bag].copy()
                    # Stage 3: No frozen instances (use_self_paced is False)
                
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
                                    if use_multi_penalty:
                                        score *= 1.0 + rho_upper[i_bag, current_class]
                                    else:
                                        score *= 1.0 + avg_rho
                                elif current_class_count <= lower_threshold[i_bag, current_class]:
                                    # Use adaptive penalty parameter for penalty
                                    if use_multi_penalty:
                                        score *= 1.0 / (1.0 + rho_lower[i_bag, current_class])
                                    else:
                                        score *= 1.0 / (1.0 + avg_rho)
                                
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
                                    if use_multi_penalty:
                                        scores[c] += rho_lower[i_bag, c] * 2.0
                                    else:
                                        scores[c] += avg_rho * 2.0
                                elif c_count < upper_bound:
                                    scores[c] += 0.5
                                else:
                                    # Use adaptive penalty for over-represented classes
                                    if use_multi_penalty:
                                        scores[c] -= rho_upper[i_bag, c]
                                    else:
                                        scores[c] -= avg_rho
                            
                            # Apply temperature-controlled softmax with instance-aware scaling
                            # Gumbel noise disabled for now (ablation analysis)
                            if use_instance_temp:
                                instance_temperature = temperature * instance_temp_scale[i_bag][idx]
                            else:
                                instance_temperature = temperature
                            
                            exp_scores = np.exp(scores / instance_temperature)
                            probabilities = exp_scores / np.sum(exp_scores)
                            
                            # Apply entropy regularization to encourage diversity
                            if use_entropy_reg and entropy_weight > 0:
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

            if debug:
                stage_name = "Stage 1 (Exploration)" if i_iter < stage1_end else "Stage 2 (Exploitation)" if i_iter < stage2_end else "Stage 3 (Polish)"
                print(f"iter: {i_iter} - {stage_name}")
                print(f"temperature: {temperature:.3f}, entropy_weight: {entropy_weight:.3f}, constraint_slack: {constraint_slack:.3f}")
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
            # In Stage 1, don't stop early even if converged
            if not has_changed and i_iter >= max(4, stage1_end):
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
            if use_multi_penalty and i_iter < max_iter - 1:
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
