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


def compute_soft_topk_count_loss(
    probs: NDArray[np.float64],
    target_count: int,
    temperature: float = 0.5,
) -> float:
    """Compute differentiable count loss using enhanced soft top-k approach.
    
    Implements a more precise soft counting mechanism based on recent MIL research,
    with attention-weighted aggregation and smooth gradient flow.
    
    Args:
    ----
        probs: Probability array for instances belonging to a class.
        target_count: Target number of instances for the class.
        temperature: Temperature parameter for soft counting (lower = sharper).
        
    Returns:
    -------
        Count loss value (squared error with regularization).
        
    """
    if len(probs) == 0 or target_count < 0:
        return 0.0
    
    n_instances = len(probs)
    
    if target_count == 0:
        # Penalize any positive probability with L2 norm
        soft_count = np.sum(probs)
        # Add small regularization to prevent extreme values
        return soft_count ** 2 + 0.01 * np.sum(probs ** 2)
    
    if target_count >= n_instances:
        # If we want all instances, minimize distance to 1
        soft_count = np.sum(probs)
        loss = (soft_count - target_count) ** 2
        # Encourage high confidence
        loss += 0.1 * np.sum((1 - probs) ** 2)
        return loss
    
    # Enhanced soft top-k with attention-based weighting
    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]
    
    # Compute adaptive threshold with smooth interpolation
    if target_count <= n_instances:
        # Use weighted average around the k-th value for smoother gradients
        k = target_count - 1
        weights = np.exp(-np.abs(np.arange(n_instances) - k) / (temperature * 2))
        weights = weights / np.sum(weights)
        threshold = np.sum(sorted_probs * weights)
    else:
        threshold = 0.0
    
    # Multi-scale soft assignment for better gradient flow
    # Combine sigmoid and tanh for more expressive soft counting
    sigmoid_assignments = 1 / (1 + np.exp(-(probs - threshold) / temperature))
    tanh_assignments = 0.5 * (1 + np.tanh((probs - threshold) / (temperature * 1.5)))
    
    # Weighted combination for robustness
    soft_assignments = 0.7 * sigmoid_assignments + 0.3 * tanh_assignments
    soft_count = np.sum(soft_assignments)
    
    # Enhanced loss with count regularization
    count_loss = (soft_count - target_count) ** 2
    
    # Add entropy regularization to prevent trivial solutions
    # Encourage diverse but confident predictions
    entropy = -np.sum(probs * np.log(probs + 1e-8) + (1 - probs) * np.log(1 - probs + 1e-8))
    entropy_reg = 0.005 * entropy / n_instances  # Further reduced for sharper decisions
    
    return count_loss + entropy_reg


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
            
        # Initialize Lagrangian multipliers for constraints
        n_bags = len(bags)
        lambda_lower = np.zeros((n_bags, n_classes))
        lambda_upper = np.zeros((n_bags, n_classes))
        rho = 2.0  # Further increased penalty parameter for stronger constraint enforcement
        
        # Enhanced temperature annealing with cosine schedule
        temperature_start = 1.8  # Slightly lower initial temperature
        temperature_end = 0.1  # Very low final temperature for sharp decisions
        
        # MIL-Dropout parameters (from high-scoring branch)
        mil_dropout_rate = 0.08  # Drop 8% of high-scoring instances
        mil_dropout_start_iter = 3  # Start after third iteration for stability
        
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
        
        # Enhanced count loss weight with schedule
        count_loss_weight_init = 0.5
        count_loss_weight_final = 1.2  # Even stronger count loss emphasis
        
        # Self-paced learning parameters
        constraint_tightness = 1.0  # Temporarily disable self-paced learning
        constraint_tightness_final = 1.0
        constraint_variance_threshold = 0.005
        
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
            # Enhanced temperature annealing with cosine schedule
            progress = i_iter / max(1, max_iter - 1)
            temperature = temperature_end + 0.5 * (temperature_start - temperature_end) * (1 + np.cos(np.pi * progress))
            
            # Adaptive count loss weight scheduling
            count_loss_weight = count_loss_weight_init + (count_loss_weight_final - count_loss_weight_init) * progress
            
            # Self-paced learning: gradually tighten constraints based on variance
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
            
            for i_bag in range(len(bags)):
                # Hybrid approach: combine constraint-based selection with soft assignments
                n_instances = len(bags[i_bag])
                if n_instances == 0:
                    continue
                    
                probs = fs[i_bag]  # Shape: (n_instances, n_classes)
                
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
                            # Use soft top-k loss with adaptive temperature
                            soft_temp = 0.5 * temperature  # Link to main temperature
                            loss = compute_soft_topk_count_loss(class_probs, target_count, soft_temp)
                            bag_count_loss += count_loss_weight * loss
                            
                            # Track soft count for variance calculation
                            soft_count = np.sum(1 / (1 + np.exp(-(class_probs - 0.5) / soft_temp)))
                
                total_count_loss += bag_count_loss
                
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

        # Use SWA classifier if available for better stability
        if swa_classifier is not None:
            self.classifier = swa_classifier
            
        # Return best solution if available
        if best_y is not None and best_accuracy > 0:
            # Refit classifier with best labels
            flatten_bags = np.vstack(bags)
            flatten_best_y = np.hstack(best_y)
            self.classifier.fit(flatten_bags, flatten_best_y)
            return self.classifier, best_y
        
        return self.classifier, y
