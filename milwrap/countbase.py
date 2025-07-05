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

        for i_iter in range(max_iter):
            # Fit
            flatten_bags = np.vstack(bags)
            flatten_y = np.hstack(y)
            self.classifier.fit(flatten_bags, flatten_y)

            # Compute outputs
            fs = [self.classifier.predict_proba(bag) for bag in bags]
            y = [self.classifier.predict(bag) for bag in bags]
            flatten_original_y = np.hstack(list(y))

            # For every bag
            has_changed = False
            for i_bag in range(len(bags)):
                target_classes = list(range(n_classes))
                # Ensure that there is no biased preference for a particular class
                random.shuffle(target_classes)
                # Prevents multiple classes from conflictingly changing the label
                changed_indice_list = []

                for i_class in target_classes:
                    class_count_dict = pd.Series(y[i_bag]).value_counts().to_dict()
                    class_count = class_count_dict.get(i_class, 0)

                    if class_count < lower_threshold[i_bag, i_class]:
                        # fs is minus
                        argsorted_with_score = np.argsort(-fs[i_bag][:, i_class])
                        # In this iteration, this instance will only be changed to this class
                        argsorted_with_score = argsorted_with_score[
                            np.isin(
                                argsorted_with_score,
                                changed_indice_list,
                                assume_unique=True,
                                invert=True,
                            )
                        ]
                        indice_should_be_positive = argsorted_with_score[: int(lower_threshold[i_bag, i_class])]
                        # Register the instance whose label has been changed
                        changed_indice_list.extend(indice_should_be_positive.tolist())
                        y[i_bag][indice_should_be_positive] = i_class
                        has_changed = True

                    elif upper_threshold[i_bag, i_class] <= class_count:
                        argsorted_with_score = np.argsort(fs[i_bag][:, i_class])
                        # In this iteration, this instance will only be changed to this class
                        argsorted_with_score = argsorted_with_score[
                            np.isin(
                                argsorted_with_score,
                                changed_indice_list,
                                assume_unique=True,
                                invert=True,
                            )
                        ]
                        indice_should_be_negative = argsorted_with_score[
                            : (n_list[i_bag] - int(upper_threshold[i_bag, i_class]))
                        ]
                        indice_should_change_to_be_negative = list(
                            set(indice_should_be_negative.tolist()).intersection(
                                set(np.argwhere(y[i_bag] == i_class).ravel().tolist()),
                            ),
                        )
                        # Register the instance whose label has been changed
                        changed_indice_list.extend(indice_should_change_to_be_negative)
                        # Instances above the upper limit will be assigned intelligently
                        # Get current class counts for this bag
                        current_counts = pd.Series(y[i_bag]).value_counts().to_dict()
                        
                        for idx in indice_should_change_to_be_negative:
                            probs = fs[i_bag][idx]
                            
                            # Calculate weighted scores for each class
                            scores = np.zeros(n_classes)
                            for c in range(n_classes):
                                if c == i_class:
                                    scores[c] = -np.inf  # Exclude current class
                                    continue
                                
                                class_count = current_counts.get(c, 0)
                                # Base score from probability (favor uncertain predictions)
                                uncertainty_score = 1.0 - abs(probs[c] - 0.5) * 2
                                
                                # Constraint satisfaction bonus
                                constraint_bonus = 0.0
                                if class_count < lower_threshold[i_bag, c]:
                                    # Strong bonus for classes below lower threshold
                                    constraint_bonus = 2.0 * (lower_threshold[i_bag, c] - class_count) / lower_threshold[i_bag, c]
                                elif class_count < upper_threshold[i_bag, c]:
                                    # Small bonus for classes within bounds
                                    constraint_bonus = 0.5
                                
                                # Combine scores: prioritize constraint satisfaction, then uncertainty
                                scores[c] = constraint_bonus + 0.3 * uncertainty_score
                            
                            # Use softmax to convert scores to probabilities
                            # Temperature parameter controls randomness (lower = more deterministic)
                            temperature = 0.5
                            exp_scores = np.exp(scores / temperature)
                            probabilities = exp_scores / np.sum(exp_scores)
                            
                            # Select class based on probabilities
                            best_class = np.argmax(probabilities)
                            
                            y[i_bag][idx] = best_class
                            # Update current counts
                            current_counts[i_class] = current_counts.get(i_class, 0) - 1
                            current_counts[best_class] = current_counts.get(best_class, 0) + 1
                        has_changed = True

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

            if not has_changed:
                break

        return self.classifier, y
