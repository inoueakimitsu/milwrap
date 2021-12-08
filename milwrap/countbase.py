from typing import List
import random

import numpy as np
import pandas as pd
import scipy.stats


def get_order_based_initial_bag_labels(
        n_instances_for_each_bag_and_class: np.ndarray) -> List[int]:
    
    assert len(n_instances_for_each_bag_and_class.shape) == 2

    n_bags = n_instances_for_each_bag_and_class.shape[0]
    n_classes = n_instances_for_each_bag_and_class.shape[1]

    bag_order = np.zeros_like(n_instances_for_each_bag_and_class, dtype=int)
    
    for i_class in range(n_classes):
        bag_order[:, i_class] = scipy.stats.rankdata(n_instances_for_each_bag_and_class[:, i_class])

    initial_bag_labels = [0 for _ in range(n_bags)]

    for i_bag in range(n_bags):
        initial_bag_labels[i_bag] = np.argmax(bag_order[i_bag, :])
    
    return initial_bag_labels


def get_order_based_initial_y(
        lower_threshold: np.ndarray,
        upper_threshold: np.ndarray,
        n_instances_of_each_bags: List[int],
        ) -> List[np.ndarray]:
    return [np.repeat(label, n) for n, label in zip(
        n_instances_of_each_bags,
        get_order_based_initial_bag_labels(
            (lower_threshold + upper_threshold) / 2))]

class MilCountBasedMultiClassLearner:

    def __init__(self, classifier):
        self.classifier = classifier
    
    def fit(
            self,
            bags,
            lower_threshold,
            upper_threshold,
            n_classes,
            max_iter=10,
            initial_y=None,
            debug_true_y=None,
            seed=None,
            debug=True):

        if seed is not None:
            random.seed(seed)
        
        # initialize y
        n_list = [len(bag) for bag in bags]
        if not initial_y:
            # number of instances of each bags
            y = [np.repeat(class_index, n_instance_in_bag) for class_index, n_instance_in_bag in zip(np.argmax(lower_threshold, axis=1), n_list)]
        else:
            y = initial_y

        for i_iter in range(max_iter):

            # fit
            flatten_bags = np.vstack(bags)
            flatten_y = np.hstack(y)
            self.classifier.fit(flatten_bags, flatten_y)

            # compute outputs
            fs = [self.classifier.predict_proba(bag) for bag in bags]
            y = [self.classifier.predict(bag) for bag in bags]
            flatten_original_y = np.hstack([s for s in y])

            # for every bag
            has_changed = False
            for i_bag in range(len(bags)):
                target_classes = list(range(n_classes))
                # Ensure that there is no biased preference for a particular class.
                random.shuffle(target_classes)                
                # Prevents multiple classes from conflictingly changing the label of a single instance in a single iteration
                changed_indice_list = []
                for i_class in target_classes:
                    class_count_dict = pd.Series(y[i_bag]).value_counts().to_dict()
                    class_count = class_count_dict.get(i_class, 0)
                    if class_count < lower_threshold[i_bag, i_class]:
                        # fs is minus
                        argsorted_with_score = np.argsort(-fs[i_bag][:, i_class])
                        # In this iteration, this instance will only be changed to this class and not to any other class
                        argsorted_with_score = argsorted_with_score[np.isin(argsorted_with_score, changed_indice_list, assume_unique=True, invert=True)]
                        indice_should_be_positive = argsorted_with_score[:int(lower_threshold[i_bag, i_class])]
                        # Register the instance whose label has been changed.
                        changed_indice_list.extend(indice_should_be_positive.tolist())
                        y[i_bag][indice_should_be_positive] = i_class
                        has_changed = True
                    elif upper_threshold[i_bag, i_class] <= class_count:
                        argsorted_with_score = np.argsort(fs[i_bag][:, i_class])
                        # In this iteration, this instance will only be changed to this class and not to any other class
                        argsorted_with_score = argsorted_with_score[np.isin(argsorted_with_score, changed_indice_list, assume_unique=True, invert=True)]
                        indice_should_be_negative = argsorted_with_score[:(n_list[i_bag] - int(upper_threshold[i_bag, i_class]))]
                        indice_should_change_to_be_negative = list(
                            set(indice_should_be_negative.tolist()).intersection(
                                set(np.argwhere(y[i_bag] == i_class).ravel().tolist())))
                        # Register the instance whose label has been changed.
                        changed_indice_list.extend(indice_should_change_to_be_negative)
                        # Instances above the upper limit will be randomly assigned to other classes.
                        y[i_bag][indice_should_change_to_be_negative] = np.random.choice(
                            n_classes, size=len(indice_should_change_to_be_negative))  # TODO
                        has_changed = True
            
            if debug:
                print("iter:", i_iter)
                predicted_count = np.nan_to_num(pd.DataFrame([pd.Series(b).value_counts().to_dict() for b in y])[list(range(n_classes))].values)
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
                    print(np.mean(np.hstack([self.classifier.predict(bag) for bag in bags]) == np.hstack(debug_true_y)))
                    print("instance unit accuracy (label adjusted)")
                    print(np.mean(np.hstack(y) == np.hstack(debug_true_y)))
                print("-----")
            
            if not has_changed:
                break
        
        return self.classifier, y
