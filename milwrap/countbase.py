import numpy as np
import pandas as pd
import sklearn
import random

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
            debug_true_y=None,
            debug=True):
        
        # number of instances of each bags
        n_list = [len(bag) for bag in bags]
        
        # initialize y
        y = [np.repeat(class_index, n_instance_in_bag) for class_index, n_instance_in_bag in zip(np.argmax(lower_threshold, axis=1), n_list)]

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
                for i_class in range(n_classes):
                    class_count_dict = pd.Series(y[i_bag]).value_counts().to_dict()
                    class_count = class_count_dict.get(i_class, 0)
                    if class_count < lower_threshold[i_bag, i_class]:
                        # fs is minus
                        indice_should_be_positive = np.argsort(-fs[i_bag][:, i_class])[:int(lower_threshold[i_bag, i_class])]
                        y[i_bag][indice_should_be_positive] = i_class
                        has_changed = True
                    elif upper_threshold[i_bag, i_class] <= class_count:
                        indice_should_be_negative = np.argsort(fs[i_bag][:, i_class])[:(n_list[i_bag] - int(upper_threshold[i_bag, i_class]))]
                        indice_should_change_to_be_negative = list(
                            set(indice_should_be_negative.tolist()).intersection(
                                set(np.argwhere(y[i_bag] == i_class).ravel().tolist())))
                        y[i_bag][indice_should_change_to_be_negative] = np.random.choice(n_classes, size=1)  # TODO
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


class MilCountBasedBinaryClassLearner:

    def __init__(self, classifier):
        self.classifier = classifier
    
    def fit(
            self,
            bags,
            lower_threshold,
            upper_threshold,
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
            if np.median(lower_threshold) == np.max(lower_threshold) or np.median(lower_threshold) == np.min(lower_threshold):
                threshold = np.mean(lower_threshold)
            else:
                threshold = np.median(lower_threshold)
            y = [np.repeat((lower_threshold_for_current_bag > threshold), n_instance_in_bag) for lower_threshold_for_current_bag, n_instance_in_bag in zip(lower_threshold, n_list)]
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
                class_count = np.sum(y[i_bag])
                if class_count < lower_threshold[i_bag]:
                    # fs is minus
                    argsorted_with_score = np.argsort(-fs[i_bag])
                    indice_should_be_positive = argsorted_with_score[:int(lower_threshold[i_bag])]
                    y[i_bag][indice_should_be_positive] = 1
                    has_changed = True
                elif upper_threshold[i_bag] <= class_count:
                    argsorted_with_score = np.argsort(fs[i_bag])
                    indice_should_be_negative = argsorted_with_score[:(n_list[i_bag] - int(upper_threshold[i_bag]))]
                    y[i_bag][indice_should_be_negative] = 0
                    has_changed = True
        
            if debug:
                print("iter:", i_iter)
                predicted_count = [np.sum(b) for b in y]
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


class OneVsRestMilCountBasedMultiClassLearner:

    def __init__(self, binary_classifiers):
        assert len(binary_classifiers) > 0
        self.binary_classifiers = binary_classifiers
    
    def fit(
            self,
            bags,
            lower_threshold,
            upper_threshold,
            n_classes,
            max_iter=10,
            initial_y=None,
            seed=None):

        if seed is not None:
            random.seed(seed)
        
        assert len(self.binary_classifiers) == n_classes, (self.binary_classifiers, n_classes)
        for i_class, binary_classifier in enumerate(self.binary_classifiers):

            lower_threshold[:, i_class]

            learner = MilCountBasedBinaryClassLearner(binary_classifier)
            learner.fit(
                bags = bags,
                lower_threshold = lower_threshold[:, i_class],
                upper_threshold = upper_threshold[:, i_class],
                max_iter = max_iter,
                initial_y = (np.array(initial_y) == i_class).astype(int) if initial_y is not None else None,
            )
            self.binary_classifiers[i_class] = learner.classifier
        
        return self.binary_classifiers



def convert_binary_classifiers_to_ovr_multiclassifier(
        n_classes, n_features, classifiers):
    
    clf = sklearn.multiclass.OneVsRestClassifier(
        sklearn.tree.DecisionTreeClassifier(min_samples_leaf=10))
    
    dummy_n_classes = n_classes
    dummy_n_sample_per_classes = 10
    dummy_n = dummy_n_sample_per_classes * dummy_n_classes
    dummy_d = n_features
    dummy_X = np.random.normal(size=(dummy_n, dummy_d))
    dummy_y = np.repeat(np.arange(dummy_n_classes), dummy_n_sample_per_classes)
    
    # fit to dummy data
    clf.fit(dummy_X, dummy_y)

    for i_estimator, estimator in enumerate(clf.estimators_):
        assert type(estimator) == type(classifiers[i_estimator])
        clf.estimators_[i_estimator] = classifiers[i_estimator]
    return clf
