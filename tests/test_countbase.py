import pandas as pd
import numpy as np

from milwrap.countbase import MilCountBasedMultiClassLearner

def generate_class_ratios(n_classes):
    
    while True:
        use_classes = np.random.choice([0, 1], size=n_classes, p=[.8, .2])
        if not np.all(use_classes == 0):
            break
    
    ratio_classes = use_classes * np.random.uniform(low=0, high=1, size=n_classes)
    ratio_classes = ratio_classes**4
    ratio_classes = ratio_classes / np.sum(ratio_classes)

    return ratio_classes

def generate_instance(n_classes, n_instances_of_each_bags):

    class_labels_of_intance_in_bags = [np.random.choice(
        np.arange(n_classes),
        size=n_instance_in_bag,
        p=generate_class_ratios(n_classes)) for n_instance_in_bag in n_instances_of_each_bags]
    
    return class_labels_of_intance_in_bags


class TestCountBase:

    def test_fit(self):

        np.random.seed(123)

        n_classes = 15
        n_bags = 100
        n_max_instance_in_one_bag = 1000
        n_instances_of_each_bags = [np.random.randint(low=0, high=n_max_instance_in_one_bag) for _ in range(n_bags)]
        class_labels_of_instance_in_bags = generate_instance(n_classes, n_instances_of_each_bags)
        count_each_class_of_instance_in_bags = [
            pd.Series(x).value_counts().to_dict() for x in class_labels_of_instance_in_bags
        ]
        count_each_class_of_instance_in_bags_matrix = \
            pd.DataFrame(count_each_class_of_instance_in_bags)[list(range(n_classes))].values
        count_each_class_of_instance_in_bags_matrix = np.nan_to_num(count_each_class_of_instance_in_bags_matrix)
        lower_threshold = np.zeros_like(count_each_class_of_instance_in_bags_matrix)
        upper_threshold = np.zeros_like(count_each_class_of_instance_in_bags_matrix)
        divisions = [0, 50, 100, 200, 1000, n_max_instance_in_one_bag]
        for i_bag in range(n_bags):
            for i_class in range(n_classes):
                positive_count = count_each_class_of_instance_in_bags_matrix[i_bag, i_class]
                for i_division in range(len(divisions)-1):
                    if divisions[i_division] <= positive_count and positive_count < divisions[i_division+1]:
                        lower_threshold[i_bag, i_class] = divisions[i_division]
                        upper_threshold[i_bag, i_class] = divisions[i_division+1]
        
        n_fatures = 7
        x_min = 0
        x_max = 100
        cov_diag = 0.1*40**2
        
        means_of_classes = [np.random.uniform(low=x_min, high=x_max, size=n_fatures) for _ in range(n_classes)]
        covs_of_classes = [np.eye(n_fatures)*cov_diag for _ in range(n_classes)]
        bags = [
            np.vstack([
                np.random.multivariate_normal(
                    means_of_classes[class_label],
                    covs_of_classes[class_label],
                    size=1) for class_label in class_labels_of_instance_in_bag
            ]) for class_labels_of_instance_in_bag in class_labels_of_instance_in_bags
        ]
        
        # from sklearn.ensemble import RandomForestClassifier
        # clf = RandomForestClassifier()

        from sklearn.tree import DecisionTreeClassifier
        clf = DecisionTreeClassifier(min_samples_leaf=10)

        # from sklearn.linear_model import LogisticRegression
        # clf = LogisticRegression()

        # from sklearn.neural_network import MLPClassifier
        # clf = MLPClassifier(alpha=1, max_iter=10)

        learner = MilCountBasedMultiClassLearner(clf)
        clf_mil, y_mil = learner.fit(
            bags,
            lower_threshold,
            upper_threshold,
            n_classes,
            debug_true_y=class_labels_of_instance_in_bags,
            max_iter=10)

        print("MIL instance unit accuracy")
        print(np.mean(clf_mil.predict(np.vstack(bags)) == np.hstack(class_labels_of_instance_in_bags)))
        print("----")

        print("MIL instance unit accuracy (label adjusted)")
        print(np.mean(np.hstack(y_mil) == np.hstack(class_labels_of_instance_in_bags)))
        print("----")

        print("SIL instance unit accuracy")
        clf.fit(np.vstack(bags), np.hstack(class_labels_of_instance_in_bags))
        print(np.mean(clf.predict(np.vstack(bags)) == np.hstack(class_labels_of_instance_in_bags)))
        print("----")


    def test_fit_initialize_externally(self):



        np.random.seed(123)

        n_classes = 15
        n_bags = 100
        n_max_instance_in_one_bag = 1000
        n_instances_of_each_bags = [np.random.randint(low=0, high=n_max_instance_in_one_bag) for _ in range(n_bags)]
        class_labels_of_instance_in_bags = generate_instance(n_classes, n_instances_of_each_bags)
        count_each_class_of_instance_in_bags = [
            pd.Series(x).value_counts().to_dict() for x in class_labels_of_instance_in_bags
        ]
        count_each_class_of_instance_in_bags_matrix = \
            pd.DataFrame(count_each_class_of_instance_in_bags)[list(range(n_classes))].values
        count_each_class_of_instance_in_bags_matrix = np.nan_to_num(count_each_class_of_instance_in_bags_matrix)
        lower_threshold = np.zeros_like(count_each_class_of_instance_in_bags_matrix)
        upper_threshold = np.zeros_like(count_each_class_of_instance_in_bags_matrix)
        divisions = [0, 50, 100, 200, 1000, n_max_instance_in_one_bag]
        for i_bag in range(n_bags):
            for i_class in range(n_classes):
                positive_count = count_each_class_of_instance_in_bags_matrix[i_bag, i_class]
                for i_division in range(len(divisions)-1):
                    if divisions[i_division] <= positive_count and positive_count < divisions[i_division+1]:
                        lower_threshold[i_bag, i_class] = divisions[i_division]
                        upper_threshold[i_bag, i_class] = divisions[i_division+1]
        
        n_fatures = 7
        x_min = 0
        x_max = 100
        cov_diag = 0.1*40**2
        
        means_of_classes = [np.random.uniform(low=x_min, high=x_max, size=n_fatures) for _ in range(n_classes)]
        covs_of_classes = [np.eye(n_fatures)*cov_diag for _ in range(n_classes)]
        bags = [
            np.vstack([
                np.random.multivariate_normal(
                    means_of_classes[class_label],
                    covs_of_classes[class_label],
                    size=1) for class_label in class_labels_of_instance_in_bag
            ]) for class_labels_of_instance_in_bag in class_labels_of_instance_in_bags
        ]
        
        # from sklearn.ensemble import RandomForestClassifier
        # clf = RandomForestClassifier()

        from sklearn.tree import DecisionTreeClassifier
        clf = DecisionTreeClassifier(min_samples_leaf=10)

        # from sklearn.linear_model import LogisticRegression
        # clf = LogisticRegression()

        # from sklearn.neural_network import MLPClassifier
        # clf = MLPClassifier(alpha=1, max_iter=10)

        learner = MilCountBasedMultiClassLearner(clf)

        # use externally initialized y
        n_list = [len(bag) for bag in bags]
        initial_y = [np.repeat(class_index, n_instance_in_bag) for class_index, n_instance_in_bag in zip(np.argmax(lower_threshold, axis=1), n_list)]
        
        clf_mil, y_mil = learner.fit(
            bags,
            lower_threshold,
            upper_threshold,
            n_classes,
            initial_y=initial_y,
            debug_true_y=class_labels_of_instance_in_bags,
            max_iter=10)

        print("MIL instance unit accuracy")
        print(np.mean(clf_mil.predict(np.vstack(bags)) == np.hstack(class_labels_of_instance_in_bags)))
        print("----")

        print("MIL instance unit accuracy (label adjusted)")
        print(np.mean(np.hstack(y_mil) == np.hstack(class_labels_of_instance_in_bags)))
        print("----")

        print("SIL instance unit accuracy")
        clf.fit(np.vstack(bags), np.hstack(class_labels_of_instance_in_bags))
        print(np.mean(clf.predict(np.vstack(bags)) == np.hstack(class_labels_of_instance_in_bags)))
        print("----")

