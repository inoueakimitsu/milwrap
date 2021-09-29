import numpy as np
import pandas as pd

class MilCountBasedMultiClassLearner:
    def __init__(self, classifier, n_classes):
        self.classifier = classifier
        self.n_classes = n_classes
    
    def fit(self, bags, lower_threshold, upper_threshold, n_classes, default_class=0):
        """

        lower_threshold = n_bags * n_classes
        upper_threshold = n_bags * n_classes    
        """
        
        # number of instances of each bags
        n_list = [len(bag) for bag in bags]
        
        # initialize y_i = Y_I for i \in I
        y = [np.repeat(class_index, n_instance_in_bag) for class_index, n_instance_in_bag in zip(np.argmax(lower_threshold, axis=1), n_list)]

        while True:
            # fit
            flatten_bags = np.vstack(bags)
            flatten_y = np.hstack(y)
            self.classifier.fit(flatten_bags, flatten_y)

            # compute outputs
            fs = [self.classifier.predict_proba(bag) for bag in bags]
            y = [self.classifier.predict(bag) for bag in bags]

            # for every bag
            has_changed = False
            for i_bag in range(len(bags)):
                class_count_dict = pd.Series(y[i_bag]).value_counts().to_dict()
                for i_class in range(n_classes):
                    class_count = class_count_dict.get(i_class, 0)
                    if class_count < lower_threshold[i_bag, i_class]:
                        # fs is minus
                        indice_should_be_positive = np.argsort(-fs[i_bag][:, i_class])[:int(lower_threshold[i_bag, i_class])]
                        y[i_bag][indice_should_be_positive] = i_class
                        has_changed = True
                    elif upper_threshold[i_bag, i_class] < class_count:
                        indice_should_be_negative = np.argsort(fs[i_bag][:, i_class])[:int(upper_threshold[i_bag, i_class])]
                        y[i_bag][indice_should_be_negative] = default_class  # TODO
                        has_changed = True
            
            if not has_changed:
                break

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


if __name__ == '__main__':
    # datasets
    # class_names = n_classes
    # lower_threshold = n_bags * n_classes
    # upper_threshold = n_bags * n_classes    
    np.random.seed(123)

    n_classes = 15
    n_bags = 100
    n_max_instance_in_one_bag = 1000
    n_instances_of_each_bags = [np.random.randint(low=0, high=n_max_instance_in_one_bag) for _ in range(n_bags)]
    class_labesl_of_instance_in_bags = generate_instance(n_classes, n_instances_of_each_bags)
    count_each_class_of_instance_in_bags = [
        pd.Series(x).value_counts().to_dict() for x in class_labesl_of_instance_in_bags
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
    
    n_fatures = 2
    x_min = 0
    x_max = 100
    cov_diag = 40**2
    
    means_of_classes = [np.random.uniform(low=x_min, high=x_max, size=n_fatures) for _ in range(n_classes)]
    covs_of_classes = [np.eye(n_fatures)*cov_diag for _ in range(n_classes)]
    bags = [
        np.vstack([
            np.random.multivariate_normal(
                means_of_classes[class_label],
                covs_of_classes[class_label],
                size=1) for class_label in class_labels_of_instance_in_bag
        ]) for class_labels_of_instance_in_bag in class_labesl_of_instance_in_bags
    ]
    
    # from sklearn.tree import DecisionTreeClassifier
    # from sklearn.ensemble import RandomForestClassifier
    # clf = DecisionTreeClassifier()
    # clf = RandomForestClassifier()

    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression()
    learner = MilCountBasedMultiClassLearner(clf, n_classes)
    learner.fit(bags, lower_threshold, upper_threshold, n_classes)



