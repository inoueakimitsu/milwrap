import numpy as np
import pandas as pd

class MilCountBasedMultiClassLearner:
    def __init__(self, classifier, n_classes):
        self.classifier = classifier
        self.n_classes = n_classes
    
    def fit(self, bags, positive_count_of_groups):
        """
        positive_count_of_groups ... n_bags * n_classes
        """
        
        # number of instances of each bags
        n_list = [len(bag) for bag in bags]
        
        # initialize y_i = Y_I for i \in I
        for i_class in range(n_classes):
            y = [np.repeat(positive_count_of_group, n) for positive_count_of_group, n in zip(positive_count_of_groups, n_list)]

        # fit
        flatten_bags = np.vstack(bags)
        flatten_y = np.vstack(y)
        self.classifier.fit(flatten_bags, flatten_y)

        # compute outputs
        fs = [self.classifier.predict_proba(bag) for bag in bags]

        y = []

if __name__ == '__main__':
    pass
