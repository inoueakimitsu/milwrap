import pandas as pd

class MilCountBasedClassifier:
    def __init__(self, classifier):
        self.classifier = classifier
    def fit(self, x, positive_count_of_groups):
        
        # initialize y_i = Y_I for i \in I
        y = [ for positive_count_of_group in positive_count_of_groups]

if __name__ == '__main__':

