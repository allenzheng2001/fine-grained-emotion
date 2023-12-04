import numpy as np
import pandas as pd
import random

from math import sqrt as sqrt
from tqdm import tqdm as tqdm

class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """
    def __init__(self, num_classes, dimensions):
        self.weights = np.zeros((num_classes, dimensions))

    def dot(self, x_i):
        return np.dot(self.weights, x_i)

    def predict(self, x_i):
        """
        :param sentence: words (List[str]) in the sentence to classify
        :return: Either 0 for negative class or 1 for positive class
        """
        raise Exception("Don't call me, call my subclasses")


    
class LogisticRegressionClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self, num_classes, dimensions, step_size):
        super().__init__(num_classes, dimensions)
        self.step_size = step_size

    def set_step_size(self, step_size):
        self.step_size = step_size

    def train(self, x_i, y_i):
        # x_i: vector of len 24 (appraisal dimensions)
        # y_i: vector of len 8 (emotions)
        res =  self.dot(x_i)
        y_label = y_i
        
        p_yes = 1/(1+np.exp(-res))
        factor = self.step_size*np.where(y_i == 1, 1-p_yes, -p_yes)

        self.weights += np.outer(factor, x_i) # shape: (8, 24)

    def predict(self, x_i) -> int:
        return (self.dot(x_i) > 0).astype(int)
    

def train_lr(train_exs) -> LogisticRegressionClassifier:
    """
    Train a logistic regression model.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained LogisticRegressionClassifier model
    """
    lr = None
    random.seed(40)
    num_epochs = 30
    step_sz_factor = 3e-3

    for t in tqdm(range(num_epochs)):
        if lr is not None:
            lr.set_step_size(1/sqrt(t+1)*step_sz_factor)
        random.shuffle(train_exs)
        for x_i, y_i in train_exs:
            if lr is None:
                lr = LogisticRegressionClassifier(num_classes=len(y_i), dimensions = len(x_i), step_size = step_sz_factor)
            lr.train(x_i, y_i)

    return lr