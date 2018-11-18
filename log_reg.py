from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
import util

class LogisticRegression():
    def __init__(self):
        self.clf = LogisticRegression(random_state=0, solver='lbfgs')

    def fit(self, X_train, Y_train):
        self.clf.fit(X_train, y_train)

    def predict(self, x):
        predictions = self.clf.predict(x)

if __name__ == '__main__':
    pass
