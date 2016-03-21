""" For practice of sklearn """
""" Source: documentation on scikit-learn.org for GaussianNB"""

import numpy as np
from sklearn.naive_bayes import GaussianNB

""" Example 1: Basic example in Guassian Naive Bayes"""
def test():
    X = np.array([[-1,-1],[-2, -1], [-3,-2],[1,1], [2,1], [3,2]])
    Y = np.array([1, 1, 1, 2, 2, 2])
    clf = GaussianNB()
    print(clf.fit(X, Y))
    print(clf.predict([[-0.8, -1]]))

def test_run():
    test()

if __name__ == '__main__':
    test_run()
