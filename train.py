import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from joblib import dump
from model import Perceptron
# dane 

# class Perceptron:
    
#     def __init__(self, eta=0.01, n_iter=10):
#         self.eta = eta
#         self.n_iter = n_iter
    
#     def fit(self, X, y):
#         self.w_ = np.zeros(1 + X.shape[1])
#         self.errors_ = []
#         for _ in range(self.n_iter):
#             errors = 0
#             for xi, target in zip(X,y):
#                 update=self.eta*(target-self.predict(xi))
#                 self.w_[1:] += update *xi
#                 self.w_[0] += update
#                 errors += int(update != 0.0)
#             self.errors_.append(errors)
#         return self
    
    
#     def net_input(self, X):
#         return np.dot(X, self.w_[1:]) + self.w_[0]
    
#     def predict(self, X):
#         return np.where(self.net_input(X) >= 0, 1, -1)



iris = load_iris()
df = pd.DataFrame(data = np.c_[iris['data'], iris['target']],
columns = iris['feature_names']+ ['target'])
X, y = df.iloc[:100, [0,2]].values, df.iloc[:100, 4]

def a(x):
    if x == 0:
        return -1
    else:
        return 1
    
y = y.map(a)



# model train
clf = Perceptron()
clf.fit(X, y)
dump(clf, './model.joblib')