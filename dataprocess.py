from sklearn.naive_bayes import *

from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
import pandas as pd
import numpy as np 
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    
    plt.figure(figsize=(12,6))
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

dataset=pd.read_csv('datanoraw.csv')

raw=dataset.values
raw=shuffle(raw)
data=raw[:raw.shape[0]-3000,:raw.shape[1]-1]
target=raw[:raw.shape[0]-3000,raw.shape[1]-1:]
pre_data=raw[raw.shape[0]-3000:,:raw.shape[1]-1]
pre_target=raw[raw.shape[0]-3000:,raw.shape[1]-1:]
gnb = GaussianNB()

y_pred = gnb.fit(data, target).predict(pre_data)
print(y_pred.T)
y=np.hstack((np.array([y_pred]).T,pre_target,np.array([y_pred]).T-pre_target))
np.savetxt('mypre.csv',y,delimiter=',')
plot_learning_curve(gnb,'The predict for Probability & Statistics',raw[:,:raw.shape[1]-1],raw[:,raw.shape[1]-1:])
