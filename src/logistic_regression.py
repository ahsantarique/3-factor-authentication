import numpy as np
from sklearn.linear_model import LogisticRegression


data = np.loadtxt( "D:/works/3 factor authentication/data/data.txt")

X =  data[:,:-1]
y = data[:,-1]

clf = LogisticRegression();
print(clf)