import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score


data = np.loadtxt( "D:/works/3 factor authentication/data/data.txt")

X =  data[:,:-1]
y = data[:,-1]

clf = LogisticRegression()
kf = KFold(n_splits=5)
for train, test in kf.split(X):
    X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
    clf.fit(X_train, y_train)

    weights = clf.coef_

    y_pred = clf.predict(X_test)

    f1 = f1_score(y_test, y_pred)

    print("Cross Validation weight", weights)
    print("Cross Validation F1-score:", f1)

print(clf)