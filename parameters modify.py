
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = load_breast_cancer()

rfc = RandomForestClassifier(n_estimators=10,random_state=90)
score_pre = cross_val_score(rfc,data.data,data.target,cv=10).mean()
print(score_pre)

score1 = []
for i in range(1,200,1):
    rfc = RandomForestClassifier(n_estimators=i,
                                 n_jobs=-1,
                                 random_state=90)
    score = cross_val_score(rfc,data.data,data.target,cv=10).mean()
    score1.append(score)
print(max(score1),(score1.index(max(score1))*10)+1)
plt.figure(figsize=[20,5])
plt.plot(range(1,200,1),score1)
plt.show()