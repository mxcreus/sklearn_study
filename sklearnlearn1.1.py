#sklearn
#如何从数据表中找出最佳节点和分支
#如何防止过拟合
#剪枝 max_depth
"""
不纯度 




"""
import sklearn
from sklearn import tree
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

wine = load_wine()

"""import pandas as pd

pd.concat([pd.dataFrame(wine.data),pd.dataFrame(wine.target)],axis=1)
"""
Xtrain,Xtest,Ytrain,Ytest = train_test_split(wine.data,wine.target,test_size=0.3)
clf = tree.DecisionTreeClassifier(criterion="entropy")
clf = clf.fit(Xtrain,Ytrain)
score = clf.score(Xtest,Ytest)
print(score)

import graphviz
feature_name = ["酒精","苹果酸","灰","碱性","镁","总酚","类黄酮","类酚类","花青素","颜色强度","1","2","3"]
dot_data = tree.export_graphviz(clf
                                ,feature_names= feature_name
                                ,class_names=["琴酒","雪梨","贝尔摩德"]
                                ,filled=True
                                ,rounded=True)
graph = graphviz.Source(dot_data)
graph.write_pdf("iris.pdf")