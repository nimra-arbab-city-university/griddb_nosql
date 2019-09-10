# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


#x= [[5.6,70,10],[5,75,12],[4,48,8],[6,55,10],[6,65,7],
#    [5,57,8],[5.6,77,11],[5,45,8],[5.4,43,5],[6,38,8]]
#y= ['male','female','female','male','female','male','male','female','female','male']
#with open() as f:

file1=pd.read_csv("C:\\Users\\user\\Desktop\\file.csv")

mylist=[list(file1['actual gender']),list(file1['height']),list(file1['weigt']),list(file1['shoe_size'])]

x=[]
y=mylist[0]
height=mylist[1]
weigt=mylist[2]
shoe_size=mylist[3]
for i in range(len(y)):
    new_list=[height[i],weigt[i],shoe_size[i]]
    x.append(new_list)

test_data = [[190, 70, 43],[154, 75, 42],[181,65,40]]
test_labels = ['male','male','male']

#DecisionTreeClassifier
dtc_clf = tree.DecisionTreeClassifier()
dtc_clf = dtc_clf.fit(x,y)
dtc_prediction = dtc_clf.predict(test_data)
print (dtc_prediction)

#RandomForestClassifier
rfc_clf = RandomForestClassifier()
rfc_clf.fit(x,y)
rfc_prediction = rfc_clf.predict(test_data)
print (rfc_prediction)

##Support Vector Classifier
s_clf = SVC()
s_clf.fit(x,y)
s_prediction = s_clf.predict(test_data)
print (s_prediction)


#LogisticRegression
l_clf = LogisticRegression()
l_clf.fit(x,y)
l_prediction = l_clf.predict(test_data)
print (l_prediction)

#accuracy scores
dtc_tree_acc = accuracy_score(dtc_prediction,test_labels)
rfc_acc = accuracy_score(rfc_prediction,test_labels)
l_acc = accuracy_score(l_prediction,test_labels)
s_acc = accuracy_score(s_prediction,test_labels)

classifiers = ['Decision Tree', 'Random Forest', 'Logistic Regression' , 'SVC']
accuracy = np.array([dtc_tree_acc, l_acc, rfc_acc, s_acc])
max_acc = np.argmax(accuracy)
print(classifiers[max_acc] + ' is the best classifier for this problem')
