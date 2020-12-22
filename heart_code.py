import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('heart.csv', header = None)

df.columns = ['age', 'sex', 'cp', 'trestbps', 'chol',
              'fbs', 'restecg', 'thalach', 'exang', 
              'oldpeak', 'slope', 'ca', 'thal', 'target']

### 1 = male, 0 = female


################################## data preprocessing ##########################################################
X=dataset.drop(['target'],axis=1)    
X.head()


y=dataset['target']
y.head()


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

from sklearn.preprocessing import StandardScaler as ss
sc = ss()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)




#########################################  KMeans  #############################################################

from sklearn.cluster import KMeans
km=KMeans(n_clusters=2,random_state=0)
km.fit(X_train,X_test)

from sklearn.metrics import confusion_matrix,accuracy_score
cm= confusion_matrix(y_pred,y_test)
print(cm)

acc= accuracy_score(y_pred,y_test)
print(acc)

from sklearn.metrics import classification_report
cs= classification_report(y_pred,y_test)
print(cs)




#########################################   KNN  #############################################################
from sklearn.svm import knn
print("Training K-Nearest Neighbors")
knn = []
for i in range(1, 21):
    knn_classifier = KNeighborsClassifier(n_neighbors = i)
    knn_classifier.fit(X_train, Y_train)
    knn.append(knn_classifier.score(X_test, Y_test))

# Predicting the Test set results
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm_test = confusion_matrix(y_pred, y_test)

y_pred_train = classifier.predict(X_train)
cm_train = confusion_matrix(y_pred_train, y_train)

print()
print('Accuracy for training set for knn = {}'.format((cm_train[0][0] + cm_train[1][1])/len(y_train)))
print('Accuracy for test set for knn = {}'.format((cm_test[0][0] + cm_test[1][1])/len(y_test)))




#########################################   SVM   #############################################################
from sklearn.svm import svm
classifier = svm(kernel = 'rbf')
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm_test = confusion_matrix(y_pred, y_test)

y_pred_train = classifier.predict(X_train)
cm_train = confusion_matrix(y_pred_train, y_train)

print()
print('Accuracy for training set for svm = {}'.format((cm_train[0][0] + cm_train[1][1])/len(y_train)))
print('Accuracy for test set for svm = {}'.format((cm_test[0][0] + cm_test[1][1])/len(y_test)))




#########################################   DecisionTreeClassifier  #############################################################

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier.fit(x_train,y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm_test = confusion_matrix(y_pred, y_test)

y_pred_train = classifier.predict(X_train)
cm_train = confusion_matrix(y_pred_train, y_train)

print()
print('Accuracy for training set for DecisionTreeClassifier = {}'.format((cm_train[0][0] + cm_train[1][1])/len(y_train)))
print('Accuracy for test set for DecisionTreeClassifier = {}'.format((cm_test[0][0] + cm_test[1][1])/len(y_test)))

