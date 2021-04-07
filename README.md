# knn-using-scikit
 #Importing libraries
 import numpy as np
 import matplotlib.pyplot as plt
 import pandas as pd

 
 
 
 #importing the dataset
 dataset = pd.read_csv("train.csv")
 data = dataset.values[:7000]
 data.shape

 #mnist = load_digit()
 #type(mnist)
 #mnist.keys()
 #pd.df(train.data).head()

#split into attributes and labels
x=dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

#training and test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

#feature scaling 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#predictions
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

#Evaluating
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

