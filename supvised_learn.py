import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import seaborn as sns

print("Supervised Learning Model")
#Importing the datasets
dataset = pd.read_csv("iris.csv")
dataset.describe()

#Splitting the dataset into the Training set and test set
x = dataset.iloc[:, [0,1,2,3]].values
y = dataset.iloc[:, 4].values


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)


sc = StandardScaler()

x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#Fitting Logistic Regression to the Training set
classifier = LogisticRegression(random_state=0, solver='lbfgs', multi_class='auto')
classifier.fit(x_train, y_train)

#Predicting the Test set results
y_pred = classifier.predict(x_test)

#Predict probabilities
probs_y = classifier.predict_proba(x_test)

cm = confusion_matrix(y_test, y_pred)
print(cm)

#Plot confusion matrix
#confusion matrix sns heatmap
ax = plt.axes()
df_cm = cm
sns.heatmap(df_cm, annot=True, annot_kws={"size":30}, fmt='d', cmap="Blues", ax = ax)
ax.set_title("Confusion Matrix")
plt.show()
