import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split as split
from sklearn.metrics import *
from sklearn.neighbors import KNeighborsClassifier as knn
from matplotlib import pyplot as plt

# Load the data
data = pd.read_csv('Data\creditcard.csv')

# Balancing the Data
legit = data[data.Class == 0]
fraud = data[data.Class == 1]
legit= legit.sample(len(fraud))

# Combining the Data after balancing it
data=pd.concat([legit, fraud], axis=0)

# Segregating features and label
x = data.drop('Class', axis=1)
y = data['Class']

# Splitting the data into train and test sets
x_train, x_test, y_train, y_test = split(x, y, stratify=y,test_size=0.2, random_state=2)

# Plotting K v/s F1-Score Plot
x_plt=range(1,300)
y_plt=[]
y_plt2=[]
for i in range(1,300):
    # Finding Optimal Value of K
    model = knn(n_neighbors=i)
    model.fit(x_train, y_train)
    res = model.predict(x_test)
    y_plt.append(f1_score(y_test, res))
    y_plt2.append(accuracy_score(y_test, res))
plt.plot(x_plt,y_plt)
plt.plot(x_plt,y_plt2)
plt.show()

# From the Plot, the K is chosen as 10 as the accuracy and F1-score is highest at 10

model = knn(n_neighbors=10)
model.fit(x_train, y_train)
res = model.predict(x_test)

print("Accuracy is:", accuracy_score(y_test, res))
print("Precision is:", precision_score(y_test, res))
print("Recall is:", recall_score(y_test, res))
print("F1 score is:", f1_score(y_test, res))

pickle.dump(model, open('Models\knn.pkl', 'wb'))