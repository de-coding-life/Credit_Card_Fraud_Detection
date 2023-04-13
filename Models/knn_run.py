import pandas as pd
import pickle
from sklearn.model_selection import train_test_split as split
from sklearn.metrics import *
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

model=pickle.load(open('Models\knn.pkl', 'rb'))
res=model.predict(x_test)

print("Accuracy is:", accuracy_score(y_test, res))
print("Precision is:", precision_score(y_test, res))
print("Recall is:", recall_score(y_test, res))
print("F1 score is:", f1_score(y_test, res))

x_bar=["Accuracy","Precision","Recall","F1 Score"]
y_bar=[accuracy_score(y_test, res),precision_score(y_test, res),recall_score(y_test, res),f1_score(y_test, res)]

plt.barh(x_bar,y_bar)
for index, value in enumerate(y_bar):
    plt.text(value, index,
             str(value))
plt.show()