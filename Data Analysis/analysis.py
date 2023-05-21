import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv('Data\creditcard.csv')

# Understanding Structure of the data
print(df.head(),'\n')

# Information about Columns
print(df.info(),'\n')

# Statistical Information
print(df.describe(),'\n')

# Checking Count of Null Values
print(df.isnull().sum(),'\n')

# Extracting Count of Real(0) and Fraud(1) Transactions
cn=df['Class'].value_counts()
print(cn,'\n')

# Percentage of Fraud Transactions
print("Percentage of Fraud Transactions",((cn[1]/(cn[0]+cn[1]))*100),'\n')

# Finding Correlation of Class with other Columns
print(df.corr()['Class'],'\n')

# Plotting Histogram for Class
plt.title('Legit='+str((cn[0]/(cn[0]+cn[1]))*100)+'%')
plt.hist(df['Class'])
plt.show()

