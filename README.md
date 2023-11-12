# Ex-07 Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee
# Date:

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard libraries. 2.Upload the dataset and check for any null values using .isnull() function.
2.Import LabelEncoder and encode the dataset.
3.Import DecisionTreeRegressor from sklearn and apply the model on the dataset. 
4.Predict the values of arrays.
5.Import metrics from sklearn and calculate the MSE and R2 of the model on the dataset 7.Predict the values of array.
6.Apply to new unknown values.
## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: S.Subashini
RegisterNumber: 212222240106 
*/
```
```
import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()

data.info()

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data["Position"]=le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
x.head()

y=data[["Salary"]]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])
```



## Output:

## Initial dataset:
![280373820-1db1e938-9d9d-4121-9f63-6dd77e01f1ac](https://github.com/SubashiniSenniappan/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119404951/7753ab60-d299-434d-97f8-12e233f0b0cf)

## Data info:
![280373948-81134bd4-85bb-4809-babd-90fb83770104](https://github.com/SubashiniSenniappan/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119404951/4481631f-6321-4b5b-8fd0-cc5a452a8eca)

## Optimization of null values:

![280378600-54e9c59b-69ce-4599-924e-0d7cd7ba3d61](https://github.com/SubashiniSenniappan/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119404951/271af81c-1ffe-4407-bf8a-dd5a545e9dc1)

## Converting string literals to numerical values using label encoder:

![280378802-b5221bcc-5bfa-4c51-aecb-33042c8cdd53](https://github.com/SubashiniSenniappan/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119404951/e4abfc3a-4e3f-4858-96c8-aff21194c817)

## Assigning x and y values:

![280378922-3ae61255-5974-4c9d-8412-3ae45d8cf60d](https://github.com/SubashiniSenniappan/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119404951/eba1d057-628b-46a5-a9ce-30a9d2807b8f)

## Mean Squared Error:
![280379034-199eec7f-580a-4b12-81eb-7a617e90ba18](https://github.com/SubashiniSenniappan/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119404951/39fb0b34-5166-4b43-be43-39d641780c44)

## R2 (variance):
![280379192-991437fc-9ae1-4435-8a9d-b1f6cc5a05a1](https://github.com/SubashiniSenniappan/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119404951/ca7d3ce9-8e05-4c70-9e06-df3420f707c1)


## Prediction:
![280379312-085838db-697d-45e0-b61c-33c5428403ae](https://github.com/SubashiniSenniappan/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119404951/13458aaa-dcf5-4c44-ac46-ee9d286fb33b)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
