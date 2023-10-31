# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Upload and read the dataset.
3. Check for any null values using the isnull() function.
4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5. Find the accuracy of the model and predict the required values by importing the required module from sklearn.

## Program:
```
 # Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
 # Developed by: Thiyagarajan A
 # RegisterNumber: 212222240110
import pandas as pd
data=pd.read_csv("dataset/Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evalution","number_project","average_montly_hours","time_spend_company","work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:
### Data Head:

![head](https://github.com/A-Thiyagarajan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118707693/466408d3-08c3-41b4-92c3-31491f0d2ddd)


### Data set info:

![info](https://github.com/A-Thiyagarajan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118707693/465f83f9-2852-4bef-b349-f20502356884)


### Null dataset:

![null](https://github.com/A-Thiyagarajan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118707693/9b75e119-f62d-43db-adfa-35b77781ad8e)


### Values count in left column:

![value](https://github.com/A-Thiyagarajan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118707693/6f638b01-b033-4a91-90b2-a0a45ce1dd2c)


### Dataset transformed head:

![transform](https://github.com/A-Thiyagarajan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118707693/9cfaed5f-5b6d-468c-875d-aa7a9a2f47c2)


### x.head:

![xhead](https://github.com/A-Thiyagarajan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118707693/c197a3bb-8235-4d10-b8a2-a732f9784b2a)


### Accuracy:

![acc](https://github.com/A-Thiyagarajan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118707693/407b6a0e-aab5-450e-85f4-b79e548e97b8)


### Data Prediction:

![predict](https://github.com/A-Thiyagarajan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118707693/9fa8b109-6f63-45e9-a72e-5fe3367587f6)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
