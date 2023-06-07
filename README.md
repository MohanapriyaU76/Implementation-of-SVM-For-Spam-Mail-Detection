# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import dataset using chardet
2. Get dataset info and check for null values
3. Assign x and y values and split the dataset into training and testing sets
4. Import CountVectorizer and transform x_train,x_test as vectors
5.Import SVC and fit it to dataset
6.Find y predict and accuracy


## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Mohanapriya U
RegisterNumber: 212220040091
*/
```
import chardet
file='/content/spam.csv'
with open(file,'rb') as rawdata:
  result = chardet.detect(rawdata.read(100000))
result

import pandas as pd
data=pd.read_csv('/content/spam.csv',encoding='Windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

*/


## Output:
![SVM For Spam Mail Detection](sam.png)

## RESULT OUTPUT:

![image](https://github.com/MohanapriyaU76/Implementation-of-SVM-For-Spam-Mail-Detection/assets/133958624/981be91e-52fc-44a8-854f-d500ee498082)

## DATA.HEAD()

![image](https://github.com/MohanapriyaU76/Implementation-of-SVM-For-Spam-Mail-Detection/assets/133958624/02486557-900f-435c-8ce0-d87c04985d31)

## DATA.INFO()

![image](https://github.com/MohanapriyaU76/Implementation-of-SVM-For-Spam-Mail-Detection/assets/133958624/75150934-5bb5-49dc-b376-b0fc1bc39645)

## DATA.ISNULL().SUM():

![image](https://github.com/MohanapriyaU76/Implementation-of-SVM-For-Spam-Mail-Detection/assets/133958624/3f3393d0-29fd-4598-8d53-d2c99a1aa482)

## Y_PREDICTION VALUE:

![image](https://github.com/MohanapriyaU76/Implementation-of-SVM-For-Spam-Mail-Detection/assets/133958624/7ab19209-3a07-4f5b-b84f-fc6f6b2bd8e6)

## ACCURACY VALUE:

![image](https://github.com/MohanapriyaU76/Implementation-of-SVM-For-Spam-Mail-Detection/assets/133958624/80635ae1-d10a-4ca1-8cfc-835f885a0fd3)



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
