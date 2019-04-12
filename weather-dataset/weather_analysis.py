import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

dataset=pd.read_csv('weatherHistory.csv')
dataset.isnull().sum()
df=dataset.fillna(value=0)
df.isnull().sum()


from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()
df['Precip Type'] = LE.fit_transform(df['Precip Type'].astype(str))

x=df.iloc[:, 2:12].values
y=df.iloc[:,1].values


x[:,9]=LE.fit_transform(x[:,9])
x=x.astype('int')
y = LE.fit_transform(y)

###Lets split values into train and test part

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)

from sklearn.preprocessing  import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

####Decision tree classifier
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_test)
y_pred = LE.inverse_transform(y_pred)
y_test = LE.inverse_transform(y_test)

classifier.score(x_test,y_test)

######Logistic regression 
from sklearn.linear_model import LogisticRegression 
classifier = LogisticRegression ()
classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_test)
classifier.score(x_test,y_test)


###Random forest
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=25, n_jobs=2, random_state=0)
clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)
clf.score(x_test,y_test)