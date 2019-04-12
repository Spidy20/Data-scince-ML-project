import pandas as pd
#Load the data
data = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

x = data.iloc[:,2:36].values
y = data.iloc[:,1:2].values


from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()
x[:,0]=LE.fit_transform(x[:,0])
x[:,2]=LE.fit_transform(x[:,2])
x[:,5]=LE.fit_transform(x[:,5])
x[:,9]=LE.fit_transform(x[:,9])
x[:,13]=LE.fit_transform(x[:,13])
x[:,15]=LE.fit_transform(x[:,15])
x[:,19]=LE.fit_transform(x[:,19])
x[:,20]=LE.fit_transform(x[:,20])
y=LE.fit_transform(y)

##Train test split data into training and testing
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

###Stndard scalar for scaling data into one scale
from sklearn.preprocessing import StandardScaler
SC=StandardScaler()
x_train=SC.fit_transform(x_train)
x_test=SC.transform(x_test)

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=25, n_jobs=2, random_state=0)
clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)

###lets make the confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

