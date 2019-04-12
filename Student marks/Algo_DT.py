import pandas as pd
from sklearn.preprocessing import LabelEncoder

dataset = pd.read_csv('edu.csv')

# =============================================================================

     

x = dataset.iloc[:, 0:16].values
y = dataset.iloc[:,16].values


LE=LabelEncoder()
x[:,0]=LE.fit_transform(x[:,0])
x[:,1]=LE.fit_transform(x[:,1])
x[:,2]=LE.fit_transform(x[:,2])
x[:,3]=LE.fit_transform(x[:,3])
x[:,4]=LE.fit_transform(x[:,4])
x[:,5]=LE.fit_transform(x[:,5])
x[:,6]=LE.fit_transform(x[:,6])
x[:,7]=LE.fit_transform(x[:,7])
x[:,8]=LE.fit_transform(x[:,8])
x[:,13]=LE.fit_transform(x[:,13])
x[:,14]=LE.fit_transform(x[:,14])
x[:,15]=LE.fit_transform(x[:,15])

y=LE.fit_transform(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

from sklearn.preprocessing import StandardScaler
SC=StandardScaler()
x_train=SC.fit_transform(x_train)
x_test=SC.transform(x_test)

###Lets build the model
from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion='entropy',random_state=0)
classifier.fit(x_train,y_train)

###Lets predict the result
y_pred=classifier.predict(x_test)
y_pred = LE.inverse_transform(y_pred)
y_test = LE.inverse_transform(y_test)


score = classifier.score(x_test,y_test)
