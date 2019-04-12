import pandas as pd

dataset = pd.read_csv('INFY.csv')

x=dataset.loc[:,'High':'Turnover (Lacs)']###Give data to model from first attribute
y=dataset.loc[:,'Open']##Give value to model from data

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)


from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
test= [[2823.8,2780.1,2815.25,2815.1,761869.0,21421.66]]
dt.fit(x_train,y_train)
print(dt.predict(test))
print(dt.score(x_test,y_test))