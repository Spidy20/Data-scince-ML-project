import pandas as pd
import numpy as np

dataset = pd.read_csv('GRE.csv')

x = dataset.iloc[:,1:8].values
y = dataset.iloc[:,8].values

from sklearn.model_selection import train_test_split

x_train , x_test ,y_train ,y_test = train_test_split(x , y ,test_size =1/5 ,random_state = 0)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train,y_train)

# GRE_SCORE = float(input('Enter your GRE Score: '))
# TOEFL_Score = float(input('Enter your TOEFL Score: '))
# UR = float(input('Enter your University ranking (between 1-5) : '))
# SOP = float(input('Enter your Statement of Purpose(between 1-5): '))
# LOR = float(input('Enter your (Latter of recommendation ) score(between 1-5): '))
# CGPA = float(input('Enter your CGPA(between 1-10): '))
# RESEARCH = float(input('Enter your Statement of Purpose(between 1(means you did research) and 0(means no)): '))
#
# VTL = []
# VTL.extend([GRE_SCORE,TOEFL_Score,UR,SOP,LOR,CGPA,RESEARCH])
# LTL = [VTL]
LTL = [[180,	100,	3,	3.5,	2.5,	8.57,	1]]
prediction = regressor.predict(LTL)
print('MODEL accuracy is : ',regressor.score(x_test,y_test))
print('Your chances of admission is :',prediction)
destinction = [0.90]
c = [0.80]
e = [0.70]
f = [0.60]
g = [0.40]
if prediction >= destinction :
    print('You have chances of get admission in Harvard university  ')

elif prediction >= c <destinction:
    print('You have chances of get admission in MIT')

elif prediction >=e <c:
    print('You have chances of get admission in Stanford university')

elif prediction >=f <e:
    print('You have chances of get admission in Caltech(California Institute of Technology)')

elif prediction >=g <f:
    print('You have chances of get admission in UOC (University of Chicago)')

elif prediction <g :
    print("You don't have chances of get admission in top colleges !! Better luck next time:))")

