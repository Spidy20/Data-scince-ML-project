import numpy as np # linear algebra
import pandas as pd # data processing, CSV file 
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures



train = pd.read_csv('Train.csv')
test = pd.read_csv('Test.csv')

train.shape, test.shape  ###for get a shape of array

train.columns  ###Get a column info
test.columns

#View the relationships between individual features as well as between the features and the dependent variable.
corr = train.corr()
#Heatmaps show the strenght of relationships between the data features.
sns.heatmap(corr, annot=True)

#Get an overall feel of the dataset.
desc = train.describe() 

#visualizing the data to detect missing values
sns.heatmap(train.isnull(),yticklabels=False,cmap='viridis')



#Changing categorical data(Outlet_Location_Type) to numerical data.
train["Outlet_Location_Type"] = pd.Categorical(train["Outlet_Location_Type"])
Outlet_Location_Type_categories = train.Outlet_Location_Type.cat.categories
train["Outlet_Location_Type"] = train.Outlet_Location_Type.cat.codes




#Changing categorical data(Outlet_Size) to numerical data.
train["Outlet_Size"] = pd.Categorical(train["Outlet_Size"])
Outlet_Size_categories = train.Outlet_Size.cat.categories
train["Outlet_Size"] = train.Outlet_Size.cat.codes

#Changing categorical data(Item_Fat_Content) to numerical data.
train["Item_Fat_Content"] = pd.Categorical(train["Item_Fat_Content"])
Item_Fat_Content_categories = train.Item_Fat_Content.cat.categories
train["Item_Fat_Content"] = train.Item_Fat_Content.cat.codes

#Changing categorical data(Item_Type) to numerical data.
train["Item_Type"] = pd.Categorical(train["Item_Type"])
Item_Type_categories = train.Item_Type.cat.categories
train["Item_Type"] = train.Item_Type.cat.codes


#Changing categorical data(Item_Type) to numerical data.
train["Outlet_Type"] = pd.Categorical(train["Outlet_Type"])
Outlet_Type_categories = train.Outlet_Type.cat.categories
train["Outlet_Type"] = train.Outlet_Type.cat.codes



#Changing categorical data(Outlet_Identifier) to numerical data.
train["Outlet_Identifier"] = pd.Categorical(train["Outlet_Identifier"])
Outlet_Identifier_categories = train.Outlet_Identifier.cat.categories
train["Outlet_Identifier"] = train.Outlet_Identifier.cat.codes



#Changing categorical data(Outlet_Establishment_Year) to numerical data.
train["Outlet_Establishment_Year"] = pd.Categorical(train["Outlet_Establishment_Year"])
Outlet_Establishment_Year_categories = train.Outlet_Establishment_Year.cat.categories
train["Outlet_Establishment_Year"] = train.Outlet_Establishment_Year.cat.codes



#Training data after categorical to numerical conversions.
train.head()

#A correlation analysis will indicate additional relationships in the dataset
corr = train.corr()
sns.heatmap(corr, annot=True)



from sklearn.model_selection import train_test_split


#Introducing Polynomial regression.
poly_features= PolynomialFeatures(degree=3)

###Now we can fit the polynomial regression    
###split the data into train and test set

###and fit the algorithm

