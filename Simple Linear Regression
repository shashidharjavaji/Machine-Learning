#simple linear regression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset=pd.read_csv('Salary_Data.csv')
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values
'''
from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer=imputer.fit(x[:, 1:3])
x[:, 1:3]=imputer.transform(x[:, 1:3])
from sklearn.preprocessing import LabelEncoder, OneHotEncoder 
labelencoder_x=LabelEncoder()
x[:, 0]=labelencoder_x.fit_transform(x[:,0])
onehotencoder=OneHotEncoder(categorical_features=[0])
x=onehotencoder.fit_transform(x).toarray() 
labelencoder_y=LabelEncoder()
y=labelencoder_y.fit_transform(y)
'''
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
'''from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform(x_test)
'''
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
'''giving user input and getting the predicted values using the regressor model 
we need to create a two dimensional array with a value such that it is similar 
to that of x dataset now we are creating a two dimensional array with a value
as 3 i.e years of experience .....and np array([]) becomes one d array and 
this gives an error so create it as np.array([[value you want to enter]])
you can also take user input  and give th output according to that value
'''
n=input("enter the years of experience:")
n=float(n)
k=np.array([[n]])
print("predicted salary:")
print(regressor.predict(k))
#visualisation
import matplotlib.patches as mpatches
plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('salary vs experience(training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
red_patch = mpatches.Patch(color='red', label='real salary')
red_patch1 = mpatches.Patch(color='blue', label='predicted values')
plt.legend(handles=[red_patch,red_patch1])
plt.show()
#test set visualisation
import matplotlib.patches as mpatches
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('salary vs experience(test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
red_patch = mpatches.Patch(color='red',label='real salary')
red_patch1 = mpatches.Patch(color='blue', label='predicted values')
plt.legend(handles=[red_patch,red_patch1])
plt.show()
