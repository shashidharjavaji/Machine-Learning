import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset=pd.read_csv('Position_Salaries.csv')
x=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

#spliting the data set into test set and the training set

'''when we have small data set then we neet not split into trainig and test set 
and we also need not use standard scaler methods'''

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
 
#using standard scaler method for feature scaling

from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform(x_test)

#building the linear regression model for our data set

from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(x,y)

#building the poynomial regression model for our data set
from sklearn.preprocessing import PolynomialFeatures 
poly_reg=PolynomialFeatures(degree=4)
x_poly=poly_reg.fit_transform(x)
lin_reg_2=LinearRegression()
lin_reg_2.fit(x_poly,y)

#visulaising the data using linear regression results

x_grid=np.arange(min(x),max(x),0.1)
x_grid=x_grid.reshape((len(x_grid),1))
plt.scatter(x,y,color='red')
plt.plot(x_grid,lin_reg_2.predict(poly_reg.fit_transform(x_grid)),color='blue')
plt.title('truth or bluff(using linear regression)')
plt.xlabel('position level')
plt.ylabel('salary')
plt.show()

#predicting the salary using linear regression model

lin_reg.predict([[6.5])
lin_reg_2.predict(poly_reg.fit_transform(np.array([[6.5]])))
#predicting the salary using polynomial regression model
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
