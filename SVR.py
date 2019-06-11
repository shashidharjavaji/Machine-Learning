
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset=pd.read_csv('Position_Salaries.csv')

x=dataset.iloc[:, 1:2].values
y=dataset.iloc[:, 2].values
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x=sc_x.fit_transform(x)
sc_y=StandardScaler()

#y=sc_y.fit_transform(y)
y = np.ravel(sc_y.fit_transform(y.reshape(-1, 1))) 
from sklearn.svm import SVR
regressor=SVR(kernel='rbf')
regressor.fit(x,y)
y_pred=sc_y.inverse_transform(regressor.predict(sc_x.transform(np.array([[6.5]]))))
plt.scatter(x,y,color='red')
plt.plot(x,regressor.predict(x),color='blue')
plt.title('truth or bluff(using svr)')
plt.xlabel('position level')
plt.ylabel('salary')
plt.show()
print(y_pred)
