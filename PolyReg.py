import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


data=pd.read_csv("/content/Position_Salaries.csv")
print(data)

sns.heatmap(data.isnull())

x=data.iloc[:,1:-1].values
print(x)
y=data.iloc[:,2].values
print(y)

from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x,y)

plt.scatter(x,y,color="green")
plt.xlabel("Position")
plt.ylabel("Salary")
plt.plot(x,model.predict(x),color="red")

from sklearn.preprocessing import PolynomialFeatures
poly_model=PolynomialFeatures(degree=4)
poly_x=poly_model.fit_transform(x)
print(poly_x)

from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(poly_x,y)

plt.scatter(x,y,color="green")
plt.xlabel("Position")
plt.ylabel("Salary")
plt.plot(x,model.predict(poly_x),color="red")

inp=poly_model.fit_transform([[11]])
y_p=model.predict(inp)
print(y_p)