

import numpy as np
import pandas as pd
import seaborn as sns

data=pd.read_csv("/content/multir.csv")
print(data)

sns.heatmap(data.isnull())

newdata=data.iloc[:,2:5]
print(newdata)
x=newdata.iloc[:,:-1].values
print(x)

y=newdata.iloc[:,2].values
print(y)

from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x,y)

inp=[[1200,1500]]
yp=model.predict(inp)
print(yp)