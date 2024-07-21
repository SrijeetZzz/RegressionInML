

import numpy as pd
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv("/content/data (1).csv")
print(data)

x=data.iloc[:,:-1].values
y=data.iloc[:,-1].values
print(x)
print(y)

sns.pairplot(data,hue="label")