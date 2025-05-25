import pandas as pd
import numpy as np

dataset=pd.read_csv('your_data_file.csv')
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

results=regressor.predict(x)

for result in results:
    with open("data_slr_scikit.txt","a") as file:
        file.write(str(result) +"\n")
