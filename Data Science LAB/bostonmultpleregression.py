import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model, metrics

boston = datasets.load_boston(return_X_y=False)

#defiining feature matrix(x) and response vector(y)
x = boston.data
y = boston.target

#splitting x and y into training and testing sets

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)

#create linear regression
reg = linear_model.LinearRegression()

reg.fit(x_train, y_train)
prediction =  reg.predict(x_test)
print("Prediction", prediction)

print("Coefficient: ", reg.coef_)

print('Variance score: {}'.format(reg.score(x_test, y_test)))



