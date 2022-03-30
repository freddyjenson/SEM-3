import pandas
from sklearn import linear_model
from sklearn.metrics import r2_score

df = pandas.read_csv('cars.csv')
x = df[['Weight', 'Volume']]
y = df[['CO2']]

reg = linear_model.LinearRegression()

reg.fit(x.values, y)

prediction = reg.predict([[2300, 1300]])
print(prediction)

