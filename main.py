import numpy as np
import matplotlib.pyplot as plt
from pandas import*
from sklearn.linear_model import LinearRegression

data = read_csv("./APPLE.csv")

open = data['Open'].tolist()
close = data['Close'].tolist()

X = np.array([open]).T

Y = np.array(close)

rmodel = LinearRegression()
rmodel = rmodel.fit(X, Y)
slope = rmodel.coef_

intercept = rmodel.intercept_

Y_predict = rmodel.predict(X)
plt.scatter(open, close, marker='*', edgecolor='r')
plt.plot(open, Y_predict, '-bo')
plt.show()


Y_predict = rmodel.predict([[0.128348]])
print(Y_predict)
