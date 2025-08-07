import pandas as pd
import matplotlib.pyplot as mt
import sklearn.linear_model as lm
df = pd.read_csv('time.csv')

X= df[ ["time"] ]
y= df[ ["speed"] ]

print(X)
print(y)

mt.scatter(X, y)
mt.show()
model = lm.LinearRegression()
model.fit(X, y)
print(model.predict([[8]]))