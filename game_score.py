import pandas as pd
import matplotlib.pyplot as mt
import sklearn.linear_model as lm
df = pd.read_csv('Game.csv')

X= df[ ["LEVEL"] ]
y= df[ ["SCORE"] ]

print(X)
print(y)

mt.scatter(X, y)
mt.show()
model = lm.LinearRegression()
model.fit(X, y)
print(model.predict([[43]]))