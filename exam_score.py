import pandas as pd
import matplotlib.pyplot as mt
import sklearn.linear_model as lm
df = pd.read_csv('data.csv')

X= df[ ["STUDY HOURS"] ]
y= df[ ["EXAM SCORE"] ]

print(X)
print(y)

mt.scatter(X, y)
mt.show()
model = lm.LinearRegression()
model.fit(X, y)
print(model.predict([[8]]))