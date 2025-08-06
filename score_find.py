import pandas as pd
import matplotlib.pyplot as mt
import sklearn.linear_model as lm
df = pd.read_csv('study_scores.csv')

X= df[ ["no_of_hours"] ]
y= df[ ["score"] ]

print(X)
print(y)

mt.scatter(X, y)
mt.show()
model = lm.LinearRegression()
model.fit(X, y)
print(model.predict([[2000000]]))