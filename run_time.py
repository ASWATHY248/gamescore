import pandas as pd
import matplotlib.pyplot as mt
import sklearn.linear_model as lm
mydata = pd.read_csv("run.csv")


X= mydata[ ["runtime"] ]
y= mydata[ ["kilomtr"] ]

print(X)
print(y)

mt.scatter(X, y)
mt.show()
model = lm.LinearRegression()
model.fit(X, y)
print(model.predict([[5]]))