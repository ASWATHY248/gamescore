import pandas as pd
import matplotlib.pyplot as mt
import sklearn.linear_model as lm
mydata = pd.read_csv("height_weight_data.csv")


X= mydata[ ["height"] ]
y= mydata[ ["weight"] ]

print(X)
print(y)

mt.scatter(X, y)
mt.show()
model = lm.LinearRegression()
model.fit(X, y)
print(model.predict([[25000]]))