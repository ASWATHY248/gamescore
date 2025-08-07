import pandas as pd
import matplotlib.pyplot as mt
import sklearn.linear_model as lm
df = pd.read_csv('time.csv')

X = df[["time"]]
y = df["speed"]  # Changed to Series for compatibility

print(X)
print(y)

mt.scatter(X["run"], y)  # Use column names for 1D arrays
mt.xlabel("Run")
mt.ylabel("Km")
mt.show()

model = lm.LinearRegression()
model.fit(X, y)