import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
mydata=pd.read_csv("gym_loss.csv")
x=mydata[["weekly_gym_hours"]]
y=mydata[["weight_loss_kg"]]
plt.scatter(x,y)
plt.show()
model=lm.LinearRegression()
model.fit(x,y)
print(model.predict([[4]]))