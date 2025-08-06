import matplotlib.pyplot as plt
import sklearn.linear_model as lm
mydata=pd.read_csv("salary.csv")
x=mydata[["years"]]
y=mydata[["salary"]]
plt.scatter(x,y)
plt.show()
model=lm.LinearRegression()
model.fit(x,y)
print(model.predict([[6]]))