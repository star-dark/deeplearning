import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
x = pd.read_csv("CarPrice_Assignment.csv", usecols=["price"])
x_val = np.array(x.values)
y1 = pd.read_csv("CarPrice_Assignment.csv", usecols=["citympg"])
y2 = pd.read_csv("CarPrice_Assignment.csv", usecols=["highwaympg"])
z1 = pd.read_csv("CarPrice_Assignment.csv", usecols=["enginesize"])
z2 = pd.read_csv("CarPrice_Assignment.csv", usecols=["horsepower"])
y1_val = np.array(y1.values)
y2_val = np.array(y2.values)
z1_val = np.array(z1.values)
z2_val = np.array(z2.values)
plt.subplot(1,2,1)
plt.plot(x_val,y1_val,"ko" ,label="citympg")
plt.plot(x_val,z1_val,"y+" ,label="enginesize")
plt.legend(loc='best')
plt.subplot(1,2,2)
plt.plot(x_val,y2_val,"sm",label="highwaympg")
plt.plot(x_val,z2_val,"r^" ,label="horsepower")
plt.legend(loc='best')
plt.show()

#X = np.arange(0, 10)
#Y = X**2
#plt.plot(X, Y)
#plt.show()