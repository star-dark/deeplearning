import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import datasets
from sklearn.metrics import mean_squared_error
# 보스턴 주택가격 데이터 갖고 오기
boston_dataset = datasets.load_boston()
# RM(방의 수, index 5)과 LSTAT(하위 계층 비율, index 12) 특성 추출
X = boston_dataset.data[:, [5, 12]]
# 주택가격 타겟 추출
y = boston_dataset.target
##################################################################~~코드 작성

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
mode = input('1. LinearRegression\n2. GradientDescent\n')
if(mode == '1'):
# LinearRegression 모델 생성
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print("MSE: %.3f" % mse)
else:
# gradient descent model
    epochs = 1000
    lrate = 0.001
    w1=0
    w2=0
    b=0
    x1 = X_train[:,0]
    x2 = X_train[:,1]
    m = float(len(x1))
    for i in range(epochs):
        y_pred = w1*x1 + w2*x2 + b # 예측값
        dw1 = (2/m) * sum(x1 * (y_pred-y_train))
        dw2 = (2/m) * sum(x2 * (y_pred-y_train))
        db = (2/m) * sum(y_pred-y_train)
        w1 = w1 - lrate * dw1 # 기울기 수정
        w2 = w2 - lrate * dw2 # 기울기 수정
        b = b - lrate * db
    mse = mean_squared_error(y_train, y_pred)
    print("MSE: %.3f" % mse)
# 데이터의 분포 그래프

fig = plt.figure(figsize=(10, 4))
# x1, y subplot
ax1 = fig.add_subplot(121)
ax1.scatter(X_test[:, 0], y_test)
ax1.set_xlabel('RM')
ax1.set_ylabel('price')
# 예측값의 그래프
ax1.plot([min(X_test[:, 0]), max(X_test[:, 0])], [min(y_pred), max(y_pred)], color='red')
# x2, y subplot
ax2 = fig.add_subplot(122)
ax2.scatter(X_test[:, 1], y_test)
ax2.set_xlabel('LSTAT')
ax2.set_ylabel('price')
# 예측값의 그래프
ax2.plot([min(X_test[:, 1]), max(X_test[:, 1])], [max(y_pred), min(y_pred)], color='red')
plt.subplots_adjust(wspace=0.4)
plt.show()