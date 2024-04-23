import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.regularizers import l1, l2, l1_l2
from tensorflow.keras import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 데이터 세트를 읽기
red_wine = pd.read_csv('winequality-red.csv', sep=';')
white_wine = pd.read_csv('winequality-white.csv', sep=';')
# column 추가
red_wine['color'] = 1.
white_wine['color'] = 0.
# 데이터 합치기
wine_data = pd.concat([red_wine, white_wine])
wine_data.reset_index(drop=True, inplace=True)
# 결손치가 있는 데이터 행은 삭제
wine_data.dropna(inplace=True)
# 특성과 레이블 분리
X = wine_data.drop(['color'], axis=1)
y = wine_data['color']
# 데이터 분할: 훈련, 검증, 테스트 세트로 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# 데이터 정규화
X_train = np.array(X_train)
y_train = np.array(y_train)
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
# Keras 모델 구성
model = Sequential(
    [
        Dense(13, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.5),
        Dense(52, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.5),
        Dense(104, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.5),
        Dense(208, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.5),
        Dense(104, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.5),
        Dense(52, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.5),
        Dense(1, activation='sigmoid', kernel_regularizer=l2(0.001)),
    ]
)
# 모델 컴파일
model.compile(optimizer='adam',
loss='binary_crossentropy',
metrics=['accuracy'])
# 모델 훈련
history = model.fit(X_train, y_train, epochs=200, validation_split=0.2)
# 모델 평가
model.evaluate(X_test, y_test)
# 훈련 손실과 검증 손실 그래프
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
# 훈련 정확도와 검증 정확도 그래프
plt.subplot(1, 2, 2)
plt.plot(history.history['acc'], label='Training Accuracy')
plt.plot(history.history['val_acc'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
