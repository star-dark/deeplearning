from sklearn import datasets, metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
X = iris.data
y = iris.target
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=4)
knn = KNeighborsClassifier(n_neighbors=6, metric='minkowski')
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(f"{metrics.classification_report(y_test, y_pred)}\n")
disp = metrics.plot_confusion_matrix(knn, X_test, y_test)
plt.show()