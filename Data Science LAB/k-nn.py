from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn import metrics
irisData = load_iris()

x = irisData.data
y = irisData.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.35, random_state=35)

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(x_train, y_train)
print(knn.predict(x_test))

acc = metrics.accuracy_score(y_test, knn.predict(x_test))

print("Accuracy: ", acc)

