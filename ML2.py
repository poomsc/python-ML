from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris_dataset = load_iris()

x_train, x_test, y_train, y_test = train_test_split(iris_dataset['data'],iris_dataset['target'],random_state = 0,test_size=0.2)

# print(x_train)
print(x_test)
# print(y_train)
print(y_test)