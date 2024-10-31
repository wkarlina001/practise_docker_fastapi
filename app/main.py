import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# load iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# train random forest classifier
model = RandomForestClassifier()
model.fit(X, y)

joblib.dump(model, 'model.joblib')
