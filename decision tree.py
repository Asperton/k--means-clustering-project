from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

default_model = DecisionTreeClassifier(random_state=42)
default_model.fit(X_train, y_train)
default_preds = default_model.predict(X_test)
default_accuracy = accuracy_score(y_test, default_preds)

tuned_model = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=3,
    min_samples_split=4,
    random_state=42
)
tuned_model.fit(X_train, y_train)
tuned_preds = tuned_model.predict(X_test)
tuned_accuracy = accuracy_score(y_test, tuned_preds)

print("Default Accuracy:", default_accuracy)
print("Tuned Accuracy:", tuned_accuracy)