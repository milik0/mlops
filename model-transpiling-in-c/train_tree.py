from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
import joblib

# generate synthetic classification data
X, y = make_classification(n_samples=100, n_features=5, n_classes=2, 
                          n_informative=3, random_state=42)

model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X, y)

# save the model
joblib.dump(model, "tree.joblib")

print("Decision Tree model trained and saved!")
print(f"Tree depth: {model.get_depth()}")
print(f"Number of leaves: {model.get_n_leaves()}")
print(f"Number of features: {model.n_features_in_}")

# test predictions
test_sample = X[0:1]
prediction = model.predict(test_sample)[0]
print(f"\nTest prediction:")
print(f"Input: {test_sample[0]}")
print(f"Class: {prediction}")
