from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
import joblib

# generate synthetic binary classification data
X, y = make_classification(n_samples=100, n_features=4, n_classes=2, random_state=42)

# train logistic regression model
model = LogisticRegression()
model.fit(X, y)

# save model
joblib.dump(model, "logistic.joblib")

print("Logistic Regression model trained and saved!")
print(f"Coefficients: {model.coef_[0]}")
print(f"Intercept: {model.intercept_[0]}")

# test predictions
test_sample = X[0:1]
prediction = model.predict(test_sample)[0]
proba = model.predict_proba(test_sample)[0]
print(f"\nTest prediction:")
print(f"Input: {test_sample[0]}")
print(f"Class: {prediction}")
print(f"Probability: {proba}")
