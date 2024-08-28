import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import logging

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
print("Starting model training...")
model.fit(X_train, y_train)
print("Model training completed.")
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print(f"Model accuracy: {accuracy:.2f}")

joblib.dump(model, 'iris_model.pkl')


try:
    
    print("Model training completed successfully.")
except Exception as e:
    print(f"An error occurred: {e}")

try:
    logging.info("Starting model training...")

    

    logging.info("Model training completed successfully.")
except Exception as e:
    logging.error(f"An error occurred: {e}")


