import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error, r2_score
import json

with open('./models/best_rf_model.pkl', 'rb') as f:
    rf = pickle.load(f)

X_test = pd.read_csv('./data/processed_data/X_test_scaled.csv')
y_test = pd.read_csv('./data/processed_data/y_test.csv').values.ravel()

predictions = rf.predict(X_test)

print("Calculating evaluation metrics...")
r2 = r2_score(y_test, predictions)
mse = mean_squared_error(y_test, predictions)

print(f"\n--- Model Performance on Test Set ---")
print(f"R-squared (R²): {r2:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"------------------------------------")

scores = {
    'r2_score': r2,
    'mean_squared_error': mse,
}

with open('./metrics/scores.json', 'w') as f:
    json.dump(scores, f, indent=4)