import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor

X_train = pd.read_csv('./data/processed_data/X_train_scaled.csv')
y_train = pd.read_csv('./data/processed_data/y_train.csv').values.ravel()

with open('./models/best_rf_params.pkl', 'rb') as f:
    best_params = pickle.load(f)

final_model = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)

final_model.fit(X_train, y_train)
model_output_path = './models/best_rf_model.pkl'
with open(model_output_path, 'wb') as f:
    pickle.dump(final_model, f)