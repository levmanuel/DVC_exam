import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

X_train = pd.read_csv('./data/processed_data/X_train_scaled.csv')
y_train = pd.read_csv('./data/processed_data/y_train.csv').values.ravel()

rf_model = RandomForestRegressor(random_state=42)

param_grid = {
    'n_estimators': [100, 200],              # Number of trees in the forest
    'max_features': ['sqrt', 'log2'],        # Number of features to consider for the best split
    'max_depth': [10, 20, None],             # Maximum depth of the tree
    'min_samples_split': [2, 5],             # Minimum number of samples required to split a node
    'min_samples_leaf': [1, 2]               # Minimum number of samples required at a leaf node
}

grid_search = GridSearchCV(
    estimator=rf_model,
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    verbose=0,
    scoring='neg_mean_squared_error'
)

grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
print(f"\nGridSearchCV complete. Best parameters found:\n{best_params}")

output_path = './models/best_rf_params.pkl'
with open(output_path, 'wb') as f:
    pickle.dump(best_params, f)