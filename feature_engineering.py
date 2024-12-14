import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load the preprocessed data (with RUL)
data_fd001 = pd.read_csv('C:/Users/Vibha/AI-Predictive-Maintenance/data/processed_train_FD001.csv')

# Define initial features (X) and target variable (y)
X = data_fd001.drop(columns=['unit_number', 'cycle', 'RUL'])  # Features
y = data_fd001['RUL']  # Target variable is RUL

# Feature Engineering
# 1. Feature Selection using Feature Importance
initial_model = RandomForestRegressor(random_state=42)
initial_model.fit(X, y)
feature_importances = initial_model.feature_importances_

# Plot feature importance
plt.figure(figsize=(12, 6))
sns.barplot(x=X.columns, y=feature_importances)
plt.title("Feature Importance")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()  # This will show the plot in your script environment

# Select features with importance > 0.01
important_features = [X.columns[i] for i in range(len(X.columns)) if feature_importances[i] > 0.01]
X_selected = X[important_features].copy()  # Create a copy of the selected features

# 2. Feature Creation: Add new features
X_selected['sensor_ratio'] = X['sensor_1'] / (X['sensor_2'] + 1e-6)  # Avoid division by zero
X_selected['rolling_mean_sensor_1'] = X['sensor_1'].rolling(window=5, min_periods=1).mean()

# 3. Feature Scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_selected)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Hyperparameter Tuning with GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

model = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best Model
best_model = grid_search.best_estimator_
print(f"Best Hyperparameters: {grid_search.best_params_}")

# Model Training
best_model.fit(X_train, y_train)

# Make Predictions
y_pred = best_model.predict(X_test)

# Evaluate the Model
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Save the Trained Model
joblib.dump(best_model, 'C:/Users/Vibha/AI-Predictive-Maintenance/model_optimized.pkl')
print("Optimized model saved successfully.")
