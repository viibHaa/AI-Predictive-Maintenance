import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import joblib

# Load the preprocessed data (with RUL)
data_fd001 = pd.read_csv('C:/Users/Vibha/AI-Predictive-Maintenance/data/processed_train_FD001.csv')

# Define the features (X) and the target variable (y)
X = data_fd001.drop(columns=['unit_number', 'cycle', 'RUL'])  # Features are everything except 'unit_number', 'cycle', 'RUL'
y = data_fd001['RUL']  # Target variable is RUL

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate the Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error (MAE): {mae}")

# Calculate the Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
# Now calculate RMSE by taking the square root of MSE
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Save the trained model
joblib.dump(model, 'C:/Users/Vibha/AI-Predictive-Maintenance/model.pkl')
print("Model saved successfully.")

# Optional: Load the saved model and test it
# Uncomment the following lines to load and test the saved model
# model = joblib.load('C:/Users/Vibha/AI-Predictive-Maintenance/model.pkl')
# test_predictions = model.predict(X_test)
# print(f"Test Predictions: {test_predictions[:5]}")
