import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Load the saved model
model = joblib.load('C:/Users/Vibha/AI-Predictive-Maintenance/model.pkl')

# Load the test dataset (for this example, using X_test from preprocessing)
data_fd001 = pd.read_csv('C:/Users/Vibha/AI-Predictive-Maintenance/data/processed_train_FD001.csv')

# Define features and target
X = data_fd001.drop(columns=['unit_number', 'cycle', 'RUL'])  # Features
y = data_fd001['RUL']  # Actual RUL values (target)

# Split the dataset into training and testing sets (use the same split as during training)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model on the test data
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Display evaluation metrics
print(f"Mean Absolute Error (MAE) on Test Data: {mae}")
print(f"Root Mean Squared Error (RMSE) on Test Data: {rmse}")

# Display some sample predictions
for i in range(5):  # Show the first 5 predictions
    print(f"Actual RUL: {y_test.iloc[i]}, Predicted RUL: {y_pred[i]}")
