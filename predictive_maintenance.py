import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def preprocess_data():
    # Correct path to your data file (update this with your actual path)
    data_fd001 = pd.read_csv('C:/Users/Vibha/AI-Predictive-Maintenance/data/processed_train_FD001.csv')  # Use the correct path to your file

    # Drop unnecessary columns and define X (features) and y (target variable)
    X = data_fd001.drop(columns=['unit_number', 'cycle', 'RUL'])  # Remove columns that are not features
    y = data_fd001['RUL']  # Target variable: Remaining Useful Life

    # Feature scaling: Apply MinMax scaling
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y

def model_training(X_scaled, y):
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Hyperparameter tuning with GridSearchCV
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    model = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Best model from GridSearchCV
    best_model = grid_search.best_estimator_

    # Train the model with the best hyperparameters
    best_model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = best_model.predict(X_test)

    # Model evaluation: Calculate MAE and RMSE
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")

    # Save the trained model to a file
    joblib.dump(best_model, 'C:/Users/Vibha/AI-Predictive-Maintenance/model.pkl')
    print("Model saved successfully!")

    return best_model

def main():
    # Preprocess the data
    X_scaled, y = preprocess_data()

    # Train the model and evaluate
    model = model_training(X_scaled, y)

if __name__ == "__main__":
    main()
