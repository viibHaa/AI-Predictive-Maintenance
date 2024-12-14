import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler

def load_data(file_path):
    print(f"Loading file from: {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file at path '{file_path}' does not exist.")
    
    data = pd.read_csv(file_path, sep=' ', header=None)
    print(f"File loaded successfully. Shape: {data.shape}")
    
    # Drop any empty columns (if any)
    data = data.dropna(axis=1)
    print(f"Data after dropping empty columns: Shape: {data.shape}")

    # Assign column names
    columns = ['unit_number', 'cycle'] + [f'setting_{i}' for i in range(1, 4)] + [f'sensor_{i}' for i in range(1, 22)]
    data.columns = columns
    
    return data

def calculate_rul(data):
    max_cycle = data.groupby('unit_number')['cycle'].max()
    data['RUL'] = data['unit_number'].map(max_cycle) - data['cycle']
    return data

def scale_data(data):
    scaler = MinMaxScaler()
    features = [col for col in data.columns if 'sensor' in col or 'setting' in col]
    data[features] = scaler.fit_transform(data[features])
    return data

# File path
file_path_fd001 = 'C:/Users/Vibha/AI-Predictive-Maintenance/data/train_FD001.txt'

# Data preprocessing
data_fd001 = load_data(file_path_fd001)
data_fd001 = calculate_rul(data_fd001)
data_fd001 = scale_data(data_fd001)

# Save the processed data
data_fd001.to_csv('C:/Users/Vibha/AI-Predictive-Maintenance/data/processed_train_FD001.csv', index=False)
print("Processed data saved successfully.")
