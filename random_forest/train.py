import os
import sys
import json
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV

current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, '..', 'src')
sys.path.append(src_path)

try:
    import Preprocess
    print("Succesfully imported Preprocess module")
except ImportError:
    print("Still cannot find the Preprocess module. Please check the directory structure.")

def load_data_from_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    X = np.array([item["features"] for item in data])
    y = np.array([item["target"] for item in data])
    return X, y

def train_rf():
    #data_path = "../data/processed/"
    data_path = os.path.join(current_dir, '..', 'data', 'processed', '')

    
    print("--- Loading data ---")
    with open(f"{data_path}train.json", "r") as f:
        train_data = json.load(f)
    
    X_train = np.array([item["features"] for item in train_data])
    y_train = np.array([item["target"] for item in train_data])





    print("--- Training Random Forest ---")
    model = RandomForestRegressor(
    n_estimators=500,       # Tăng số cây
    max_depth=10,           # Cho phép cây sâu hơn để học đặc trưng mưa
    min_samples_leaf=4,     # Giảm để nhạy bén hơn với các mẫu nhỏ
    max_features='sqrt',    # Giảm tương quan giữa các cây
    random_state=42,
    n_jobs=-1               # Chạy đa nhân để tiết kiệm thời gian
)
    model.fit(X_train, y_train)

    joblib.dump(model, "rf_model.pkl")
    print("Model saved in random_forest/rf_model.pkl")

if __name__ == "__main__":
    train_rf()