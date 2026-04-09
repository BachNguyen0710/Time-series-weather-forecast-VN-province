import os
import sys
import json
from xml.parsers.expat import model
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

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
    y = np.array([item["target"] for item in data], dtype=int)
    return X, y

def train_rf():
    # Xác định đường dẫn dữ liệu
    data_path = os.path.join(current_dir, '..', 'data', 'processed', '')
    
    print("--- Loading data ---")
    with open(f"{data_path}train.json", "r") as f:
        train_data = json.load(f)
    
    X_train = np.array([item["features"] for item in train_data])
    y_train = np.array([item["target"] for item in train_data], dtype=int)

    print("--- Training Random Forest Classifier ---")
    # Sử dụng Classifier thay vì Regressor cho bài toán phân loại nhị phân
    model = RandomForestClassifier(
        n_estimators=500,       # Số lượng cây quyết định
        max_depth=10,           # Độ sâu tối đa của cây
        min_samples_leaf=4,     # Số lượng mẫu tối thiểu ở nút lá
        max_features='sqrt',    # Số lượng đặc trưng xét đến khi tách nút
        random_state=42,
        n_jobs=-1,              # Dùng tất cả CPU cores để train nhanh hơn
        class_weight='balanced' # Rất quan trọng: Giúp mô hình cân bằng giữa lớp Có mưa và Không mưa
    )
    
    model.fit(X_train, y_train)

    #joblib.dump(model, "rf_model.pkl")
    model_path = os.path.join(current_dir, 'rf_model.pkl')
    joblib.dump(model, model_path)
    print("Model saved in rf_model.pkl")


if __name__ == "__main__":
    train_rf()