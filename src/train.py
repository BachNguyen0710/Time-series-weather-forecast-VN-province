import os
import sys
import json
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

# Thiết lập đường dẫn
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.abspath(os.path.join(current_dir, '..', 'src'))
if src_path not in sys.path:
    sys.path.append(src_path)

def train_rf():
    # 1. Đường dẫn tới folder chứa dữ liệu đã xử lý
    data_path = os.path.abspath(os.path.join(current_dir, '..', 'data', 'processed'))
    train_file = os.path.join(data_path, 'train.json')
    
    print(f"--- Loading data from {train_file} ---")
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Không tìm thấy file: {train_file}")

    with open(train_file, "r") as f:
        train_data = json.load(f)
    
    # ĐỊNH NGHĨA X_train và y_train TẠI ĐÂY
    X_train = np.array([item["features"] for item in train_data])
    y_train = np.array([item["target"] for item in train_data], dtype=int)

    print("--- Training Random Forest Classifier ---")
    random_model = RandomForestClassifier(
        n_estimators=500,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced' 
    )
    
    # Bây giờ X_train đã tồn tại, lệnh này sẽ chạy ok
    random_model.fit(X_train, y_train)
    
    return random_model # Trả model về để lưu

if __name__ == "__main__":
    # Thực hiện huấn luyện
    model = train_rf()

    # Thiết lập đường dẫn và lưu model
    models_dir = os.path.abspath(os.path.join(current_dir, '..', 'models'))
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, 'rf_model.pkl')

    print(f"--- Saving model to {model_path} ---")
    joblib.dump(model, model_path)
    print("Successfully saved!")