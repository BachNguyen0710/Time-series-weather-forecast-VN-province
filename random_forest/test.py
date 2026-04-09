import os
import json
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def load_data(path):
    """Đọc dữ liệu từ file JSON"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cannot find data file: {path}")
        
    with open(path, "r") as f:
        data = json.load(f)
    X = np.array([item["features"] for item in data])
    y = np.array([item["target"] for item in data], dtype=int)
    return X, y

def run_test():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'rf_model.pkl')
    test_data_path = os.path.join(current_dir, '..', 'data', 'processed', 'test.json')
    
    print("--- Loading model and test data ---")
    
    if not os.path.exists(model_path):
        print(f" Error: Cannot find file {model_path}. Please run train.py first!")
        return

    model = joblib.load(model_path)
    
    X_test, y_test = load_data(test_data_path)
    print(f"5 nhãn thực tế đầu tiên (y_test): {y_test[:5]}")
    print(f"5 nhãn dự đoán đầu tiên (y_pred): {model.predict(X_test)[:5]}")

    print("--- Computing predictions ---")
    y_pred = model.predict(X_test)

    print("\n" + "="*40)
    print("RESULTS ON TEST SET (CLASSIFICATION)")
    print("="*40)
    
    # In báo cáo phân loại chi tiết (Precision, Recall, F1-score)
    print(classification_report(y_test, y_pred, target_names=['Không mưa (0)', 'Có mưa (1)']))
    print("="*40)

    # Trực quan hóa kết quả
    while True:
        print("\n--- CHẾ ĐỘ HIỂN THỊ BIỂU ĐỒ ---")
        print("1. Xem Ma trận nhầm lẫn (Confusion Matrix)")
        print("2. Xem 100 mẫu đầu tiên (Thực tế vs Dự đoán)")
        print("3. Thoát")
        
        choice = input("Nhập lựa chọn của bạn (1-3): ")

        if choice == '1':
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=['Đoán Không (0)', 'Đoán Có (1)'], 
                        yticklabels=['Thực tế Không (0)', 'Thực tế Có (1)'])
            plt.title('Confusion Matrix')
            plt.ylabel('Thực tế')
            plt.xlabel('Dự đoán')
            plt.tight_layout()
            plt.show()

        elif choice == '2':
            plt.figure(figsize=(15, 3))
            limit = min(100, len(y_test)) # Lấy 100 mẫu hoặc ít hơn nếu dữ liệu không đủ
            
            # Vẽ scatter plot để dễ nhìn sự trùng khớp của nhãn 0 và 1
            plt.scatter(range(limit), y_test[:limit], label='Thực tế', color='blue', alpha=0.6, marker='o', s=50)
            plt.scatter(range(limit), y_pred[:limit], label='Dự đoán', color='red', alpha=0.6, marker='x', s=50)
            
            plt.yticks([0, 1], ['Không mưa (0)', 'Có mưa (1)'])
            plt.title(f'So sánh {limit} mẫu đầu tiên')
            plt.xlabel('Chỉ số mẫu')
            plt.legend()
            plt.tight_layout()
            plt.show()

        elif choice == '3':
            print("Kết thúc chương trình kiểm thử.")
            break
        else:
            print("Lựa chọn không hợp lệ, vui lòng nhập lại!")

if __name__ == "__main__":
    run_test()