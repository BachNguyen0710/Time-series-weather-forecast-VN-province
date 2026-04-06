import os
import json
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

def load_data(path):
    """Đọc dữ liệu từ file JSON"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cannot find data file: {path}")
        
    with open(path, "r") as f:
        data = json.load(f)
    X = np.array([item["features"] for item in data])
    y = np.array([item["target"] for item in data])
    return X, y

def run_test():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    #model_path = "rf_model.pkl"
    model_path = os.path.join(current_dir, 'rf_model.pkl')

    #test_data_path = "../data/processed/test.json"
    test_data_path = os.path.join(current_dir, '..', 'data', 'processed', 'test.json')
    print("--- Loading model and test data ---")
    
    if not os.path.exists(model_path):
        print(f" Error: Cannot find file {model_path}. Please run train.py first!")
        return

    model = joblib.load(model_path)
    
    X_test, y_test = load_data(test_data_path)

    print("--- Computing predictions ---")
    y_pred = model.predict(X_test)




    #Tính toán các chỉ số đánh giá
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)



    print("\n" + "="*40)
    print("RESULTS ON TEST SET")
    print("="*40)

    print(f"- MAE (Sai số tuyệt đối TB): {mae:.4f} mm")
    print(f"- MSE (Sai số bình phương TB): {mse:.4f}")
    print(f"- R2 Score (Độ khớp): {r2:.4f}")
    print("="*40)

    # 5. Trực quan hóa TOÀN BỘ kết quả
    while True:
        print("\n--- CHẾ ĐỘ HIỂN THỊ BIỂU ĐỒ ---")
        print("1. Xem 100 mẫu đầu tiên (Rõ nét)")
        print("2. Xem TOÀN BỘ các mẫu (Tổng quan)")
        print("3. Xem biểu đồ Phân tán (Kiểm tra độ khớp R2)")
        print("4. Thoát")
        
        choice = input("Nhập lựa chọn của bạn (1-4): ")

        if choice == '1':
            plt.figure(figsize=(12, 6))
            plt.plot(y_test[:100], label='Thực tế', color='blue', marker='o')
            plt.plot(y_pred[:100], label='Dự đoán', color='red', linestyle='--', marker='x')
            plt.title('So sánh 100 mẫu đầu tiên')
            plt.legend()
            plt.show()

        elif choice == '2':
            plt.figure(figsize=(20, 8))
            plt.plot(y_test, label='Thực tế', color='blue', alpha=0.6)
            plt.plot(y_pred, label='Dự đoán', color='red', alpha=0.6, linestyle='--')
            plt.title(f'Toàn bộ {len(y_test)} mẫu dữ liệu')
            plt.legend()
            plt.show()

        elif choice == '3':
            plt.figure(figsize=(8, 8))
            plt.scatter(y_test, y_pred, alpha=0.3, color='green')
            # Đường 45 độ lý tưởng
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            plt.xlabel('Thực tế (mm)')
            plt.ylabel('Dự đoán (mm)')
            plt.title('Biểu đồ Scatter: Actual vs Predicted')
            plt.show()

        elif choice == '4':
            print("Kết thúc chương trình kiểm thử.")
            break
        else:
            print("Lựa chọn không hợp lệ, vui lòng nhập lại!")
if __name__ == "__main__":
    run_test()