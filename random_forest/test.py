import os
import json
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

FEATURE_NAMES = [
    'temperature', 'humidity', 'wind', 'pressure',
    'hour', 'dayofweek', 'month',
    'rain_lag1', 'rain_lag3', 'rain_lag24',
    'rain_roll3', 'rain_roll24'
]

def load_data(path):
    """Đọc dữ liệu từ file JSON"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cannot find data file: {path}")
        
    with open(path, "r") as f:
        data = json.load(f)
    X = np.array([item["features"] for item in data])
    y = np.array([item["target"] for item in data], dtype=int)
    return X, y

def plot_feature_importance(model, save_path=None):
    """Vẽ biểu đồ Feature Importance từ mô hình Random Forest"""
    importances = model.feature_importances_
    n_features = len(importances)

    # Tạo tên feature: dùng FEATURE_NAMES nếu khớp, ngược lại đánh số
    if n_features == len(FEATURE_NAMES):
        feature_names = FEATURE_NAMES
    else:
        # Nếu có thêm city_ dummies, đánh số phần còn lại
        feature_names = FEATURE_NAMES + [f'city_{i}' for i in range(n_features - len(FEATURE_NAMES))]

    # Sắp xếp theo importance giảm dần
    indices = np.argsort(importances)[::-1]
    sorted_names  = [feature_names[i] for i in indices]
    sorted_values = importances[indices]

    # Màu sắc: highlight top 5
    colors = ['#1f77b4' if i < 5 else '#aec7e8' for i in range(len(sorted_names))]

    fig, ax = plt.subplots(figsize=(10, max(5, n_features * 0.4)))
    bars = ax.barh(range(len(sorted_names)), sorted_values[::-1], color=colors[::-1])
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names[::-1], fontsize=10)
    ax.set_xlabel('Feature Importance (Mean Decrease in Gini)', fontsize=11)
    ax.set_title('Random Forest — Feature Importance', fontsize=13, fontweight='bold')

    # Thêm giá trị số trên mỗi bar
    for i, (bar, val) in enumerate(zip(bars, sorted_values[::-1])):
        ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                f'{val:.4f}', va='center', fontsize=8.5)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Da luu bieu do: {save_path}")

    plt.show()

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
    print(f"5 nhan thuc te dau tien (y_test): {y_test[:5]}")
    print(f"5 nhan du doan dau tien (y_pred): {model.predict(X_test).astype(int)[:5]}")

    print("--- Computing predictions ---")
    y_pred = model.predict(X_test).astype(int)

    print("\n" + "="*40)
    print("RESULTS ON TEST SET (CLASSIFICATION)")
    print("="*40)
    print(classification_report(y_test, y_pred, target_names=['Khong mua (0)', 'Co mua (1)']))
    print("="*40)

    while True:
        print("\n--- CHE DO HIEN THI BIEU DO ---")
        print("1. Xem Ma tran nham lan (Confusion Matrix)")
        print("2. Xem 100 mau dau tien (Thuc te vs Du doan)")
        print("3. Xem Feature Importance")
        print("4. Luu Feature Importance ra file PNG")
        print("5. Thoat")
        
        choice = input("Nhap lua chon cua ban (1-5): ")

        if choice == '1':
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=['Doan Khong (0)', 'Doan Co (1)'], 
                        yticklabels=['Thuc te Khong (0)', 'Thuc te Co (1)'])
            plt.title('Confusion Matrix')
            plt.ylabel('Thuc te')
            plt.xlabel('Du doan')
            plt.tight_layout()
            plt.show()

        elif choice == '2':
            plt.figure(figsize=(15, 3))
            limit = min(100, len(y_test))
            plt.scatter(range(limit), y_test[:limit], label='Thuc te', color='blue', alpha=0.6, marker='o', s=50)
            plt.scatter(range(limit), y_pred[:limit], label='Du doan', color='red',  alpha=0.6, marker='x', s=50)
            plt.yticks([0, 1], ['Khong mua (0)', 'Co mua (1)'])
            plt.title(f'So sanh {limit} mau dau tien')
            plt.xlabel('Chi so mau')
            plt.legend()
            plt.tight_layout()
            plt.show()

        elif choice == '3':
            plot_feature_importance(model)

        elif choice == '4':
            save_path = os.path.join(current_dir, 'feature_importance.png')
            plot_feature_importance(model, save_path=save_path)

        elif choice == '5':
            print("Ket thuc chuong trinh kiem thu.")
            break
        else:
            print("Lua chon khong hop le, vui long nhap lai!")

if __name__ == "__main__":
    run_test()