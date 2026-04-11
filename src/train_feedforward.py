import json
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_json_data(path):
    """Load preprocessed data from JSON file."""
    with open(path, 'r') as f:
        data = json.load(f)
    
    X = np.array([item['features'] for item in data])
    # Ensure targets are float32 for BCEWithLogitsLoss
    y = np.array([item['target'] for item in data], dtype=np.float32)
    
    return X, y

class FeedForwardClassifier(nn.Module):
    """Simple feed-forward neural network for binary classification."""
    def __init__(self, input_size):
        super(FeedForwardClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(32, 16)
        
        # Outputs raw logits (no activation needed here due to BCEWithLogitsLoss)
        self.fc4 = nn.Linear(16, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        
        x = torch.relu(self.fc3(x))
        
        x = self.fc4(x)
        return x

def train_model(X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
    """Train the feed-forward classifier."""
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1).to(device)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Build model
    model = FeedForwardClassifier(X_train.shape[1]).to(device)
    
    # Print model summary
    print("\n" + "="*60)
    print("MODEL ARCHITECTURE")
    print("="*60)
    print(model)
    print("\nTotal parameters:", sum(p.numel() for p in model.parameters()))
    
    # Optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    # Binary Cross Entropy with Logits Loss
    # 1. Calculate the ratio of 0s to 1s in your training data
    num_negatives = (y_train == 0).sum()
    num_positives = (y_train == 1).sum()
    
    # If you have 900 zeros and 100 ones, pos_weight will be 9.0
    weight = torch.tensor([num_negatives / num_positives], dtype=torch.float32).to(device)
    
    # 2. Pass the weight to the loss function
    criterion = nn.BCEWithLogitsLoss(pos_weight=weight)
    
    # Training loop with early stopping
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)
    
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 10
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_acc = 0
        
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred_logits = model(X_batch)
            loss = criterion(y_pred_logits, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Calculate accuracy for this batch
            y_pred_class = (torch.sigmoid(y_pred_logits) > 0.5).float()
            train_acc += (y_pred_class == y_batch).float().mean().item()
        
        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        
        # Validation
        model.eval()
        with torch.no_grad():
            y_val_logits = model(X_val_tensor)
            val_loss = criterion(y_val_logits, y_val_tensor).item()
            
            y_val_class = (torch.sigmoid(y_val_logits) > 0.5).float()
            val_acc = (y_val_class == y_val_tensor).float().mean().item()
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - "
                  f"Acc: {train_acc:.4f} - Val Acc: {val_acc:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                model.load_state_dict(best_model_state)
                break
    
    return model, history

def evaluate_model(model, X_test, y_test, model_name="Feed-Forward Classifier"):
    """Evaluate model on test set."""
    model.eval()
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    
    with torch.no_grad():
        y_pred_logits = model(X_test_tensor)
        # Apply sigmoid to convert logits to probabilities (0.0 to 1.0)
        y_probs = torch.sigmoid(y_pred_logits).cpu().numpy().flatten()
    
    # Threshold probabilities to get exactly 0 or 1
    y_pred = (y_probs > 0.5).astype(int)
    y_test_int = y_test.astype(int)
    
    # Classification Metrics
    acc = accuracy_score(y_test_int, y_pred)
    prec = precision_score(y_test_int, y_pred, zero_division=0)
    rec = recall_score(y_test_int, y_pred, zero_division=0)
    f1 = f1_score(y_test_int, y_pred, zero_division=0)
    
    try:
        auc = roc_auc_score(y_test_int, y_probs)
    except ValueError:
        auc = float('nan') # Handles cases where test set only has 1 class
    
    print("\n" + "="*60)
    print(f"TEST SET EVALUATION - {model_name}")
    print("="*60)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC AUC:   {auc:.4f}")
    print("="*60 + "\n")
    
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'auc': auc, 'y_pred': y_pred}

def plot_training_history(history, save_path="../results/training_history.png"):
    """Plot training and validation loss and accuracy."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    
    # Loss
    ax1.plot(history['train_loss'], label='Training Loss', linewidth=2)
    ax1.plot(history['val_loss'], label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (BCE)')
    ax1.set_title('Model Loss Over Epochs')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy
    ax2.plot(history['train_acc'], label='Training Accuracy', linewidth=2)
    ax2.plot(history['val_acc'], label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Model Accuracy Over Epochs')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    print(f"Training history saved to {save_path}")
    plt.close()

def plot_confusion_matrix(y_test, y_pred, save_path="../results/confusion_matrix.png"):
    """Plot confusion matrix for binary predictions."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Class 0', 'Class 1'])
    
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    ax.set_title('Confusion Matrix')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    print(f"Confusion matrix saved to {save_path}")
    plt.close()

def main():
    # Set paths
    data_dir = Path("../data/processed")
    
    print("Loading data...")
    X_train, y_train = load_json_data(data_dir / "train.json")
    X_val, y_val = load_json_data(data_dir / "val.json")
    X_test, y_test = load_json_data(data_dir / "test.json")
    
    print(f"Training set shape:   {X_train.shape}, Target: {y_train.shape}")
    print(f"Validation set shape: {X_val.shape}, Target: {y_val.shape}")
    print(f"Test set shape:       {X_test.shape}, Target: {y_test.shape}")
    
    # Train model
    model, history = train_model(X_train, y_train, X_val, y_val, epochs=100, batch_size=32)
    
    # Evaluate on test set
    results = evaluate_model(model, X_test, y_test, "Feed-Forward Classifier")
    
    # Save model
    model_path = Path("../models/classification_model.pth")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # Plot results
    plot_training_history(history)
    plot_confusion_matrix(y_test, results['y_pred'])
    
    # Save results
    results_dir = Path("../results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results_df = pd.DataFrame({
        'model': ['Feed-Forward Classifier'],
        'accuracy': [results['accuracy']],
        'precision': [results['precision']],
        'recall': [results['recall']],
        'f1_score': [results['f1']],
        'roc_auc': [results['auc']]
    })
    results_df.to_csv(results_dir / "model_results.csv", index=False)
    print(f"Results saved to {results_dir / 'model_results.csv'}")

if __name__ == "__main__":
    main()