import pandas as pd
from sklearn.preprocessing import StandardScaler

def create_features(df):
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values(['city', 'datetime'])

    df['hour'] = df['datetime'].dt.hour
    df['dayofweek'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.month

    #Lag features 
    df['rain_lag1'] = df.groupby('city')['rain'].shift(1)
    df['rain_lag3'] = df.groupby('city')['rain'].shift(3)
    df['rain_lag24'] = df.groupby('city')['rain'].shift(24)
    df['rain_roll3'] = df.groupby('city')['rain'] \
        .shift(1).rolling(3).mean().reset_index(0, drop=True)

    df['rain_roll24'] = df.groupby('city')['rain'] \
        .shift(1).rolling(24).mean().reset_index(0, drop=True)
    df['target'] = df.groupby('city')['rain'].shift(-1)
    df['target'] = (df['target'] > 0).astype(int)
    df = pd.get_dummies(df, columns=['city'], drop_first=True)
    df = df.dropna().reset_index(drop=True)

    return df

def split_and_scale(df):
    df = df.copy()

    df = df.sort_values('datetime').reset_index(drop=True)
    features = [
        'temperature', 'humidity', 'wind', 'pressure',
        'hour', 'dayofweek', 'month',
        'rain_lag1', 'rain_lag3', 'rain_lag24',
        'rain_roll3', 'rain_roll24'
    ] + [col for col in df.columns if 'city_' in col]

    target = 'target'
    train_size = int(len(df) * 0.7)
    val_size = int(len(df) * 0.15)

    train = df.iloc[:train_size]
    val = df.iloc[train_size:train_size + val_size]
    test = df.iloc[train_size + val_size:]

    X_train, y_train = train[features], train[target]
    X_val, y_val = val[features], val[target]
    X_test, y_test = test[features], test[target]
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler

def preprocess_pipeline(df):
    df = create_features(df)
    return split_and_scale(df)

def save_to_json(X, y, path):
    import json
    data = []

    for i in range(len(X)):
        data.append({
            "features": X[i].tolist(),
            "target": float(y.iloc[i])
        })

    with open(path, "w") as f:
        json.dump(data, f)

if __name__ == "__main__":
    df = pd.read_csv("../data/processed/data.csv")

    (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler = preprocess_pipeline(df)

    print("Train:", X_train.shape)
    print("Val:", X_val.shape)
    print("Test:", X_test.shape)

    # Lưu JSON
    save_to_json(X_train, y_train, "../data/processed/train.json")
    save_to_json(X_val, y_val, "../data/processed/val.json")
    save_to_json(X_test, y_test, "../data/processed/test.json")