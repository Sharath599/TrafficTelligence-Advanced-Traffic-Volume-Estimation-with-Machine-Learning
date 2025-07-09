import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
def load_and_preprocess(filepath):
    df = pd.read_csv(filepath)
    if 'date_time' in df.columns:
        df['date_time'] = pd.to_datetime(df['date_time'])
        df['hour'] = df['date_time'].dt.hour
        df['dayofweek'] = df['date_time'].dt.dayofweek
        df.drop(columns=['date_time'], inplace=True)
    target = 'traffic_volume'
    if target not in df.columns:
        raise ValueError("Target column 'traffic_volume' not found!")
    X = df.drop(columns=[target])
    y = df[target]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, scaler
