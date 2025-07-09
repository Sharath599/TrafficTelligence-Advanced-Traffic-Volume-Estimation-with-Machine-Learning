import pandas as pd
import numpy as np
import pickle
import joblib  # Added for saving feature columns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
data = pd.read_csv("traffic volume.csv")
data['temp'] = data['temp'].fillna(data['temp'].mean())
data['rain'] = data['rain'].fillna(data['rain'].mean())
data['snow'] = data['snow'].fillna(data['snow'].mean())
data['weather'] = data['weather'].fillna(data['weather'].mode()[0])
data['date'] = pd.to_datetime(data['date'], format='%d-%m-%Y')
data['year'] = data['date'].dt.year
data['month'] = data['date'].dt.month
data['day'] = data['date'].dt.day
data['hour'] = pd.to_datetime(data['Time']).dt.hour
data.drop(['date', 'Time'], axis=1, inplace=True)
data = pd.get_dummies(data)
X = data.drop('traffic_volume', axis=1)
y = data['traffic_volume']
pd.DataFrame({"columns": X.columns}).to_csv("training_columns.csv", index=False)
joblib.dump(X.columns.tolist(), "feature_columns.pkl")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=0)
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
print(" Model, Scaler, and Feature Columns saved successfully.")
print(f" R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.2f}")
