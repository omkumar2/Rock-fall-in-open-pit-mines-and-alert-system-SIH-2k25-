import joblib
import os

model_path = "rockfall_model.pkl"
scaler_path = "scaler.pkl"

print(f"Current working directory: {os.getcwd()}")
print(f"Attempting to load model from: {os.path.abspath(model_path)}")
print(f"Attempting to load scaler from: {os.path.abspath(scaler_path)}")

try:
    model = joblib.load(model_path)
    print(f"Successfully loaded model: {type(model)}")
except Exception as e:
    print(f"Failed to load model: {e}")

try:
    scaler = joblib.load(scaler_path)
    print(f"Successfully loaded scaler: {type(scaler)}")
except Exception as e:
    print(f"Failed to load scaler: {e}")