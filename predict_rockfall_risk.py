import pandas as pd
import numpy as np
import joblib
import rasterio
from rasterio.transform import from_origin

# --- Simulation Functions (Replace with actual GEE/API calls) ---

def simulate_dem_features(latitude, longitude):
    """
    Simulates fetching DEM-derived features for a given location.
    In a real application, this would involve GEE API calls to extract
    mean elevation, slope, aspect, and roughness from a DEM image.
    """
    print(f"Simulating DEM feature extraction for Lat: {latitude}, Lon: {longitude}...")
    # For demonstration, generate random but somewhat plausible values
    mean_elevation = np.random.uniform(500, 3000) # meters
    max_elevation = mean_elevation + np.random.uniform(50, 500)
    min_elevation = mean_elevation - np.random.uniform(50, 500)
    elevation_range = max_elevation - min_elevation
    slope_mean = np.random.uniform(5, 45) # degrees
    aspect_mean = np.random.uniform(0, 360) # degrees
    roughness = np.random.uniform(5, 50) # std deviation of elevation

    return {
        'mean_elevation': mean_elevation,
        'max_elevation': max_elevation,
        'min_elevation': min_elevation,
        'elevation_range': elevation_range,
        'slope_mean': slope_mean,
        'aspect_mean': aspect_mean,
        'roughness': roughness
    }

def simulate_environmental_features(latitude, longitude):
    """
    Simulates fetching environmental features for a given location.
    In a real application, this would involve calls to weather APIs,
    seismic data APIs, etc.
    """
    print(f"Simulating environmental feature extraction for Lat: {latitude}, Lon: {longitude}...")
    # For demonstration, generate random but somewhat plausible values
    rainfall = np.random.uniform(0, 150) # mm
    temperature = np.random.uniform(5, 35) # Celsius
    vibration = np.random.uniform(0, 4) # arbitrary unit

    return {
        'rainfall': rainfall,
        'temperature': temperature,
        'vibration': vibration
    }

# --- Prediction Function ---

def predict_rockfall_risk(latitude, longitude, model_path='rockfall_model.pkl', scaler_path='scaler.pkl'):
    """
    Fetches features for a given location, scales them, and predicts rockfall risk.

    Args:
        latitude (float): Latitude of the location.
        longitude (float): Longitude of the location.
        model_path (str): Path to the trained machine learning model (.pkl).
        scaler_path (str): Path to the fitted StandardScaler (.pkl).

    Returns:
        str: Predicted rockfall risk level (Stable, Moderate, High) or an error message.
    """
    try:
        # Load the trained model and scaler
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)

        # Simulate fetching features for the given location
        dem_features = simulate_dem_features(latitude, longitude)
        environmental_features = simulate_environmental_features(latitude, longitude)

        # Combine all features into a single DataFrame row
        all_features = {**dem_features, **environmental_features}
        features_df = pd.DataFrame([all_features])

        # Ensure the order of columns matches the training data
        # This is crucial for correct scaling and prediction
        # We need to know the feature names from the training data.
        # For this simulation, we'll assume the order is consistent.
        # In a real scenario, you'd get X.columns from your training script.
        # For now, let's hardcode based on dem_feature_extraction.py's output order
        feature_order = [
            'mean_elevation', 'max_elevation', 'min_elevation', 'elevation_range',
            'slope_mean', 'aspect_mean', 'roughness',
            'rainfall', 'temperature', 'vibration'
        ]
        features_df = features_df[feature_order]


        # Scale the features
        scaled_features = scaler.transform(features_df)

        # Make prediction
        prediction = model.predict(scaled_features)
        prediction_proba = model.predict_proba(scaled_features)

        risk_map = {0: "Stable", 1: "Moderate", 2: "High"}
        predicted_risk = risk_map.get(prediction[0], "Unknown")

        print(f"\n--- Prediction Results for Lat: {latitude}, Lon: {longitude} ---")
        print(f"Predicted Rockfall Risk: {predicted_risk}")
        print(f"Prediction Probabilities: {prediction_proba[0]}") # Probabilities for each class

        return predicted_risk

    except FileNotFoundError:
        return "Error: Model or scaler file not found. Please ensure 'rockfall_model.pkl' and 'scaler.pkl' exist."
    except Exception as e:
        return f"An error occurred during prediction: {e}"

if __name__ == '__main__':
    # Example usage:
    test_latitude = 27.9881  # Example: Near Mount Everest
    test_longitude = 86.9250

    predict_rockfall_risk(test_latitude, test_longitude)

    print("\n--- Another example ---")
    predict_rockfall_risk(34.0522, -118.2437) # Example: Los Angeles