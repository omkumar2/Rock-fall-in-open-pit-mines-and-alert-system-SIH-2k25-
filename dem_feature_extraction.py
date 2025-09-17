import rasterio
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def extract_dem_features(dem_path):
    """
    Extracts numerical features from a DEM raster and stores them in a Pandas DataFrame.

    Args:
        dem_path (str): Path to the DEM GeoTIFF file.

    Returns:
        pandas.DataFrame: DataFrame containing the extracted features.
    """
    try:
        with rasterio.open(dem_path) as src:
            dem_array = src.read(1)

            # Calculate features
            mean_elevation = np.mean(dem_array)
            max_elevation = np.max(dem_array)
            min_elevation = np.min(dem_array)
            elevation_range = max_elevation - min_elevation
            roughness = np.std(dem_array)

            # Calculate slope and aspect using numpy gradient
            gradient_x, gradient_y = np.gradient(dem_array)
            slope = np.arctan(np.sqrt(gradient_x**2 + gradient_y**2))
            aspect = np.arctan2(gradient_y, gradient_x)

            # Convert slope and aspect to degrees
            slope_degrees = np.degrees(slope)
            aspect_degrees = np.degrees(aspect)

            # Handle edge cases for aspect (optional)
            aspect_degrees = (aspect_degrees + 360) % 360  # Ensure values between 0 and 360

            # Add synthetic environmental features
            rainfall = np.random.rand() * 100  # Example rainfall value
            temperature = np.random.rand() * 30 + 10  # Example temperature value (10-40)
            vibration = np.random.rand() * 5  # Example vibration value

            # Create DataFrame
            data = {
                'mean_elevation': [mean_elevation],
                'max_elevation': [max_elevation],
                'min_elevation': [min_elevation],
                'elevation_range': [elevation_range],
                'slope_mean': [np.mean(slope_degrees)],
                'aspect_mean': [np.mean(aspect_degrees)],
                'roughness': [roughness],
                'rainfall': [rainfall],
                'temperature': [temperature],
                'vibration': [vibration]
            }
            df = pd.DataFrame(data)

            # Add rockfall_risk label based on a more deterministic (but still synthetic) logic
            # This is a simplified example to demonstrate how features could influence risk
            risk_level = 0 # Stable

            if np.mean(slope_degrees) > 20 and roughness > 15:
                risk_level = 1 # Moderate
            if np.mean(slope_degrees) > 35 and roughness > 25 and rainfall > 70 and vibration > 3:
                risk_level = 2 # High

            df['rockfall_risk'] = risk_level

            return df

    except rasterio.RasterioIOError as e:
        print(f"Error opening or reading the DEM file: {e}")
        print("Please ensure the GeoTIFF file exists at the specified path and is valid.")
        return pd.DataFrame()  # Return an empty DataFrame in case of error


if __name__ == '__main__':
    # Replace 'real_dem.tif' with the actual path to your downloaded GEE GeoTIFF file
    dem_file = 'dummy_dem.tif'
    features_df = extract_dem_features(dem_file)

    if not features_df.empty:
        print(features_df)
    else:
        print(f"No features extracted. Ensure '{dem_file}' is a valid GeoTIFF.")