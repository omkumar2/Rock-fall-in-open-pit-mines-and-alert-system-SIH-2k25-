import ee
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
from typing import Optional, Dict, Any, List

def geocode_location(location_name: str) -> Optional[Dict[str, float]]:
    """
    Converts a location name to latitude and longitude coordinates using Nominatim.

    Args:
        location_name (str): The name of the location (e.g., "Mount Everest").

    Returns:
        Optional[Dict[str, float]]: A dictionary with 'latitude' and 'longitude'
                                    if geocoding is successful, otherwise None.
    """
    geolocator = Nominatim(user_agent="rockfall_prediction_app")
    try:
        location = geolocator.geocode(location_name, timeout=10)
        if location:
            print(f"Geocoded '{location_name}' to Lat: {location.latitude}, Lon: {location.longitude}")
            return {"latitude": location.latitude, "longitude": location.longitude}
        else:
            print(f"Could not geocode location: {location_name}")
            return None
    except GeocoderTimedOut:
        print("Geocoding service timed out. Please try again later.")
        return None
    except GeocoderServiceError as e:
        print(f"Geocoding service error: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during geocoding: {e}")
        return None

def fetch_dem_features_from_gee(location_name: str, buffer_meters: float = 1000) -> pd.DataFrame:
    """
    Fetches DEM data for a given location using Google Earth Engine (GEE) API,
    extracts elevation, slope, aspect, and roughness features, and returns them
    as a Pandas DataFrame.

    Args:
        location_name (str): The name of the location (e.g., "Mount Everest").
        buffer_meters (float): The buffer distance in meters around the point
                                to calculate features over.

    Returns:
        pd.DataFrame: A DataFrame with a single row containing the extracted
                      DEM features, or an empty DataFrame if an error occurs.
    """
    try:
        # Initialize Earth Engine (if not already initialized)
        try:
            ee.Initialize()
        except Exception:
            print("Earth Engine already initialized or authentication required. Attempting to proceed.")
            # If ee.Initialize() fails, it might be due to already initialized or auth issues.
            # User needs to run `earthengine authenticate` if not done.

        coords = geocode_location(location_name)
        if not coords:
            return pd.DataFrame()

        point = ee.Geometry.Point(coords['longitude'], coords['latitude'])
        buffered_point = point.buffer(buffer_meters)

        # Load SRTM DEM data
        dem = ee.Image('USGS/SRTMGL1_003')

        # Calculate terrain features
        # Elevation
        elevation = dem.select('elevation')

        # Slope and Aspect
        # ee.Terrain.products computes slope, aspect, and hillshade
        terrain = ee.Terrain.products(elevation)
        slope = terrain.select('slope')
        aspect = terrain.select('aspect')

        # Roughness (using standard deviation of elevation within a neighborhood)
        # A simple way to estimate roughness is the standard deviation of elevation
        # within a small window. Here, we'll use a larger buffer for overall roughness.
        # For a more precise roughness, a smaller kernel would be used.
        roughness_image = elevation.reduceNeighborhood(
            reducer=ee.Reducer.stdDev(),
            kernel=ee.Kernel.square(radius=buffer_meters / 30, units='meters') # Adjust radius as needed
        )

        # Combine features for reduction
        combined_image = elevation.addBands(slope).addBands(aspect).addBands(roughness_image.rename('roughness'))

        # Define reducers for each feature
        reducers = ee.Reducer.mean().combine(
            reducer2=ee.Reducer.max(),
            sharedInputs=True
        ).combine(
            reducer2=ee.Reducer.min(),
            sharedInputs=True
        )

        # Extract statistics over the buffered point
        stats = combined_image.reduceRegion(
            reducer=reducers,
            geometry=buffered_point,
            scale=30,  # SRTM resolution is 30 meters
            maxPixels=1e9
        )

        # Convert GEE results to a dictionary
        feature_data: Dict[str, float] = {}
        if stats.getInfo():
            info = stats.getInfo()
            feature_data['mean_elevation'] = info.get('elevation_mean', 0.0)
            feature_data['max_elevation'] = info.get('elevation_max', 0.0)
            feature_data['min_elevation'] = info.get('elevation_min', 0.0)
            feature_data['elevation_range'] = feature_data['max_elevation'] - feature_data['min_elevation']
            feature_data['slope_mean'] = info.get('slope_mean', 0.0)
            feature_data['aspect_mean'] = info.get('aspect_mean', 0.0)
            feature_data['roughness'] = info.get('roughness_stdDev', 0.0) # Assuming stdDev is used for roughness

            # Create a DataFrame
            return pd.DataFrame([feature_data])
        else:
            print(f"No GEE data found for {location_name} within the specified buffer.")
            return pd.DataFrame()

    except ee.EEException as e:
        print(f"Google Earth Engine error: {e}")
        print("Please ensure you have authenticated with GEE (`earthengine authenticate`) and have an active project.")
        return pd.DataFrame()
    except Exception as e:
        print(f"An unexpected error occurred during GEE data fetching: {e}")
        return pd.DataFrame()

if __name__ == '__main__':
    # Example usage
    print("--- Fetching DEM features for Mount Everest ---")
    mount_everest_features = fetch_dem_features_from_gee("Mount Everest")
    if not mount_everest_features.empty:
        print(mount_everest_features)
    else:
        print("Failed to fetch features for Mount Everest.")

    print("\n--- Fetching DEM features for Grand Canyon ---")
    grand_canyon_features = fetch_dem_features_from_gee("Grand Canyon")
    if not grand_canyon_features.empty:
        print(grand_canyon_features)
    else:
        print("Failed to fetch features for Grand Canyon.")

    print("\n--- Fetching DEM features for a non-existent location (error handling) ---")
    non_existent_location_features = fetch_dem_features_from_gee("NonExistentPlace12345")
    if non_existent_location_features.empty:
        print("Successfully handled non-existent location.")