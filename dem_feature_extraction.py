import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def extract_dem_features(location_str: str) -> pd.DataFrame:
    """Generate deterministic dummy features for a given location string.

    This function intentionally avoids heavy GIS dependencies so the API
    remains testable even when real DEM processing is not available.

    The returned DataFrame contains the exact columns that `app._predict_risk`
    expects:
      - mean_elevation, max_elevation, min_elevation, elevation_range,
        slope_mean, aspect_mean, roughness, rainfall, temperature, vibration
    Additionally it may include:
      - rockfall_risk (0=Stable,1=Moderate,2=High) used as a fallback label
      - latitude, longitude (floats)

    Args:
        location_str: arbitrary location identifier (used to seed RNG deterministically)

    Returns:
        pd.DataFrame with one row of features, or empty DataFrame on fatal error.
    """
    try:
        # Use a hash of the location string to produce deterministic but varied results
        seed = abs(hash(location_str)) % (2 ** 32)
        rng = np.random.default_rng(seed)

        mean_elevation = float(rng.uniform(100, 1000))
        elevation_range = float(rng.uniform(50, 200))
        max_elevation = mean_elevation + elevation_range / 2.0
        min_elevation = mean_elevation - elevation_range / 2.0
        roughness = float(rng.uniform(0.1, 5.0))

        slope_mean = float(rng.uniform(0, 45))
        aspect_mean = float(rng.uniform(0, 360))

        rainfall = float(rng.uniform(0, 100))
        temperature = float(rng.uniform(10, 40))
        vibration = float(rng.uniform(0, 5))

        data = {
            'mean_elevation': [mean_elevation],
            'max_elevation': [max_elevation],
            'min_elevation': [min_elevation],
            'elevation_range': [elevation_range],
            'slope_mean': [slope_mean],
            'aspect_mean': [aspect_mean],
            'roughness': [roughness],
            'rainfall': [rainfall],
            'temperature': [temperature],
            'vibration': [vibration],
        }

        df = pd.DataFrame(data)

        # Simple deterministic heuristic label for fallback when model is absent
        if slope_mean > 35 and roughness > 2.5 and rainfall > 70 and vibration > 3:
            df['rockfall_risk'] = 2
        elif slope_mean > 20 and roughness > 1.5:
            df['rockfall_risk'] = 1
        else:
            df['rockfall_risk'] = 0

        # Synthetic coordinates derived from seed so location->coords consistent
        lat = 25.0 + (seed % 1000) / 1000.0 - 0.5
        lon = 85.0 + ((seed // 1000) % 1000) / 1000.0 - 0.5
        df['latitude'] = float(lat)
        df['longitude'] = float(lon)

        return df

    except Exception as exc:
        logger.exception("Failed to generate dummy DEM features for %s", location_str)
        return pd.DataFrame()
