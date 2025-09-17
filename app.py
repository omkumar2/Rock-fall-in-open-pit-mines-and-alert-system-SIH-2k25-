import os
import logging
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from dem_feature_extraction import extract_dem_features
from alerts import send_email_alert, send_sms_alert


logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class PredictRequest(BaseModel):
    location: str


class PredictResponse(BaseModel):
    location: str
    risk_level: str
    probability: float
    alert_sent: bool
    latitude: float
    longitude: float


def _load_model_and_scaler(model_path: str = "rockfall_model.pkl", scaler_path: str = "scaler.pkl"):
    try:
        logger.info("Attempting to load model from absolute path: %s", os.path.abspath(model_path))
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except FileNotFoundError as e:
        logger.error("Model or scaler file not found: %s", e)
        return None, None
    except Exception:
        logger.error("Failed to load model/scaler", exc_info=True)
        return None, None


def _compute_probability(model, scaled_features: np.ndarray) -> float:
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(scaled_features)
            return float(np.max(proba[0]))
        # Fallback: attempt decision_function → convert to pseudo probability
        if hasattr(model, "decision_function"):
            decision = model.decision_function(scaled_features)
            # Handle binary or multiclass
            arr = np.array(decision)
            if arr.ndim == 1:
                # binary sigmoid
                prob_pos = 1.0 / (1.0 + np.exp(-arr[0]))
                return float(max(prob_pos, 1.0 - prob_pos))
            # multiclass softmax
            exps = np.exp(arr[0] - np.max(arr[0]))
            softmax = exps / np.sum(exps)
            return float(np.max(softmax))
        # Last resort
        return 0.5
    except Exception:
        logger.exception("Failed to compute probability; defaulting to 0.0")
        return 0.0


def _predict_risk(features_df: pd.DataFrame, model, scaler) -> tuple[str, float]:
    feature_order = [
        'mean_elevation', 'max_elevation', 'min_elevation', 'elevation_range',
        'slope_mean', 'aspect_mean', 'roughness', 'rainfall', 'temperature', 'vibration'
    ]
    try:
        X = features_df[feature_order]
    except KeyError as e:
        missing = set([*feature_order]) - set(features_df.columns)
        raise HTTPException(status_code=500, detail=f"Missing features for prediction: {sorted(missing)}") from e

    try:
        X_scaled = scaler.transform(X)
        pred = model.predict(X_scaled)
        risk_map = {0: "Stable", 1: "Moderate", 2: "High"}
        risk_label = risk_map.get(int(pred[0]), "Unknown")
        probability = _compute_probability(model, X_scaled)
        return risk_label, probability
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


def _get_dem_path_from_env() -> str:
    # Allow override via DEM_PATH; fallback to dummy_dem.tif
    return os.getenv("DEM_PATH", "dummy_dem.tif")


app = FastAPI(title="Rockfall Risk API")

# CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    location = req.location
    logger.info(f"Received prediction request for location: {location}")

    try:
        # 1. Extract features using dummy data
        features_df = extract_dem_features(location)
        if features_df.empty:
            logger.error("Failed to generate features")
            raise HTTPException(status_code=500, detail="Failed to generate features")

        # 2. Load model and make prediction
        model, scaler = _load_model_and_scaler()
        if model is None or scaler is None:
            logger.error("Failed to load model or scaler")
            raise HTTPException(status_code=500, detail="Model unavailable")

        # 3. Make prediction
        risk_level, probability = _predict_risk(features_df, model, scaler)
        logger.info(f"Prediction: {risk_level} with probability {probability:.2f}")

        # 4. Send alerts if high risk
        alert_sent = False
        if risk_level == "High":
            try:
                email_ok = send_email_alert(location, risk_level, probability)
                sms_ok = send_sms_alert(location, risk_level, probability)
                alert_sent = bool(email_ok or sms_ok)
            except Exception as e:
                logger.error(f"Failed to send alerts: {e}")
                # Continue even if alerts fail

        # 5. Get coordinates (either from features or defaults)
        lat = features_df['latitude'].iloc[0] if 'latitude' in features_df.columns else 25.0
        lon = features_df['longitude'].iloc[0] if 'longitude' in features_df.columns else 85.0

        return PredictResponse(
            location=location,
            risk_level=risk_level,
            probability=float(probability),
            alert_sent=alert_sent,
            latitude=lat,
            longitude=lon
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unexpected error during prediction")
        raise HTTPException(status_code=500, detail=str(e))

    # 2. Load model and scaler, then predict; if unavailable, fall back to DEM-derived label
    model, scaler = _load_model_and_scaler()
    if model is not None and scaler is not None:
        risk_level, probability = _predict_risk(features_df, model, scaler)
    else:
        # Fallback: use embedded heuristic label from DEM features if present
        if 'rockfall_risk' in features_df.columns:
            risk_map = {0: "Stable", 1: "Moderate", 2: "High"}
            numeric_label = int(features_df['rockfall_risk'].iloc[0])
            risk_level = risk_map.get(numeric_label, "Unknown")
            probability = {"Stable": 0.2, "Moderate": 0.6, "High": 0.9}.get(risk_level, 0.5)
            logger.info("Using fallback prediction (no model/scaler). risk=%s prob=%.2f", risk_level, probability)
        else:
            raise HTTPException(status_code=500, detail="Model/scaler unavailable and no fallback label present")

    # 3. Alerts if High
    alert_sent = False
    if risk_level == "High":
        email_ok = send_email_alert(location, risk_level, probability)
        sms_ok = send_sms_alert(location, risk_level, probability)
        alert_sent = bool(email_ok or sms_ok)

    # 4. Return JSON
    # TODO: Implement actual geocoding for latitude and longitude based on 'location'
    # For now, using placeholder values or deriving from DEM features if possible.
    # Assuming a default or a way to get these from the DEM features.
    # For this example, I'll use fixed values or try to get them from features_df if available.
    # If features_df contains 'latitude' and 'longitude' columns, use them.
    # Otherwise, use default values.
    lat = features_df['latitude'].iloc[0] if 'latitude' in features_df.columns else 25.0
    lon = features_df['longitude'].iloc[0] if 'longitude' in features_df.columns else 85.0

    return PredictResponse(
        location=location,
        risk_level=risk_level,
        probability=float(probability),
        alert_sent=alert_sent,
        latitude=lat,
        longitude=lon,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)


