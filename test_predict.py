import os
import sys
import json
from typing import Any, Dict

import requests


def main() -> int:
    url = os.getenv("API_URL", "http://127.0.0.1:8000/predict")
    payload: Dict[str, Any] = {"location": "Goa, India"}

    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"Request failed: {exc}")
        return 1

    try:
        data = response.json()
    except ValueError:
        print("Response was not JSON:")
        print(response.text)
        return 1

    print(json.dumps(data, indent=2, ensure_ascii=False))

    # Validate expected keys
    expected_keys = {"location", "risk_level", "probability", "alert_sent"}
    missing = expected_keys - set(data.keys())
    if missing:
        print(f"Missing expected keys in response: {sorted(missing)}")
        return 1

    # Optional simple type checks
    if not isinstance(data.get("risk_level"), str):
        print("Invalid type: 'risk_level' should be a string")
        return 1
    if not isinstance(data.get("probability"), (int, float)):
        print("Invalid type: 'probability' should be a number")
        return 1
    if not isinstance(data.get("alert_sent"), bool):
        print("Invalid type: 'alert_sent' should be a boolean")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())


