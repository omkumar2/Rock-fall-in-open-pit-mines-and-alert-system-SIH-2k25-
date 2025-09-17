# Rockfall Risk Prediction System

This project implements a system for predicting rockfall risk using a FastAPI backend for predictions and a Streamlit frontend for user interaction.

## Project Structure

- [`app.py`](app.py): FastAPI application that exposes the `/predict` endpoint for rockfall risk prediction.
- [`streamlit_app.py`](streamlit_app.py): Streamlit application that provides a user interface to interact with the prediction system.
- [`dem_feature_extraction.py`](dem_feature_extraction.py): Contains logic for extracting features from Digital Elevation Models (DEMs).
- [`alerts.py`](alerts.py): Handles sending email and SMS alerts based on risk levels.
- [`rockfall_model.pkl`](rockfall_model.pkl) and [`scaler.pkl`](scaler.pkl): Pre-trained machine learning model and scaler for predictions.
- [`test_predict.py`](test_predict.py): A client script to test the `/predict` endpoint of the FastAPI application.
- `requirements.txt`: Lists Python dependencies for the project.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd SIH 2k25
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv .venv
    .venv\Scripts\activate  # On Windows
    source .venv/bin/activate  # On macOS/Linux
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    (Note: If `requirements.txt` is not found, try `pip install -r "requirements (2).txt"` or check for the correct requirements file name.)

## How to Run

To run the full application, you need to start both the FastAPI backend and the Streamlit frontend.

### 1. Start the FastAPI Backend

Open a terminal and run the FastAPI application:

```bash
python app.py
```

This will start the API server, typically on `http://0.0.0.0:8000`. Keep this terminal running.

### 2. Start the Streamlit Frontend

Open a **new** terminal and run the Streamlit application:

```bash
streamlit run streamlit_app.py
```

This will launch the Streamlit application in your web browser, usually at `http://localhost:8501`.