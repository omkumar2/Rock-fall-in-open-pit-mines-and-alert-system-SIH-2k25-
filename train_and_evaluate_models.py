import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import joblib

# Assuming dem_feature_extraction.py is in the same directory
from dem_feature_extraction import extract_dem_features

def train_and_evaluate_models(dem_path='dummy_dem.tif'):
    """
    Loads DEM and environmental features, splits data, normalizes features,
    trains and evaluates multiple classification models.

    Args:
        dem_path (str): Path to the DEM GeoTIFF file.

    Returns:
        pandas.DataFrame: DataFrame containing the evaluation results.
    """
    # 1. Load the DataFrame
    df = extract_dem_features(dem_path)

    if df.empty:
        print("Failed to load features. Exiting.")
        return pd.DataFrame()

    # Prepare data for ML
    X = df.drop('rockfall_risk', axis=1)
    y = df['rockfall_risk']

    # Ensure there's enough data for splitting (especially if only one row is generated)
    if len(df) < 2:
        print("Warning: Not enough data to split into training and testing sets. Generating more synthetic data.")
        # Generate more synthetic data for demonstration purposes
        num_additional_samples = 100 - len(df)
        additional_data = []
        for _ in range(num_additional_samples):
            # Re-run feature extraction to get a new row of data
            new_df_row = extract_dem_features(dem_path)
            if not new_df_row.empty:
                additional_data.append(new_df_row.iloc[0])
        if additional_data:
            df = pd.concat([df, pd.DataFrame(additional_data)], ignore_index=True)
            X = df.drop('rockfall_risk', axis=1)
            y = df['rockfall_risk']
        else:
            print("Could not generate enough synthetic data. Exiting.")
            return pd.DataFrame()


    # 2. Split into training (80%) and testing (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 3. Apply StandardScaler for feature normalization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save the scaler
    joblib.dump(scaler, 'scaler.pkl')
    print("\nStandardScaler saved to scaler.pkl")

    # Define parameter grids for GridSearchCV
    param_grid_rf = {
        'n_estimators': [50, 100],
        'max_depth': [None, 10]
    }
    param_grid_gb = {
        'n_estimators': [50, 100],
        'learning_rate': [0.05, 0.1]
    }

    models = {
        'RandomForestClassifier': (RandomForestClassifier(random_state=42), param_grid_rf),
        'LogisticRegression': (LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=42, max_iter=1000), {}), # No GridSearchCV for Logistic Regression
        'GradientBoostingClassifier': (GradientBoostingClassifier(random_state=42), param_grid_gb)
    }

    results = []
    best_model = None
    best_accuracy = -1.0
    best_model_name = ""

    for name, (model, param_grid) in models.items():
        print(f"\nTraining and optimizing {name}...")
        if param_grid: # Apply GridSearchCV if param_grid is not empty
            grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
            grid_search.fit(X_train_scaled, y_train)
            model = grid_search.best_estimator_
            print(f"Best parameters for {name}: {grid_search.best_params_}")

        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        results.append({
            'Model': name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        })

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_model_name = name

    results_df = pd.DataFrame(results)
    print("\n--- Model Evaluation Results ---")
    print(results_df.to_string(index=False))

    # Save best model if accuracy > 85%
    if best_accuracy > 0.85:
        model_filename = 'rockfall_model.pkl'
        joblib.dump(best_model, model_filename)
        print(f"\nBest model ({best_model_name}) with accuracy {best_accuracy:.2f} saved to {model_filename}")
    else:
        print(f"\nBest model accuracy ({best_accuracy:.2f}) did not exceed 85%. Model not saved.")

    return results_df

if __name__ == '__main__':
    train_and_evaluate_models()