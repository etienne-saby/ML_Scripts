# 01_Scripts/Python/03_ML/predict_climate.py
import pandas as pd
import numpy as np

def predict_climate(lat, lon, main_crop, climate_models, n_years=40):
    """Predict climate features using surrogate models."""
    df = pd.DataFrame({
        "latitude":              np.repeat(lat, n_years),
        "longitude":             np.repeat(lon, n_years),
        "main_crop":             pd.Categorical([main_crop] * n_years),
        "Harvest_Year_Absolute": np.arange(1, n_years + 1, dtype=np.int64),
    })
    
    results = {}
    for target, model in climate_models.items():
        results[target] = model.predict(df).tolist()
    
    return results