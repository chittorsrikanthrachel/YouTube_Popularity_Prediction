import pandas as pd
import numpy as np
import pickle
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# -----------------------------
# Load models
# -----------------------------
with open(os.path.join(MODEL_DIR, "views_rfr.pkl"), "rb") as f:
    rf_model = pickle.load(f)

with open(os.path.join(MODEL_DIR, "views_gbr.pkl"), "rb") as f:
    gbr_model = pickle.load(f)

with open(os.path.join(MODEL_DIR, "views_xgb.pkl"), "rb") as f:
    xgb_model = pickle.load(f)

with open(os.path.join(MODEL_DIR, "views_lgb.pkl"), "rb") as f:
    lgb_model = pickle.load(f)

# -----------------------------
# Load preprocessing
# -----------------------------
with open(os.path.join(MODEL_DIR, "imputer.pkl"), "rb") as f:
    imputer = pickle.load(f)

with open(os.path.join(MODEL_DIR, "pt.pkl"), "rb") as f:
    pt = pickle.load(f)

with open(os.path.join(MODEL_DIR, "X_columns.pkl"), "rb") as f:
    X_columns = pickle.load(f)

with open(os.path.join(MODEL_DIR, "numerical_cols.pkl"), "rb") as f:
    numerical_cols = pickle.load(f)

# -----------------------------
# Helper
# -----------------------------
def human_readable(n):
    if n >= 1_000_000_000:
        return f"{n/1_000_000_000:.1f}B"
    elif n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    elif n >= 1_000:
        return f"{n/1_000:.1f}K"
    else:
        return str(int(n))

# -----------------------------
# Predict Views
# -----------------------------
def predict_views(input_dict):
    df = pd.DataFrame([input_dict])

    # add missing columns
    for col in X_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[X_columns]

    # preprocessing (SAME as training)
    df[numerical_cols] = imputer.transform(df[numerical_cols])
    df[numerical_cols] = np.log1p(df[numerical_cols])
    df[numerical_cols] = pt.transform(df[numerical_cols])

    output = {
        "RandomForest": human_readable(np.expm1(rf_model.predict(df)[0])),
        "GradientBoosting": human_readable(np.expm1(gbr_model.predict(df)[0])),
        "XGBoost": human_readable(np.expm1(xgb_model.predict(df)[0])),
        "LightGBM": human_readable(np.expm1(lgb_model.predict(df)[0]))
    }

    return output
