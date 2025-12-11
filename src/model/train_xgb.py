from pathlib import Path
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

DATA = Path("data/processed/ml")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def main():
    df = pd.read_parquet(DATA / "dataset_xgb.parquet")

    target_col = "err_days"

    # on exclut les colonnes non numériques et les identifiants
    drop_cols = ["serial", "serial_norm", "color"]
    feature_cols = [
        c for c in df.columns
        if c not in drop_cols + [target_col]
        and not pd.api.types.is_datetime64_any_dtype(df[c])
    ]

    X = df[feature_cols].fillna(0.0)
    y = df[target_col].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBRegressor(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        n_jobs=-1,
        random_state=42,
    )

    print("[train_xgb] training...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    r2 = r2_score(y_test, y_pred)

    print("===== XGB Offset Model =====")
    print(f"MAE   : {mae:.2f} jours")
    print(f"RMSE  : {rmse:.2f} jours")
    print(f"R²    : {r2:.3f}")

    # sauvegarde
    joblib.dump(
        {
            "model": model,
            "feature_cols": feature_cols,
            "target_col": target_col,
        },
        MODEL_DIR / "xgb_offset_model.pkl",
    )
    print(f"[train_xgb] modèle sauvegardé -> {MODEL_DIR/'xgb_offset_model.pkl'}")

if __name__ == "__main__":
    main()
