# src/features/smooth_levels.py
import pandas as pd
import numpy as np

def exponential_smoothing(series: pd.Series, alpha: float = 0.2):
    """
    Applique un lissage exponentiel simple sur une série temporelle.
    Paramètres:
      - series: niveaux d’encre (%)
      - alpha: poids du dernier point (0.1 à 0.3 recommandé)
    """
    result = []
    s = None

    for value in series:
        if pd.isna(value):
            result.append(s)
            continue

        if s is None:  # premier point
            s = value
        else:
            s = alpha * value + (1 - alpha) * s

        result.append(s)

    return pd.Series(result, index=series.index)


def compute_smoothed_slopes(df_kpax, alpha=0.2):
    """
    Calcule une pente lissée par (serial, color).
    df_kpax = consommables WITH resets → contient:
        serial_norm, date, color, level_pct
    Retour:
        DataFrame avec slope_smoothed par cycle.
    """

    rows = []

    for (serial, color), sub in df_kpax.groupby(["serial_norm", "color"]):
        sub = sub.sort_values("date")

        # lissage expo
        sub["smooth_level"] = exponential_smoothing(sub["level_pct"], alpha=alpha)

        # pente par regression linéaire
        if len(sub) >= 3:
            x = (sub["date"] - sub["date"].min()).dt.total_seconds() / 86400.0
            y = sub["smooth_level"]

            slope = np.polyfit(x, y, 1)[0] if y.notna().sum() >= 3 else np.nan
        else:
            slope = np.nan

        rows.append({
            "serial_norm": serial,
            "color": color,
            "slope_smoothed": slope
        })

    return pd.DataFrame(rows)
