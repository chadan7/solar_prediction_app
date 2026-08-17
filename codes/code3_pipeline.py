import joblib
import pandas as pd
from src.code1_theoretical import calculate_theoretical_power
from src.code2_weather import fetch_and_prepare_features


def predict_solar_yield_pipeline(
    user_inputs, model_path="Model/random_forest_k_pv.pkl"
):
    # 1. Fetch Open-Meteo features & timezone
    raw_features_df, times = fetch_and_prepare_features(user_inputs)

    # 2. Run Code 1: Theoretical power
    code1_out = calculate_theoretical_power(user_inputs, times)

    # 3. Run Code 2: Compute irradiance ratios
    features_df, _ = fetch_and_prepare_features(
        user_inputs, code1_outputs=code1_out
    )

    # 4. Model inference
    model = joblib.load(model_path)
    model_features = [
        "nwp_globalirrad",
        "nwp_directirrad",
        "nwp_temperature",
        "nwp_humidity",
        "nwp_windspeed",
        "nwp_winddirection",
        "nwp_pressure",
        "GHI_ratio",
        "DNI_ratio",
    ]

    predicted_k = model.predict(features_df[model_features])
    predicted_k_series = pd.Series(predicted_k, index=times)

    p_actual = code1_out["p_theoretical"] * predicted_k_series

    return pd.DataFrame(
        {
            "Timestamp": times.strftime("%Y-%m-%d %H:%M:%S"),
            "P_Theoretical_kW": (code1_out["p_theoretical"] / 1000.0).round(2),
            "Predicted_K_PV": predicted_k_series.round(4),
            "Predicted_Actual_Power_kW": (p_actual / 1000.0).round(2),
        }
    )
