import datetime
import requests
import pandas as pd
import numpy as np

def fetch_and_prepare_features(user_inputs, code1_outputs=None):
    """
    Code 2: Fetches Open-Meteo NWP features and computes GHI/DNI ratios.
    Handles historical archives vs future forecasts dynamically.
    """
    start_dt = datetime.datetime.strptime(user_inputs["start_date"], "%Y-%m-%d").date()
    today_dt = datetime.date.today()

    # Route request to historical archive or forecast endpoint
    if start_dt < today_dt:
        base_url = "https://archive-api.open-meteo.com/v1/archive"
    else:
        base_url = "https://api.open-meteo.com/v1/forecast"

    url = (
        f"{base_url}?"
        f"latitude={user_inputs['latitude']}&"
        f"longitude={user_inputs['longitude']}&"
        f"start_date={user_inputs['start_date']}&"
        f"end_date={user_inputs['end_date']}&"
        f"hourly=temperature_2m,relative_humidity_2m,surface_pressure,"
        f"wind_speed_10m,wind_direction_10m,direct_normal_irradiance,"
        f"global_tilted_irradiance&"
        f"timezone=auto"
    )

    response = requests.get(url).json()
    hourly = response["hourly"]
    timezone_str = response.get("timezone", "UTC")
    user_inputs["timezone"] = timezone_str

    # Create timezone-aware DatetimeIndex
    times = pd.DatetimeIndex(pd.to_datetime(hourly["time"])).tz_localize(timezone_str)

    # Construct base NWP feature set
    features_df = pd.DataFrame(index=times)
    features_df["nwp_globalirrad"] = hourly["global_tilted_irradiance"]
    features_df["nwp_directirrad"] = hourly["direct_normal_irradiance"]
    features_df["nwp_temperature"] = hourly["temperature_2m"]
    features_df["nwp_humidity"] = hourly["relative_humidity_2m"]
    features_df["nwp_windspeed"] = hourly["wind_speed_10m"]
    features_df["nwp_winddirection"] = hourly["wind_direction_10m"]
    features_df["nwp_pressure"] = hourly["surface_pressure"]

    # Compute GHI_ratio and DNI_ratio if Code 1 outputs are provided
    if code1_outputs is not None:
        ghi_clear_safe = code1_outputs["ghi_clear"].replace(0, np.nan)
        dni_clear_safe = code1_outputs["dni_clear"].replace(0, np.nan)

        features_df["GHI_ratio"] = (features_df["nwp_globalirrad"] / ghi_clear_safe).fillna(0.0).clip(0.0, 1.2)
        features_df["DNI_ratio"] = (features_df["nwp_directirrad"] / dni_clear_safe).fillna(0.0).clip(0.0, 1.2)

    return features_df, times