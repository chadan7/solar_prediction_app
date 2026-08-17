import datetime
import os
import re
from geopy.geocoders import Nominatim
import pandas as pd
import streamlit as st

from src.code3_pipeline import predict_solar_yield_pipeline

# Page Configuration
st.set_page_config(
    page_title="Solar Energy Yield Predictor - Lebanon",
    page_icon="☀️",
    layout="wide",
)

st.title("☀️ Solar Energy Yield & Loss Predictor")
st.markdown(
    "Estimate theoretical clear-sky PV power output and actual yield based on weather forecast features."
)

st.sidebar.header("📍 Location Selection")


# Initialize Nominatim Geocoder
@st.cache_resource
def get_geolocator():
    return Nominatim(user_agent="lebanon_solar_yield_app")


geolocator = get_geolocator()


# Helper function to parse Google Maps links
def parse_google_maps_url(url_string):
    match_at = re.search(r"@(-?\d+\.\d+),(-?\d+\.\d+)", url_string)
    if match_at:
        return float(match_at.group(1)), float(match_at.group(2))

    match_q = re.search(r"(?:q|ll)=(-?\d+\.\d+),(-?\d+\.\d+)", url_string)
    if match_q:
        return float(match_q.group(1)), float(match_q.group(2))

    return None


# Selection Mode
location_mode = st.sidebar.radio(
    "Choose Location Method",
    [
        "Search Any Town in Lebanon",
        "Major Regional Centers",
        "Google Maps Link / Manual",
    ],
)

default_lat, default_lng = 33.88863211, 35.49547925  # High-precision default

if location_mode == "Search Any Town in Lebanon":
    town_query = st.sidebar.text_input(
        "Enter Town or Village Name",
        placeholder="e.g. Kfarhata, Bcharre, Douma, Jezzine, Qaraoun",
    )
    if town_query.strip():
        with st.sidebar.spinner(f"Geocoding '{town_query}'..."):
            try:
                location = geolocator.geocode(
                    f"{town_query}, Lebanon", timeout=5
                )
                if location:
                    default_lat = location.latitude
                    default_lng = location.longitude
                    st.sidebar.success(f"📍 Found: {location.address}")
                else:
                    st.sidebar.error(
                        "Town not found. Please check spelling or use manual inputs."
                    )
            except Exception:
                st.sidebar.warning(
                    "Geocoding lookup timed out. Using default coordinates."
                )

elif location_mode == "Major Regional Centers":
    MAJOR_CITIES = {
        "Beirut (Capital)": (33.88863211, 35.49547925),
        "Tripoli (North)": (34.43666670, 35.83444440),
        "Sidon / Saida (South)": (33.56041670, 35.37527780),
        "Zahle (Bekaa)": (33.84388890, 35.90722220),
        "Jounieh (Mount Lebanon)": (33.96972220, 35.61555560),
        "Tyre / Sour (South)": (33.27361110, 35.19472220),
        "Byblos / Jbeil (Mount Lebanon)": (34.11944440, 35.64666670),
        "Baalbek (Baalbek-Hermel)": (34.00583330, 36.20861110),
        "Nabatieh (Nabatieh)": (33.37888890, 35.48305560),
        "Halba (Akkar)": (34.54277780, 36.07972220),
        "Bcharre (North)": (34.25138890, 36.01083330),
        "Jezzine (South)": (33.54250000, 35.58416670),
    }
    selected_city = st.sidebar.selectbox(
        "Select Regional City", options=list(MAJOR_CITIES.keys())
    )
    default_lat, default_lng = MAJOR_CITIES[selected_city]

elif location_mode == "Google Maps Link / Manual":
    maps_url = st.sidebar.text_input(
        "Paste Google Maps Link",
        placeholder="https://maps.google.com/?q=33.88863211,35.49547925",
    )
    if maps_url.strip():
        parsed_coords = parse_google_maps_url(maps_url)
        if parsed_coords:
            default_lat, default_lng = parsed_coords
            st.sidebar.success(
                f"Parsed Coords: {default_lat:.8f}, {default_lng:.8f}"
            )
        else:
            st.sidebar.warning("Could not parse coordinates from link.")

# Displays final coordinate fields with 8-decimal precision
col_lat, col_lng = st.sidebar.columns(2)
latitude = col_lat.number_input(
    "Latitude",
    value=float(default_lat),
    format="%.8f",
    step=0.00001,
)
longitude = col_lng.number_input(
    "Longitude",
    value=float(default_lng),
    format="%.8f",
    step=0.00001,
)

st.sidebar.subheader("⚙️ System Specifications")
number_of_panels = st.sidebar.number_input(
    "Number of Panels", value=100, step=1
)
panel_wattage = st.sidebar.number_input(
    "Panel Wattage (W)", value=250, step=10
)
tilt = st.sidebar.number_input("Tilt Angle (°)", value=31, min_value=0, max_value=90)

# Azimuth Selection Dropdown
AZIMUTH_PRESETS = {
    "South (180°) - Ideal for Lebanon": 180,
    "South-East (135°)": 135,
    "South-West (225°)": 225,
    "East (90°)": 90,
    "West (270°)": 270,
    "North (0°)": 0,
    "Custom Angle": -1,
}

azimuth_selection = st.sidebar.selectbox(
    "Panel Azimuth Direction", options=list(AZIMUTH_PRESETS.keys())
)

if AZIMUTH_PRESETS[azimuth_selection] != -1:
    azimuth = AZIMUTH_PRESETS[azimuth_selection]
else:
    azimuth = st.sidebar.number_input(
        "Custom Azimuth Angle (°)", value=180, min_value=0, max_value=360
    )

st.sidebar.subheader("📅 Prediction Date")
selected_date = st.sidebar.date_input("Select Date", datetime.date.today())

# Model Path Verification
model_path = os.path.join("Model", "random_forest_k_pv.pkl")

# Run Pipeline
if st.sidebar.button("🚀 Predict Solar Yield", type="primary"):
    if not os.path.exists(model_path):
        st.error(
            f"Model file not found at `{model_path}`. Please verify your folder structure."
        )
    else:
        user_inputs = {
            "latitude": float(latitude),
            "longitude": float(longitude),
            "number_of_panels": int(number_of_panels),
            "panel_wattage": int(panel_wattage),
            "tilt": int(tilt),
            "azimuth": int(azimuth),
            "start_date": selected_date.strftime("%Y-%m-%d"),
            "end_date": selected_date.strftime("%Y-%m-%d"),
        }

        with st.spinner("Fetching weather data and running ML pipeline..."):
            try:
                results_df = predict_solar_yield_pipeline(
                    user_inputs, model_path=model_path
                )

                st.success(
                    f"Yield prediction calculated for location (**{latitude:.8f}, {longitude:.8f}**) on **{selected_date}**!"
                )

                # -------------------------------------------------------------
                # 1. Bar Chart (Histogram): 24-Hour Range (00:00 to 23:00)
                # -------------------------------------------------------------
                st.subheader(
                    "📊 Expected Power Output by Hour (00:00 - 23:00)"
                )

                results_df["Hour_Num"] = pd.to_datetime(
                    results_df["Timestamp"]
                ).dt.hour

                full_24h_df = pd.DataFrame({"Hour_Num": list(range(24))})
                full_24h_df["Hour_Label"] = full_24h_df["Hour_Num"].apply(
                    lambda h: f"{h:02d}:00"
                )

                bar_data_merged = pd.merge(
                    full_24h_df,
                    results_df[["Hour_Num", "Predicted_Actual_Power_kW"]],
                    on="Hour_Num",
                    how="left",
                ).fillna(0.0)

                bar_data = bar_data_merged.set_index("Hour_Label")[
                    ["Predicted_Actual_Power_kW"]
                ]
                st.bar_chart(bar_data)

                st.divider()

                # -------------------------------------------------------------
                # 2. Results Data Table & CSV Download Button
                # -------------------------------------------------------------
                st.subheader("📋 Detailed Hourly Prediction Sheet")
                st.dataframe(results_df, use_container_width=True)

                csv_data = results_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Predictions CSV",
                    data=csv_data,
                    file_name=f"solar_yield_{selected_date}.csv",
                    mime="text/csv",
                )

                st.divider()

                # -------------------------------------------------------------
                # 3. Summary Metrics Banner (Theoretical, Expected, Loss)
                # -------------------------------------------------------------
                st.subheader("⚡ Summary Energy Yield Metrics")
                total_theoretical_kwh = results_df["P_Theoretical_kW"].sum()
                total_predicted_kwh = results_df[
                    "Predicted_Actual_Power_kW"
                ].sum()
                overall_loss_percent = (
                    (
                        (total_theoretical_kwh - total_predicted_kwh)
                        / total_theoretical_kwh
                    )
                    * 100
                    if total_theoretical_kwh > 0
                    else 0.0
                )

                col1, col2, col3 = st.columns(3)
                col1.metric(
                    "Total Theoretical Energy",
                    f"{total_theoretical_kwh:,.2f} kWh",
                )
                col2.metric(
                    "Total Expected Power",
                    f"{total_predicted_kwh:,.2f} kWh",
                )
                col3.metric(
                    "Estimated Yield Loss", f"{overall_loss_percent:.2f} %"
                )

                st.divider()

                # -------------------------------------------------------------
                # 4. Line Chart: Theoretical vs. Expected Comparison
                # -------------------------------------------------------------
                st.subheader(
                    "📈 Theoretical vs. Expected Power Comparison (kW)"
                )
                line_data = results_df[
                    [
                        "Timestamp",
                        "P_Theoretical_kW",
                        "Predicted_Actual_Power_kW",
                    ]
                ].set_index("Timestamp")
                st.line_chart(line_data)

            except Exception as e:
                st.error(f"Execution Error: {e}")
