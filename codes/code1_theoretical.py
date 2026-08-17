import numpy as np
import pandas as pd
import pvlib
from pvlib.location import Location

def calculate_theoretical_power(user_inputs, times):
    """
    Code 1: Calculates clear-sky radiation and theoretical power output.
    Inputs: 
        user_inputs: dict (latitude, longitude, tilt, azimuth, number_of_panels, panel_wattage, timezone)
        times: pd.DatetimeIndex
    Outputs:
        dict containing p_theoretical, ghi_clear, and dni_clear pd.Series
    """
    location = Location(
        latitude=user_inputs["latitude"],
        longitude=user_inputs["longitude"],
        tz=user_inputs["timezone"]
    )

    # 1. Clear-sky model (Ineichen)
    clear_sky = location.get_clearsky(times, model="ineichen")

    # 2. Solar position & extraterrestrial radiation
    solar_position = location.get_solarposition(times)
    dni_extra = pvlib.irradiance.get_extra_radiation(times)

    # 3. Perez Transposition for Plane of Array (POA) Irradiance
    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=user_inputs["tilt"],
        surface_azimuth=user_inputs["azimuth"],
        solar_zenith=solar_position["apparent_zenith"],
        solar_azimuth=solar_position["azimuth"],
        dni=clear_sky["dni"],
        ghi=clear_sky["ghi"],
        dhi=clear_sky["dhi"],
        dni_extra=dni_extra,
        model="perez"
    )

    # 4. Capacity & Inverter Efficiency (clipping at max capacity)
    total_capacity = user_inputs["number_of_panels"] * user_inputs["panel_wattage"]  # Watts
    dc_power = (total_capacity * poa["poa_global"] / 1000.0).clip(lower=0)

    inverter_efficiency = 0.98
    p_theoretical = dc_power * inverter_efficiency
    p_theoretical = np.minimum(p_theoretical, total_capacity)

    return {
        "p_theoretical": p_theoretical,
        "ghi_clear": clear_sky["ghi"],
        "dni_clear": clear_sky["dni"]
    }