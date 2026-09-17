import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os


# PAGE CONFIGURATION & STYLING

st.set_page_config(
    page_title="Mars Pressure Predictor",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# css

st.markdown("""
<style>
    /* Clean dark workspace */
    .stApp {
        background-color: #0e1117;
        color: #e6edf3;
    }
    
    /* Top Banner Styling */
    .header-box {
        text-align: center;
        padding: 24px 10px 10px 10px;
        margin-bottom: 20px;
    }
    .header-title {
        color: #f97316;
        font-size: 2.3rem;
        font-weight: 700;
        margin-bottom: 6px;
        letter-spacing: -0.5px;
    }
    .header-subtitle {
        color: #9ca3af;
        font-size: 1.05rem;
        font-weight: 400;
        margin-bottom: 0px;
    }

    /* Result Card Styling */
    .result-box {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 16px;
        padding: 30px;
        text-align: center;
        margin: 15px 0 25px 0;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4);
    }
    .result-value {
        font-size: 3.6rem;
        font-weight: 800;
        margin: 10px 0;
        letter-spacing: -1px;
    }
    .result-subtext {
        font-size: 1.15rem;
        color: #cbd5e1;
    }

    /* Primary Action Button */
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #ea580c, #c2410c);
        color: #ffffff;
        font-weight: 600;
        font-size: 1.1rem;
        border: none;
        border-radius: 10px;
        padding: 12px 24px;
        width: 100%;
        transition: all 0.2s ease-in-out;
    }
    div.stButton > button:first-child:hover {
        background: linear-gradient(90deg, #f97316, #ea580c);
        transform: translateY(-1px);
        box-shadow: 0 4px 14px rgba(234, 88, 12, 0.4);
    }
</style>
""", unsafe_allow_html=True)


# LOAD MODEL & PREPROCESSING

@st.cache_resource  
def load_model_and_preprocessing():
    model        = joblib.load('model_FINAL.pkl')
    imputer      = joblib.load('imputer.pkl')
    feature_cols = joblib.load('feature_cols.pkl')
    
    # Ensure cross-version compatibility for SimpleImputer
    if hasattr(imputer, '_fit_dtype') and not hasattr(imputer, '_fill_dtype'):
        imputer._fill_dtype = imputer._fit_dtype
    if hasattr(imputer, '_fill_dtype') and not hasattr(imputer, '_fit_dtype'):
        imputer._fit_dtype = imputer._fill_dtype
        
    return model, imputer, feature_cols

try:
    model, imputer, feature_cols = load_model_and_preprocessing()
    model_loaded = True
except Exception as e:
    model_loaded = False
    load_error = str(e)


# HELPER FUNCTIONS

def predict_pressure(input_values):
    input_df = pd.DataFrame([input_values], columns=feature_cols)
    if hasattr(imputer, '_fit_dtype') and not hasattr(imputer, '_fill_dtype'):
        imputer._fill_dtype = imputer._fit_dtype
    if hasattr(imputer, '_fill_dtype') and not hasattr(imputer, '_fit_dtype'):
        imputer._fit_dtype = imputer._fill_dtype
    input_imputed = imputer.transform(input_df)
    prediction = model.predict(input_imputed)[0]
    return float(prediction)

def get_pressure_context(pressure_pa):
    if pressure_pa < 720:
        return {
            'level': 'Very Low Pressure',
            'badge_color': '#ef4444',
            'status': 'Low Pressure Event',
            'explanation': 'Pressure is significantly below average. This typical occurrence corresponds to transient low-pressure vortices or dust devils passing across Jezero Crater.'
        }
    elif pressure_pa < 740:
        return {
            'level': 'Below Average Pressure',
            'badge_color': '#f97316',
            'status': 'Seasonal Low',
            'explanation': 'Pressure is lower than average, consistent with early mission sols or winter when CO₂ freezes at the polar caps, reducing atmospheric mass.'
        }
    elif pressure_pa < 758:
        return {
            'level': 'Normal / Typical Pressure',
            'badge_color': '#22c55e',
            'status': 'Optimal Conditions',
            'explanation': 'Pressure is within the baseline range for Jezero Crater. Standard Martian atmospheric conditions with stable weather.'
        }
    elif pressure_pa < 768:
        return {
            'level': 'Above Average Pressure',
            'badge_color': '#3b82f6',
            'status': 'Seasonal High',
            'explanation': 'Pressure is elevated above baseline. Typically occurs as polar CO₂ ice sublimates back into gas during warmer seasons, swelling atmospheric mass.'
        }
    else:
        return {
            'level': 'High Pressure',
            'badge_color': '#a855f7',
            'status': 'Peak Season',
            'explanation': 'Pressure is near the top of observed mission ranges, corresponding to maximum solar heating and CO₂ atmosphere replenishment.'
        }

def create_gauge_chart(pressure, min_val=713, max_val=772):
    fig, ax = plt.subplots(figsize=(8, 1.8))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    
    # Range bar
    ax.barh(0, max_val - min_val, left=min_val, height=0.35, color='#1e293b', alpha=0.9)
    
    # Zones
    zones = [
        (min_val, 720,  '#ef4444', 'Very Low'),
        (720,     740,  '#f97316', 'Low'),
        (740,     758,  '#22c55e', 'Normal'),
        (758,     768,  '#3b82f6', 'High'),
        (768,     max_val, '#a855f7', 'Very High'),
    ]
    for start, end, color, label in zones:
        ax.barh(0, end - start, left=start, height=0.35, color=color, alpha=0.35)
    
    # Prediction marker
    ax.barh(0, 0.6, left=pressure - 0.3, height=0.55, color='#ffffff', alpha=1.0)
    
    ax.text(min_val, -0.4, f'{min_val} Pa', color='#9ca3af', fontsize=9, ha='left')
    ax.text(max_val, -0.4, f'{max_val} Pa', color='#9ca3af', fontsize=9, ha='right')
    ax.text(pressure, 0.42, f'{pressure:.1f} Pa', color='#ffffff', fontsize=11, ha='center', fontweight='bold')
    
    ax.set_xlim(min_val - 2, max_val + 2)
    ax.set_ylim(-0.6, 0.7)
    ax.axis('off')
    plt.tight_layout(pad=0.1)
    return fig


# APP HEADER

st.markdown("""
<div class='header-box'>
    <div class='header-title'>Mars Atmospheric Pressure Predictor</div>
    <div class='header-subtitle'>Virtual Sensor Recovery System for NASA's Perseverance Rover</div>
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.error(f"Unable to load predictive model: {load_error}. Please ensure model files are present.")
    st.stop()


# SIDEBAR - INPUT CONTROLS & PRESETS

st.sidebar.header("Control Panel")

# Scenario Preset Selector
scenario = st.sidebar.selectbox(
    "Quick Preset Scenario",
    options=[
        "Default Mission Baseline",
        "Cold Winter Night",
        "Warm Summer Afternoon",
        "Dust Vortex / Low Pressure Event",
        "Custom Parameters"
    ],
    help="Select a predefined Martian scenario to quickly load typical sensor values."
)

# Preset definitions
if scenario == "Cold Winter Night":
    default_sol = 30.0
    default_sclk = 668000000
    default_ls = -45.0
    default_zenith = 140.0
    default_temp = 195.0
    default_humidity = 3.5
    default_vmr = 15.0
    default_down_lw = 22.0
    default_up_lw = 70.0
elif scenario == "Warm Summer Afternoon":
    default_sol = 140.0
    default_sclk = 674000000
    default_ls = 90.0
    default_zenith = 25.0
    default_temp = 265.0
    default_humidity = 0.1
    default_vmr = 45.0
    default_down_lw = 45.0
    default_up_lw = 240.0
elif scenario == "Dust Vortex / Low Pressure Event":
    default_sol = 80.0
    default_sclk = 671000000
    default_ls = 20.0
    default_zenith = 85.0
    default_temp = 215.0
    default_humidity = 1.2
    default_vmr = 28.0
    default_down_lw = 32.0
    default_up_lw = 110.0
else: # Default Mission Baseline or Custom
    default_sol = 60.0
    default_sclk = 672226200
    default_ls = 3.2
    default_zenith = 82.0
    default_temp = 223.56
    default_humidity = 0.62
    default_vmr = 21.65
    default_down_lw = 28.90
    default_up_lw = 118.95

st.sidebar.subheader("Primary Sensors")

sol = st.sidebar.number_input(
    "Mission Sol (Martian Day)",
    min_value=1.0, max_value=200.0, value=default_sol, step=1.0,
    help="Days elapsed since Perseverance landing on Mars."
)

c_temp = default_temp - 273.15
air_temp_c = st.sidebar.slider(
    "Air / Sensor Temperature (°C)",
    min_value=-80.0, max_value=5.0, value=round(c_temp, 1), step=0.5,
    help="Ambient temperature recorded at the sensor site."
)
humidity_temp = air_temp_c + 273.15

humidity = st.sidebar.slider(
    "Relative Humidity (%)",
    min_value=0.0, max_value=20.0, value=default_humidity, step=0.1,
    help="Atmospheric humidity ratio measured near surface."
)

solar_longitude = st.sidebar.slider(
    "Martian Season (Solar Longitude °)",
    min_value=-180.0, max_value=180.0, value=default_ls, step=1.0,
    help="Orbital position around the Sun driving seasonal pressure changes."
)

# Advanced Controls Expander
with st.sidebar.expander("Advanced Sensor Controls", expanded=False):
    st.caption("Fine-tune individual rover telemetry & sensor metrics:")
    
    sclk = st.number_input("Spacecraft Clock (SCLK)", min_value=667042464, max_value=675857058, value=default_sclk, step=1000)
    solar_zenithal = st.slider("Solar Zenith Angle (°)", min_value=0.0, max_value=156.0, value=default_zenith, step=0.5)
    transducer = st.selectbox("Sensor Transducer ID", options=[1, 2], index=0)
    volume_mixing_ratio = st.slider("Water Vapor (ppm)", min_value=5.0, max_value=75.0, value=default_vmr, step=0.5)
    
    st.markdown("**Rover Location & Orientation**")
    col_x, col_y = st.columns(2)
    with col_x:
        rover_x = st.number_input("Pos X (m)", value=-2.19, step=0.1)
        rover_z = st.number_input("Pos Z (m)", value=1.15, step=0.01)
    with col_y:
        rover_y = st.number_input("Pos Y (m)", value=-58.95, step=0.1)
        rover_velocity = st.number_input("Speed (m/s)", value=0.0, step=0.01)
    
    rover_pitch = st.slider("Pitch (°)", min_value=-3.6, max_value=3.6, value=0.14, step=0.01)
    rover_yaw = st.slider("Yaw (°)", min_value=-180.0, max_value=180.0, value=66.34, step=0.1)
    rover_roll = st.slider("Roll (°)", min_value=-3.6, max_value=2.5, value=0.40, step=0.01)
    
    st.markdown("**Thermal Irradiance & Quality**")
    downward_lw = st.slider("Downward Radiation (W/m²)", min_value=20.0, max_value=80.0, value=default_down_lw, step=0.5)
    downward_lw_unc = st.number_input("Downward Rad Uncertainty", value=3.31, step=0.01)
    upward_lw = st.slider("Upward Radiation (W/m²)", min_value=60.0, max_value=360.0, value=default_up_lw, step=0.5)
    upward_lw_unc = st.number_input("Upward Rad Uncertainty", value=1.03, step=0.01)
    humidity_temp_unc = st.number_input("Temp Uncertainty", value=0.17, step=0.01)
    
    sun_outside_fov = st.selectbox("Sun Clear of Sensor FOV", options=[1, 0], format_func=lambda x: "Yes (Valid)" if x == 1 else "No (Obstructed)")
    ground_not_in_shadow = st.selectbox("Ground Area Sunlit", options=[1, 0], format_func=lambda x: "Yes (Valid)" if x == 1 else "No (Shadowed)")
    rover_still = st.selectbox("Rover Stationary", options=[1, 0], format_func=lambda x: "Yes (Stationary)" if x == 1 else "No (Moving)")


# MAIN APPLICATION INTERFACE

input_values = {
    'SCLK':                          sclk,
    'SOLAR_LONGITUDE_ANGLE':         solar_longitude,
    'SOLAR_ZENITHAL_ANGLE':          solar_zenithal,
    'ROVER_POSITION_X':              rover_x,
    'ROVER_POSITION_Y':              rover_y,
    'ROVER_POSITION_Z':              rover_z,
    'ROVER_VELOCITY':                rover_velocity,
    'ROVER_PITCH':                   rover_pitch,
    'ROVER_YAW':                     rover_yaw,
    'ROVER_ROLL':                    rover_roll,
    'sol':                           sol,
    'TRANSDUCER':                    transducer,
    'LOCAL_RELATIVE_HUMIDITY':       humidity,
    'HUMIDITY_LOCAL_TEMP':           humidity_temp,
    'HUMIDITY_LOCAL_TEMP_UNCERTAINTY': humidity_temp_unc,
    'VOLUME_MIXING_RATIO':           volume_mixing_ratio,
    'DOWNWARD_LW_IRRADIANCE':        downward_lw,
    'DOWNWARD_LW_IRRADIANCE_UNCERTAINTY': downward_lw_unc,
    'UPWARD_LW_IRRADIANCE':          upward_lw,
    'UPWARD_LW_UNCERTAINTY':         upward_lw_unc,
    'SUN_OUTSIDE_TIRS_FOV':          sun_outside_fov,
    'TIRS_GROUND_FOOTPRINT_NOT_IN_SHADOW': ground_not_in_shadow,
    'ROVER_STILL':                   rover_still,
}

col_action, col_space = st.columns([1, 1])
with col_action:
    predict_button = st.button("Predict Atmospheric Pressure")

if predict_button:
    with st.spinner("Calculating Virtual Pressure Prediction..."):
        try:
            predicted_pressure = predict_pressure(input_values)
            context = get_pressure_context(predicted_pressure)
            
            # Big Clean Display Banner
            st.markdown(f"""
            <div class='result-box' style='border: 2px solid {context["badge_color"]};'>
                <div class='result-subtext'>Predicted Martian Atmospheric Pressure</div>
                <div class='result-value' style='color: {context["badge_color"]};'>
                    {predicted_pressure:.2f} Pa
                </div>
                <div style='font-size: 1.1rem; color: #f8fafc;'>
                    Status: <strong style='color: {context["badge_color"]};'>{context["level"]}</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Key Summary Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    label="Pressure (Pascals)",
                    value=f"{predicted_pressure:.2f} Pa",
                    delta=f"{predicted_pressure - 749.65:.2f} Pa vs mission mean"
                )
            with col2:
                earth_pressure = 101325
                ratio = (predicted_pressure / earth_pressure) * 100
                st.metric(
                    label="Comparison to Earth Surface",
                    value=f"{ratio:.3f}% of Earth",
                    help="Mars has less than 1% of Earth's atmospheric pressure at sea level."
                )
            with col3:
                mbar_val = predicted_pressure / 100.0
                st.metric(
                    label="Pressure in Millibars (hPa)",
                    value=f"{mbar_val:.2f} mbar",
                    help="Standard meteorological pressure unit."
                )
            
            # Visual Gauge
            st.markdown("#### Pressure Gauge Range")
            gauge_fig = create_gauge_chart(predicted_pressure)
            st.pyplot(gauge_fig, use_container_width=True)
            plt.close()
            
            # Simple Context Explanation
            st.info(f"**{context['status']}:** {context['explanation']}")
            
        except Exception as e:
            st.error(f"Prediction could not be completed: {str(e)}")

else:
    # Initial Friendly Welcome State
    st.markdown("""
    <div style='text-align: center; padding: 45px; background: #161b22; border-radius: 12px; border: 1px solid #30363d;'>
        <h3 style='color: #f97316; margin-bottom: 10px;'>Ready to Predict</h3>
        <p style='color: #9ca3af; font-size: 1.05rem; max-width: 600px; margin: 0 auto 15px auto;'>
            Adjust the environmental controls in the left sidebar or select a <strong>Quick Preset Scenario</strong>, then click <strong>Predict Atmospheric Pressure</strong>.
        </p>
    </div>
    """, unsafe_allow_html=True)


# FOOTER

st.markdown("""
<div style='text-align: center; color: #6b7280; font-size: 0.85rem; margin-top: 30px;'>
    Data Source: NASA Mars 2020 Perseverance Rover - MEDA Instrument
</div>
""", unsafe_allow_html=True)