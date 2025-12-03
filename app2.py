<<<<<<< HEAD
# To run this app:
# 1) pip install -r requirements.txt
# 2) streamlit run app.py
import streamlit as st
import pandas as pd
from catboost import CatBoostRegressor
import shap
import requests


@st.cache_resource
def load_model():
    model_path = r"D:\DATA-SCIENCE\project\catboost_model.cbm"
    model = CatBoostRegressor()
    model.load_model(model_path)
    return model

model = load_model()

# Initialize prediction history in session state
if "history" not in st.session_state:
    st.session_state.history = []


# Reasonable ranges & defaults for sliders
FEATURE_RANGES = {
    'distance-to-solar-noon': (0.0, 60.0, 5.0, 0.01),      # minutes from solar noon
    'temperature': (0.0, 50.0, 30.0, 0.1),                 # °C
    'wind-speed': (0.0, 25.0, 3.0, 0.1),                   # m/s
    'visibility': (0.0, 10.0, 8.0, 0.1),                   # km
    'humidity': (0.0, 100.0, 40.0, 1.0),                   # %
    'average-wind-speed-(period)': (0.0, 25.0, 3.0, 0.1),
    'average-pressure-(period)': (980.0, 1040.0, 1013.0, 0.1),  # hPa
    'wind-dir-sin': (-1.0, 1.0, 0.0, 0.01),
    'wind-dir-cos': (-1.0, 1.0, 1.0, 0.01),
    'sky-cover_1': (0.0, 1.0, 0.0, 0.01),
    'sky-cover_2': (0.0, 1.0, 0.0, 0.01),
    'sky-cover_3': (0.0, 1.0, 0.0, 0.01),
    'sky-cover_4': (0.0, 1.0, 0.0, 0.01),
}

# Short explanations for each input (shown as tooltips)
FEATURE_HELP = {
    'distance-to-solar-noon': "Time difference from solar noon (minutes). 0 = sun at highest point.",
    'temperature': "Ambient air temperature in °C.",
    'wind-speed': "Instantaneous wind speed (m/s).",
    'visibility': "Visibility distance in km (higher = clearer sky).",
    'humidity': "Relative humidity in percent.",
    'average-wind-speed-(period)': "Average wind speed over recent period (m/s).",
    'average-pressure-(period)': "Average air pressure over recent period (hPa).",
    'wind-dir-sin': "Sine of wind direction angle (encoded direction).",
    'wind-dir-cos': "Cosine of wind direction angle (encoded direction).",
    'sky-cover_1': "Low-level cloud fraction (0–1).",
    'sky-cover_2': "Mid-level cloud fraction (0–1).",
    'sky-cover_3': "High-level cloud fraction (0–1).",
    'sky-cover_4': "Total/other cloud cover (0–1).",
}


st.title("Solar Power Prediction (CatBoost)")
st.write("Predicting solar energy generation using the trained CatBoost model.")

st.sidebar.header("Input Features")

input_data = {}

for feature in FEATURES:
    min_val, max_val, default, step = FEATURE_RANGES[feature]
    input_data[feature] = st.sidebar.slider(
        label=feature,
        min_value=float(min_val),
        max_value=float(max_val),
        value=float(default),
        step=float(step),
        help=FEATURE_HELP.get(feature, "")
    )


input_df = pd.DataFrame([input_data])

st.subheader("Input Data")
st.dataframe(input_df)

if st.button("Predict"):
    prediction = model.predict(input_df)[0]

    # Save to history
    record = input_data.copy()
    record["prediction"] = float(prediction)
    st.session_state.history.append(record)

    st.subheader("Prediction Output")
    st.success(f"Estimated Solar Power: {prediction:.2f} W")

st.subheader("Prediction History")
if st.session_state.history:
    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df)
else:
    st.write("No predictions yet.")


st.subheader("What-if Analysis: Effect of Solar Noon Distance")

# Use current inputs as baseline, vary only distance-to-solar-noon
distances = np.linspace(0, 60, 30)  # 0 to 60 minutes
what_if_rows = []
for d in distances:
    row = input_data.copy()
    row['distance-to-solar-noon'] = float(d)
    what_if_rows.append(row)

what_if_df = pd.DataFrame(what_if_rows)
what_if_preds = model.predict(what_if_df)

chart_df = pd.DataFrame({
    "distance_to_solar_noon": distances,
    "predicted_power_W": what_if_preds
})

st.line_chart(chart_df.set_index("distance_to_solar_noon"))


st.header("Batch Prediction from CSV")

uploaded_file = st.file_uploader(
    "Upload a CSV with the same feature columns as the training data",
    type=["csv"]
)

if uploaded_file is not None:
    batch_df = pd.read_csv(uploaded_file)

    st.write("Preview of uploaded data:")
    st.dataframe(batch_df.head())

    # Ensure only known columns are used
    batch_df = batch_df[FEATURES]

    batch_preds = model.predict(batch_df)
    batch_df["predicted_power_W"] = batch_preds

    st.subheader("Batch Predictions")
    st.dataframe(batch_df.head())

    csv_out = batch_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download predictions as CSV",
        data=csv_out,
        file_name="solar_predictions.csv",
        mime="text/csv"
    )


st.header("Global Feature Importance")

import numpy as np

importances = model.get_feature_importance()
imp_df = pd.DataFrame({
    "feature": FEATURES,
    "importance": importances
}).sort_values("importance", ascending=False)

st.bar_chart(imp_df.set_index("feature"))

def get_weather_features(lat, lon):
    # Example placeholder – you’d adapt to your real API
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=temperature_2m,relativehumidity_2m"
    resp = requests.get(url)
    data = resp.json()
    # parse into your feature values and return a dict shaped like input_data
    ...

if st.sidebar.button("Use Live Weather (demo placeholder)"):
    # call get_weather_features and overwrite input_data values
    ...
=======
# To run this app:
# 1) pip install -r requirements.txt
# 2) streamlit run app.py
import streamlit as st
import pandas as pd
from catboost import CatBoostRegressor
import shap
import requests


@st.cache_resource
def load_model():
    model_path = r"D:\DATA-SCIENCE\project\catboost_model.cbm"
    model = CatBoostRegressor()
    model.load_model(model_path)
    return model

model = load_model()

# Initialize prediction history in session state
if "history" not in st.session_state:
    st.session_state.history = []


# Reasonable ranges & defaults for sliders
FEATURE_RANGES = {
    'distance-to-solar-noon': (0.0, 60.0, 5.0, 0.01),      # minutes from solar noon
    'temperature': (0.0, 50.0, 30.0, 0.1),                 # °C
    'wind-speed': (0.0, 25.0, 3.0, 0.1),                   # m/s
    'visibility': (0.0, 10.0, 8.0, 0.1),                   # km
    'humidity': (0.0, 100.0, 40.0, 1.0),                   # %
    'average-wind-speed-(period)': (0.0, 25.0, 3.0, 0.1),
    'average-pressure-(period)': (980.0, 1040.0, 1013.0, 0.1),  # hPa
    'wind-dir-sin': (-1.0, 1.0, 0.0, 0.01),
    'wind-dir-cos': (-1.0, 1.0, 1.0, 0.01),
    'sky-cover_1': (0.0, 1.0, 0.0, 0.01),
    'sky-cover_2': (0.0, 1.0, 0.0, 0.01),
    'sky-cover_3': (0.0, 1.0, 0.0, 0.01),
    'sky-cover_4': (0.0, 1.0, 0.0, 0.01),
}

# Short explanations for each input (shown as tooltips)
FEATURE_HELP = {
    'distance-to-solar-noon': "Time difference from solar noon (minutes). 0 = sun at highest point.",
    'temperature': "Ambient air temperature in °C.",
    'wind-speed': "Instantaneous wind speed (m/s).",
    'visibility': "Visibility distance in km (higher = clearer sky).",
    'humidity': "Relative humidity in percent.",
    'average-wind-speed-(period)': "Average wind speed over recent period (m/s).",
    'average-pressure-(period)': "Average air pressure over recent period (hPa).",
    'wind-dir-sin': "Sine of wind direction angle (encoded direction).",
    'wind-dir-cos': "Cosine of wind direction angle (encoded direction).",
    'sky-cover_1': "Low-level cloud fraction (0–1).",
    'sky-cover_2': "Mid-level cloud fraction (0–1).",
    'sky-cover_3': "High-level cloud fraction (0–1).",
    'sky-cover_4': "Total/other cloud cover (0–1).",
}


st.title("Solar Power Prediction (CatBoost)")
st.write("Predicting solar energy generation using the trained CatBoost model.")

st.sidebar.header("Input Features")

input_data = {}

for feature in FEATURES:
    min_val, max_val, default, step = FEATURE_RANGES[feature]
    input_data[feature] = st.sidebar.slider(
        label=feature,
        min_value=float(min_val),
        max_value=float(max_val),
        value=float(default),
        step=float(step),
        help=FEATURE_HELP.get(feature, "")
    )


input_df = pd.DataFrame([input_data])

st.subheader("Input Data")
st.dataframe(input_df)

if st.button("Predict"):
    prediction = model.predict(input_df)[0]

    # Save to history
    record = input_data.copy()
    record["prediction"] = float(prediction)
    st.session_state.history.append(record)

    st.subheader("Prediction Output")
    st.success(f"Estimated Solar Power: {prediction:.2f} W")

st.subheader("Prediction History")
if st.session_state.history:
    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df)
else:
    st.write("No predictions yet.")


st.subheader("What-if Analysis: Effect of Solar Noon Distance")

# Use current inputs as baseline, vary only distance-to-solar-noon
distances = np.linspace(0, 60, 30)  # 0 to 60 minutes
what_if_rows = []
for d in distances:
    row = input_data.copy()
    row['distance-to-solar-noon'] = float(d)
    what_if_rows.append(row)

what_if_df = pd.DataFrame(what_if_rows)
what_if_preds = model.predict(what_if_df)

chart_df = pd.DataFrame({
    "distance_to_solar_noon": distances,
    "predicted_power_W": what_if_preds
})

st.line_chart(chart_df.set_index("distance_to_solar_noon"))


st.header("Batch Prediction from CSV")

uploaded_file = st.file_uploader(
    "Upload a CSV with the same feature columns as the training data",
    type=["csv"]
)

if uploaded_file is not None:
    batch_df = pd.read_csv(uploaded_file)

    st.write("Preview of uploaded data:")
    st.dataframe(batch_df.head())

    # Ensure only known columns are used
    batch_df = batch_df[FEATURES]

    batch_preds = model.predict(batch_df)
    batch_df["predicted_power_W"] = batch_preds

    st.subheader("Batch Predictions")
    st.dataframe(batch_df.head())

    csv_out = batch_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download predictions as CSV",
        data=csv_out,
        file_name="solar_predictions.csv",
        mime="text/csv"
    )


st.header("Global Feature Importance")

import numpy as np

importances = model.get_feature_importance()
imp_df = pd.DataFrame({
    "feature": FEATURES,
    "importance": importances
}).sort_values("importance", ascending=False)

st.bar_chart(imp_df.set_index("feature"))

def get_weather_features(lat, lon):
    # Example placeholder – you’d adapt to your real API
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=temperature_2m,relativehumidity_2m"
    resp = requests.get(url)
    data = resp.json()
    # parse into your feature values and return a dict shaped like input_data
    ...

if st.sidebar.button("Use Live Weather (demo placeholder)"):
    # call get_weather_features and overwrite input_data values
    ...
>>>>>>> 2882346c349d0347d69fa756a59b4da33dddd557
