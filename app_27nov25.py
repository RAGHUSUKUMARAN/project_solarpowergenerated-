<<<<<<< HEAD
# To run this app:
# 1) cd D:\DATA-SCIENCE\project
# 2) streamlit run app.py
# Make sure these files are in the SAME folder:
#   - catboost_model.cbm
#   - encoder.joblib
#   - scaler.joblib
#   - power_transformer.joblib

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from catboost import CatBoostRegressor

# ---------------------------------------------------------
# Load model + preprocessing artifacts (cached)
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    # Load trained CatBoost model
    model = CatBoostRegressor()
    model.load_model("catboost_model.cbm")

    # Load preprocessing objects
    encoder = joblib.load("encoder.joblib")
    scaler = joblib.load("scaler.joblib")
    pt = joblib.load("power_transformer.joblib")

    return model, encoder, scaler, pt


model, oe, scaler, pt = load_artifacts()

# Final feature order used during training
FEATURES = [
    "distance-to-solar-noon",
    "temperature",
    "wind-speed",
    "visibility",
    "humidity",
    "average-wind-speed-(period)",
    "average-pressure-(period)",
    "wind-dir-sin",
    "wind-dir-cos",
    "sky-cover_1",
    "sky-cover_2",
    "sky-cover_3",
    "sky-cover_4",
]

# ---------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------
st.title("Solar Power Prediction (CatBoost)")
st.write("Predict solar power generation using your trained CatBoost model.")

st.sidebar.header("Input Features (Raw)")

# These are the RAW features from the dataset (before preprocessing)
distance = st.sidebar.number_input(
    "distance-to-solar-noon", value=0.0, format="%.5f"
)
temperature = st.sidebar.number_input(
    "temperature", value=0.0, format="%.2f"
)
wind_direction = st.sidebar.number_input(
    "wind-direction (same coding as dataset, e.g. 36 = 360°)", value=0.0, format="%.2f"
)
wind_speed = st.sidebar.number_input(
    "wind-speed", value=0.0, format="%.2f"
)
# ✅ original sky-cover values 0–4 via selectbox
sky_cover = st.sidebar.selectbox(
    "sky-cover", options=[0, 1, 2, 3, 4], index=0
)
visibility = st.sidebar.number_input(
    "visibility", value=0.0, format="%.2f"
)
humidity = st.sidebar.number_input(
    "humidity", value=0.0, format="%.2f"
)
avg_wind_period = st.sidebar.number_input(
    "average-wind-speed-(period)", value=0.0, format="%.2f"
)
avg_pressure_period = st.sidebar.number_input(
    "average-pressure-(period)", value=0.0, format="%.2f"
)

# Create raw input DataFrame (matches original columns before preprocessing)
raw_df = pd.DataFrame(
    {
        "distance-to-solar-noon": [distance],
        "temperature": [temperature],
        "wind-direction": [wind_direction],
        "wind-speed": [wind_speed],
        "sky-cover": [sky_cover],
        "visibility": [visibility],
        "humidity": [humidity],
        "average-wind-speed-(period)": [avg_wind_period],
        "average-pressure-(period)": [avg_pressure_period],
    }
)

st.subheader("Raw Input Data")
st.dataframe(raw_df)

# ---- Raw input summary cards ----
st.markdown("### Raw Input Summary")

c1, c2, c3 = st.columns(3)

with c1:
    st.metric("Distance to solar noon", f"{distance:.2f}")
    st.metric("Temperature (°C)", f"{temperature:.2f}")
    st.metric("Wind speed (m/s)", f"{wind_speed:.2f}")

with c2:
    st.metric("Wind direction (code)", f"{wind_direction:.2f}")
    st.metric("Sky cover (0–4)", str(sky_cover))
    st.metric("Visibility", f"{visibility:.2f}")

with c3:
    st.metric("Humidity (%)", f"{humidity:.2f}")
    st.metric("Avg wind speed (period)", f"{avg_wind_period:.2f}")
    st.metric("Avg pressure (period)", f"{avg_pressure_period:.2f}")

# ---- Bar chart of raw inputs ----
st.markdown("### Raw Input Profile")

plot_df = raw_df.T.reset_index()
plot_df.columns = ["Feature", "Value"]

# Keep only numeric for chart (sky-cover is numeric too but ok)
plot_df = plot_df.astype({"Value": "float"})

st.bar_chart(
    data=plot_df.set_index("Feature")
)


# ---------------------------------------------------------
# Apply SAME preprocessing as in the notebook
# ---------------------------------------------------------
def preprocess_input(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    # 1) Wind direction transformation
    # In notebook: data1["wind-direction"] = data1["wind-direction"] * 10
    df["wind-direction"] = df["wind-direction"] * 10
    df["wind-dir-rad"] = np.deg2rad(df["wind-direction"])
    df["wind-dir-sin"] = np.sin(df["wind-dir-rad"])
    df["wind-dir-cos"] = np.cos(df["wind-dir-rad"])
    df.drop(columns=["wind-direction", "wind-dir-rad"], inplace=True)

    # 2) One-hot encode sky-cover using fitted encoder (oe)
    encoded = oe.transform(df[["sky-cover"]])
    encoded_cols = oe.get_feature_names_out(["sky-cover"])
    encoded_df = pd.DataFrame(encoded, columns=encoded_cols, index=df.index)
    df = pd.concat([df.drop("sky-cover", axis=1), encoded_df], axis=1)

    # 3) PowerTransform visibility & humidity (pt)
    df[["visibility", "humidity"]] = pt.transform(df[["visibility", "humidity"]])

    # 4) StandardScaler for selected numeric cols (scaler)
    cols_to_scale = [
        "distance-to-solar-noon",
        "temperature",
        "wind-speed",
        "average-wind-speed-(period)",
        "average-pressure-(period)",
    ]
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])

    # 5) Reorder columns to match training feature order
    df = df[FEATURES]

    return df


if st.button("Predict"):
    try:
        processed_df = preprocess_input(raw_df)

        st.subheader("Processed Features (Model Input)")
        st.dataframe(processed_df)

        prediction = model.predict(processed_df)[0]
        st.subheader("Prediction Output")
        st.success(f"Estimated Solar Power: {prediction:.2f} W")

    except Exception as e:
        st.error(f"Something went wrong during preprocessing/prediction: {e}")
=======
# To run this app:
# 1) cd D:\DATA-SCIENCE\project
# 2) streamlit run app.py
# Make sure these files are in the SAME folder:
#   - catboost_model.cbm
#   - encoder.joblib
#   - scaler.joblib
#   - power_transformer.joblib

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from catboost import CatBoostRegressor

# ---------------------------------------------------------
# Load model + preprocessing artifacts (cached)
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    # Load trained CatBoost model
    model = CatBoostRegressor()
    model.load_model("catboost_model.cbm")

    # Load preprocessing objects
    encoder = joblib.load("encoder.joblib")
    scaler = joblib.load("scaler.joblib")
    pt = joblib.load("power_transformer.joblib")

    return model, encoder, scaler, pt


model, oe, scaler, pt = load_artifacts()

# Final feature order used during training
FEATURES = [
    "distance-to-solar-noon",
    "temperature",
    "wind-speed",
    "visibility",
    "humidity",
    "average-wind-speed-(period)",
    "average-pressure-(period)",
    "wind-dir-sin",
    "wind-dir-cos",
    "sky-cover_1",
    "sky-cover_2",
    "sky-cover_3",
    "sky-cover_4",
]

# ---------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------
st.title("Solar Power Prediction (CatBoost)")
st.write("Predict solar power generation using your trained CatBoost model.")

st.sidebar.header("Input Features (Raw)")

# These are the RAW features from the dataset (before preprocessing)
distance = st.sidebar.number_input(
    "distance-to-solar-noon", value=0.0, format="%.5f"
)
temperature = st.sidebar.number_input(
    "temperature", value=0.0, format="%.2f"
)
wind_direction = st.sidebar.number_input(
    "wind-direction (same coding as dataset, e.g. 36 = 360°)", value=0.0, format="%.2f"
)
wind_speed = st.sidebar.number_input(
    "wind-speed", value=0.0, format="%.2f"
)
# ✅ original sky-cover values 0–4 via selectbox
sky_cover = st.sidebar.selectbox(
    "sky-cover", options=[0, 1, 2, 3, 4], index=0
)
visibility = st.sidebar.number_input(
    "visibility", value=0.0, format="%.2f"
)
humidity = st.sidebar.number_input(
    "humidity", value=0.0, format="%.2f"
)
avg_wind_period = st.sidebar.number_input(
    "average-wind-speed-(period)", value=0.0, format="%.2f"
)
avg_pressure_period = st.sidebar.number_input(
    "average-pressure-(period)", value=0.0, format="%.2f"
)

# Create raw input DataFrame (matches original columns before preprocessing)
raw_df = pd.DataFrame(
    {
        "distance-to-solar-noon": [distance],
        "temperature": [temperature],
        "wind-direction": [wind_direction],
        "wind-speed": [wind_speed],
        "sky-cover": [sky_cover],
        "visibility": [visibility],
        "humidity": [humidity],
        "average-wind-speed-(period)": [avg_wind_period],
        "average-pressure-(period)": [avg_pressure_period],
    }
)

st.subheader("Raw Input Data")
st.dataframe(raw_df)

# ---- Raw input summary cards ----
st.markdown("### Raw Input Summary")

c1, c2, c3 = st.columns(3)

with c1:
    st.metric("Distance to solar noon", f"{distance:.2f}")
    st.metric("Temperature (°C)", f"{temperature:.2f}")
    st.metric("Wind speed (m/s)", f"{wind_speed:.2f}")

with c2:
    st.metric("Wind direction (code)", f"{wind_direction:.2f}")
    st.metric("Sky cover (0–4)", str(sky_cover))
    st.metric("Visibility", f"{visibility:.2f}")

with c3:
    st.metric("Humidity (%)", f"{humidity:.2f}")
    st.metric("Avg wind speed (period)", f"{avg_wind_period:.2f}")
    st.metric("Avg pressure (period)", f"{avg_pressure_period:.2f}")

# ---- Bar chart of raw inputs ----
st.markdown("### Raw Input Profile")

plot_df = raw_df.T.reset_index()
plot_df.columns = ["Feature", "Value"]

# Keep only numeric for chart (sky-cover is numeric too but ok)
plot_df = plot_df.astype({"Value": "float"})

st.bar_chart(
    data=plot_df.set_index("Feature")
)


# ---------------------------------------------------------
# Apply SAME preprocessing as in the notebook
# ---------------------------------------------------------
def preprocess_input(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    # 1) Wind direction transformation
    # In notebook: data1["wind-direction"] = data1["wind-direction"] * 10
    df["wind-direction"] = df["wind-direction"] * 10
    df["wind-dir-rad"] = np.deg2rad(df["wind-direction"])
    df["wind-dir-sin"] = np.sin(df["wind-dir-rad"])
    df["wind-dir-cos"] = np.cos(df["wind-dir-rad"])
    df.drop(columns=["wind-direction", "wind-dir-rad"], inplace=True)

    # 2) One-hot encode sky-cover using fitted encoder (oe)
    encoded = oe.transform(df[["sky-cover"]])
    encoded_cols = oe.get_feature_names_out(["sky-cover"])
    encoded_df = pd.DataFrame(encoded, columns=encoded_cols, index=df.index)
    df = pd.concat([df.drop("sky-cover", axis=1), encoded_df], axis=1)

    # 3) PowerTransform visibility & humidity (pt)
    df[["visibility", "humidity"]] = pt.transform(df[["visibility", "humidity"]])

    # 4) StandardScaler for selected numeric cols (scaler)
    cols_to_scale = [
        "distance-to-solar-noon",
        "temperature",
        "wind-speed",
        "average-wind-speed-(period)",
        "average-pressure-(period)",
    ]
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])

    # 5) Reorder columns to match training feature order
    df = df[FEATURES]

    return df


if st.button("Predict"):
    try:
        processed_df = preprocess_input(raw_df)

        st.subheader("Processed Features (Model Input)")
        st.dataframe(processed_df)

        prediction = model.predict(processed_df)[0]
        st.subheader("Prediction Output")
        st.success(f"Estimated Solar Power: {prediction:.2f} W")

    except Exception as e:
        st.error(f"Something went wrong during preprocessing/prediction: {e}")
>>>>>>> 2882346c349d0347d69fa756a59b4da33dddd557
