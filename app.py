# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from catboost import CatBoostRegressor

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="Solar Power Prediction",
    page_icon="☀️",
    layout="wide"
)

# ---------------------------------------------------------
# LOAD ARTIFACTS
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = CatBoostRegressor()
    model.load_model("catboost_model.cbm")

    encoder = joblib.load("encoder.joblib")
    scaler = joblib.load("scaler.joblib")
    pt = joblib.load("power_transformer.joblib")

    return model, encoder, scaler, pt


model, oe, scaler, pt = load_artifacts()

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

REQUIRED_COLS = [
    "distance-to-solar-noon",
    "temperature",
    "wind-direction",
    "wind-speed",
    "sky-cover",
    "visibility",
    "humidity",
    "average-wind-speed-(period)",
    "average-pressure-(period)",
]

# ---------------------------------------------------------
# PREPROCESSING
# ---------------------------------------------------------
def preprocess_input(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    # wind-direction: *10 -> rad -> sin/cos
    df["wind-direction"] = df["wind-direction"] * 10
    df["wind-dir-rad"] = np.deg2rad(df["wind-direction"])
    df["wind-dir-sin"] = np.sin(df["wind-dir-rad"])
    df["wind-dir-cos"] = np.cos(df["wind-dir-rad"])
    df.drop(columns=["wind-direction", "wind-dir-rad"], inplace=True)

    # One-hot encode sky-cover
    encoded = oe.transform(df[["sky-cover"]])
    encoded_cols = oe.get_feature_names_out(["sky-cover"])
    encoded_df = pd.DataFrame(encoded, columns=encoded_cols, index=df.index)
    df = pd.concat([df.drop("sky-cover", axis=1), encoded_df], axis=1)

    # PowerTransform visibility & humidity
    df[["visibility", "humidity"]] = pt.transform(df[["visibility", "humidity"]])

    # Scale numeric cols
    cols_to_scale = [
        "distance-to-solar-noon",
        "temperature",
        "wind-speed",
        "average-wind-speed-(period)",
        "average-pressure-(period)",
    ]
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])

    # Reorder to model feature order
    df = df[FEATURES]
    return df

# ---------------------------------------------------------
# SIDEBAR – RAW INPUTS
# ---------------------------------------------------------
st.sidebar.title("⚙️ Input Features (Raw)")

distance = st.sidebar.number_input(
    "distance-to-solar-noon", value=0.0, format="%.5f"
)
temperature = st.sidebar.number_input(
    "temperature (°C)", value=0.0, format="%.2f"
)
wind_direction = st.sidebar.number_input(
    "wind-direction (code, e.g. 36 ≈ 360°)",
    value=0.0,
    format="%.2f",
)
wind_speed = st.sidebar.number_input(
    "wind-speed", value=0.0, format="%.2f"
)
sky_cover = st.sidebar.selectbox(
    "sky-cover (0–4)", options=[0, 1, 2, 3, 4], index=0
)
visibility = st.sidebar.number_input(
    "visibility", value=0.0, format="%.2f"
)
humidity = st.sidebar.number_input(
    "humidity (%)", value=0.0, format="%.2f"
)
avg_wind_period = st.sidebar.number_input(
    "average-wind-speed-(period)", value=0.0, format="%.2f"
)
avg_pressure_period = st.sidebar.number_input(
    "average-pressure-(period)", value=0.0, format="%.2f"
)


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

# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
st.markdown(
    """
    <h1 style="text-align:center;">Solar Power Prediction (CatBoost)</h1>
    <p style="text-align:center; color: #bbbbbb;">
        End-to-end deployed model with automatic preprocessing and interactive visualization.
    </p>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# TABS
# ---------------------------------------------------------
tab_pred, tab_info, tab_features, tab_about = st.tabs(
    ["🔮 Prediction", "📊 Model Info", "ℹ️ Feature Guide", "📘 About App"]
)

# =========================================================
# TAB 1 – PREDICTION
# =========================================================
with tab_pred:
    sub_single, sub_csv = st.tabs(["🎛️ Manual Input", "📂 CSV Upload"])

    # -------------------------------------------------
    # SUBTAB 1 – MANUAL INPUT
    # -------------------------------------------------
    with sub_single:
        st.markdown("### Raw Inputs & Profile")

        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("#### Input Features & Values")
            raw_display = raw_df.T.reset_index()
            raw_display.columns = ["Feature", "Value"]
            st.dataframe(raw_display, use_container_width=True)

        with col_right:
            st.markdown("#### Raw Input Profile")
            plot_df = raw_display.copy()
            plot_df["Value"] = plot_df["Value"].astype(float)
            st.bar_chart(plot_df.set_index("Feature"))

        st.markdown("---")
        st.markdown("### Model Input & Prediction")

        if st.button("Predict", type="primary", key="predict_single"):
            processed_df = preprocess_input(raw_df)

            col_proc, col_out = st.columns([2, 1])

            with col_proc:
                st.markdown("#### Processed Features (Model Input)")
                st.dataframe(processed_df, use_container_width=True)

            with col_out:
                prediction = float(model.predict(processed_df)[0])

                st.markdown("#### Prediction Output")
                st.metric("Estimated Solar Power (W)", f"{prediction:,.2f}")

                st.markdown(
                    """
                    <div style="background-color:#123524; padding:10px; border-radius:8px; margin-top:10px;">
                        <span style="color:#bfffc5; font-size: 13px;">
                            This prediction is based on the current CatBoost model trained on historical data.
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("Set the input values in the sidebar and click **Predict** to see results.")

    # -------------------------------------------------
    # SUBTAB 2 – CSV UPLOAD (batch prediction)
    # -------------------------------------------------
    with sub_csv:
        st.markdown("### Batch Prediction from CSV")

        st.write("Upload a CSV file containing the raw input columns:")
        st.code(", ".join(REQUIRED_COLS), language="text")

        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"], key="csv_uploader")

        if uploaded_file is not None:
            try:
                df_raw_csv = pd.read_csv(uploaded_file)

                st.subheader("Preview of Uploaded Data")
                st.dataframe(df_raw_csv.head(), use_container_width=True)

                # check columns
                missing = [c for c in REQUIRED_COLS if c not in df_raw_csv.columns]
                if missing:
                    st.error(f"These required columns are missing in the CSV: {missing}")
                else:
                    # preprocess and predict
                    processed_csv = preprocess_input(df_raw_csv)
                    preds = model.predict(processed_csv)

                    result_df = df_raw_csv.copy()
                    result_df["predicted_power_W"] = preds

                    st.subheader("Predictions (first 10 rows)")
                    st.dataframe(result_df.head(10), use_container_width=True)

                    # download button
                    csv_out = result_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="⬇️ Download predictions as CSV",
                        data=csv_out,
                        file_name="solar_power_predictions.csv",
                        mime="text/csv",
                    )
            except Exception as e:
                st.error(f"Error processing CSV: {e}")
        else:
            st.info("Upload a CSV file to generate batch predictions.")

# =========================================================
# TAB 2 – MODEL INFO
# =========================================================
with tab_info:
    st.subheader("Model Overview")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Model type:** CatBoostRegressor")
        st.markdown("**Loss function:** RMSE")
        st.markdown("**Features used:** 13 (after preprocessing)")
    with col_b:
        st.markdown("**Artifacts:** encoder.joblib, scaler.joblib, power_transformer.joblib, catboost_model.cbm")
        st.markdown("**Training done in:** Modeldeployment.ipynb")

    st.markdown("---")
    st.subheader("Global Feature Importance")

    try:
        importances = model.get_feature_importance()
        fi_df = pd.DataFrame({"Feature": FEATURES, "Importance": importances})
        fi_df = fi_df.sort_values("Importance", ascending=False)

        st.dataframe(fi_df, use_container_width=True)
        st.bar_chart(fi_df.set_index("Feature"))
    except Exception as e:
        st.warning(f"Could not compute feature importance: {e}")

# =========================================================
# TAB 3 – FEATURE GUIDE
# =========================================================
with tab_features:
    st.subheader("Input Feature Guide")

    st.markdown("""
    **distance-to-solar-noon**  
    • Time-based feature representing how far the timestamp is from local solar noon.  

    **temperature**  
    • Ambient air temperature in °C.  

    **wind-direction**  
    • Encoded direction of wind; internally transformed to sine/cosine so the model understands circular behavior.  

    **wind-speed / average-wind-speed-(period)**  
    • Instant wind speed and average over a recent time window.  

    **sky-cover (0–4)**  
    • Categorical level for cloud cover (0 = clear, 4 = very cloudy).  
    • Internally converted into one-hot columns `sky-cover_1..4`.  

    **visibility, humidity**  
    • Transformed using PowerTransformer to stabilize variance before scaling.  

    **average-pressure-(period)**  
    • Average atmospheric pressure over a period.
    """)

    st.info("The UI always takes **raw values**, and the app automatically applies all transformations used during training.")

# =========================================================
# TAB 4 – ABOUT APP
# =========================================================
with tab_about:
    st.subheader("About This App")

    st.markdown("""
    This dashboard wraps a trained **CatBoost regression model** for predicting solar power generation.

    **Pipeline steps handled automatically:**
    - Data cleaning & preprocessing (done in the training notebook)
    - Wind direction → sine & cosine
    - One-hot encoding for *sky-cover*
    - Power transformation for *visibility* & *humidity*
    - Standard scaling for numeric features
    - Final prediction via CatBoost

    You can use this app to:
    - Experiment with different environmental conditions  
    - Understand how features affect predicted generation  
    - Demonstrate a full ML deployment workflow
    """)

    # Optional image: place e.g. "solar_panels.jpg" in the same folder
    # and uncomment the next lines:
    # from PIL import Image
    # img = Image.open("solar_panels.jpg")
    # st.image(img, caption="Solar power generation", use_column_width=True)

    st.markdown("---")
    st.caption("Built with Streamlit, CatBoost, and a frankly unreasonable amount of preprocessing 🙂")
