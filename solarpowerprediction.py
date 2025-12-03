<<<<<<< HEAD
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
# GLOBAL CUSTOM CSS (WEBSITE-STYLE LOOK)
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    /* Overall app background */
    .stApp {
        background: radial-gradient(circle at top, #1f2933 0, #020617 45%, #000000 100%);
        color: #e5e7eb;
        font-family: "Segoe UI", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Main container width + padding */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: #020617;
        border-right: 1px solid #111827;
    }
    [data-testid="stSidebar"] * {
        color: #e5e7eb !important;
    }

    /* Headings */
    h1, h2, h3, h4 {
        color: #f9fafb;
    }

    /* Section card styling */
    .glass-card {
        background: rgba(15, 23, 42, 0.85);
        border-radius: 16px;
        padding: 18px 20px;
        border: 1px solid rgba(148, 163, 184, 0.3);
        box-shadow: 0 18px 45px rgba(0, 0, 0, 0.45);
        margin-bottom: 1rem;
    }

    .glass-card-soft {
        background: rgba(15, 23, 42, 0.7);
        border-radius: 14px;
        padding: 14px 16px;
        border: 1px solid rgba(55, 65, 81, 0.6);
        margin-bottom: 1rem;
    }

    /* Metric block */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #022c22, #064e3b);
        padding: 14px 16px;
        border-radius: 12px;
        border: 1px solid #10b98155;
    }

    /* Tabs styling */
    button[role="tab"] {
        border-radius: 999px !important;
        padding: 0.4rem 1.1rem !important;
    }

    /* Code blocks */
    code, pre {
        background: #020617 !important;
        border-radius: 8px !important;
    }

    </style>
    """,
    unsafe_allow_html=True,
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

    # Drop missing values (if any)
    df = df.dropna()

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

st.sidebar.markdown(
    """
    Configure the raw environmental inputs here.  
    The app will apply the same preprocessing used during model training.
    """
)

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
    <div style="text-align:center; margin-bottom: 0.5rem;">
        <h1 style="margin-bottom: 0.2rem;">Solar Power Prediction ☀️</h1>
        <p style="color: #9ca3af; font-size: 14px;">
            Production-style dashboard built with CatBoost, Streamlit, and a full preprocessing pipeline.
        </p>
    </div>
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
        st.markdown("### Manual Input & Live Preview")
        with st.container():
            col_left, col_right = st.columns(2)

            with col_left:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown("#### Input Features & Values")
                raw_display = raw_df.T.reset_index()
                raw_display.columns = ["Feature", "Value"]
                st.dataframe(raw_display, use_container_width=True, height=320)
                st.markdown("</div>", unsafe_allow_html=True)

            with col_right:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown("#### Raw Input Profile")
                plot_df = raw_display.copy()
                plot_df["Value"] = plot_df["Value"].astype(float)
                st.bar_chart(plot_df.set_index("Feature"))
                st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("### Model Input & Prediction")

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        if st.button("🚀 Run Prediction", type="primary", key="predict_single"):
            processed_df = preprocess_input(raw_df)

            col_proc, col_out = st.columns([2, 1])

            with col_proc:
                st.markdown("#### Processed Features (Model Input)")
                st.dataframe(processed_df, use_container_width=True)

            with col_out:
                prediction = float(model.predict(processed_df)[0])

                # Enforce non-negative solar power
                if prediction < 0:
                    prediction = 0.0

                st.markdown("#### Prediction Output")
                st.metric("Estimated Solar Power (W)", f"{prediction:,.2f}")

                st.markdown(
                    """
                    <div style="background-color:#022c22; padding:10px; border-radius:10px; margin-top:10px; border: 1px solid #16a34a55;">
                        <span style="color:#bbf7d0; font-size: 13px;">
                            Prediction is generated using the deployed CatBoost model with the same preprocessing as training.
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("Configure inputs in the sidebar and click **Run Prediction** to see the estimated solar power.")
        st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------------------------------
    # SUBTAB 2 – CSV UPLOAD (batch prediction)
    # -------------------------------------------------
    with sub_csv:
        st.markdown("### Batch Prediction from CSV")
        st.markdown(
            '<div class="glass-card-soft">',
            unsafe_allow_html=True,
        )
        st.write("Upload a CSV file containing the raw input columns:")
        st.code(", ".join(REQUIRED_COLS), language="text")
        st.markdown("</div>", unsafe_allow_html=True)

        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"], key="csv_uploader")

        if uploaded_file is not None:
            try:
                df_raw_csv = pd.read_csv(uploaded_file)

                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.subheader("Preview of Uploaded Data")
                st.dataframe(df_raw_csv.head(), use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

                # check columns
                missing = [c for c in REQUIRED_COLS if c not in df_raw_csv.columns]
                if missing:
                    st.error(f"These required columns are missing in the CSV: {missing}")
                else:
                    # preprocess and predict
                    processed_csv = preprocess_input(df_raw_csv)

                    # if everything got dropped due to NaNs
                    if processed_csv.empty:
                        st.warning("All rows had missing values after cleaning. No predictions could be generated.")
                    else:
                        preds = model.predict(processed_csv)

                        # align back to original non-dropped rows
                        result_df = df_raw_csv.loc[processed_csv.index].copy()
                        result_df["predicted_power_W"] = preds
                        result_df["predicted_power_W"] = result_df["predicted_power_W"].clip(lower=0)

                        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                        st.subheader("Predictions (first 10 rows)")
                        st.dataframe(result_df.head(10), use_container_width=True)

                        csv_out = result_df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label="⬇️ Download predictions as CSV",
                            data=csv_out,
                            file_name="solar_power_predictions.csv",
                            mime="text/csv",
                        )
                        st.markdown("</div>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error processing CSV: {e}")
        else:
            st.info("Upload a CSV file to generate batch predictions.")

# =========================================================
# TAB 2 – MODEL INFO
# =========================================================
with tab_info:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Model Overview")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Model type:** CatBoostRegressor")
        st.markdown("**Loss function:** RMSE")
        st.markdown("**Features used:** 13 (after preprocessing)")
    with col_b:
        st.markdown("**Artifacts:** `encoder.joblib`, `scaler.joblib`, `power_transformer.joblib`, `catboost_model.cbm`")
        st.markdown("**Training notebook:** `Modeldeployment.ipynb`")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Global Feature Importance")

    try:
        importances = model.get_feature_importance()
        fi_df = pd.DataFrame({"Feature": FEATURES, "Importance": importances})
        fi_df = fi_df.sort_values("Importance", ascending=False)

        st.dataframe(fi_df, use_container_width=True)
        st.bar_chart(fi_df.set_index("Feature"))
    except Exception as e:
        st.warning(f"Could not compute feature importance: {e}")
    st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# TAB 3 – FEATURE GUIDE
# =========================================================
with tab_features:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
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
    st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# TAB 4 – ABOUT APP
# =========================================================
with tab_about:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
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

    st.markdown("---")
    st.caption("Built with Streamlit, CatBoost, and a frankly unreasonable amount of preprocessing 🙂")
    st.markdown("</div>", unsafe_allow_html=True)
=======
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

    # Drop missing values (if any)
    df = df.dropna()

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

                # Enforce non-negative solar power
                if prediction < 0:
                    prediction = 0.0

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

                    # if everything got dropped due to NaNs
                    if processed_csv.empty:
                        st.warning("All rows had missing values after cleaning. No predictions could be generated.")
                    else:
                        preds = model.predict(processed_csv)

                        # align back to original non-dropped rows
                        result_df = df_raw_csv.loc[processed_csv.index].copy()
                        result_df["predicted_power_W"] = preds
                        result_df["predicted_power_W"] = result_df["predicted_power_W"].clip(lower=0)

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
>>>>>>> 2882346c349d0347d69fa756a59b4da33dddd557
