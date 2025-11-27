
# Solar Power Prediction Application

This application predicts solar power generation using a machine learning model trained on historical environmental and atmospheric features. The frontend is built with Streamlit and supports both single-input predictions and batch CSV-based predictions.

---

## Installation and Execution

1. Install Python 3.11.
2. Install the required dependencies:
```

pip install -r requirements.txt

```
3. Launch the application:
```

streamlit run app.py

```

The interface will be available at:
```

[http://localhost:8501](http://localhost:8501)

```

---

## Model Information

- Algorithm: CatBoostRegressor  
- Training details:
  - Hyperparameter tuning using GridSearchCV
  - Includes full preprocessing pipeline from the original dataset
- Model file:
```

catboost_model.cbm

```

---

## Required Files

Ensure the following files are located in the same directory as `app.py`:

```

app.py
catboost_model.cbm
encoder.joblib
scaler.joblib
power_transformer.joblib
requirements.txt

```

Optional but recommended:

```

Modeldeployment.ipynb
data_clean.csv

```

---

## Input Methods

### Manual Input Mode

Users can provide individual values in the user interface for:

- distance-to-solar-noon  
- temperature  
- wind-direction  
- wind-speed  
- sky-cover  
- visibility  
- humidity  
- average-wind-speed-(period)  
- average-pressure-(period)

These raw values are internally transformed before prediction.

### CSV Batch Prediction

Users may upload a CSV containing the following columns:

```

distance-to-solar-noon
temperature
wind-direction
wind-speed
sky-cover
visibility
humidity
average-wind-speed-(period)
average-pressure-(period)

```

The application returns a downloadable CSV containing model predictions.

---

## Features

- Interactive Streamlit UI
- Live visualization of input parameters
- CSV batch processing
- Model transparency via feature importance
- Built-in documentation of feature meanings
- Fully automated preprocessing, including:
  - One-hot encoding
  - Standard scaling
  - PowerTransform
  - Wind direction conversion to sine/cosine representation

---

## Troubleshooting

If running the app triggers dependency errors, manually install:

```

pip install streamlit pandas numpy scikit-learn catboost joblib

```

If model files are missing, ensure the extracted project includes all `.joblib` and `.cbm` artifacts.

---

## Author

Maddy  
Python Developer — Machine Learning & Data Applications

---

This application demonstrates a complete ML deployment workflow, from preprocessing and feature engineering to model inference and interactive visualization.
```

---

