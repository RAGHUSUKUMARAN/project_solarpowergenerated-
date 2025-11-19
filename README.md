# Solar Power Prediction Project

This project focuses on predicting solar power output using machine learning models. The workflow includes data preprocessing, exploratory data analysis (EDA), feature engineering, model training, hyperparameter tuning, and evaluation. Multiple regression algorithms were implemented and compared, including Ridge Regression, Support Vector Regression (SVR), and Multi-Layer Perceptron (MLP) Regressor.

## Objective

The goal of this project is to build a reliable predictive model that can estimate solar power generation based on environmental and meteorological factors such as temperature, humidity, wind speed, sky cover, and distance to solar noon.

## Project Structure

```

project/
│
├── data/                     # Raw and processed datasets
├── notebooks/                # Jupyter notebooks for EDA, modelling, and evaluation
├── models/                   # Saved trained model files (.pkl)
├── src/                      # Python scripts for preprocessing and modelling
├── visuals/                  # Plots and charts generated during EDA and analysis
├── requirements.txt          # Python dependency list
└── README.md                 # Project documentation

````

## Dataset Description

The dataset includes meteorological and environmental variables such as:

- Temperature  
- Humidity  
- Wind speed and direction  
- Visibility  
- Sky cover  
- Pressure  
- Distance to solar noon  
- Power generated (target variable)

## Models Implemented

Three regression models were trained and evaluated:

1. **Ridge Regression**  
   Linear model with L2 regularization to reduce overfitting and stabilize predictions.

2. **Support Vector Regression (SVR)**  
   Captures non-linear relationships using kernel methods. Useful for complex patterns in solar data.

3. **Multi-Layer Perceptron (MLP) Regressor**  
   A feed-forward neural network capable of learning non-linear structures in the dataset.

## How to Run

1. Clone the repository:

   ```bash
   git clone [https://github.com/USERNAME/REPO-NAME.git](https://github.com/RAGHUSUKUMARAN/project_solarpowergenerated-/new/main?filename=README.md)
````

2. Move into the project directory:

   ```bash
   cd project
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Open the notebooks and run the workflow:

   ```bash
   jupyter notebook
   ```

## Evaluation

The models were evaluated using metrics such as:

* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* R² Score

Comparative performance plots and scatter diagrams are included inside the `visuals/` or notebook outputs.

## Saved Models

Exported `.pkl` files for Ridge, SVR, and MLP models are stored in:

```
models/
```

These can be reused directly for prediction or deployment.

## Future Improvements

* Add feature selection techniques
* Hyperparameter optimization using GridSearch or Bayesian search
* Add LSTM or Transformer-based models for time-series prediction
* API endpoint for real-time solar power prediction

## License

This project is licensed under the MIT License.

```

---

If you want, I can also create a **requirements.txt**, a **.gitignore**, or a visually appealing GitHub project description badge section.
```
