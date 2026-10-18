# Vehicle CO₂ Emissions Prediction

**Machine learning regression project | Python · scikit-learn · XGBoost**

Predict vehicle CO₂ emissions (g/km) from vehicle specifications **without using fuel-consumption measurements as model inputs**. This project explores the data, engineers features, tunes and compares four regression models, and evaluates them on a held-out test set.

**[Explore the notebook](vehicle_co2_prediction.ipynb)**

## Project overview

Fuel-consumption measurements are closely related to vehicle CO₂ emissions. Instead of relying on those measurements, this experiment investigates how well technical attributes alone can predict emissions.

- **Target:** `CO2 Emissions(g/km)`
- **Inputs:** engine size, number of cylinders, vehicle class (mapped to an approximate weight), transmission type, and fuel type
- **Excluded inputs:** all fuel-consumption columns, as well as make and model
- **Models:** XGBoost, Decision Tree, Random Forest, and Linear Regression (baseline)
- **Evaluation:** mean absolute error (MAE), root mean squared error (RMSE), and R²

## Dataset

**Source:** [CO2 Emission of Vehicles in Canada — Kaggle (isaacfemiogunniyi)](https://www.kaggle.com/datasets/isaacfemiogunniyi/co2-emission-of-vehicles-in-canada)

The notebook downloads the dataset directly using `kagglehub` and reads `CO2 Emissions_Canada.csv`. No manual CSV placement is necessary, although an internet connection is needed for the initial download.

The processed dataset in the saved run contains **7,385 rows**, divided into **5,169 training**, **1,108 validation**, and **1,108 test** examples.

## Methodology

1. **Exploratory data analysis:** visualize the target distribution, vehicle-class frequencies, engine size versus emissions, and numeric-feature correlations.
2. **Feature engineering:** map each vehicle class to an *estimated* representative vehicle weight; extract the transmission category from its code; one-hot encode transmission category and fuel type.
3. **Feature selection:** retain only vehicle-specification predictors; exclude fuel-consumption features.
4. **Data splitting:** split the processed data into training (70%), validation (15%), and test (15%) sets, with `random_state=42`.
5. **Model tuning:** use validation RMSE to select Decision Tree depth, Random Forest tree count, and XGBoost learning rate / maximum depth. XGBoost also uses early stopping (20 rounds).
6. **Evaluation:** compare the four fitted models on the held-out test set and inspect actual-versus-predicted plots and residuals.

## Results

| Model | MAE (g/km) ↓ | RMSE (g/km) ↓ | R² ↑ |
|---|---:|---:|---:|
| XGBoost | 11.217 | 15.405 | 0.9296 |
| Decision Tree | **11.106** | 15.451 | 0.9292 |
| Random Forest | 11.202 | **15.399** | **0.9296** |
| Linear Regression | 19.610 | 25.962 | 0.8000 |

The three tree-based models perform similarly, with R² around **0.93**, and outperform the Linear Regression baseline on this test split. Decision Tree has the lowest MAE; Random Forest has the lowest RMSE and highest R², though the differences among tree-based models are small.

In the saved run, validation tuning selected:

- **XGBoost:** learning rate `0.2`, maximum depth `6`, with early stopping
- **Decision Tree:** maximum depth `19`
- **Random Forest:** `50` trees

These are results for this particular split and experiment; they do not establish real-world on-road accuracy.

## Limitations

- Vehicle-class weights are heuristic estimates; individual vehicle weights are not provided as model inputs.
- The model uses the dataset's available technical attributes; additional real-world factors are not modeled.
- Test-set performance reflects one random split and may vary with other datasets or conditions.
