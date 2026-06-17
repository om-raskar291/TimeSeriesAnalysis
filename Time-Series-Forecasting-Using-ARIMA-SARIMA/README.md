# 📈 Retail Sales Time Series Forecasting

## 📌 Project Overview

This project focuses on forecasting retail sales using Time Series Analysis techniques.

The objective is to analyze historical sales data, identify trends and seasonality, and develop forecasting models capable of predicting future sales.

The following forecasting models were implemented and compared:

- Autoregressive Model (AR)
- ARIMA Model
- SARIMA Model

Model performance was evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

---

## 🎯 Problem Statement

Accurate sales forecasting helps businesses:

- Optimize inventory levels
- Improve staffing decisions
- Enhance budgeting and financial planning
- Support strategic business decisions

The goal of this project is to build and compare statistical forecasting models capable of predicting future retail sales.

---

## 📊 Dataset

**Dataset:** Superstore Sales Dataset

### Features Used

- Order Date
- Sales
- Profit

### Target Variable

- Sales

---

## 🔄 Project Workflow

1. Project Objective
2. Data Loading and Inspection
3. Data Cleaning and Preprocessing
4. Exploratory Data Analysis (EDA)
5. Stationarity Analysis
6. Time Series Transformation
7. Time Series Decomposition
8. Autocorrelation Plots (ACF & PACF)
9. Autoregressive Model (AR)
10. ARIMA Modeling
11. SARIMA Modeling
12. Model Evaluation & Comparison
13. Conclusion

---

## 📈 Model Performance

| Model  | MAE       | RMSE      |
|--------|---------- |---------- |
| AR 	 | 13,560.76 | 14,915.16 |
| ARIMA  | 25,686.16 | 30,556.25 |
| SARIMA | 10,220.60 | 11,662.17 |

---

## 🏆 Best Performing Model

The **SARIMA Model** achieved the lowest MAE and RMSE values among all forecasting models.

Therefore, SARIMA was selected as the final forecasting model for future sales prediction.

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Statsmodels
- Scikit-learn
- SciPy

---

## 👨‍💻 Author

**Om Raskar**

LinkedIn: http://linkedin.com/in/om-raskar-566338220

GitHub: https://github.com/om-raskar291
