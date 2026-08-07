import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 1. Page Configuration
st.set_page_config(page_title="Retail Sales Forecaster - AR, ARIMA, SARIMA", layout="wide")
st.title("📈 Retail Sales Time Series Forecasting (AR, ARIMA, SARIMA)")
st.write("Train and compare different time series models dynamically on your Superstore dataset without requiring `.pkl` files.")

# 2. Data Loading & Preprocessing (Cached for performance)
@st.cache_data
def load_and_preprocess_data():
    df = pd.read_csv('Superstore_Data.csv')
    # Convert 'Order Date' to actual datetime objects (DD-MM-YYYY format)
    df['Order Date'] = pd.to_datetime(df['Order Date'], format='%d-%m-%Y')
    df = df.sort_values('Order Date')
    df.set_index('Order Date', inplace=True)
    
    # Resample data to Monthly Sales ('MS' = Month Start)
    monthly_sales = df['Sales'].resample('MS').sum()
    return monthly_sales

try:
    monthly_sales = load_and_preprocess_data()
except Exception as e:
    st.error(f"Error loading 'Superstore_Data.csv': {e}")
    st.stop()

# Display historical data chart
st.subheader("📊 Historical Monthly Sales Trend")
st.line_chart(monthly_sales)

# 3. Sidebar for Model Selection & Hyperparameters
st.sidebar.header("⚙️ Model Configuration")
model_type = st.sidebar.selectbox("Choose Time Series Model", ["AR (Autoregressive)", "ARIMA", "SARIMA"])

# Dynamic parameters based on model type
if model_type == "AR (Autoregressive)":
    st.sidebar.info("AR model is configured as ARIMA(p, 0, 0)")
    p = st.sidebar.slider("AR Lags (p)", 1, 5, 2)
    d, q = 0, 0
elif model_type == "ARIMA":
    p = st.sidebar.slider("AR Lags (p)", 0, 3, 1)
    d = st.sidebar.slider("Differencing (d)", 0, 1, 1)
    q = st.sidebar.slider("MA Window (q)", 0, 3, 1)
else:  # SARIMA
    p = st.sidebar.slider("AR Lags (p)", 0, 2, 1)
    d = st.sidebar.slider("Differencing (d)", 0, 1, 1)
    q = st.sidebar.slider("MA Window (q)", 0, 2, 1)
    st.sidebar.markdown("---")
    st.sidebar.subheader("Seasonal Parameters")
    P = st.sidebar.slider("Seasonal AR (P)", 0, 1, 1)
    D = st.sidebar.slider("Seasonal Diff (D)", 0, 1, 1)
    Q = st.sidebar.slider("Seasonal MA (Q)", 0, 1, 1)
    seasonal_period = st.sidebar.selectbox("Seasonal Period (s)", [4, 6, 12], index=2)

forecast_steps = st.sidebar.slider("Forecast Horizon (Months)", 1, 24, 12)

# 4. Model Training & Forecasting Execution
if st.button(f"Train {model_type} & Generate Forecast 🚀", type="primary"):
    with st.spinner(f'Fitting {model_type} model on data...'):
        try:
            if model_type == "SARIMA":
                model = SARIMAX(monthly_sales, 
                                order=(p, d, q), 
                                seasonal_order=(P, D, Q, seasonal_period),
                                enforce_stationarity=False, 
                                enforce_invertibility=False)
            else:
                model = ARIMA(monthly_sales, order=(p, d, q))
                
            fitted_model = model.fit()
            
            # Forecast future steps
            forecast = fitted_model.forecast(steps=forecast_steps)
            
            # In-sample predictions for evaluation metrics
            in_sample_preds = fitted_model.predict(start=monthly_sales.index[1], end=monthly_sales.index[-1])
            rmse = np.sqrt(mean_squared_error(monthly_sales[1:], in_sample_preds))
            mae = mean_absolute_error(monthly_sales[1:], in_sample_preds)
            
            # Display Evaluation Metrics in Sidebar
            st.sidebar.success(f"{model_type} Trained Successfully!")
            st.sidebar.metric("RMSE (Root Mean Square Error)", f"${rmse:,.2f}")
            st.sidebar.metric("MAE (Mean Absolute Error)", f"${mae:,.2f}")
            
            # 5. Plotting Historical + Forecast Chart
            st.subheader(f"🔮 {model_type} Forecast for the Next {forecast_steps} Months")
            fig, ax = plt.subplots(figsize=(12, 5))
            
            ax.plot(monthly_sales.index, monthly_sales.values, label='Historical Sales', color='#0068c9', linewidth=2)
            ax.plot(forecast.index, forecast.values, label=f'{model_type} Forecast', color='#ff2b2b', linestyle='dashed', linewidth=2)
            
            ax.set_xlabel("Order Date")
            ax.set_ylabel("Total Sales ($)")
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.6)
            
            st.pyplot(fig)
            
            # 6. Data Table Expander
            with st.expander("View Forecasted Data Table"):
                forecast_df = forecast.reset_index()
                forecast_df.columns = ['Date', 'Predicted Sales ($)']
                forecast_df['Date'] = forecast_df['Date'].dt.strftime('%B %Y')
                st.dataframe(forecast_df, use_container_width=True)
                
        except Exception as error:
            st.error(f"An error occurred while training the model: {error}")
            st.info("Try adjusting your parameters (e.g., lower p/q values or changing differencing 'd') and click train again.")
