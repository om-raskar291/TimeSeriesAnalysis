import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 1. Page Configuration
st.set_page_config(page_title="Retail Sales Forecaster", layout="wide")
st.title("📈 Retail Sales Time Series Forecasting")
st.write("This app forecasts future sales based on the Superstore dataset (Training dynamically without .pkl files).")

# 2. Data Loading & Preprocessing (Cached for speed)
@st.cache_data
def load_and_preprocess_data():
    # Load dataset
    df = pd.read_csv('Superstore_Data.csv')
    
    # Convert 'Order Date' to actual datetime objects
    # The CSV format appears to be DD-MM-YYYY
    df['Order Date'] = pd.to_datetime(df['Order Date'], format='%d-%m-%Y')
    
    # Sort by date and set as index
    df = df.sort_values('Order Date')
    df.set_index('Order Date', inplace=True)
    
    # Resample data to Monthly Sales ('MS' = Month Start)
    monthly_sales = df['Sales'].resample('MS').sum()
    return monthly_sales

monthly_sales = load_and_preprocess_data()

# Display historical data
st.subheader("📊 Historical Monthly Sales")
st.line_chart(monthly_sales)

# 3. Sidebar for Model Hyperparameters
st.sidebar.header("⚙️ ARIMA Parameters")
p = st.sidebar.slider("AR (Lag observations - p)", 0, 5, 1)
d = st.sidebar.slider("I (Degree of differencing - d)", 0, 2, 1)
q = st.sidebar.slider("MA (Moving average window - q)", 0, 5, 1)

forecast_steps = st.sidebar.slider("Forecast Horizon (Months)", 1, 24, 12)

# 4. Model Training Function (Cached to prevent retraining on every click)
@st.cache_resource
def train_model(data, order):
    model = ARIMA(data, order=order)
    fitted_model = model.fit()
    return fitted_model

# 5. Execution & Forecasting
if st.button("Train Model & Generate Forecast 🚀", type="primary"):
    with st.spinner('Training ARIMA Model on the fly...'):
        # Train model
        model = train_model(monthly_sales, (p, d, q))
        
        # Generate Forecast
        forecast = model.forecast(steps=forecast_steps)
        
        # Evaluate model on historical data (In-sample predictions)
        # We skip the first row because differencing creates NaNs
        in_sample_preds = model.predict(start=monthly_sales.index[1], end=monthly_sales.index[-1])
        rmse = np.sqrt(mean_squared_error(monthly_sales[1:], in_sample_preds))
        mae = mean_absolute_error(monthly_sales[1:], in_sample_preds)
        
        # Display Metrics in Sidebar
        st.sidebar.success("Model Trained Successfully!")
        st.sidebar.metric("RMSE", f"{rmse:,.2f}")
        st.sidebar.metric("MAE", f"{mae:,.2f}")
        
        # 6. Plotting Historical + Forecast
        st.subheader(f"🔮 Forecast for the Next {forecast_steps} Months")
        fig, ax = plt.subplots(figsize=(12, 5))
        
        # Plot Historical
        ax.plot(monthly_sales.index, monthly_sales.values, label='Historical Sales', color='blue')
        # Plot Forecast
        ax.plot(forecast.index, forecast.values, label='Forecasted Sales', color='red', linestyle='dashed')
        
        ax.set_xlabel("Date")
        ax.set_ylabel("Total Sales ($)")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        
        st.pyplot(fig)
        
        # 7. Data Table
        with st.expander("View Forecasted Data Table"):
            forecast_df = forecast.reset_index()
            forecast_df.columns = ['Date', 'Predicted Sales ($)']
            # Convert date to display cleanly
            forecast_df['Date'] = forecast_df['Date'].dt.strftime('%B %Y') 
            st.dataframe(forecast_df, use_container_width=True)