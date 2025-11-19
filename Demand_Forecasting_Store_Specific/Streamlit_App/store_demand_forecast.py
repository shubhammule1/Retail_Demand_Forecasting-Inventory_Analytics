import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.api import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import warnings
import os

warnings.simplefilter(action='ignore', category=FutureWarning)

# --- CONFIGURATION ---
FILE_PATH = r'D:\Slooze Dataset\slooze_challenge\SalesFINAL12312016.csv'
STORES = sorted([1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2, 20, 21, 22, 23, 24,
                 25, 26, 27, 28, 29, 3, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 4,
                 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 5, 50, 51, 52, 53, 54, 55,
                 56, 57, 58, 59, 6, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 7, 70,
                 71, 72, 73, 74, 75, 76, 77, 78, 79, 8, 9])
XGB_N_ESTIMATORS = [50, 70, 100, 120, 150, 170, 200]
XGB_MAX_DEPTHS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
FORECAST_DAYS = 2

# --- DATA LOADING ---

@st.cache_data
def load_and_preprocess_data():
    st.info(f"Attempting to load data from: {FILE_PATH}")
    
    try:
        df_raw = pd.read_csv(FILE_PATH)
        df_raw['SalesDate'] = pd.to_datetime(df_raw['SalesDate'])
        df_ts = df_raw.groupby(['Store', 'SalesDate'])['SalesQuantity'].sum().reset_index()
        df_ts = df_ts.rename(columns={'SalesQuantity': 'y'})
        
        st.success(f"Successfully loaded and aggregated data. Total daily entries: {len(df_ts)}")
        return df_ts

    except FileNotFoundError:
        st.error(f"❌ File not found at `{FILE_PATH}`")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error: {e}")
        return pd.DataFrame()


def create_xgb_features(df):
    df['dayofweek'] = df['SalesDate'].dt.dayofweek
    df['dayofyear'] = df['SalesDate'].dt.dayofyear
    df['day'] = df['SalesDate'].dt.day
    df['month'] = df['SalesDate'].dt.month
    df['year'] = df['SalesDate'].dt.year
    df['weekofyear'] = df['SalesDate'].dt.isocalendar().week.astype(int)
    return df


def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse


# --- XGBOOST MODEL ---

@st.cache_resource
def run_xgb_tuning_and_forecast(store_data, n_estimators_list, max_depth_list, forecast_days):

    store_data_features = create_xgb_features(store_data.copy())

    VAL_SIZE = 7
    if len(store_data_features) < VAL_SIZE + 1:
        return None, "Not enough data.", 0, 0, None, None

    train_df = store_data_features.iloc[:-VAL_SIZE]
    val_df = store_data_features.iloc[-VAL_SIZE:]

    FEATURES = ['dayofweek', 'dayofyear', 'day', 'month', 'year', 'weekofyear']
    TARGET = 'y'

    X_train, y_train = train_df[FEATURES], train_df[TARGET]
    X_val, y_val = val_df[FEATURES], val_df[TARGET]

    best_mae = float('inf')
    best_params = {}
    best_model = None

    tuning_status = st.status(f"Tuning XGBoost Model for Store {store_data['Store'].iloc[0]}...")

    for n_est in n_estimators_list:
        for m_depth in max_depth_list:
            model = xgb.XGBRegressor(
                n_estimators=n_est,
                max_depth=m_depth,
                random_state=42,
                objective='reg:absoluteerror',
                tree_method='hist'
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train, y_train)
            
            val_pred = model.predict(X_val)
            current_mae, _ = calculate_metrics(y_val.values, val_pred)

            if current_mae < best_mae:
                best_mae = current_mae
                best_params = {'n_estimators': n_est, 'max_depth': m_depth}
                best_model = model

    tuning_status.update(label=f"Tuning complete. Best MAE: {best_mae:.2f}", state="complete")

    # Final evaluation
    val_pred_best = best_model.predict(X_val)
    mae, rmse = calculate_metrics(y_val.values, val_pred_best)

    # Future forecast
    last_date = store_data['SalesDate'].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)
    future_df = pd.DataFrame({'SalesDate': future_dates})
    future_df = create_xgb_features(future_df)

    xgb_forecast = best_model.predict(future_df[FEATURES])
    xgb_forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Forecast': np.maximum(0, xgb_forecast.round(0)).astype(int)
    })

    val_df['Prediction'] = np.maximum(0, val_pred_best.round(0)).astype(int)
    val_plot = val_df[['SalesDate', 'y', 'Prediction']].rename(columns={'y': 'Actual'})

    return best_model, best_params, mae, rmse, xgb_forecast_df, val_plot


# --- HOLT-WINTERS MODEL ---

@st.cache_resource
def run_hw_forecast(store_data, forecast_days):

    VAL_SIZE = 7
    if len(store_data) < 14:
        return None, "Not enough data.", 0, 0, None, None

    train_series = store_data['y'].iloc[:-VAL_SIZE].reset_index(drop=True)
    val_series = store_data['y'].iloc[-VAL_SIZE:].reset_index(drop=True)
    val_dates = store_data['SalesDate'].iloc[-VAL_SIZE:]

    try:
        hw_model = ExponentialSmoothing(
            train_series,
            seasonal_periods=7,
            trend='mul',
            seasonal='mul',
            initialization_method="estimated"
        ).fit()
    except Exception as e:
        return None, str(e), 0, 0, None, None

    val_pred = hw_model.forecast(VAL_SIZE)
    mae, rmse = calculate_metrics(val_series.values, val_pred.values)

    hw_forecast = hw_model.forecast(len(store_data) + forecast_days)[-forecast_days:]

    last_date = store_data['SalesDate'].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)

    hw_forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Forecast': np.maximum(0, hw_forecast.round(0)).astype(int)
    })

    val_plot = pd.DataFrame({
        'SalesDate': val_dates.values,
        'Actual': val_series.values,
        'Prediction': np.maximum(0, val_pred.values.round(0)).astype(int)
    })

    return hw_model, "Successful", mae, rmse, hw_forecast_df, val_plot


# --- STREAMLIT APP ---

def app():
    st.set_page_config(layout="wide", page_title="Demand Forecasting Analyzer")
    st.title("Sales Demand Forecasting Analyzer")
    st.markdown("---")

    full_df = load_and_preprocess_data()
    if full_df.empty:
        return

    st.sidebar.header("Store Selection & Configuration")
    available_stores = sorted(full_df['Store'].unique().tolist())

    selected_store = st.sidebar.selectbox("Select Store ID:", options=available_stores)
    store_data = full_df[full_df['Store'] == selected_store].sort_values('SalesDate').reset_index(drop=True)

    st.sidebar.markdown(f"**Data Span:** {store_data['SalesDate'].min().strftime('%Y-%m-%d')} → {store_data['SalesDate'].max().strftime('%Y-%m-%d')}")
    st.sidebar.markdown(f"**Total Days:** {len(store_data)}")

    if st.sidebar.button("🔬 Run Analysis & Forecast Generation", type="primary"):
        st.subheader(f"Forecast Results for Store ID: {selected_store}")

        # Holt-Winters
        with st.spinner("Preparing Holt-Winters Forecast..."):
            run_hw_forecast.clear()
            hw_model, hw_status, hw_mae, hw_rmse, hw_forecast_df, hw_val_plot = run_hw_forecast(store_data, FORECAST_DAYS)

        # XGBoost
        with st.spinner("Training XGBoost Model..."):
            run_xgb_tuning_and_forecast.clear()
            xgb_model, xgb_params, xgb_mae, xgb_rmse, xgb_forecast_df, xgb_val_plot = run_xgb_tuning_and_forecast(
                store_data, XGB_N_ESTIMATORS, XGB_MAX_DEPTHS, FORECAST_DAYS
            )

        st.markdown("---")
        col1, col2 = st.columns(2)

        # XGBoost
        with col1:
            st.markdown("#### 🌲 XGBoost Results")
            if xgb_model:
                st.markdown("##### Forecast")
                st.dataframe(xgb_forecast_df, hide_index=True)
                st.markdown("##### Validation Metrics")
                st.dataframe(pd.DataFrame({
                    'Metric': ['MAE', 'RMSE'],
                    'Value': [f"{xgb_mae:.2f}", f"{xgb_rmse:.2f}"]
                }).set_index('Metric'))
                with st.expander("Optimal Parameters"):
                    st.json(xgb_params)
            else:
                st.error(xgb_params)

        # Holt-Winters
        with col2:
            st.markdown("#### 🌊 Holt-Winters Results")
            if hw_model:
                st.markdown("##### Forecast")
                st.dataframe(hw_forecast_df, hide_index=True)
                st.markdown("##### Validation Metrics")
                st.dataframe(pd.DataFrame({
                    'Metric': ['MAE', 'RMSE'],
                    'Value': [f"{hw_mae:.2f}", f"{hw_rmse:.2f}"]
                }).set_index('Metric'))
                with st.expander("Show Model Configuration"):
                    st.info("Configuration: Seasonal Periods (S=7), Trend: Multiplicative, Seasonality: Multiplicative")

            else:
                st.error(hw_status)

        st.markdown("---")
        st.subheader("Model Performance Visualization")

        if xgb_model and hw_model:
            best_model = "XGBoost" if xgb_mae < hw_mae else "Holt-Winters"
            st.balloons()
            st.markdown(f"🎉 **Best Model: {best_model} (Based on MAE)**")

            plot_data = []

            if hw_val_plot is not None:
                plot_data.append(hw_val_plot.rename(columns={'Prediction': 'HW_Prediction'}))

            if xgb_val_plot is not None:
                plot_data.append(xgb_val_plot.drop(columns=['Actual']).rename(columns={'Prediction': 'XGB_Prediction'}))

            combined = plot_data[0].set_index('SalesDate')
            if len(plot_data) > 1:
                combined = combined.merge(plot_data[1].set_index('SalesDate'),
                                          left_index=True, right_index=True, how='outer')

            combined = combined.loc[:, ~combined.columns.duplicated()]
            st.line_chart(combined)
            st.caption("Last 7 Days Validation: Actual vs Predictions")


if __name__ == "__main__":
    app()
