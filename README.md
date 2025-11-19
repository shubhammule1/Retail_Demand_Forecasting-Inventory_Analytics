# 🚀 Retail Demand Forecasting & Inventory Analysis
This project provides a comprehensive system for predicting store-level demand and analyzing inventory using historical sales and purchase data. It combines statistical and machine learning models to forecast sales, classify products, calculate safety stock, and analyze lead times for optimized inventory management.

# ✨ Features
📈 Store-specific demand forecasting using Holt-Winters and XGBoost
🗂️ ABC classification of products based on revenue contribution
🛒 Reorder Point (ROP) calculation with safety stock estimation
⏱️ Lead time analysis to identify procurement efficiency
📊 Visualizations for forecast, validation, and inventory insights
🖥️ Interactive Streamlit app for exploring store forecasts

# 🧰 Tech Stack
🐍 Python 3.x
📊 pandas, numpy, matplotlib
📈 statsmodels (Holt-Winters)
🤖 xgboost (regression forecasting)
🧪 scikit-learn (metrics & evaluation)
🖥️ streamlit (interactive dashboards)

# ⚡ How It Works

Load historical sales and purchase data.

Preprocess and aggregate data at store/product level.

Forecast store-level demand using Holt-Winters and XGBoost.

Classify products into A/B/C categories based on total revenue.

Calculate safety stock and reorder points using demand variability and lead time.

Generate visual summaries and interactive dashboards for analysis.
