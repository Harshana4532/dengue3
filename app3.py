# -*- coding: utf-8 -*-
"""
Created on Sun Mar 30 14:17:16 2025

@author: owner
"""

# -*- coding: utf-8 -*-
"""
Dengue Forecasting Streamlit App
"""

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input, Bidirectional, LayerNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import io

# Set page config
st.set_page_config(page_title="Dengue Forecasting", layout="wide", page_icon="🦟")

# Cache data loading and preprocessing
@st.cache_data
def load_data(file_path=None, sample_data=False):
    """Load and preprocess data"""
    if sample_data:
        # Create sample data if no file is uploaded
        date_range = pd.date_range(start='2015-01-01', end='2024-03-01', freq='MS')
        np.random.seed(42)
        data = {
            "Cases": np.random.poisson(lam=100, size=len(date_range)) + 
                   np.random.randint(0, 50, size=len(date_range)),
            "RainFl": np.random.uniform(100, 300, size=len(date_range)),
            "RainDy": np.random.randint(5, 20, size=len(date_range)),
            "Temp": np.random.uniform(25, 32, size=len(date_range)),
            "Rhumid": np.random.uniform(60, 90, size=len(date_range))
        }
        df = pd.DataFrame(data, index=date_range)
    else:
        # Read uploaded file
        df = pd.read_csv(file_path, sep="\t", header=0)
        df.columns = ["Year", "Month", "Cases", "RainFl", "RainDy", "Temp", "Rhumid", "PrIndex", "CnIndex"]
        df = df[["Year", "Month", "Cases", "RainFl", "RainDy", "Temp", "Rhumid"]]
        df["Date"] = pd.to_datetime(df[["Year", "Month"]].assign(Day=1))
        df.set_index("Date", inplace=True)
        df.drop(columns=["Year", "Month"], inplace=True)
    
    # Handle missing values
    df.fillna(df.median(), inplace=True)
    
    return df

def create_features(df, lags=3):
    """Create lagged features"""
    for lag in range(1, lags + 1):
        df[f"RainFl_lag_{lag}"] = df["RainFl"].shift(lag)
        df[f"RainDy_lag_{lag}"] = df["RainDy"].shift(lag)
        df[f"Temp_lag_{lag}"] = df["Temp"].shift(lag)
        df[f"Rhumid_lag_{lag}"] = df["Rhumid"].shift(lag)
    
    df.dropna(inplace=True)
    return df

def build_model(input_shape):
    """Build LSTM model"""
    model = Sequential([
        Input(shape=input_shape),
        LayerNormalization(),
        Bidirectional(LSTM(128, activation="relu", return_sequences=True, kernel_regularizer=l2(0.005))),
        Dropout(0.3),
        Bidirectional(LSTM(64, activation="relu", return_sequences=True, kernel_regularizer=l2(0.005))),
        Dropout(0.3),
        Bidirectional(LSTM(32, activation="relu", kernel_regularizer=l2(0.005))),
        Dropout(0.3),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    return model

def create_sequences(X, y, seq_length):
    """Create time series sequences"""
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i+seq_length])
        y_seq.append(y[i+seq_length])
    return np.array(X_seq), np.array(y_seq)

def inverse_transform_cases(y, scaler, n_features):
    """Inverse transform scaled cases"""
    dummy = np.zeros((len(y), n_features + 1))
    dummy[:, 0] = y.flatten()
    return scaler.inverse_transform(dummy)[:, 0]

def calculate_lag_importance(model, X, y, features, lags=3):
    """Calculate lag feature importance"""
    baseline_score = r2_score(y, model.predict(X).flatten())
    importance_scores = {}
    
    for feature in ['RainFl', 'RainDy', 'Temp', 'Rhumid']:
        for lag in range(1, lags+1):
            col_name = f"{feature}_lag_{lag}"
            if col_name in features:
                col_idx = features.get_loc(col_name)
                X_shuffled = X.copy()
                for i in range(X_shuffled.shape[0]):
                    np.random.shuffle(X_shuffled[i, :, col_idx])
                shuffled_score = r2_score(y, model.predict(X_shuffled).flatten())
                importance_scores[col_name] = baseline_score - shuffled_score
                
    return importance_scores

def main():
    st.title("🦟 Dengue Forecasting Analysis")
    st.write("""
    This app performs comprehensive analysis and forecasting of dengue cases using environmental factors.
    Upload your data or use sample data to get started.
    """)
    
    # Sidebar controls
    st.sidebar.header("Settings")
    use_sample_data = st.sidebar.checkbox("Use sample data", value=True)
    
    uploaded_file = None
    if not use_sample_data:
        uploaded_file = st.sidebar.file_uploader("Upload your data file", type=["txt", "csv"])
        if uploaded_file is None:
            st.warning("Please upload a file or select 'Use sample data'")
            return
    
    lags = st.sidebar.slider("Number of lag months", 1, 6, 3)
    seq_length = st.sidebar.slider("Sequence length (months)", 1, 24, 12)
    forecast_period = st.sidebar.slider("Forecast period (months)", 1, 12, 6)
    
    # Load data
    with st.spinner("Loading and preprocessing data..."):
        df = load_data(uploaded_file, sample_data=use_sample_data)
        df = create_features(df, lags=lags)
    
    st.success("Data loaded successfully!")
    
    # Show raw data
    if st.checkbox("Show raw data"):
        st.dataframe(df.head())
        
        # Download sample data template
        if use_sample_data:
            buffer = io.StringIO()
            sample_df = df[['Cases', 'RainFl', 'RainDy', 'Temp', 'Rhumid']].copy()
            sample_df['Year'] = sample_df.index.year
            sample_df['Month'] = sample_df.index.month
            sample_df = sample_df[['Year', 'Month', 'Cases', 'RainFl', 'RainDy', 'Temp', 'Rhumid']]
            sample_df.to_csv(buffer, sep="\t", index=False)
            st.download_button(
                label="Download sample data template",
                data=buffer.getvalue(),
                file_name="dengue_data_template.txt",
                mime="text/plain"
            )
    
    # Correlation Analysis Section
    st.header("1. Correlation Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 8))
        mask = np.triu(np.ones_like(df.corr(), dtype=bool))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f", mask=mask, ax=ax)
        st.pyplot(fig)
    
    with col2:
        st.subheader("Covariance Heatmap")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df.cov(), annot=True, cmap='viridis', fmt=".2f", mask=mask, ax=ax)
        st.pyplot(fig)
    
    # Correlation with dengue cases
    st.subheader("Correlation with Dengue Cases")
    correlation_results = []
    for col in df.columns:
        if col != 'Cases':
            r, p = stats.pearsonr(df['Cases'], df[col])
            correlation_results.append({
                'Variable': col,
                'Correlation': r,
                'P-value': p
            })
    
    correlation_df = pd.DataFrame(correlation_results).round(4)
    st.dataframe(correlation_df)
    
    # Regression plots
    st.subheader("Regression Plots")
    n_features = len(df.columns) - 1  # Exclude 'Cases'
    n_cols = 3
    n_rows = int(np.ceil(n_features / n_cols))
    
    fig = plt.figure(figsize=(15, 5*n_rows))
    for i, col in enumerate(df.columns[1:]):
        plt.subplot(n_rows, n_cols, i+1)
        sns.regplot(x=df[col], y=df['Cases'], line_kws={'color': 'red'})
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(df[col], df['Cases'])
        plt.title(f'{col} vs Cases\nR²: {r_value**2:.3f}, p-value: {p_value:.4f}')
        plt.xlabel(col)
        plt.ylabel('Cases')
    plt.tight_layout()
    st.pyplot(fig)
    
    # Model Training Section
    st.header("2. LSTM Model Training")
    
    with st.spinner("Preparing data and training model..."):
        # Scale data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df)
        
        X = scaled_data[:, 1:]
        y = scaled_data[:, 0]
        
        # Create sequences
        X_seq, y_seq = create_sequences(X, y, seq_length)
        
        # Split data
        train_size = int(len(X_seq) * 0.8)
        X_train, X_test = X_seq[:train_size], X_seq[train_size:]
        y_train, y_test = y_seq[:train_size], y_seq[train_size:]
        
        # Build model
        model = build_model((seq_length, X_train.shape[2]))
        
        # Train model
        history = model.fit(
            X_train, y_train,
            epochs=400,
            batch_size=32,
            validation_data=(X_test, y_test),
            callbacks=[EarlyStopping(monitor="val_loss", patience=25, restore_best_weights=True)],
            verbose=0
        )
    
    st.success("Model training completed!")
    
    # Training history plot
    st.subheader("Training History")
    fig = plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    st.pyplot(fig)
    
    # Model Evaluation Section
    st.header("3. Model Evaluation")
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    y_train_actual = inverse_transform_cases(y_train, scaler, X.shape[1])
    y_train_pred_actual = inverse_transform_cases(y_train_pred, scaler, X.shape[1])
    y_test_actual = inverse_transform_cases(y_test, scaler, X.shape[1])
    y_test_pred_actual = inverse_transform_cases(y_test_pred, scaler, X.shape[1])
    
    # Training and test set plots
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Training Set Performance")
        fig = plt.figure(figsize=(10, 5))
        plt.plot(y_train_actual, label='Actual Cases', color='blue')
        plt.plot(y_train_pred_actual, label='Predicted Cases', color='red', linestyle='--')
        plt.title('Training Set: Actual vs Predicted Dengue Cases')
        plt.xlabel('Time Index')
        plt.ylabel('Cases')
        plt.legend()
        plt.grid()
        st.pyplot(fig)
    
    with col2:
        st.subheader("Test Set Performance")
        fig = plt.figure(figsize=(10, 5))
        plt.plot(y_test_actual, label='Actual Cases', color='blue')
        plt.plot(y_test_pred_actual, label='Predicted Cases', color='red', linestyle='--')
        plt.title('Test Set: Actual vs Predicted Dengue Cases')
        plt.xlabel('Time Index')
        plt.ylabel('Cases')
        plt.legend()
        plt.grid()
        st.pyplot(fig)
    
    # Performance metrics
    st.subheader("Model Performance Metrics")
    metrics_train = {
        'Dataset': 'Training',
        'MSE': mean_squared_error(y_train_actual, y_train_pred_actual),
        'RMSE': np.sqrt(mean_squared_error(y_train_actual, y_train_pred_actual)),
        'MAE': mean_absolute_error(y_train_actual, y_train_pred_actual),
        'R²': r2_score(y_train_actual, y_train_pred_actual)
    }
    
    metrics_test = {
        'Dataset': 'Test',
        'MSE': mean_squared_error(y_test_actual, y_test_pred_actual),
        'RMSE': np.sqrt(mean_squared_error(y_test_actual, y_test_pred_actual)),
        'MAE': mean_absolute_error(y_test_actual, y_test_pred_actual),
        'R²': r2_score(y_test_actual, y_test_pred_actual)
    }
    
    metrics_df = pd.DataFrame([metrics_train, metrics_test])
    st.dataframe(metrics_df.round(4))
    
    # Feature Importance Section
    st.header("4. Feature Importance Analysis")
    
    with st.spinner("Calculating feature importance..."):
        feature_names = df.columns[1:]
        lag_importance = calculate_lag_importance(model, X_test, y_test, feature_names, lags=lags)
    
    st.subheader("Lag Variable Importance")
    fig = plt.figure(figsize=(10, 6))
    pd.Series(lag_importance).sort_values().plot(kind='barh', color='teal')
    plt.title('Lag Variable Importance (Permutation Importance)')
    plt.xlabel('Importance Score (Drop in R² when shuffled)')
    plt.ylabel('Lag Variables')
    plt.grid(True)
    st.pyplot(fig)
    
    # Show importance scores in a table
    lag_importance_df = pd.DataFrame.from_dict(lag_importance, orient='index', columns=['Importance']).sort_values('Importance', ascending=False)
    st.dataframe(lag_importance_df.round(4))
    
    # Forecasting Section
    st.header("5. Future Forecasting")
    
    with st.spinner("Generating forecast..."):
        current_sequence = X_seq[-1].copy()
        forecast_scaled = []
        
        for _ in range(forecast_period):
            predicted = model.predict(current_sequence[np.newaxis, ...])[0][0]
            forecast_scaled.append(predicted)
            
            new_features = np.zeros(X.shape[1])
            new_features[:4] = current_sequence[-1, :4]  # Current values of main features
            
            # Update lag features
            for lag in range(1, lags+1):
                if lag == 1:
                    new_features[4:8] = current_sequence[-1, :4]
                else:
                    src_start = 4 + (lag-2)*4
                    dest_start = 4 + (lag-1)*4
                    if dest_start + 4 <= X.shape[1]:
                        new_features[dest_start:dest_start+4] = current_sequence[-1, src_start:src_start+4]
            
            current_sequence = np.vstack([current_sequence[1:], new_features])
        
        forecast_actual = inverse_transform_cases(np.array(forecast_scaled), scaler, X.shape[1])
        future_dates = [df.index[-1] + pd.DateOffset(months=i) for i in range(1, forecast_period+1)]
    
    st.subheader(f"{forecast_period}-Month Dengue Case Forecast")
    
    # Show forecast in a table
    forecast_df = pd.DataFrame({
        'Month': [date.strftime('%Y-%m') for date in future_dates],
        'Predicted Cases': forecast_actual.round(1)
    })
    st.dataframe(forecast_df)
    
    # Plot forecast
    fig = plt.figure(figsize=(12, 6))
    plt.plot(df['2022-01-01':].index, df['2022-01-01':]['Cases'], 'b-', label='Historical Cases')
    plt.plot(future_dates, forecast_actual, 'r--', marker='o', label=f'{forecast_period}-Month Forecast')
    plt.axvline(x=df.index[-1], color='gray', linestyle=':', alpha=0.7)
    plt.title(f'Dengue Cases with {forecast_period}-Month Forecast')
    plt.xlabel('Date')
    plt.ylabel('Cases')
    plt.legend()
    plt.grid(True)
    
    for date, value in zip(future_dates, forecast_actual):
        plt.annotate(f'{value:.0f}', (date, value), textcoords="offset points", xytext=(0,10), ha='center')
    
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    st.pyplot(fig)

if __name__ == "__main__":
    main()