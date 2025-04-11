import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import time

# Disabled OpenAI API to prevent quota errors
OPENAI_AVAILABLE = False

# Function to check if sentiment analysis with OpenAI is possible
def is_openai_configured():
    """Check if OpenAI API key is configured properly"""
    import os
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return False
    if OPENAI_AVAILABLE:
        return True
    return False

def prepare_features(df, prediction_days=30, feature_days=60):
    """
    Prepare features for machine learning models.
    
    Args:
        df: DataFrame with historical stock data
        prediction_days: Number of days to predict into the future
        feature_days: Number of historical days to use as features
        
    Returns:
        X_train, X_test, y_train, y_test, scaler
    """
    # Create feature set with technical indicators and price data
    feature_columns = []
    
    # Use previous N days' closing prices as features
    for i in range(1, feature_days + 1):
        col_name = f'close_lag_{i}'
        df[col_name] = df['Close'].shift(i)
        feature_columns.append(col_name)
    
    # Add technical indicators if available
    if 'rsi' in df.columns:
        df['rsi_lag_1'] = df['rsi'].shift(1)
        df['rsi_lag_5'] = df['rsi'].shift(5)
        feature_columns.extend(['rsi_lag_1', 'rsi_lag_5'])
    
    if 'macd' in df.columns and 'macd_signal' in df.columns:
        df['macd_diff'] = df['macd'] - df['macd_signal']
        df['macd_diff_lag_1'] = df['macd_diff'].shift(1)
        feature_columns.extend(['macd_diff_lag_1'])
    
    if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        df['bb_width_lag_1'] = df['bb_width'].shift(1)
        feature_columns.extend(['bb_width_lag_1'])
    
    # Add volume features
    df['volume_lag_1'] = df['Volume'].shift(1)
    df['volume_lag_5'] = df['Volume'].shift(5)
    feature_columns.extend(['volume_lag_1', 'volume_lag_5'])
    
    # Add trend features
    df['price_5d_pct'] = df['Close'].pct_change(periods=5)
    df['price_10d_pct'] = df['Close'].pct_change(periods=10)
    df['price_20d_pct'] = df['Close'].pct_change(periods=20)
    feature_columns.extend(['price_5d_pct', 'price_10d_pct', 'price_20d_pct'])
    
    # Calculate price range features
    df['range'] = df['High'] - df['Low']
    df['range_5d_avg'] = df['range'].rolling(window=5).mean().shift(1)
    feature_columns.append('range_5d_avg')
    
    # Calculate future price for prediction target (next X days return)
    df['future_return'] = df['Close'].pct_change(periods=-prediction_days)
    df['future_direction'] = np.where(df['future_return'] > 0, 1, 0)
    
    # Drop NaN values
    df.dropna(inplace=True)
    
    # Create feature matrix and target vector
    X = df[feature_columns].values
    y_reg = df['future_return'].values
    y_cls = df['future_direction'].values
    
    # Split into training and testing sets (80/20)
    X_train, X_test, y_train_reg, y_test_reg = train_test_split(
        X, y_reg, test_size=0.2, shuffle=False
    )
    
    _, _, y_train_cls, y_test_cls = train_test_split(
        X, y_cls, test_size=0.2, shuffle=False
    )
    
    # Scale the features for better model performance
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return {
        'X_train': X_train_scaled, 
        'X_test': X_test_scaled, 
        'y_train_reg': y_train_reg, 
        'y_test_reg': y_test_reg,
        'y_train_cls': y_train_cls,
        'y_test_cls': y_test_cls,
        'scaler': scaler,
        'feature_columns': feature_columns,
        'latest_features': scaler.transform(df[feature_columns].iloc[-1:].values)
    }

def train_prediction_models(ticker, prediction_days=30, feature_days=60):
    """
    Train machine learning models for price prediction.
    
    Args:
        ticker: Stock symbol
        prediction_days: Number of days to predict into the future
        feature_days: Number of historical days to use as features
        
    Returns:
        Dictionary with trained models and performance metrics
    """
    # Get extra historical data to have enough for training
    start_date = (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Download historical data
    df = yf.download(ticker, start=start_date, end=end_date)
    
    if len(df) == 0 or len(df) < feature_days + prediction_days + 50:
        return None
    
    # Calculate basic technical indicators for features
    df['rsi'] = calculate_rsi(df['Close'])
    df['macd'], df['macd_signal'], _ = calculate_macd(df['Close'])
    df['bb_upper'], _, df['bb_lower'] = calculate_bollinger_bands(df['Close'])
    
    # Prepare features and target
    feature_data = prepare_features(df, prediction_days, feature_days)
    
    # Train regression models (for price movement prediction)
    models = {}
    
    # Linear Regression
    start_time = time.time()
    lr_model = LinearRegression()
    lr_model.fit(feature_data['X_train'], feature_data['y_train_reg'])
    lr_pred = lr_model.predict(feature_data['X_test'])
    lr_mse = mean_squared_error(feature_data['y_test_reg'], lr_pred)
    lr_r2 = r2_score(feature_data['y_test_reg'], lr_pred)
    lr_time = time.time() - start_time
    
    models['linear_regression'] = {
        'model': lr_model,
        'mse': lr_mse,
        'r2': lr_r2,
        'training_time': lr_time,
        'prediction': lr_model.predict(feature_data['latest_features'])[0]
    }
    
    # Random Forest
    start_time = time.time()
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(feature_data['X_train'], feature_data['y_train_reg'])
    rf_pred = rf_model.predict(feature_data['X_test'])
    rf_mse = mean_squared_error(feature_data['y_test_reg'], rf_pred)
    rf_r2 = r2_score(feature_data['y_test_reg'], rf_pred)
    rf_time = time.time() - start_time
    
    models['random_forest'] = {
        'model': rf_model,
        'mse': rf_mse,
        'r2': rf_r2,
        'training_time': rf_time,
        'prediction': rf_model.predict(feature_data['latest_features'])[0]
    }
    
    # Gradient Boosting
    start_time = time.time()
    gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb_model.fit(feature_data['X_train'], feature_data['y_train_reg'])
    gb_pred = gb_model.predict(feature_data['X_test'])
    gb_mse = mean_squared_error(feature_data['y_test_reg'], gb_pred)
    gb_r2 = r2_score(feature_data['y_test_reg'], gb_pred)
    gb_time = time.time() - start_time
    
    models['gradient_boosting'] = {
        'model': gb_model,
        'mse': gb_mse,
        'r2': gb_r2,
        'training_time': gb_time,
        'prediction': gb_model.predict(feature_data['latest_features'])[0]
    }
    
    # SVR (Support Vector Regression)
    start_time = time.time()
    svr_model = SVR(kernel='rbf')
    svr_model.fit(feature_data['X_train'], feature_data['y_train_reg'])
    svr_pred = svr_model.predict(feature_data['X_test'])
    svr_mse = mean_squared_error(feature_data['y_test_reg'], svr_pred)
    svr_r2 = r2_score(feature_data['y_test_reg'], svr_pred)
    svr_time = time.time() - start_time
    
    models['svr'] = {
        'model': svr_model,
        'mse': svr_mse,
        'r2': svr_r2,
        'training_time': svr_time,
        'prediction': svr_model.predict(feature_data['latest_features'])[0]
    }
    
    # Create combined prediction based on model weights
    # Weight based on inverse MSE (better models have more weight)
    total_weight = sum(1/model_info['mse'] for model_info in models.values())
    ensemble_prediction = sum(
        (model_info['prediction'] * (1/model_info['mse'])) / total_weight 
        for model_info in models.values()
    )
    
    # Calculate confidence score based on model agreement
    predictions = [model_info['prediction'] for model_info in models.values()]
    mean_pred = np.mean(predictions)
    std_pred = np.std(predictions)
    
    # Calculate confidence score (higher agreement = higher confidence)
    if std_pred == 0:
        confidence = 1.0  # Perfect agreement
    else:
        # Convert standard deviation to confidence (inversely related)
        # Normalize to 0-1 range where 1 is high confidence
        confidence = 1.0 / (1.0 + 5 * std_pred)
    
    # Determine trend direction and strength
    if ensemble_prediction > 0:
        trend = "bullish"
        strength = min(abs(ensemble_prediction) * 100, 100)
    else:
        trend = "bearish"
        strength = min(abs(ensemble_prediction) * 100, 100)
    
    return {
        'models': models,
        'ensemble_prediction': ensemble_prediction,
        'prediction_std': std_pred,
        'confidence': confidence,
        'trend': trend,
        'strength': strength,
        'feature_importance': {
            'random_forest': dict(zip(feature_data['feature_columns'], rf_model.feature_importances_))
        },
        'latest_price': float(df['Close'].iloc[-1]) if len(df) > 0 else 0.0,  # Convert last price to float
        'forecast_price': float(df['Close'].iloc[-1]) * (1 + ensemble_prediction) if len(df) > 0 else 0.0,  # Calculate forecasted price
        'prediction_days': prediction_days
    }

def analyze_sentiment_with_ai(ticker, news_items):
    """
    This function is disabled to prevent OpenAI API quota errors
    
    Args:
        ticker: Stock symbol
        news_items: List of news article dictionaries
        
    Returns:
        None (OpenAI API integration is disabled)
    """
    # Always return None to skip OpenAI API calls
    return None

def calculate_rsi(price_series, period=14):
    """Calculate RSI technical indicator"""
    from utils import calculate_rsi
    return calculate_rsi(price_series, period)

def calculate_macd(price_series, fast_period=12, slow_period=26, signal_period=9):
    """Calculate MACD technical indicator"""
    from utils import calculate_macd
    return calculate_macd(price_series, fast_period, slow_period, signal_period)

def calculate_bollinger_bands(price_series, period=20, num_std=2):
    """Calculate Bollinger Bands technical indicator"""
    from utils import calculate_bollinger_bands
    return calculate_bollinger_bands(price_series, period, num_std)

def plot_prediction_confidence(prediction_results):
    """
    Create a gauge chart for prediction confidence
    
    Args:
        prediction_results: Dictionary with prediction results
        
    Returns:
        Plotly figure
    """
    confidence = prediction_results['confidence'] * 100
    trend = prediction_results['trend']
    strength = prediction_results['strength']
    
    # Create gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=confidence,
        title={"text": "Prediction Confidence"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "darkblue"},
            "steps": [
                {"range": [0, 30], "color": "red"},
                {"range": [30, 70], "color": "yellow"},
                {"range": [70, 100], "color": "green"}
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": 90
            }
        },
        domain={"x": [0, 1], "y": [0, 1]}
    ))
    
    # Add subtitle with prediction details
    direction = "↑" if trend == "bullish" else "↓"
    subtitle = f"{trend.capitalize()} ({direction}) with {strength:.1f}% strength"
    
    fig.update_layout(
        height=250,
        margin=dict(l=10, r=10, t=50, b=10),
        annotations=[
            dict(
                x=0.5,
                y=0.25,
                text=subtitle,
                showarrow=False,
                font=dict(size=14)
            )
        ]
    )
    
    return fig

def plot_model_comparison(prediction_results):
    """
    Create bar chart comparing different model predictions
    
    Args:
        prediction_results: Dictionary with prediction results
        
    Returns:
        Plotly figure
    """
    models = []
    predictions = []
    colors = []
    
    for model_name, model_info in prediction_results['models'].items():
        pred = model_info['prediction'] * 100  # Convert to percentage
        models.append(model_name.replace('_', ' ').title())
        predictions.append(pred)
        colors.append('green' if pred > 0 else 'red')
    
    # Add ensemble prediction
    models.append('Ensemble')
    predictions.append(prediction_results['ensemble_prediction'] * 100)
    colors.append('blue')
    
    fig = go.Figure(go.Bar(
        x=models,
        y=predictions,
        marker_color=colors,
        text=[f"{p:.2f}%" for p in predictions],
        textposition='auto'
    ))
    
    fig.update_layout(
        title="Model Predictions (% Return)",
        xaxis_title="Model",
        yaxis_title="Predicted Return (%)",
        height=350,
        margin=dict(l=10, r=10, t=50, b=10)
    )
    
    return fig

def plot_feature_importance(prediction_results, num_features=10):
    """
    Create bar chart of feature importance
    
    Args:
        prediction_results: Dictionary with prediction results
        num_features: Number of top features to show
        
    Returns:
        Plotly figure
    """
    if 'feature_importance' not in prediction_results or 'random_forest' not in prediction_results['feature_importance']:
        return None
    
    # Get feature importance from Random Forest model
    importance_dict = prediction_results['feature_importance']['random_forest']
    
    # Sort features by importance
    sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    
    # Select top N features
    top_features = sorted_features[:num_features]
    
    # Create horizontal bar chart
    features = [f.replace('_lag_', ' (t-') for f, _ in top_features]
    features = [f + ')' if '(t-' in f else f for f in features]
    features = [f.replace('_', ' ').title() for f in features]
    
    importance = [i for _, i in top_features]
    
    fig = go.Figure(go.Bar(
        y=features,
        x=importance,
        orientation='h',
        marker_color='lightblue'
    ))
    
    fig.update_layout(
        title="Top Predictive Factors",
        xaxis_title="Importance",
        yaxis_title="Feature",
        height=400,
        margin=dict(l=10, r=10, t=50, b=10)
    )
    
    return fig

def ml_predictions_section():
    """Main function for the ML predictions page"""
    st.header("🧠 ML-Powered Price Predictions")
    
    st.markdown("""
    Using machine learning to predict future price movements based on historical patterns.
    These predictions combine multiple models and technical indicators for more robust forecasting.
    """)
    
    # Get all selected stocks for prediction
    all_stocks = st.session_state.selected_stocks if 'selected_stocks' in st.session_state else ['AAPL']
    
    # Prediction settings
    st.subheader("Prediction Settings")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Stock selector
        selected_ticker = st.selectbox(
            "Select Stock",
            options=all_stocks,
            index=0 if all_stocks else None
        )
    
    with col2:
        # Prediction timeframe
        prediction_days = st.select_slider(
            "Prediction Timeframe",
            options=[5, 10, 15, 20, 30, 60, 90],
            value=30
        )
    
    with col3:
        # Feature lookback period
        feature_days = st.select_slider(
            "Historical Lookback Period",
            options=[30, 45, 60, 90, 120, 180],
            value=60
        )
    
    # Run prediction button
    run_prediction = st.button("Run ML Prediction", type="primary", use_container_width=True)
    
    # Check if we should run prediction
    run_prediction = run_prediction or (
        'ml_predictions' in st.session_state and 
        selected_ticker in st.session_state.ml_predictions and
        st.session_state.ml_predictions[selected_ticker].get('prediction_days') != prediction_days
    )
    
    # Initialize ML predictions in session state if not exists
    if 'ml_predictions' not in st.session_state:
        st.session_state.ml_predictions = {}
    
    # Run prediction if requested
    if run_prediction and selected_ticker:
        with st.spinner(f"Training ML models for {selected_ticker}... This may take a minute."):
            prediction_results = train_prediction_models(
                selected_ticker,
                prediction_days=prediction_days,
                feature_days=feature_days
            )
            
            if prediction_results:
                st.session_state.ml_predictions[selected_ticker] = prediction_results
                
                # Also update the news sentiment analysis
                if 'stocks_data' in st.session_state and selected_ticker in st.session_state.stocks_data:
                    from sentiment_tracker import fetch_recent_news
                    news_items = fetch_recent_news(selected_ticker, limit=10)
                    
                    if news_items:
                        sentiment_results = analyze_sentiment_with_ai(selected_ticker, news_items)
                        if sentiment_results:
                            prediction_results['sentiment_analysis'] = sentiment_results
            else:
                st.error(f"Could not generate predictions for {selected_ticker}. Insufficient historical data.")
    
    # Display prediction results if available
    if selected_ticker in st.session_state.ml_predictions:
        prediction_results = st.session_state.ml_predictions[selected_ticker]
        
        # Get current price and predicted price
        current_price = prediction_results['latest_price']
        predicted_price = prediction_results['forecast_price']
        
        # Make sure we have scalar values, not Series objects
        if hasattr(current_price, 'iloc'):
            current_price = current_price.iloc[0] if len(current_price) > 0 else 0
        if hasattr(predicted_price, 'iloc'):
            predicted_price = predicted_price.iloc[0] if len(predicted_price) > 0 else 0
            
        price_change = predicted_price - current_price
        price_change_pct = (price_change / current_price) * 100 if current_price > 0 else 0
        
        # Display predicted price and confidence
        st.subheader(f"{prediction_days}-Day Price Prediction")
        
        metric_cols = st.columns([1, 1, 1])
        
        with metric_cols[0]:
            st.metric(
                "Current Price",
                f"${current_price:.2f}"
            )
        
        with metric_cols[1]:
            st.metric(
                f"Predicted Price ({prediction_days} days)",
                f"${predicted_price:.2f}",
                f"{price_change_pct:+.2f}%"
            )
        
        with metric_cols[2]:
            st.metric(
                "ML Confidence",
                f"{prediction_results['confidence']*100:.1f}%",
                f"{prediction_results['trend'].title()}"
            )
        
        # Display prediction confidence gauge
        confidence_col, comparison_col = st.columns([1, 2])
        
        with confidence_col:
            confidence_fig = plot_prediction_confidence(prediction_results)
            st.plotly_chart(confidence_fig, use_container_width=True)
        
        with comparison_col:
            comparison_fig = plot_model_comparison(prediction_results)
            st.plotly_chart(comparison_fig, use_container_width=True)
        
        # Display feature importance
        st.subheader("What's Driving This Prediction?")
        
        # Split for feature importance and sentiment
        feature_col, sentiment_col = st.columns([3, 2])
        
        with feature_col:
            importance_fig = plot_feature_importance(prediction_results)
            if importance_fig:
                st.plotly_chart(importance_fig, use_container_width=True)
            else:
                st.info("Feature importance data not available.")
        
        with sentiment_col:
            if 'sentiment_analysis' in prediction_results:
                sentiment = prediction_results['sentiment_analysis']
                st.subheader("News Sentiment Analysis")
                
                # Display sentiment score with color
                sentiment_score = sentiment['sentiment_score']
                sentiment_color = "green" if sentiment_score >= 70 else "red" if sentiment_score <= 30 else "orange"
                
                st.markdown(f"""
                <div style="text-align: center; margin-bottom: 10px;">
                    <h1 style="color: {sentiment_color}; font-size: 3rem;">{sentiment_score}</h1>
                    <p>Sentiment Score (0-100)</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Display key sentiment drivers
                st.subheader("Key Sentiment Drivers")
                for driver in sentiment['key_drivers']:
                    st.markdown(f"• {driver}")
                
                # Display price impact
                impact = sentiment['price_impact']
                impact_icon = "📈" if impact == "bullish" else "📉" if impact == "bearish" else "📊"
                st.info(f"{impact_icon} News suggests {impact.upper()} sentiment with {sentiment['confidence']}% confidence")
                
                st.caption(f"Based on analysis of {sentiment['analyzed_count']} recent news items")
            else:
                st.info("Enable OpenAI API for enhanced sentiment analysis")
                
                # Display the option to set up OpenAI API
                if st.button("Set Up Enhanced Sentiment Analysis", use_container_width=True):
                    st.session_state.setup_openai = True
        
        # Technical interpretation
        st.subheader("Prediction Interpretation")
        
        # Create a markdown table with model metrics
        metrics_md = "| Model | Predicted Return | MSE | R² Score |\n|-------|----------------|-----|-------|\n"
        
        for model_name, model_info in prediction_results['models'].items():
            model_label = model_name.replace('_', ' ').title()
            pred_return = f"{model_info['prediction']*100:+.2f}%"
            mse = f"{model_info['mse']:.6f}"
            r2 = f"{model_info['r2']:.4f}"
            
            metrics_md += f"| {model_label} | {pred_return} | {mse} | {r2} |\n"
        
        st.markdown(metrics_md)
        
        st.markdown(f"""
        ### Key Takeaways
        
        - The ensemble prediction suggests a {prediction_results['trend']} trend with {prediction_results['strength']:.1f}% strength
        - Prediction confidence is {prediction_results['confidence']*100:.1f}%
        - Forecasted {prediction_days}-day price target: ${predicted_price:.2f} ({price_change_pct:+.2f}%)
        """)
        
        st.markdown("""
        #### Important Notes:
        
        - ML predictions are based on historical patterns and technical indicators
        - Past performance does not guarantee future results
        - These predictions should be used as one of many tools in your investment decision process
        - Higher confidence scores indicate greater agreement among models
        """)
    else:
        st.info("Click 'Run ML Prediction' to generate price forecasts for the selected stock.")
        
    # Check if we should show OpenAI API key setup
    if 'setup_openai' in st.session_state and st.session_state.setup_openai:
        st.subheader("Set Up OpenAI API Key")
        st.markdown("""
        Enhanced sentiment analysis uses OpenAI's powerful AI models to analyze news sentiment.
        To enable this feature, you'll need to provide an OpenAI API key.
        """)
        
        if st.button("Set Up OpenAI API Key", use_container_width=True):
            # This will be handled by the ask_secrets tool in the main app
            # The button here just indicates user intention
            st.session_state.openai_api_requested = True
            st.success("API key request has been logged. Please wait for the administrator to set up the API key.")
            st.rerun()
