imimport streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
import yfinance as yf
from utils import (calculate_rsi, check_buy_signal, get_stock_data, 
               calculate_macd, calculate_bollinger_bands, calculate_ema, get_last_signal_time,
               calculate_stochastic_oscillator, calculate_adx, calculate_obv, detect_chart_patterns,
               create_custom_indicator)
from backtest import run_backtest, get_benchmark_performance, optimize_strategy_parameters, walk_forward_optimization, create_mean_reversion_strategy, create_momentum_strategy, create_breakout_strategy, create_dual_moving_average_strategy, create_volatility_breakout_strategy
from gamification_minimal import learning_section, check_watchlist_achievements, check_backtest_achievements, check_badge_unlocks, init_gamification
from sms_alerts import is_twilio_configured, send_price_alert, validate_phone_number
from email_alerts import is_email_configured, send_email_alert, validate_email, create_buy_signal_email
from signal_alerts import SignalAlertManager, signal_alerts_section
from portfolio import portfolio_section
from signal_tracker import signal_tracker_section, initialize_signal_tracker, detect_and_record_signals, update_signal_performance
from sentiment_tracker import sentiment_tracker_section

# Import new animation and visualization modules
from animations import add_page_transitions, loading_animation, show_animated_notification
from animated_badges import display_animated_badge_progress, award_badge_with_animation, display_achievement_statistics
from mood_ring_visualization import mood_ring_section
from sentiment_animation_simple import sentiment_animation_section
from watchlist import watchlist_page
from ml_predictions import ml_predictions_section
from fundamental_analysis import (fundamental_analysis_section, get_company_info, 
                              get_key_metrics, get_financials, get_peer_comparison, 
                              plot_metric_comparison, format_financial_statement, 
                              parse_screening_filters, apply_screening_filters)
from social_features import social_features_section
from advanced_visualization import advanced_visualization_section

# Page config
st.set_page_config(
    page_title="Stock Market Analysis Tool",
    page_icon="📈",
    layout="wide"
)

# Initialize app state for navigation
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'analysis'

# Add page transition animations
add_page_transitions()

# Initialize gamification and check for badge unlocks
init_gamification()
check_badge_unlocks()

# Main navigation
st.title("📈 Stock Market Analysis Tool")

# Set the theme to dark mode
st.markdown("""
<style>
    .main {
        background-color: #1E1E1E;
        color: #FFFFFF;
    }
    .stButton>button {
        width: 100%;
        background-color: #2E2E2E;
        color: #FFFFFF;
        border: 1px solid #3E3E3E;
        border-radius: 4px;
        padding: 10px 24px;
        margin: 5px 0;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #4E4E4E;
        color: #FFFFFF;
    }
    .stButton>button:disabled {
        background-color: #0C7BDC;
        color: #FFFFFF;
        border: 1px solid #0C7BDC;
    }
    /* Global styling for tutorial content */
    .tutorial-content, 
    .tutorial-content h3, 
    .tutorial-content p, 
    .tutorial-content ul, 
    .tutorial-content li,
    .tutorial-content ol,
    .tutorial-content strong,
    .tutorial-content em {
        color: #000000 !important;
        background-color: #FFFFFF !important;
    }
    .tutorial-content {
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
        border: 1px solid #DDDDDD;
    }
    .streamlit-expanderHeader {
        background-color: #2E2E2E;
        color: #FFFFFF;
        border-radius: 4px;
        padding: 10px 15px;
        font-weight: bold;
    }
    .streamlit-expanderContent {
        background-color: #1E1E1E;
        border: 1px solid #3E3E3E;
        border-radius: 0 0 4px 4px;
        padding: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Create collapsible sections with expanders
with st.expander("📊 ANALYSIS TOOLS", expanded=True):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Stock Analysis", use_container_width=True, 
                    disabled=st.session_state.current_page == 'analysis'):
            st.session_state.current_page = 'analysis'
            st.rerun()
    
    with col2:
        if st.button("Sentiment Analysis", use_container_width=True,
                    disabled=st.session_state.current_page == 'sentiment'):
            st.session_state.current_page = 'sentiment'
            st.rerun()
    
    with col3:
        if st.button("Advanced Visualization", use_container_width=True,
                    disabled=st.session_state.current_page == 'advanced_viz'):
            st.session_state.current_page = 'advanced_viz'
            st.rerun()

with st.expander("🛠 OPERATION TOOLS", expanded=True):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Signal Tracker", use_container_width=True,
                    disabled=st.session_state.current_page == 'signals'):
            st.session_state.current_page = 'signals'
            st.rerun()
    
    with col2:
        if st.button("Buy/Sell Alerts", use_container_width=True,
                    disabled=st.session_state.current_page == 'alerts'):
            st.session_state.current_page = 'alerts'
            st.rerun()
    
    with col3:
        if st.button("ML Predictions", use_container_width=True,
                    disabled=st.session_state.current_page == 'ml_predictions'):
            st.session_state.current_page = 'ml_predictions'
            st.rerun()

with st.expander("🔁 STRATEGY", expanded=True):
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("Backtesting", use_container_width=True,
                    disabled=st.session_state.current_page == 'backtesting'):
            st.session_state.current_page = 'backtesting'
            st.rerun()
    
    # Additional columns are empty but keep the layout balanced
    with col2:
        pass
    
    with col3:
        pass

with st.expander("🎯 LEARNING", expanded=True):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("My Analysis Hub", use_container_width=True, 
                    disabled=st.session_state.current_page == 'analysis_hub'):
            st.session_state.current_page = 'analysis_hub'
            st.rerun()
    
    with col2:
        if st.button("Learning Center", use_container_width=True,
                    disabled=st.session_state.current_page == 'learning'):
            st.session_state.current_page = 'learning'
            st.rerun()
    
    # Empty column for balance
    with col3:
        pass

# Community Interaction section
with st.expander("🌐 COMMUNITY INTERACTION", expanded=True):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Social Hub", use_container_width=True,
                    disabled=st.session_state.current_page == 'social'):
            st.session_state.current_page = 'social'
            st.rerun()
    
    # Empty columns for balance
    with col2:
        pass
    
    with col3:
        pass

# Add a separator after navigation
st.markdown("---")

# Description based on current page
if st.session_state.current_page == 'analysis':
    st.markdown("""
    This application monitors selected stocks, calculates technical indicators like RSI (Relative Strength Index),
    and provides buy signals based on market conditions.
    """)
elif st.session_state.current_page == 'watchlists':
    st.markdown("""
    Create and manage custom watchlists to track different groups of stocks.
    Organize stocks by sector, strategy, or your own custom categories.
    """)
elif st.session_state.current_page == 'learning':
    st.markdown("""
    Learn about stock market concepts and technical analysis through interactive modules.
    Complete quizzes and earn badges as you master trading concepts.
    """)
elif st.session_state.current_page == 'backtesting':
    st.markdown("""
    Test trading strategies against historical data to see how they would have performed.
    Compare multiple strategies and optimize parameters for better returns.
    """)
elif st.session_state.current_page == 'portfolio':
    st.markdown("""
    Track your stock portfolio, analyze performance metrics, assess risk, and visualize your investments.
    Get insights on sector allocation and personalized risk management recommendations.
    """)
elif st.session_state.current_page == 'signals':
    st.markdown("""
    Track the performance of historical buy signals generated by the system.
    Analyze signal success rates and see which indicators have historically been most accurate.
    """)
elif st.session_state.current_page == 'alerts':
    st.markdown("""
    Set up automated email and SMS alerts for strong buy and sell signals.
    Configure alert thresholds, frequency, and customize which technical indicators trigger notifications.
    """)
elif st.session_state.current_page == 'sentiment':
    st.markdown("""
    Analyze market sentiment from news, social media, and price action.
    Track sentiment for individual stocks, visualize sentiment waves, and gauge overall market mood to improve your timing.
    """)
elif st.session_state.current_page == 'ml_predictions':
    st.markdown("""
    🧠 Use machine learning to predict future price movements based on historical patterns.
    Combine multiple ML models for price forecasting with AI-powered sentiment analysis.
    """)
elif st.session_state.current_page == 'social':
    st.markdown("""
    Connect with other investors, share watchlists and trading strategies, discuss stocks,
    and track the top-performing traders on the leaderboard.
    """)
elif st.session_state.current_page == 'analysis_hub':
    st.markdown("""
    Your personalized analysis center that combines watchlists, portfolio tracking, buy/sell alerts, and backtesting in one place.
    Track your investments, test strategies, and get notified of trading opportunities all from a single interface.
    """)
elif st.session_state.current_page == 'fundamental':
    st.markdown("""
    Analyze key financial metrics, company statements, and compare with industry peers.
    Screen stocks based on fundamental criteria and evaluate long-term investment potential.
    """)
elif st.session_state.current_page == 'advanced_viz':
    st.markdown("""
    Explore stocks using advanced visualization tools including Renko charts, Point & Figure charts,
    multi-stock comparison, and interactive charts with annotations for important events.
    These specialized visualizations help identify patterns that might not be visible in traditional charts.
    """)

# Initialize session state variables
if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now() - timedelta(minutes=16)
if 'update_counter' not in st.session_state:
    st.session_state.update_counter = 0
if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = False
if 'stocks_data' not in st.session_state:
    st.session_state.stocks_data = {}
if 'selected_stocks' not in st.session_state:
    st.session_state.selected_stocks = ['NVDA', 'TSLA', 'META']
if 'rsi_period' not in st.session_state:
    st.session_state.rsi_period = 14
if 'time_period' not in st.session_state:
    st.session_state.time_period = "1mo"
if 'drop_threshold' not in st.session_state:
    st.session_state.drop_threshold = 5.0
if 'show_macd' not in st.session_state:
    st.session_state.show_macd = True
if 'show_ema' not in st.session_state:
    st.session_state.show_ema = True
if 'show_bollinger' not in st.session_state:
    st.session_state.show_bollinger = True
if 'show_stochastic' not in st.session_state:
    st.session_state.show_stochastic = True
if 'show_adx' not in st.session_state:
    st.session_state.show_adx = True
if 'show_obv' not in st.session_state:
    st.session_state.show_obv = True
if 'show_pattern_recognition' not in st.session_state:
    st.session_state.show_pattern_recognition = True
if 'show_custom_indicator' not in st.session_state:
    st.session_state.show_custom_indicator = False
if 'ema_period' not in st.session_state:
    st.session_state.ema_period = 20
if 'bb_period' not in st.session_state:
    st.session_state.bb_period = 20
if 'stoch_k_period' not in st.session_state:
    st.session_state.stoch_k_period = 14
if 'stoch_d_period' not in st.session_state:
    st.session_state.stoch_d_period = 3
if 'adx_period' not in st.session_state:
    st.session_state.adx_period = 14
if 'custom_indicator_weights' not in st.session_state:
    st.session_state.custom_indicator_weights = {
        'rsi': 1.0,
        'macd': 1.0,
        'stoch': 1.0,
        'obv': 0.5,
        'bb': 0.5
    }
if 'last_signals' not in st.session_state:
    st.session_state.last_signals = {}

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    
    # Stock selection
    stock_input = st.text_input("Add a stock symbol:", placeholder="e.g., AAPL")
    if st.button("Add Stock", use_container_width=True) and stock_input:
        stock_input = stock_input.strip().upper()
        if stock_input not in st.session_state.selected_stocks:
            try:
                # Verify the stock exists by fetching its data
                stock = yf.Ticker(stock_input)
                hist = stock.history(period="1d")
                if len(hist) > 0:
                    st.session_state.selected_stocks.append(stock_input)
                    st.success(f"Added {stock_input} to your watchlist")
                else:
                    st.error(f"Could not find stock with symbol {stock_input}")
            except Exception as e:
                st.error(f"Error adding stock: {str(e)}")
    
    # Display and manage selected stocks
    st.subheader("Your Watchlist")
    for i, stock in enumerate(st.session_state.selected_stocks):
        # Display stock name
        st.write(f"{i+1}. {stock}")
        
        # Make remove button vertical (full width)
        if st.button("Remove", key=f"remove_{stock}", use_container_width=True):
            st.session_state.selected_stocks.remove(stock)
            st.session_state.stocks_data.pop(stock, None)
            st.rerun()
    
    # Technical parameters
    st.subheader("Technical Parameters")
    st.session_state.time_period = st.selectbox(
        "Time Period",
        options=["1d", "5d", "1mo", "3mo", "6mo", "1y"],
        index=2
    )
    
    # Create tabs for different technical indicators
    tabs = st.tabs(["RSI", "MACD", "Bollinger", "Stochastic", "ADX", "OBV", "Patterns", "Custom"])
    
    with tabs[0]:  # RSI Tab
        st.session_state.rsi_period = st.slider("RSI Period", min_value=2, max_value=30, value=st.session_state.rsi_period)
        st.session_state.drop_threshold = st.slider(
            "Price Drop Threshold (%)",
            min_value=1.0,
            max_value=10.0,
            value=st.session_state.drop_threshold,
            step=0.5
        )
    
    with tabs[1]:  # MACD Tab
        st.session_state.show_macd = st.checkbox("Show MACD", value=st.session_state.show_macd)
        if 'macd_fast_period' not in st.session_state:
            st.session_state.macd_fast_period = 12
        if 'macd_slow_period' not in st.session_state:
            st.session_state.macd_slow_period = 26
        if 'macd_signal_period' not in st.session_state:
            st.session_state.macd_signal_period = 9
            
        st.session_state.macd_fast_period = st.slider("Fast EMA Period", min_value=5, max_value=30, value=st.session_state.macd_fast_period)
        st.session_state.macd_slow_period = st.slider("Slow EMA Period", min_value=15, max_value=50, value=st.session_state.macd_slow_period)
        st.session_state.macd_signal_period = st.slider("Signal Period", min_value=5, max_value=20, value=st.session_state.macd_signal_period)
        st.info("MACD is calculated as the difference between Fast EMA and Slow EMA, with the Signal line as EMA of MACD.")
    
    with tabs[2]:  # Bollinger Tab
        st.session_state.show_bollinger = st.checkbox("Show Bollinger Bands", value=st.session_state.show_bollinger)
        st.session_state.bb_period = st.slider("Bollinger Period", min_value=5, max_value=50, value=st.session_state.bb_period)
        
        # Add standard deviation parameter
        if 'bb_std' not in st.session_state:
            st.session_state.bb_std = 2.0
        st.session_state.bb_std = st.slider("Standard Deviations", min_value=1.0, max_value=3.0, value=st.session_state.bb_std, step=0.1)
        
        st.session_state.show_ema = st.checkbox("Show EMA", value=st.session_state.show_ema)
        st.session_state.ema_period = st.slider("EMA Period", min_value=5, max_value=50, value=st.session_state.ema_period)
        
        st.info("Bollinger Bands consist of a middle band (SMA) and two outer bands that are standard deviations away from the middle band.")
    
    with tabs[3]:  # Stochastic Tab
        st.session_state.show_stochastic = st.checkbox("Show Stochastic Oscillator", value=st.session_state.show_stochastic)
        st.session_state.stoch_k_period = st.slider("%K Period", min_value=5, max_value=30, value=st.session_state.stoch_k_period)
        st.session_state.stoch_d_period = st.slider("%D Period", min_value=1, max_value=10, value=st.session_state.stoch_d_period)
        
    with tabs[4]:  # ADX Tab
        st.session_state.show_adx = st.checkbox("Show ADX (Average Directional Index)", value=st.session_state.show_adx)
        st.session_state.adx_period = st.slider("ADX Period", min_value=7, max_value=30, value=st.session_state.adx_period)
        st.info("ADX above 25 indicates a strong trend (regardless of direction).")
        
    with tabs[5]:  # OBV Tab
        st.session_state.show_obv = st.checkbox("Show On-Balance Volume (OBV)", value=st.session_state.show_obv)
        st.info("OBV measures buying and selling pressure as a cumulative indicator that adds volume on up days and subtracts volume on down days.")
        
    with tabs[6]:  # Chart Pattern Recognition Tab
        st.session_state.show_pattern_recognition = st.checkbox("Enable Pattern Recognition", value=st.session_state.show_pattern_recognition)
        st.info("Detects chart patterns like Head & Shoulders, Double Top/Bottom, Triangle patterns, Support/Resistance levels, etc.")
        
    with tabs[7]:  # Custom Indicator Tab
        st.session_state.show_custom_indicator = st.checkbox("Create Custom Indicator", value=st.session_state.show_custom_indicator)
        if st.session_state.show_custom_indicator:
            st.write("Set weights for each indicator in the custom combined indicator:")
            
            custom_weights = st.session_state.custom_indicator_weights
            
            custom_weights['rsi'] = st.slider("RSI Weight", 0.0, 2.0, custom_weights['rsi'], 0.1)
            custom_weights['macd'] = st.slider("MACD Weight", 0.0, 2.0, custom_weights['macd'], 0.1)
            custom_weights['stoch'] = st.slider("Stochastic Weight", 0.0, 2.0, custom_weights['stoch'], 0.1)
            custom_weights['obv'] = st.slider("OBV Weight", 0.0, 2.0, custom_weights['obv'], 0.1)
            custom_weights['bb'] = st.slider("Bollinger Weight", 0.0, 2.0, custom_weights['bb'], 0.1)
            
            st.session_state.custom_indicator_weights = custom_weights
            
            st.info("Custom indicator combines multiple indicators with weighted averages to create a personalized signal.")
    
    # SMS Price Alerts
    st.subheader("📱 SMS Price Alerts")
    
    # Initialize session state variables for SMS alerts
    if 'sms_alerts_enabled' not in st.session_state:
        st.session_state.sms_alerts_enabled = False
    if 'phone_number' not in st.session_state:
        st.session_state.phone_number = ""
    
    # Check if Twilio is configured
    twilio_configured = is_twilio_configured()
    
    if not twilio_configured:
        st.warning("SMS alerts require Twilio credentials. Please contact the administrator to set up this feature.")
        
        # Only show a button to request setup rather than exposing missing credentials
        if st.button("Request SMS Alert Setup", use_container_width=True):
            st.info("The administrator has been notified. Twilio credentials need to be configured for SMS alerts.")
    else:
        # SMS alerts toggle and phone number input
        st.session_state.sms_alerts_enabled = st.checkbox(
            "Enable SMS Buy Signal Alerts",
            value=st.session_state.sms_alerts_enabled,
            help="Receive SMS notifications when buy signals are detected"
        )
        
        if st.session_state.sms_alerts_enabled:
            st.session_state.phone_number = st.text_input(
                "Your Phone Number (with country code)",
                value=st.session_state.phone_number,
                placeholder="e.g., +1XXXXXXXXXX",
                help="Enter your phone number with country code (e.g., +1 for US)"
            )
            
            # Validate phone number format
            if st.session_state.phone_number:
                validated_number = validate_phone_number(st.session_state.phone_number)
                if validated_number:
                    st.session_state.phone_number = validated_number
                    st.success("Phone number format is valid")
                else:
                    st.error("Please enter a valid phone number with country code")
            
            # Alert threshold settings
            if 'sms_signal_threshold' not in st.session_state:
                st.session_state.sms_signal_threshold = 75
                
            st.session_state.sms_signal_threshold = st.slider(
                "Minimum Signal Strength for Alerts (%)",
                min_value=25,
                max_value=100,
                value=st.session_state.sms_signal_threshold,
                step=25,
                help="Only send alerts when signal strength exceeds this threshold"
            )
            
            # Test SMS button - make it full width for vertical layout
            if st.button("Send Test SMS", use_container_width=True) and st.session_state.phone_number:
                validated_number = validate_phone_number(st.session_state.phone_number)
                if validated_number:
                    result = send_price_alert(
                        validated_number,
                        "TEST",
                        100.00,
                        "TEST",
                        100
                    )
                    if "Error" in result:
                        st.error(result)
                    else:
                        st.success("Test SMS sent successfully!")
    
    # Auto refresh settings
    st.subheader("Auto Refresh")
    st.session_state.auto_refresh = st.checkbox("Enable 15-minute Auto Refresh", value=st.session_state.auto_refresh)
    
    # Manual refresh button - make it full width for vertical layout
    if st.button("Refresh Data Now", use_container_width=True):
        st.session_state.last_update = datetime.now()
        st.session_state.update_counter += 1

# Auto-refresh logic
current_time = datetime.now()
time_diff = (current_time - st.session_state.last_update).total_seconds() / 60

# Check if we should auto-refresh (15 minutes have passed and auto-refresh is enabled)
if st.session_state.auto_refresh and time_diff >= 15:
    st.session_state.last_update = current_time
    st.session_state.update_counter += 1
    st.rerun()

# Check if OpenAI API key was requested from the ML Predictions page
if 'openai_api_requested' in st.session_state and st.session_state.openai_api_requested:
    # Reset the flag
    st.session_state.openai_api_requested = False

# Display last update time if on analysis page
if st.session_state.current_page == 'analysis':
    st.caption(f"Last updated: {st.session_state.last_update.strftime('%Y-%m-%d %H:%M:%S')}")

# Check for achievements when adding stocks to watchlist
check_watchlist_achievements()

def get_current_price(ticker):
    """
    Get the current price for a given ticker symbol
    
    Args:
        ticker: Stock symbol
        
    Returns:
        Current price or None if unable to retrieve
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1d")
        if len(hist) > 0:
            return hist['Close'].iloc[-1]
        return None
    except Exception as e:
        print(f"Error getting current price for {ticker}: {str(e)}")
        return None

# Function to display the integrated Analysis Hub page
def analysis_hub_page():
    """
    Displays the integrated Analysis Hub page that combines watchlists, portfolio tracking,
    buy/sell alerts, and backtesting in one interface.
    """
    st.title("🌟 My Analysis Hub")
    st.markdown("""
    Your personalized analysis center that combines your watchlists, portfolio, signals, and backtesting in one place.
    Get a comprehensive view of your investments and trading opportunities.
    """)
    
    # Create tabs for the different components
    hub_tabs = st.tabs(["Watchlists", "Portfolio", "Buy/Sell Alerts", "Quick Backtest"])
    
    # Tab 1: Watchlists with stock selection for analysis
    with hub_tabs[0]:
        st.subheader("My Watchlists")
        
        # Get all watchlists
        if 'watchlists' in st.session_state:
            all_watchlists = list(st.session_state.watchlists.keys())
            
            # Left column for watchlist selection and management
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # Select watchlist
                if all_watchlists:
                    selected_watchlist = st.selectbox(
                        "Select a watchlist:", 
                        all_watchlists,
                        key="hub_watchlist_select"
                    )
                    
                    # Create a new watchlist
                    new_watchlist_name = st.text_input("Create new watchlist:", key="hub_new_watchlist")
                    if st.button("Create Watchlist", key="hub_create_watchlist") and new_watchlist_name:
                        if new_watchlist_name not in st.session_state.watchlists:
                            st.session_state.watchlists[new_watchlist_name] = []
                            st.success(f"Created watchlist '{new_watchlist_name}'")
                            st.rerun()
                        else:
                            st.error(f"Watchlist '{new_watchlist_name}' already exists")
                else:
                    st.info("No watchlists yet. Create your first watchlist.")
                    new_watchlist_name = st.text_input("Create new watchlist:", key="hub_first_watchlist")
                    if st.button("Create Watchlist", key="hub_create_first_watchlist") and new_watchlist_name:
                        st.session_state.watchlists[new_watchlist_name] = []
                        st.success(f"Created watchlist '{new_watchlist_name}'")
                        st.rerun()
            
            # Right column for watchlist stocks and quick analysis
            with col2:
                if all_watchlists:
                    selected_watchlist = selected_watchlist if 'selected_watchlist' in locals() else all_watchlists[0]
                    watchlist_stocks = st.session_state.watchlists[selected_watchlist]
                    
                    if watchlist_stocks:
                        st.write(f"Stocks in '{selected_watchlist}':")
                        
                        # Display stocks in a table with quick actions
                        for i, stock in enumerate(watchlist_stocks):
                            stock_col1, stock_col2, stock_col3 = st.columns([3, 1, 1])
                            
                            with stock_col1:
                                st.write(f"{stock}")
                                
                            with stock_col2:
                                if st.button("Analyze", key=f"hub_analyze_{stock}_{i}"):
                                    if stock not in st.session_state.selected_stocks:
                                        st.session_state.selected_stocks.append(stock)
                                    st.session_state.current_page = 'analysis'
                                    st.rerun()
                                    
                            with stock_col3:
                                if st.button("Remove", key=f"hub_remove_{stock}_{i}"):
                                    st.session_state.watchlists[selected_watchlist].remove(stock)
                                    st.rerun()
                                    
                        # Add a stock to this watchlist
                        st.write("Add a stock to this watchlist:")
                        stock_input = st.text_input("Stock symbol:", key=f"hub_add_stock_{selected_watchlist}")
                        if st.button("Add Stock", key=f"hub_add_btn_{selected_watchlist}") and stock_input:
                            stock_input = stock_input.strip().upper()
                            if stock_input not in watchlist_stocks:
                                try:
                                    # Verify the stock exists
                                    stock = yf.Ticker(stock_input)
                                    hist = stock.history(period="1d")
                                    if len(hist) > 0:
                                        st.session_state.watchlists[selected_watchlist].append(stock_input)
                                        st.success(f"Added {stock_input} to {selected_watchlist}")
                                        st.rerun()
                                    else:
                                        st.error(f"Could not find stock with symbol {stock_input}")
                                except Exception as e:
                                    st.error(f"Error adding stock: {str(e)}")
                    else:
                        st.info(f"No stocks in '{selected_watchlist}' yet. Add your first stock.")
                        stock_input = st.text_input("Stock symbol:", key=f"hub_first_stock_{selected_watchlist}")
                        if st.button("Add Stock", key=f"hub_add_first_{selected_watchlist}") and stock_input:
                            stock_input = stock_input.strip().upper()
                            try:
                                # Verify the stock exists
                                stock = yf.Ticker(stock_input)
                                hist = stock.history(period="1d")
                                if len(hist) > 0:
                                    st.session_state.watchlists[selected_watchlist].append(stock_input)
                                    st.success(f"Added {stock_input} to {selected_watchlist}")
                                    st.rerun()
                                else:
                                    st.error(f"Could not find stock with symbol {stock_input}")
                            except Exception as e:
                                st.error(f"Error adding stock: {str(e)}")
        else:
            st.warning("Watchlists not initialized.")
    
    # Tab 2: Portfolio summary and management
    with hub_tabs[1]:
        st.subheader("My Portfolio")
        
        # Initialize portfolio if needed
        if 'portfolio' not in st.session_state:
            st.session_state.portfolio = []
            
        # Display current portfolio
        if len(st.session_state.portfolio) > 0:
            # Create a table with portfolio data
            portfolio_data = []
            for position in st.session_state.portfolio:
                # Get current price
                current_price = get_current_price(position['ticker'])
                if current_price:
                    position_value = position['shares'] * current_price
                    gain_loss = position_value - (position['shares'] * position['purchase_price'])
                    gain_loss_pct = (gain_loss / (position['shares'] * position['purchase_price'])) * 100 if position['purchase_price'] > 0 else 0
                    
                    portfolio_data.append({
                        'Ticker': position['ticker'],
                        'Shares': position['shares'],
                        'Purchase Price': f"${position['purchase_price']:.2f}",
                        'Current Price': f"${current_price:.2f}",
                        'Position Value': f"${position_value:.2f}",
                        'Gain/Loss': f"${gain_loss:.2f}",
                        'Gain/Loss %': f"{gain_loss_pct:.2f}%",
                        'Purchase Date': position.get('purchase_date', 'N/A')
                    })
            
            # Create a DataFrame for display
            if portfolio_data:
                portfolio_df = pd.DataFrame(portfolio_data)
                st.dataframe(portfolio_df)
                
                # Total portfolio value and performance
                total_value = sum([float(data['Position Value'].replace('$', '')) for data in portfolio_data])
                total_cost = sum([position['shares'] * position['purchase_price'] for position in st.session_state.portfolio])
                total_gain_loss = total_value - total_cost
                total_gain_loss_pct = (total_gain_loss / total_cost) * 100 if total_cost > 0 else 0
                
                # Display totals
                st.write(f"**Total Portfolio Value:** ${total_value:.2f}")
                st.write(f"**Total Gain/Loss:** ${total_gain_loss:.2f} ({total_gain_loss_pct:.2f}%)")
                
                # Quick actions for portfolio stocks
                st.subheader("Quick Actions")
                for position in st.session_state.portfolio:
                    action_col1, action_col2, action_col3 = st.columns([3, 1, 1])
                    
                    with action_col1:
                        st.write(f"{position['ticker']} ({position['shares']} shares)")
                        
                    with action_col2:
                        if st.button("Analyze", key=f"portfolio_analyze_{position['ticker']}"):
                            if position['ticker'] not in st.session_state.selected_stocks:
                                st.session_state.selected_stocks.append(position['ticker'])
                            st.session_state.current_page = 'analysis'
                            st.rerun()
                            
                    with action_col3:
                        if st.button("Remove", key=f"portfolio_remove_{position['ticker']}"):
                            st.session_state.portfolio = [p for p in st.session_state.portfolio if p['ticker'] != position['ticker']]
                            st.rerun()
            else:
                st.info("Unable to retrieve current prices for portfolio stocks.")
        else:
            st.info("Your portfolio is empty. Add your first position.")
            
        # Add new position form
        st.subheader("Add Position")
        with st.form("add_position_form"):
            ticker = st.text_input("Ticker Symbol", key="portfolio_ticker")
            shares = st.number_input("Number of Shares", min_value=0.1, step=0.1, key="portfolio_shares")
            purchase_price = st.number_input("Purchase Price per Share ($)", min_value=0.01, step=0.01, key="portfolio_price")
            purchase_date = st.date_input("Purchase Date", key="portfolio_date")
            
            submit_button = st.form_submit_button("Add Position")
            
            if submit_button and ticker and shares > 0 and purchase_price > 0:
                ticker = ticker.strip().upper()
                
                # Validate ticker
                try:
                    stock = yf.Ticker(ticker)
                    hist = stock.history(period="1d")
                    if len(hist) > 0:
                        # Check if ticker already exists in portfolio
                        existing_position = next((p for p in st.session_state.portfolio if p['ticker'] == ticker), None)
                        
                        if existing_position:
                            # Update existing position with average price
                            total_shares = existing_position['shares'] + shares
                            total_cost = (existing_position['shares'] * existing_position['purchase_price']) + (shares * purchase_price)
                            avg_price = total_cost / total_shares
                            
                            existing_position['shares'] = total_shares
                            existing_position['purchase_price'] = avg_price
                            existing_position['purchase_date'] = purchase_date.strftime("%Y-%m-%d")
                            
                            st.success(f"Updated position for {ticker}. New total: {total_shares} shares at avg price ${avg_price:.2f}")
                        else:
                            # Add new position
                            st.session_state.portfolio.append({
                                'ticker': ticker,
                                'shares': shares,
                                'purchase_price': purchase_price,
                                'purchase_date': purchase_date.strftime("%Y-%m-%d")
                            })
                            st.success(f"Added {shares} shares of {ticker} at ${purchase_price:.2f}")
                        
                        st.rerun()
                    else:
                        st.error(f"Could not find stock with symbol {ticker}")
                except Exception as e:
                    st.error(f"Error adding position: {str(e)}")
    
    # Tab 3: Buy/Sell Alerts
    with hub_tabs[2]:
        st.subheader("Buy/Sell Alerts")
        
        # Initialize the signal alert manager
        alert_manager = SignalAlertManager()
        
        # Get current alert preferences
        alert_preferences = alert_manager.load_preferences()
        
        # Display alert history
        st.write("### Recent Alerts")
        alert_manager.display_alert_history()
        
        # Alert setup section
        st.write("### Alert Settings")
        
        # Create form for alert preferences
        with st.form("alert_preferences_form"):
            # Enable email alerts
            email_alerts_enabled = st.checkbox(
                "Enable Email Alerts", 
                value=alert_preferences.get('email_enabled', False),
                help="Receive email alerts when buy/sell signals are detected"
            )
            
            # Email settings (only shown if email alerts are enabled)
            if email_alerts_enabled:
                email_address = st.text_input(
                    "Email Address", 
                    value=alert_preferences.get('email', ''),
                    placeholder="your.email@example.com"
                )
            else:
                email_address = alert_preferences.get('email', '')
            
            # Enable SMS alerts
            sms_alerts_enabled = st.checkbox(
                "Enable SMS Alerts", 
                value=alert_preferences.get('sms_enabled', False),
                help="Receive SMS alerts when buy/sell signals are detected"
            )
            
            # SMS settings (only shown if SMS alerts are enabled)
            if sms_alerts_enabled:
                phone_number = st.text_input(
                    "Phone Number (with country code)", 
                    value=alert_preferences.get('phone', ''),
                    placeholder="e.g., +1XXXXXXXXXX"
                )
            else:
                phone_number = alert_preferences.get('phone', '')
            
            # Alert threshold settings
            alert_threshold = st.slider(
                "Minimum Signal Strength for Alerts (%)",
                min_value=25,
                max_value=100,
                value=alert_preferences.get('threshold', 75),
                step=5,
                help="Only send alerts when signal strength exceeds this threshold"
            )
            
            # Alert frequency settings
            cooldown_hours = st.number_input(
                "Hours Between Alerts (Cooldown)",
                min_value=1,
                max_value=72,
                value=alert_preferences.get('cooldown_hours', 24),
                help="Minimum time between alerts for the same stock"
            )
            
            # Select stocks to monitor
            selected_watchlists = []
            if 'watchlists' in st.session_state and st.session_state.watchlists:
                # Multi-select for watchlists to monitor
                all_watchlist_names = list(st.session_state.watchlists.keys())
                selected_watchlists = st.multiselect(
                    "Monitor Watchlists",
                    options=all_watchlist_names,
                    default=alert_preferences.get('watchlists', []),
                    help="Select which watchlists to monitor for signals"
                )
            
            alert_types = st.multiselect(
                "Types of Alerts",
                options=["BUY", "SELL", "STRONG BUY", "STRONG SELL"],
                default=alert_preferences.get('alert_types', ["BUY", "STRONG BUY"]),
                help="Select which types of signals should trigger alerts"
            )
            
            # Submit button
            submit_button = st.form_submit_button("Save Alert Preferences")
            
            if submit_button:
                # Save alert preferences
                new_preferences = {
                    'email_enabled': email_alerts_enabled,
                    'email': email_address,
                    'sms_enabled': sms_alerts_enabled,
                    'phone': phone_number,
                    'threshold': alert_threshold,
                    'cooldown_hours': cooldown_hours,
                    'watchlists': selected_watchlists,
                    'alert_types': alert_types
                }
                
                alert_manager.save_preferences(new_preferences)
                st.success("Alert preferences saved successfully!")
        
        # Test alert section
        st.write("### Test Alerts")
        test_col1, test_col2 = st.columns(2)
        
        with test_col1:
            if st.button("Send Test Email Alert", use_container_width=True):
                if email_alerts_enabled and email_address:
                    if validate_email(email_address):
                        # Create a test email
                        subject = "Test Buy Signal Alert"
                        message_html = create_buy_signal_email(
                            "TEST",
                            100.00,
                            "TEST BUY",
                            95,
                            {
                                "RSI": "28 (Oversold)",
                                "MACD": "Bullish Crossover",
                                "Price": "5% drop in last 3 days"
                            }
                        )
                        
                        # Send the test email
                        result = send_email_alert(email_address, subject, message_html)
                        if "Error" in result:
                            st.error(result)
                        else:
                            st.success("Test email sent successfully!")
                    else:
                        st.error("Please enter a valid email address")
                else:
                    st.error("Email alerts are not enabled or no email address provided")
        
        with test_col2:
            if st.button("Send Test SMS Alert", use_container_width=True):
                if sms_alerts_enabled and phone_number:
                    validated_number = validate_phone_number(phone_number)
                    if validated_number:
                        result = send_price_alert(
                            validated_number,
                            "TEST",
                            100.00,
                            "TEST BUY",
                            95
                        )
                        if "Error" in result:
                            st.error(result)
                        else:
                            st.success("Test SMS sent successfully!")
                    else:
                        st.error("Please enter a valid phone number with country code")
                else:
                    st.error("SMS alerts are not enabled or no phone number provided")
    
    # Tab 4: Quick Backtest
    with hub_tabs[3]:
        st.subheader("Quick Backtest")
        
        # Create form for backtest parameters
        with st.form("quick_backtest_form"):
            # Select stock for backtest
            backtest_ticker_options = []
            
            # Add stocks from watchlists
            if 'watchlists' in st.session_state:
                for watchlist_name, stocks in st.session_state.watchlists.items():
                    for stock in stocks:
                        if stock not in backtest_ticker_options:
                            backtest_ticker_options.append(stock)
            
            # Add stocks from portfolio
            if 'portfolio' in st.session_state:
                for position in st.session_state.portfolio:
                    if position['ticker'] not in backtest_ticker_options:
                        backtest_ticker_options.append(position['ticker'])
            
            # Add stocks from main watchlist if no stocks found
            if not backtest_ticker_options and 'selected_stocks' in st.session_state:
                backtest_ticker_options = st.session_state.selected_stocks
            
            # If still no stocks, add default
            if not backtest_ticker_options:
                backtest_ticker_options = ['AAPL']
            
            # Sort tickers alphabetically
            backtest_ticker_options.sort()
            
            # Stock selection
            backtest_ticker = st.selectbox(
                "Select Stock to Backtest",
                options=backtest_ticker_options,
                index=0
            )
            
            # Strategy selection
            strategy_options = [
                'rsi', 'macd', 'bbands', 'combined', 'meanrev', 
                'momentum', 'breakout', 'dualma', 'volbreakout'
            ]
            
            strategy_names = {
                'rsi': 'RSI Strategy',
                'macd': 'MACD Strategy',
                'bbands': 'Bollinger Bands Strategy',
                'combined': 'Combined Strategy',
                'meanrev': 'Mean Reversion Strategy',
                'momentum': 'Momentum Strategy',
                'breakout': 'Breakout Strategy',
                'dualma': 'Dual Moving Average Strategy',
                'volbreakout': 'Volatility Breakout Strategy'
            }
            
            backtest_strategy = st.selectbox(
                "Select Strategy",
                options=strategy_options,
                format_func=lambda x: strategy_names.get(x, x),
                index=0
            )
            
            # Time period
            col1, col2 = st.columns(2)
            
            with col1:
                start_date = st.date_input(
                    "Start Date",
                    value=(datetime.now() - timedelta(days=365)).date()
                )
            
            with col2:
                end_date = st.date_input(
                    "End Date",
                    value=datetime.now().date()
                )
            
            # Capital and cost settings
            initial_capital = st.number_input(
                "Initial Capital ($)",
                min_value=1000,
                max_value=1000000,
                value=10000,
                step=1000
            )
            
            cost_col1, cost_col2 = st.columns(2)
            
            with cost_col1:
                commission = st.number_input(
                    "Commission (%)",
                    min_value=0.0,
                    max_value=5.0,
                    value=0.1,
                    step=0.1,
                    help="Commission percentage per trade"
                ) / 100.0  # Convert percentage to decimal
            
            with cost_col2:
                slippage = st.number_input(
                    "Slippage (%)",
                    min_value=0.0,
                    max_value=5.0,
                    value=0.1,
                    step=0.1,
                    help="Slippage percentage per trade"
                ) / 100.0  # Convert percentage to decimal
            
            # Submit button
            submit_button = st.form_submit_button("Run Backtest")
        
        # Run backtest if form is submitted
        if submit_button and backtest_ticker and backtest_strategy:
            try:
                with st.spinner(f"Running {strategy_names.get(backtest_strategy, backtest_strategy)} backtest for {backtest_ticker}..."):
                    # Run the backtest
                    results, performance = run_backtest(
                        backtest_ticker,
                        backtest_strategy,
                        start_date,
                        end_date,
                        initial_capital,
                        commission,
                        slippage
                    )
                    
                    # Get benchmark performance
                    benchmark_performance = get_benchmark_performance(
                        backtest_ticker,
                        start_date,
                        end_date,
                        initial_capital,
                        commission,
                        slippage
                    )
                    
                    # Display the results
                    st.write("### Backtest Results")
                    
                    # Performance metrics
                    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                    
                    with metrics_col1:
                        st.metric(
                            "Final Value",
                            f"${performance['final_value']:.2f}",
                            f"{((performance['final_value'] / initial_capital) - 1) * 100:.2f}%"
                        )
                        st.metric(
                            "CAGR",
                            f"{performance['cagr'] * 100:.2f}%"
                        )
                    
                    with metrics_col2:
                        st.metric(
                            "Sharpe Ratio",
                            f"{performance['sharpe_ratio']:.2f}"
                        )
                        st.metric(
                            "Max Drawdown",
                            f"{performance['max_drawdown'] * 100:.2f}%"
                        )
                    
                    with metrics_col3:
                        st.metric(
                            "Total Trades",
                            f"{performance['total_trades']}"
                        )
                        st.metric(
                            "Win Rate",
                            f"{performance['win_rate'] * 100:.2f}%" if 'win_rate' in performance else "N/A"
                        )
                    
                    # Strategy vs Benchmark
                    st.write("### Strategy vs Buy & Hold")
                    compare_col1, compare_col2 = st.columns(2)
                    
                    with compare_col1:
                        st.metric(
                            "Strategy Return",
                            f"{((performance['final_value'] / initial_capital) - 1) * 100:.2f}%"
                        )
                    
                    with compare_col2:
                        st.metric(
                            "Buy & Hold Return",
                            f"{((benchmark_performance['final_value'] / initial_capital) - 1) * 100:.2f}%"
                        )
                    
                    # Display the equity curve
                    st.write("### Equity Curve")
                    if 'equity_curve' in results:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=results['equity_curve'].index,
                            y=results['equity_curve'].values,
                            mode='lines',
                            name='Strategy',
                            line=dict(color='blue', width=2)
                        ))
                        
                        if 'benchmark' in results:
                            fig.add_trace(go.Scatter(
                                x=results['benchmark'].index,
                                y=results['benchmark'].values,
                                mode='lines',
                                name='Buy & Hold',
                                line=dict(color='gray', width=2, dash='dash')
                            ))
                        
                        fig.update_layout(
                            title=f"{backtest_ticker} - {strategy_names.get(backtest_strategy, backtest_strategy)} Performance",
                            xaxis_title="Date",
                            yaxis_title="Portfolio Value ($)",
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Display the drawdown chart
                    st.write("### Drawdown")
                    if 'drawdown' in results:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=results['drawdown'].index,
                            y=results['drawdown'].values * 100,  # Convert to percentage
                            mode='lines',
                            name='Drawdown',
                            line=dict(color='red', width=2),
                            fill='tozeroy'
                        ))
                        
                        fig.update_layout(
                            title=f"{backtest_ticker} - Drawdown Analysis",
                            xaxis_title="Date",
                            yaxis_title="Drawdown (%)",
                            yaxis=dict(tickformat=".2f"),
                            height=300
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Detailed trade history
                    if 'trades' in results and len(results['trades']) > 0:
                        st.write("### Trade History")
                        trades_df = results['trades']
                        st.dataframe(trades_df)
                        
                # Update achievements based on backtest results
                check_backtest_achievements(results, performance, benchmark_performance, backtest_strategy)
                
            except Exception as e:
                st.error(f"Error running backtest: {str(e)}")

# Show the appropriate page content based on the current page
if st.session_state.current_page == 'watchlists':
    # Redirect to Analysis Hub for watchlists
    st.session_state.current_page = 'analysis_hub'
    st.rerun()

elif st.session_state.current_page == 'analysis_hub':
    # My Analysis Hub page
    analysis_hub_page()
    
elif st.session_state.current_page == 'learning':
    # Learning Center page
    learning_section()
    
elif st.session_state.current_page == 'backtesting':
    # Backtesting section moved to its own page
    st.header("📊 Enhanced Backtesting Strategies")
    st.markdown("""
    Test how different trading strategies would have performed historically. 
    Backtest your strategies with various parameters, transaction costs, slippage, and parameter optimization.
    """)
    
    # Initialize backtesting session state variables
    if 'backtest_strategy' not in st.session_state:
        st.session_state.backtest_strategy = 'rsi'
    if 'backtest_ticker' not in st.session_state:
        st.session_state.backtest_ticker = 'AAPL' if 'AAPL' in st.session_state.selected_stocks else st.session_state.selected_stocks[0] if st.session_state.selected_stocks else 'AAPL'
    if 'backtest_start_date' not in st.session_state:
        st.session_state.backtest_start_date = (datetime.now() - timedelta(days=365)).date()
    if 'backtest_end_date' not in st.session_state:
        st.session_state.backtest_end_date = datetime.now().date()
    if 'backtest_initial_capital' not in st.session_state:
        st.session_state.backtest_initial_capital = 10000
    if 'backtest_commission' not in st.session_state:
        st.session_state.backtest_commission = 0.001  # 0.1% commission per trade
    if 'backtest_slippage' not in st.session_state:
        st.session_state.backtest_slippage = 0.001    # 0.1% slippage per trade
    if 'backtest_rsi_period' not in st.session_state:
        st.session_state.backtest_rsi_period = 14
    if 'backtest_rsi_lower' not in st.session_state:
        st.session_state.backtest_rsi_lower = 30
    if 'backtest_rsi_upper' not in st.session_state:
        st.session_state.backtest_rsi_upper = 70
    if 'backtest_macd_fast' not in st.session_state:
        st.session_state.backtest_macd_fast = 12
    if 'backtest_macd_slow' not in st.session_state:
        st.session_state.backtest_macd_slow = 26
    if 'backtest_macd_signal' not in st.session_state:
        st.session_state.backtest_macd_signal = 9
    if 'backtest_bb_period' not in st.session_state:
        st.session_state.backtest_bb_period = 20
    if 'backtest_bb_std' not in st.session_state:
        st.session_state.backtest_bb_std = 2
    if 'backtest_meanrev_lookback' not in st.session_state:
        st.session_state.backtest_meanrev_lookback = 20
    if 'backtest_meanrev_zscore' not in st.session_state:
        st.session_state.backtest_meanrev_zscore = 1.0
    if 'backtest_momentum_period' not in st.session_state:
        st.session_state.backtest_momentum_period = 90
    if 'backtest_momentum_top_pct' not in st.session_state:
        st.session_state.backtest_momentum_top_pct = 25
    if 'backtest_breakout_window' not in st.session_state:
        st.session_state.backtest_breakout_window = 50
    if 'backtest_breakout_threshold' not in st.session_state:
        st.session_state.backtest_breakout_threshold = 2.0
    if 'backtest_dualma_fast' not in st.session_state:
        st.session_state.backtest_dualma_fast = 50
    if 'backtest_dualma_slow' not in st.session_state:
        st.session_state.backtest_dualma_slow = 200
    if 'backtest_volbreak_period' not in st.session_state:
        st.session_state.backtest_volbreak_period = 20
    if 'backtest_volbreak_mult' not in st.session_state:
        st.session_state.backtest_volbreak_mult = 2.0
    if 'backtest_mode' not in st.session_state:
        st.session_state.backtest_mode = "standard"  # standard, optimization, walk_forward
    if 'backtest_optimization_target' not in st.session_state:
        st.session_state.backtest_optimization_target = "sharpe"  # sharpe, total_return, cagr, max_drawdown
    if 'backtest_window_size' not in st.session_state:
        st.session_state.backtest_window_size = 365  # days for walk-forward window
    if 'backtest_step_size' not in st.session_state:
        st.session_state.backtest_step_size = 90    # days for walk-forward step
        
    # Create tabs for different backtesting modes
    backtest_tabs = st.tabs(["Standard Backtest", "Parameter Optimization", "Walk-Forward Analysis"])
    
    with backtest_tabs[0]:
        # Standard backtesting tab
        st.subheader("Standard Backtest")
        st.markdown("Configure your backtest parameters and run a backtest to evaluate strategy performance.")
        
        # Define labels and options outside the form
        strategy_options = ['rsi', 'macd', 'bbands', 'combined', 'meanrev', 'momentum', 'breakout', 'dualma', 'volbreakout']
        strategy_labels = {
            'rsi': 'RSI Strategy', 
            'macd': 'MACD Strategy', 
            'bbands': 'Bollinger Bands Strategy',
            'combined': 'Combined Strategy',
            'meanrev': 'Mean Reversion Strategy',
            'momentum': 'Momentum Strategy',
            'breakout': 'Breakout Strategy',
            'dualma': 'Dual Moving Average Strategy',
            'volbreakout': 'Volatility Breakout Strategy'
        }
        
        # Parameters outside the form
        selected_strategy = st.selectbox(
            "Strategy",
            options=strategy_options,
            format_func=lambda x: strategy_labels.get(x, x),
            index=strategy_options.index(st.session_state.backtest_strategy),
            key='bt_strategy_selector'
        )
        st.session_state.backtest_strategy = selected_strategy
        
        # Stock ticker
        stock_options = st.session_state.selected_stocks if st.session_state.selected_stocks else ['AAPL']
        if st.session_state.backtest_ticker not in stock_options:
            stock_options.append(st.session_state.backtest_ticker)
            
        selected_ticker = st.selectbox(
            "Stock",
            options=stock_options,
            index=stock_options.index(st.session_state.backtest_ticker),
            key='bt_ticker_selector'
        )
        st.session_state.backtest_ticker = selected_ticker
        
        # Date range
        date_cols = st.columns(2)
        with date_cols[0]:
            start_date = st.date_input(
                "Start Date",
                value=st.session_state.backtest_start_date,
                max_value=datetime.now().date() - timedelta(days=30),
                key='bt_start_date'
            )
            st.session_state.backtest_start_date = start_date
        
        with date_cols[1]:
            end_date = st.date_input(
                "End Date",
                value=st.session_state.backtest_end_date,
                min_value=start_date + timedelta(days=30),
                max_value=datetime.now().date(),
                key='bt_end_date'
            )
            st.session_state.backtest_end_date = end_date
            
        # Initial capital and transaction costs
        cost_cols = st.columns(3)
        with cost_cols[0]:
            initial_capital = st.number_input(
                "Initial Capital ($)",
                min_value=1000,
                max_value=1000000,
                value=st.session_state.backtest_initial_capital,
                step=1000,
                key='bt_initial_capital'
            )
            st.session_state.backtest_initial_capital = initial_capital
        
        with cost_cols[1]:
            commission = st.number_input(
                "Commission (%)",
                min_value=0.0,
                max_value=2.0,
                value=st.session_state.backtest_commission * 100,
                step=0.05,
                help="Commission percentage per trade (e.g., 0.1% = 0.001)",
                key="bt_commission"
            ) / 100
            st.session_state.backtest_commission = commission
        
        with cost_cols[2]:
            slippage = st.number_input(
                "Slippage (%)",
                min_value=0.0,
                max_value=2.0,
                value=st.session_state.backtest_slippage * 100,
                step=0.05,
                help="Slippage percentage per trade (e.g., 0.1% = 0.001)",
                key="bt_slippage"
            ) / 100
            st.session_state.backtest_slippage = slippage
            
        # Create a form just for the submit button
        with st.form(key="backtest_simple_form"):
            # Just include the submit button in a minimal form
            submit_backtest = st.form_submit_button("Run Backtest", help="Click to run the backtest with current parameters")
        
        # Display strategy-specific parameters in the sidebar
        if selected_strategy in ['rsi', 'combined']:
            st.subheader("RSI Parameters")
            rsi_period = st.slider(
                "RSI Period",
                min_value=2,
                max_value=30,
                value=st.session_state.backtest_rsi_period,
                key="bt_sidebar_rsi_period"
            )
            st.session_state.backtest_rsi_period = rsi_period
            
            rsi_lower = st.slider(
                "RSI Lower Threshold",
                min_value=10,
                max_value=40,
                value=st.session_state.backtest_rsi_lower,
                key="bt_sidebar_rsi_lower"
            )
            st.session_state.backtest_rsi_lower = rsi_lower
            
            rsi_upper = st.slider(
                "RSI Upper Threshold",
                min_value=60,
                max_value=90,
                value=st.session_state.backtest_rsi_upper,
                key="bt_sidebar_rsi_upper"
            )
            st.session_state.backtest_rsi_upper = rsi_upper
        
        # Add transaction costs and additional parameters
        cost_cols = st.columns(2)
        with cost_cols[0]:
            # Commission
            commission = st.number_input(
                "Commission (%)",
                min_value=0.0,
                max_value=2.0,
                value=st.session_state.backtest_commission * 100,
                step=0.05,
                help="Commission percentage per trade (e.g., 0.1% = 0.001)",
                key="bt_basic_commission"
            ) / 100
            st.session_state.backtest_commission = commission
            
        with cost_cols[1]:
            # Slippage
            slippage = st.number_input(
                "Slippage (%)",
                min_value=0.0,
                max_value=2.0,
                value=st.session_state.backtest_slippage * 100,
                step=0.05,
                help="Slippage percentage per trade (e.g., 0.1% = 0.001)",
                key="bt_basic_slippage"
            ) / 100
            st.session_state.backtest_slippage = slippage
        
        # Create expanders for different strategy parameters
        if selected_strategy in ['rsi', 'combined']:
            with st.expander("RSI Parameters", expanded=selected_strategy == 'rsi'):
                rsi_cols = st.columns(3)
                with rsi_cols[0]:
                    rsi_period = st.slider(
                        "RSI Period",
                        min_value=2,
                        max_value=30,
                        value=st.session_state.backtest_rsi_period,
                        key="bt_expander_rsi_period"
                    )
                    st.session_state.backtest_rsi_period = rsi_period
                
                with rsi_cols[1]:
                    rsi_lower = st.slider(
                        "RSI Lower Threshold",
                        min_value=10,
                        max_value=40,
                        value=st.session_state.backtest_rsi_lower,
                        key="bt_expander_rsi_lower"
                    )
                    st.session_state.backtest_rsi_lower = rsi_lower
                
                with rsi_cols[2]:
                    rsi_upper = st.slider(
                        "RSI Upper Threshold",
                        min_value=60,
                        max_value=90,
                        value=st.session_state.backtest_rsi_upper,
                        key="bt_expander_rsi_upper"
                    )
                    st.session_state.backtest_rsi_upper = rsi_upper
        
        if selected_strategy in ['macd', 'combined']:
            with st.expander("MACD Parameters", expanded=selected_strategy == 'macd'):
                macd_cols = st.columns(3)
                with macd_cols[0]:
                    macd_fast = st.slider(
                        "Fast Period",
                        min_value=5,
                        max_value=20,
                        value=st.session_state.backtest_macd_fast,
                        key="bt_expander_macd_fast"
                    )
                    st.session_state.backtest_macd_fast = macd_fast
                
                with macd_cols[1]:
                    macd_slow = st.slider(
                        "Slow Period",
                        min_value=15,
                        max_value=50,
                        value=st.session_state.backtest_macd_slow,
                        key="bt_expander_macd_slow"
                    )
                    st.session_state.backtest_macd_slow = macd_slow
                
                with macd_cols[2]:
                    macd_signal = st.slider(
                        "Signal Period",
                        min_value=5,
                        max_value=20,
                        value=st.session_state.backtest_macd_signal,
                        key="bt_expander_macd_signal"
                    )
                    st.session_state.backtest_macd_signal = macd_signal
        
        if selected_strategy in ['bbands', 'combined']:
            with st.expander("Bollinger Bands Parameters", expanded=selected_strategy == 'bbands'):
                bb_cols = st.columns(2)
                with bb_cols[0]:
                    bb_period = st.slider(
                        "BB Period",
                        min_value=5,
                        max_value=50,
                        value=st.session_state.backtest_bb_period,
                        key="bt_expander_bb_period"
                    )
                    st.session_state.backtest_bb_period = bb_period
                
                with bb_cols[1]:
                    bb_std = st.slider(
                        "BB Standard Deviations",
                        min_value=1.0,
                        max_value=3.0,
                        value=st.session_state.backtest_bb_std,
                        step=0.1,
                        key="bt_expander_bb_std"
                    )
                    st.session_state.backtest_bb_std = bb_std
                    
        # Add parameters for new strategies
        if selected_strategy == 'meanrev':
            with st.expander("Mean Reversion Parameters", expanded=True):
                meanrev_cols = st.columns(2)
                with meanrev_cols[0]:
                    meanrev_lookback = st.slider(
                        "Lookback Period",
                        min_value=5,
                        max_value=50,
                        value=st.session_state.backtest_meanrev_lookback,
                        key="bt_expander_meanrev_lookback"
                    )
                    st.session_state.backtest_meanrev_lookback = meanrev_lookback
                
                with meanrev_cols[1]:
                    meanrev_zscore = st.slider(
                        "Z-Score Threshold",
                        min_value=0.5,
                        max_value=3.0,
                        value=st.session_state.backtest_meanrev_zscore,
                        step=0.1,
                        help="Z-score threshold for mean reversion entry/exit",
                        key="bt_expander_meanrev_zscore"
                    )
                    st.session_state.backtest_meanrev_zscore = meanrev_zscore
        
        if selected_strategy == 'momentum':
            with st.expander("Momentum Parameters", expanded=True):
                momentum_cols = st.columns(2)
                with momentum_cols[0]:
                    momentum_period = st.slider(
                        "Momentum Period (days)",
                        min_value=30,
                        max_value=180,
                        value=st.session_state.backtest_momentum_period,
                        step=10,
                        key="bt_expander_momentum_period"
                    )
                    st.session_state.backtest_momentum_period = momentum_period
                
                with momentum_cols[1]:
                    momentum_top_pct = st.slider(
                        "Top Percentile Threshold",
                        min_value=5,
                        max_value=50,
                        value=st.session_state.backtest_momentum_top_pct,
                        help="Percentile threshold for selecting top momentum performers",
                        key="bt_expander_momentum_top_pct"
                    )
                    st.session_state.backtest_momentum_top_pct = momentum_top_pct
        
        if selected_strategy == 'breakout':
            with st.expander("Breakout Parameters", expanded=True):
                breakout_cols = st.columns(2)
                with breakout_cols[0]:
                    breakout_window = st.slider(
                        "Resistance Window (days)",
                        min_value=20,
                        max_value=100,
                        value=st.session_state.backtest_breakout_window,
                        step=5,
                        key="bt_expander_breakout_window"
                    )
                    st.session_state.backtest_breakout_window = breakout_window
                
                with breakout_cols[1]:
                    breakout_threshold = st.slider(
                        "Breakout Threshold (%)",
                        min_value=0.5,
                        max_value=5.0,
                        value=st.session_state.backtest_breakout_threshold,
                        step=0.1,
                        help="Percentage above resistance to trigger breakout",
                        key="bt_expander_breakout_threshold"
                    )
                    st.session_state.backtest_breakout_threshold = breakout_threshold
        
        if selected_strategy == 'dualma':
            with st.expander("Dual Moving Average Parameters", expanded=True):
                dualma_cols = st.columns(2)
                with dualma_cols[0]:
                    dualma_fast = st.slider(
                        "Fast MA Period",
                        min_value=20,
                        max_value=100,
                        value=st.session_state.backtest_dualma_fast,
                        step=5,
                        key="bt_expander_dualma_fast"
                    )
                    st.session_state.backtest_dualma_fast = dualma_fast
                
                with dualma_cols[1]:
                    dualma_slow = st.slider(
                        "Slow MA Period",
                        min_value=100,
                        max_value=300,
                        value=st.session_state.backtest_dualma_slow,
                        step=10,
                        key="bt_expander_dualma_slow"
                    )
                    st.session_state.backtest_dualma_slow = dualma_slow
                    
        if selected_strategy == 'volbreakout':
            with st.expander("Volatility Breakout Parameters", expanded=True):
                volbreak_cols = st.columns(2)
                with volbreak_cols[0]:
                    volbreak_period = st.slider(
                        "Lookback Period",
                        min_value=10,
                        max_value=50,
                        value=st.session_state.backtest_volbreak_period,
                        step=5,
                        key="bt_expander_volbreak_period"
                    )
                    st.session_state.backtest_volbreak_period = volbreak_period
                
                with volbreak_cols[1]:
                    volbreak_mult = st.slider(
                        "Volatility Multiplier",
                        min_value=1.0,
                        max_value=4.0,
                        value=st.session_state.backtest_volbreak_mult,
                        step=0.1,
                        help="Multiplier for volatility to set breakout threshold",
                        key="bt_expander_volbreak_mult"
                    )
                    st.session_state.backtest_volbreak_mult = volbreak_mult
        
        # Process backtest when submitted
        if submit_backtest:
            with st.spinner("Running backtest - this may take a moment..."):
                try:
                    # Prepare strategy parameters
                    strategy_params = {}
                    
                    if st.session_state.backtest_strategy in ['rsi', 'combined']:
                        strategy_params.update({
                            'rsi_period': st.session_state.backtest_rsi_period,
                            'rsi_lower': st.session_state.backtest_rsi_lower,
                            'rsi_upper': st.session_state.backtest_rsi_upper
                        })
                    
                    if st.session_state.backtest_strategy in ['macd', 'combined']:
                        strategy_params.update({
                            'fast_period': st.session_state.backtest_macd_fast,
                            'slow_period': st.session_state.backtest_macd_slow,
                            'signal_period': st.session_state.backtest_macd_signal
                        })
                    
                    if st.session_state.backtest_strategy in ['bbands', 'combined']:
                        strategy_params.update({
                            'period': st.session_state.backtest_bb_period,
                            'num_std': st.session_state.backtest_bb_std
                        })
                    
                    if st.session_state.backtest_strategy == 'meanrev':
                        strategy_params.update({
                            'lookback_period': st.session_state.backtest_meanrev_lookback,
                            'z_score_threshold': st.session_state.backtest_meanrev_zscore
                        })
                    
                    if st.session_state.backtest_strategy == 'momentum':
                        strategy_params.update({
                            'momentum_period': st.session_state.backtest_momentum_period,
                            'top_pct': st.session_state.backtest_momentum_top_pct
                        })
                    
                    if st.session_state.backtest_strategy == 'breakout':
                        strategy_params.update({
                            'window': st.session_state.backtest_breakout_window,
                            'threshold_pct': st.session_state.backtest_breakout_threshold
                        })
                    
                    if st.session_state.backtest_strategy == 'dualma':
                        strategy_params.update({
                            'fast_period': st.session_state.backtest_dualma_fast,
                            'slow_period': st.session_state.backtest_dualma_slow
                        })
                    
                    if st.session_state.backtest_strategy == 'volbreakout':
                        strategy_params.update({
                            'lookback_period': st.session_state.backtest_volbreak_period,
                            'volatility_multiplier': st.session_state.backtest_volbreak_mult
                        })
                    
                    # Convert dates to string format
                    start_date_str = st.session_state.backtest_start_date.strftime('%Y-%m-%d')
                    end_date_str = st.session_state.backtest_end_date.strftime('%Y-%m-%d')
                    
                    # Run the backtest
                    backtest_results = run_backtest(
                        st.session_state.backtest_ticker,
                        st.session_state.backtest_strategy,
                        start_date_str,
                        end_date_str,
                        st.session_state.backtest_initial_capital,
                        st.session_state.backtest_commission,
                        st.session_state.backtest_slippage,
                        **strategy_params
                    )
                    
                    # Get benchmark performance for the same period
                    benchmark_results = get_benchmark_performance(
                        st.session_state.backtest_ticker,
                        start_date_str,
                        end_date_str,
                        st.session_state.backtest_initial_capital,
                        st.session_state.backtest_commission,
                        st.session_state.backtest_slippage
                    )
                    
                    # Store results for display
                    st.session_state.backtest_results = backtest_results
                    st.session_state.benchmark_results = benchmark_results
                    
                    # Set flag to indicate results are available
                    st.session_state.backtest_results_available = True
                
                except Exception as e:
                    st.error(f"Error running backtest: {str(e)}")
    
    # Parameter Optimization tab
    with backtest_tabs[1]:
        # Parameter Optimization tab
        st.markdown("""
        ## Parameter Optimization
        
        Use this tab to automatically find the optimal parameters for your selected strategy. 
        The system will test multiple parameter combinations and identify the one that maximizes your chosen performance metric.
        """)
        
        with st.form(key='optimization_form'):
            opt_cols = st.columns(3)
            
            with opt_cols[0]:
                # Strategy selector for optimization
                opt_strategy = st.selectbox(
                    "Strategy to Optimize",
                    options=strategy_options,
                    format_func=lambda x: strategy_labels.get(x, x),
                    index=strategy_options.index(st.session_state.backtest_strategy),
                    key="opt_strategy"
                )
                
                # Stock ticker for optimization
                opt_ticker = st.selectbox(
                    "Stock",
                    options=stock_options,
                    index=stock_options.index(st.session_state.backtest_ticker),
                    key="opt_ticker"
                )
                
                # Optimization target metric
                opt_target = st.selectbox(
                    "Optimization Target",
                    options=["sharpe", "total_return", "cagr", "max_drawdown"],
                    format_func=lambda x: {
                        "sharpe": "Sharpe Ratio", 
                        "total_return": "Total Return", 
                        "cagr": "Annual Return (CAGR)", 
                        "max_drawdown": "Max Drawdown (minimize)"
                    }.get(x, x),
                    index=0,
                    key="opt_target"
                )
                st.session_state.backtest_optimization_target = opt_target
            
            with opt_cols[1]:
                # Date range selectors for optimization
                opt_start_date = st.date_input(
                    "Start Date",
                    value=st.session_state.backtest_start_date,
                    max_value=datetime.now().date() - timedelta(days=30),
                    key="opt_start_date"
                )
                
                opt_end_date = st.date_input(
                    "End Date",
                    value=st.session_state.backtest_end_date,
                    min_value=opt_start_date + timedelta(days=30),
                    max_value=datetime.now().date(),
                    key="opt_end_date"
                )
                
                # Initial capital for optimization
                opt_capital = st.number_input(
                    "Initial Capital ($)",
                    min_value=1000,
                    max_value=1000000,
                    value=st.session_state.backtest_initial_capital,
                    step=1000,
                    key="opt_capital"
                )
            
            with opt_cols[2]:
                # Transaction costs
                opt_commission = st.number_input(
                    "Commission (%)",
                    min_value=0.0,
                    max_value=2.0,
                    value=st.session_state.backtest_commission * 100,
                    step=0.05,
                    help="Commission percentage per trade (e.g., 0.1% = 0.001)",
                    key="opt_commission"
                ) / 100
                
                opt_slippage = st.number_input(
                    "Slippage (%)",
                    min_value=0.0,
                    max_value=2.0,
                    value=st.session_state.backtest_slippage * 100,
                    step=0.05,
                    help="Slippage percentage per trade (e.g., 0.1% = 0.001)",
                    key="opt_slippage"
                ) / 100
            
            # Submit button for optimization
            submit_optimization = st.form_submit_button("Run Parameter Optimization")
        
        if submit_optimization:
            with st.spinner("Running parameter optimization - this may take several minutes..."):
                try:
                    # Convert dates to string format
                    opt_start_date_str = opt_start_date.strftime('%Y-%m-%d')
                    opt_end_date_str = opt_end_date.strftime('%Y-%m-%d')
                    
                    # Call the optimization function
                    optimization_results = optimize_strategy_parameters(
                        opt_ticker,
                        opt_strategy,
                        opt_start_date_str,
                        opt_end_date_str,
                        opt_capital,
                        opt_commission,
                        opt_slippage,
                        opt_target
                    )
                    
                    if optimization_results:
                        # Display optimization results
                        st.subheader("Optimization Results")
                        
                        # Display the best parameters
                        st.markdown("### Best Parameters")
                        
                        # Create a DataFrame for the best parameters
                        best_params = optimization_results.get('best_params', {})
                        best_performance = optimization_results.get('best_performance', {})
                        
                        # Format parameter display based on strategy type
                        if opt_strategy == 'rsi':
                            st.write(f"RSI Period: **{best_params.get('rsi_period')}**")
                            st.write(f"RSI Lower Threshold: **{best_params.get('rsi_lower')}**")
                            st.write(f"RSI Upper Threshold: **{best_params.get('rsi_upper')}**")
                        
                        elif opt_strategy == 'macd':
                            st.write(f"Fast Period: **{best_params.get('fast_period')}**")
                            st.write(f"Slow Period: **{best_params.get('slow_period')}**")
                            st.write(f"Signal Period: **{best_params.get('signal_period')}**")
                        
                        elif opt_strategy == 'bbands':
                            st.write(f"Period: **{best_params.get('period')}**")
                            st.write(f"Standard Deviations: **{best_params.get('num_std')}**")
                        
                        elif opt_strategy == 'meanrev':
                            st.write(f"Lookback Period: **{best_params.get('lookback_period')}**")
                            st.write(f"Z-Score Threshold: **{best_params.get('z_score_threshold')}**")
                        
                        elif opt_strategy == 'momentum':
                            st.write(f"Momentum Period: **{best_params.get('momentum_period')}**")
                            st.write(f"Top Percentile: **{best_params.get('top_pct')}**")
                        
                        elif opt_strategy == 'breakout':
                            st.write(f"Window: **{best_params.get('window')}**")
                            st.write(f"Threshold: **{best_params.get('threshold_pct')}%**")
                        
                        elif opt_strategy == 'dualma':
                            st.write(f"Fast MA Period: **{best_params.get('fast_period')}**")
                            st.write(f"Slow MA Period: **{best_params.get('slow_period')}**")
                        
                        elif opt_strategy == 'volbreakout':
                            st.write(f"Lookback Period: **{best_params.get('lookback_period')}**")
                            st.write(f"Volatility Multiplier: **{best_params.get('volatility_multiplier')}**")
                        
                        # Display performance metrics for the best parameters
                        st.markdown("### Performance with Best Parameters")
                        
                        # Display the performance metrics in a table
                        metrics_df = pd.DataFrame({
                            "Metric": [
                                "Total Return (%)", 
                                "Annual Return (CAGR) (%)", 
                                "Sharpe Ratio", 
                                "Max Drawdown (%)",
                                "Win Rate (%)",
                                "Total Trades"
                            ],
                            "Value": [
                                f"{best_performance.get('total_return', 0)*100:.2f}",
                                f"{best_performance.get('cagr', 0)*100:.2f}",
                                f"{best_performance.get('sharpe', 0):.2f}",
                                f"{best_performance.get('max_drawdown', 0)*100:.2f}",
                                f"{best_performance.get('win_rate', 0)*100:.2f}",
                                f"{best_performance.get('total_trades', 0)}"
                            ]
                        })
                        
                        st.table(metrics_df)
                        
                        # Display parameter search space visualization if available
                        if 'parameter_space' in optimization_results:
                            st.markdown("### Parameter Search Space")
                            
                            param_space = optimization_results['parameter_space']
                            if len(param_space) > 0:
                                # Choose the two most important parameters to plot
                                plot_params = list(param_space[0].keys())
                                # Remove the performance metric key
                                if opt_target in plot_params:
                                    plot_params.remove(opt_target)
                                
                                if len(plot_params) >= 2:
                                    param1 = plot_params[0]
                                    param2 = plot_params[1]
                                    
                                    # Create scatter plot of parameter space
                                    fig = go.Figure()
                                    
                                    x_vals = [p[param1] for p in param_space]
                                    y_vals = [p[param2] for p in param_space]
                                    z_vals = [p.get(opt_target, 0) for p in param_space]
                                    
                                    # Create custom color scale (reverse for max_drawdown since lower is better)
                                    if opt_target == 'max_drawdown':
                                        z_vals = [-z for z in z_vals]  # Invert for visualization
                                    
                                    fig.add_trace(
                                        go.Scatter(
                                            x=x_vals,
                                            y=y_vals,
                                            mode='markers',
                                            marker=dict(
                                                size=10,
                                                color=z_vals,
                                                colorscale='Viridis',
                                                showscale=True,
                                                colorbar=dict(
                                                    title=opt_target
                                                )
                                            ),
                                            text=[f"{opt_target}: {z:.4f}" for z in z_vals],
                                            hoverinfo='text'
                                        )
                                    )
                                    
                                    fig.update_layout(
                                        title=f"Parameter Optimization Results for {opt_strategy.capitalize()} Strategy",
                                        xaxis_title=param1,
                                        yaxis_title=param2,
                                        height=500
                                    )
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                        
                        # Option to apply these parameters
                        if st.button("Apply Optimal Parameters to Standard Backtest"):
                            # Update session state with optimal parameters
                            st.session_state.backtest_strategy = opt_strategy
                            st.session_state.backtest_ticker = opt_ticker
                            
                            # Update strategy-specific parameters
                            if opt_strategy == 'rsi':
                                st.session_state.backtest_rsi_period = best_params.get('rsi_period', st.session_state.backtest_rsi_period)
                                st.session_state.backtest_rsi_lower = best_params.get('rsi_lower', st.session_state.backtest_rsi_lower)
                                st.session_state.backtest_rsi_upper = best_params.get('rsi_upper', st.session_state.backtest_rsi_upper)
                            
                            elif opt_strategy == 'macd':
                                st.session_state.backtest_macd_fast = best_params.get('fast_period', st.session_state.backtest_macd_fast)
                                st.session_state.backtest_macd_slow = best_params.get('slow_period', st.session_state.backtest_macd_slow)
                                st.session_state.backtest_macd_signal = best_params.get('signal_period', st.session_state.backtest_macd_signal)
                            
                            elif opt_strategy == 'bbands':
                                st.session_state.backtest_bb_period = best_params.get('period', st.session_state.backtest_bb_period)
                                st.session_state.backtest_bb_std = best_params.get('num_std', st.session_state.backtest_bb_std)
                            
                            elif opt_strategy == 'meanrev':
                                st.session_state.backtest_meanrev_lookback = best_params.get('lookback_period', st.session_state.backtest_meanrev_lookback)
                                st.session_state.backtest_meanrev_zscore = best_params.get('z_score_threshold', st.session_state.backtest_meanrev_zscore)
                            
                            elif opt_strategy == 'momentum':
                                st.session_state.backtest_momentum_period = best_params.get('momentum_period', st.session_state.backtest_momentum_period)
                                st.session_state.backtest_momentum_top_pct = best_params.get('top_pct', st.session_state.backtest_momentum_top_pct)
                            
                            elif opt_strategy == 'breakout':
                                st.session_state.backtest_breakout_window = best_params.get('window', st.session_state.backtest_breakout_window)
                                st.session_state.backtest_breakout_threshold = best_params.get('threshold_pct', st.session_state.backtest_breakout_threshold)
                            
                            elif opt_strategy == 'dualma':
                                st.session_state.backtest_dualma_fast = best_params.get('fast_period', st.session_state.backtest_dualma_fast)
                                st.session_state.backtest_dualma_slow = best_params.get('slow_period', st.session_state.backtest_dualma_slow)
                            
                            elif opt_strategy == 'volbreakout':
                                st.session_state.backtest_volbreak_period = best_params.get('lookback_period', st.session_state.backtest_volbreak_period)
                                st.session_state.backtest_volbreak_mult = best_params.get('volatility_multiplier', st.session_state.backtest_volbreak_mult)
                            
                            # Notify the user
                            st.success("Optimal parameters applied! Go to the Standard Backtest tab and run the backtest to see results.")
                            st.balloons()
                    else:
                        st.error("Optimization failed. Please check your parameters and try again.")
                        
                except Exception as e:
                    st.error(f"Error running parameter optimization: {str(e)}")
    
    # Walk-Forward Analysis tab
    with backtest_tabs[2]:
        st.markdown("""
        ## Walk-Forward Analysis
        
        Walk-Forward Analysis evaluates strategy robustness by testing if optimized parameters continue to work on unseen data.
        This helps identify if your strategy is truly reliable or just fitted to historical data.
        """)
        
        with st.form(key='walkforward_form'):
            wf_cols = st.columns(3)
            
            with wf_cols[0]:
                # Strategy selector for walk-forward
                wf_strategy = st.selectbox(
                    "Strategy to Analyze",
                    options=strategy_options,
                    format_func=lambda x: strategy_labels.get(x, x),
                    index=strategy_options.index(st.session_state.backtest_strategy),
                    key="wf_strategy"
                )
                
                # Stock ticker for walk-forward
                wf_ticker = st.selectbox(
                    "Stock",
                    options=stock_options,
                    index=stock_options.index(st.session_state.backtest_ticker),
                    key="wf_ticker"
                )
                
                # Optimization target
                wf_target = st.selectbox(
                    "Optimization Target",
                    options=["sharpe", "total_return", "cagr", "max_drawdown"],
                    format_func=lambda x: {
                        "sharpe": "Sharpe Ratio", 
                        "total_return": "Total Return", 
                        "cagr": "Annual Return (CAGR)", 
                        "max_drawdown": "Max Drawdown (minimize)"
                    }.get(x, x),
                    index=0,
                    key="wf_target"
                )
                
            with wf_cols[1]:
                # Date range selectors for walk-forward
                wf_start_date = st.date_input(
                    "Start Date",
                    value=(datetime.now() - timedelta(days=365*2)).date(),  # 2 years of data for walk-forward
                    max_value=datetime.now().date() - timedelta(days=365),  # Need at least 1 year
                    key="wf_start_date"
                )
                
                wf_end_date = st.date_input(
                    "End Date",
                    value=datetime.now().date(),
                    min_value=wf_start_date + timedelta(days=365),  # Need at least 1 year
                    max_value=datetime.now().date(),
                    key="wf_end_date"
                )
                
                # Window and step size
                wf_window_size = st.slider(
                    "Window Size (days)",
                    min_value=90,
                    max_value=365,
                    value=st.session_state.backtest_window_size,
                    step=30,
                    help="Size of each analysis window in days",
                    key="wf_window_size"
                )
                st.session_state.backtest_window_size = wf_window_size
                
                wf_step_size = st.slider(
                    "Step Size (days)",
                    min_value=30,
                    max_value=180,
                    value=st.session_state.backtest_step_size,
                    step=15,
                    help="Number of days between consecutive windows",
                    key="wf_step_size"
                )
                st.session_state.backtest_step_size = wf_step_size
            
            with wf_cols[2]:
                # Initial capital
                wf_capital = st.number_input(
                    "Initial Capital ($)",
                    min_value=1000,
                    max_value=1000000,
                    value=st.session_state.backtest_initial_capital,
                    step=1000,
                    key="wf_capital"
                )
                
                # Transaction costs
                wf_commission = st.number_input(
                    "Commission (%)",
                    min_value=0.0,
                    max_value=2.0,
                    value=st.session_state.backtest_commission * 100,
                    step=0.05,
                    help="Commission percentage per trade (e.g., 0.1% = 0.001)",
                    key="wf_commission"
                ) / 100
                
                wf_slippage = st.number_input(
                    "Slippage (%)",
                    min_value=0.0,
                    max_value=2.0,
                    value=st.session_state.backtest_slippage * 100,
                    step=0.05,
                    help="Slippage percentage per trade (e.g., 0.1% = 0.001)",
                    key="wf_slippage"
                ) / 100
            
            # Submit button for walk-forward
            submit_walkforward = st.form_submit_button("Run Walk-Forward Analysis")
        
        if submit_walkforward:
            with st.spinner("Running walk-forward analysis - this may take several minutes..."):
                try:
                    # Convert dates to string format
                    wf_start_date_str = wf_start_date.strftime('%Y-%m-%d')
                    wf_end_date_str = wf_end_date.strftime('%Y-%m-%d')
                    
                    # Call the walk-forward optimization function
                    wf_results = walk_forward_optimization(
                        wf_ticker,
                        wf_strategy,
                        wf_start_date_str,
                        wf_end_date_str,
                        wf_window_size,
                        wf_step_size,
                        wf_capital,
                        wf_commission,
                        wf_slippage,
                        wf_target
                    )
                    
                    if wf_results:
                        # Display walk-forward results
                        st.subheader("Walk-Forward Analysis Results")
                        
                        # Display summary statistics
                        st.markdown("### Overall Performance")
                        
                        # Extract key metrics
                        total_windows = wf_results.get('total_windows', 0)
                        successful_windows = wf_results.get('successful_windows', 0)
                        success_rate = wf_results.get('success_rate', 0)
                        avg_return = wf_results.get('average_return', 0)
                        
                        # Create metrics display
                        wf_cols = st.columns(4)
                        with wf_cols[0]:
                            st.metric("Success Rate", f"{success_rate*100:.1f}%", help="Percentage of windows where the strategy outperformed buy & hold")
                        
                        with wf_cols[1]:
                            st.metric("Total Windows", total_windows, help="Number of test windows analyzed")
                        
                        with wf_cols[2]:
                            st.metric("Successful Windows", successful_windows, help="Windows where strategy outperformed benchmark")
                        
                        with wf_cols[3]:
                            st.metric("Avg. Return", f"{avg_return*100:.2f}%", help="Average return across all windows")
                        
                        # Display window-by-window results
                        st.markdown("### Window-by-Window Results")
                        
                        windows = wf_results.get('windows', [])
                        if windows:
                            # Create DataFrame for window results
                            window_data = []
                            for window in windows:
                                window_data.append({
                                    "Window": f"{window.get('in_sample_start')} to {window.get('out_sample_end')}",
                                    "Strategy Return (%)": f"{window.get('strategy_return', 0)*100:.2f}",
                                    "Benchmark Return (%)": f"{window.get('benchmark_return', 0)*100:.2f}",
                                    "Difference (%)": f"{(window.get('strategy_return', 0) - window.get('benchmark_return', 0))*100:.2f}",
                                    "Success": "✅" if window.get('success', False) else "❌"
                                })
                            
                            window_df = pd.DataFrame(window_data)
                            st.table(window_df)
                        
                        # Display strategy robustness graph
                        st.markdown("### Strategy Performance Over Time")
                        
                        # Create the performance comparison chart
                        if 'cumulative_strategy' in wf_results and 'cumulative_benchmark' in wf_results:
                            fig = go.Figure()
                            
                            # Add strategy performance
                            fig.add_trace(
                                go.Scatter(
                                    x=list(range(len(wf_results['cumulative_strategy']))),
                                    y=wf_results['cumulative_strategy'],
                                    mode='lines',
                                    name=f"{wf_strategy.capitalize()} Strategy",
                                    line=dict(color='blue', width=2)
                                )
                            )
                            
                            # Add benchmark performance
                            fig.add_trace(
                                go.Scatter(
                                    x=list(range(len(wf_results['cumulative_benchmark']))),
                                    y=wf_results['cumulative_benchmark'],
                                    mode='lines',
                                    name='Buy & Hold',
                                    line=dict(color='gray', width=2, dash='dash')
                                )
                            )
                            
                            # Update the layout
                            fig.update_layout(
                                title=f"Walk-Forward Performance: {wf_strategy.capitalize()} Strategy vs Buy & Hold",
                                xaxis_title="Window Number",
                                yaxis_title="Cumulative Return (%)",
                                height=500,
                                legend=dict(
                                    yanchor="top",
                                    y=0.99,
                                    xanchor="left",
                                    x=0.01
                                )
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Display parameter stability analysis
                        if 'parameter_stability' in wf_results:
                            st.markdown("### Parameter Stability Analysis")
                            st.write("This chart shows how optimal parameters changed across different time windows, indicating strategy robustness.")
                            
                            param_stability = wf_results['parameter_stability']
                            
                            # Create parameter stability charts for each parameter
                            for param_name, param_values in param_stability.items():
                                fig = go.Figure()
                                
                                fig.add_trace(
                                    go.Scatter(
                                        x=list(range(len(param_values))),
                                        y=param_values,
                                        mode='lines+markers',
                                        name=param_name,
                                        line=dict(width=2)
                                    )
                                )
                                
                                fig.update_layout(
                                    title=f"Optimal {param_name} Across Windows",
                                    xaxis_title="Window Number",
                                    yaxis_title="Parameter Value",
                                    height=300
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                        
                        # Conclusion and recommendation
                        st.markdown("### Conclusion")
                        if success_rate >= 0.6:  # 60% success rate threshold
                            st.success(f"✅ The {wf_strategy} strategy shows good robustness with a {success_rate*100:.1f}% success rate across different time periods.")
                            st.markdown("**Recommendation**: This strategy appears reliable and suitable for live trading, demonstrating consistent performance across different market conditions.")
                        else:
                            st.warning(f"⚠️ The {wf_strategy} strategy shows limited robustness with only a {success_rate*100:.1f}% success rate across different time periods.")
                            st.markdown("**Recommendation**: Consider refining this strategy or exploring alternative approaches, as it may be overfitted to specific historical data.")
                    else:
                        st.error("Walk-forward analysis failed. Please check your parameters and try again.")
                        
                except Exception as e:
                    st.error(f"Error running walk-forward analysis: {str(e)}")

    # Redirect to Analysis Hub for portfolio
    st.session_state.current_page = 'analysis_hub'
    st.rerun()
    
elif st.session_state.current_page == 'signals':
    # Signal tracking section
    signal_tracker_section()

elif st.session_state.current_page == 'alerts':
    # Signal alerts setup section
    signal_alerts_section()

elif st.session_state.current_page == 'sentiment':
    # Create tabs for different sentiment visualization methods
    sentiment_tabs = st.tabs(["Sentiment Tracker", "Mood Ring Visualization"])
    
    with sentiment_tabs[0]:
        # Original sentiment tracker section
        sentiment_tracker_section()
    
    with sentiment_tabs[1]:
        # New animated mood ring visualization
        mood_ring_section()

# Sentiment waves is now part of the Sentiment tab

elif st.session_state.current_page == 'ml_predictions':
    # ML Predictions section
    ml_predictions_section()

elif st.session_state.current_page == 'social':
    # Social trading features section
    social_features_section()

elif st.session_state.current_page == 'advanced_viz':
    # Advanced Visualization section
    advanced_visualization_section()

# Removed the separate handler for fundamental analysis since we're integrating it into Stock Analysis

elif st.session_state.current_page == 'analysis':
    # Analysis section is the default view with current stock monitoring
    st.header("📊 Stock Analysis")
    st.markdown("""
    Monitor stocks with technical indicators and receive signals when good buying opportunities arise.
    Customize your watchlist, adjust technical parameters, and get SMS alerts for important signals.
    """)
    
    # Create tabs for technical and fundamental analysis
    analysis_tabs = st.tabs(["Technical Analysis", "Fundamental Analysis"])
    
    # Initialize fundamental_ticker in session state
    if 'fundamental_ticker' not in st.session_state:
        st.session_state.fundamental_ticker = st.session_state.selected_stocks[0] if st.session_state.selected_stocks else 'AAPL'
        
    # Variable to track the active tab
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = 0
        
    # Add the Fundamental Analysis tab after the Technical Analysis Tab
    with analysis_tabs[1]:  # Fundamental Analysis Tab
        fundamental_analysis_section()
    
    with analysis_tabs[0]:  # Technical Analysis Tab
        # Create a container for selecting stocks from watchlists and portfolio
        stock_selection_container = st.container()
        with stock_selection_container:
            st.subheader("Select Stocks from Watchlists or Portfolio")
            
            # Create columns for watchlist and portfolio
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("##### Watchlists")
                # Get all watchlists
                if 'watchlists' in st.session_state:
                    all_watchlists = list(st.session_state.watchlists.keys())
                    if all_watchlists:
                        selected_watchlist = st.selectbox(
                            "Select a watchlist:", 
                            all_watchlists,
                            key="analysis_watchlist_select"
                        )
                        
                        # Show tickers in the selected watchlist
                        if selected_watchlist in st.session_state.watchlists:
                            watchlist_tickers = st.session_state.watchlists[selected_watchlist]
                            if watchlist_tickers:
                                selected_ticker = st.selectbox(
                                    "Select a stock:", 
                                    watchlist_tickers,
                                    key="analysis_watchlist_ticker"
                                )
                                
                                if selected_ticker and selected_ticker not in st.session_state.selected_stocks:
                                    if st.button("Add to Analysis", key="add_from_watchlist"):
                                        st.session_state.selected_stocks.append(selected_ticker)
                                        st.rerun()
                            else:
                                st.info("No stocks in this watchlist.")
                    else:
                        st.info("No watchlists created yet.")
                else:
                    st.info("Watchlists not initialized.")
            
            with col2:
                st.write("##### Portfolio")
                # Check if portfolio exists
                if 'portfolio' in st.session_state and st.session_state.portfolio:
                    # Get list of tickers from portfolio
                    portfolio_tickers = [stock['ticker'] for stock in st.session_state.portfolio]
                    if portfolio_tickers:
                        selected_portfolio_ticker = st.selectbox(
                            "Select a stock from your portfolio:", 
                            portfolio_tickers,
                            key="analysis_portfolio_ticker"
                        )
                        
                        if selected_portfolio_ticker and selected_portfolio_ticker not in st.session_state.selected_stocks:
                            if st.button("Add to Analysis", key="add_from_portfolio"):
                                st.session_state.selected_stocks.append(selected_portfolio_ticker)
                                st.rerun()
                    else:
                        st.info("No stocks in your portfolio.")
                else:
                    st.info("Portfolio not initialized or empty.")
        
        # Display separator
        st.markdown("---")
        
        # Fetch data for all selected stocks
        for ticker in st.session_state.selected_stocks:
            try:
                # Get stock data
                df = get_stock_data(ticker, period=st.session_state.time_period)
            
                if df is not None and len(df) > 0:
                    # Calculate RSI
                    df['rsi'] = calculate_rsi(df['Close'], period=st.session_state.rsi_period)
                
                    # Calculate standard indicators if enabled
                    if st.session_state.show_macd:
                        df['macd'], df['macd_signal'], df['macd_hist'] = calculate_macd(
                            df['Close'], 
                            fast_period=st.session_state.macd_fast_period,
                            slow_period=st.session_state.macd_slow_period,
                            signal_period=st.session_state.macd_signal_period
                        )
                
                    if st.session_state.show_bollinger:
                        df['bb_upper'], df['bb_middle'], df['bb_lower'] = calculate_bollinger_bands(
                            df['Close'], 
                            period=st.session_state.bb_period,
                            num_std=st.session_state.bb_std
                        )
                
                    if st.session_state.show_ema:
                        df['ema'] = calculate_ema(df['Close'], period=st.session_state.ema_period)
                
                    # Calculate advanced indicators if enabled
                    if st.session_state.show_stochastic:
                        df['stoch_k'], df['stoch_d'] = calculate_stochastic_oscillator(
                            df['High'], 
                            df['Low'], 
                            df['Close'],
                            k_period=st.session_state.stoch_k_period,
                            d_period=st.session_state.stoch_d_period
                        )
                
                    if st.session_state.show_adx:
                        df['adx'], df['plus_di'], df['minus_di'] = calculate_adx(
                            df['High'], 
                            df['Low'], 
                            df['Close'],
                            period=st.session_state.adx_period
                        )
                
                    if st.session_state.show_obv:
                        df['obv'] = calculate_obv(df['Close'], df['Volume'])
                        # Calculate OBV slope for signal generation (using last 5 periods)
                        if len(df) >= 5:
                            obv_recent = df['obv'].iloc[-5:].values
                            df['obv_slope'] = np.polyfit(np.arange(len(obv_recent)), obv_recent, 1)[0]
                        else:
                            df['obv_slope'] = 0
                
                    # Detect chart patterns if enabled
                    if st.session_state.show_pattern_recognition and len(df) >= 20:
                        patterns = detect_chart_patterns(df)
                        # Store the detected patterns in the dataframe
                        for pattern_name, pattern_value in patterns.items():
                            if pattern_name not in ['support_levels', 'resistance_levels']:
                                df[f'pattern_{pattern_name}'] = pattern_value
                        
                        # Store the full patterns dictionary for later reference
                        df.attrs['patterns'] = patterns
                    
                    # Create custom indicator if enabled
                    if st.session_state.show_custom_indicator:
                        # Collect all available indicators for the custom indicator
                        available_indicators = []
                        weights = []
                        
                        if 'rsi' in df.columns:
                            available_indicators.append('rsi')
                            weights.append(st.session_state.custom_indicator_weights['rsi'])
                        
                        if 'macd' in df.columns:
                            # Normalize MACD to 0-100 range for custom indicator
                            macd_range = df['macd'].max() - df['macd'].min() if len(df['macd']) > 0 else 1
                            if macd_range > 0:
                                df['macd_norm'] = ((df['macd'] - df['macd'].min()) / macd_range) * 100
                                available_indicators.append('macd_norm')
                                weights.append(st.session_state.custom_indicator_weights['macd'])
                        
                        if 'stoch_k' in df.columns:
                            available_indicators.append('stoch_k')
                            weights.append(st.session_state.custom_indicator_weights['stoch'])
                        
                        if 'obv_slope' in df.columns:
                            # Normalize OBV slope to 0-100 range
                            obv_slope_max = max(abs(df['obv_slope'].max() if len(df) > 0 else 1), 1)
                            df['obv_slope_norm'] = ((df['obv_slope'] / obv_slope_max) + 1) * 50  # Center at 50
                            available_indicators.append('obv_slope_norm')
                            weights.append(st.session_state.custom_indicator_weights['obv'])
                        
                        if 'bb_lower' in df.columns and 'bb_upper' in df.columns:
                            # Calculate position within Bollinger Bands (0=lower, 100=upper)
                            df['bb_position'] = ((df['Close'] - df['bb_lower']) / 
                                              (df['bb_upper'] - df['bb_lower'])) * 100
                            df['bb_position'] = df['bb_position'].clip(0, 100)  # Ensure values are within 0-100
                            available_indicators.append('bb_position')
                            weights.append(st.session_state.custom_indicator_weights['bb'])
                        
                        # Create custom indicator if we have at least 2 indicators
                        if len(available_indicators) >= 2:
                            df['custom_indicator'] = create_custom_indicator(
                                df,
                                available_indicators,
                                weights
                            )
                
                    # Store in session state
                    st.session_state.stocks_data[ticker] = df
                    
                    # Initialize signal tracker if not done yet
                    initialize_signal_tracker()
                    
                    # Detect and record potential buy signals for the signal tracker
                    detect_and_record_signals(
                        ticker,
                        df,
                        rsi_threshold=30,
                        drop_threshold=st.session_state.drop_threshold
                    )
                    
                    # Calculate last signal time if not already in session state
                    if ticker not in st.session_state.last_signals:
                        days_since_signal, signal_date = get_last_signal_time(
                            ticker, 
                            rsi_threshold=30,
                            drop_threshold=st.session_state.drop_threshold
                        )
                        st.session_state.last_signals[ticker] = {
                            'days': days_since_signal,
                            'date': signal_date
                        }
                else:
                    st.warning(f"Could not retrieve data for {ticker}")
            
            except Exception as e:
                st.error(f"Error processing {ticker}: {str(e)}")
    
        # Display stocks data with technical indicators
        for ticker in st.session_state.selected_stocks:
            if ticker in st.session_state.stocks_data and len(st.session_state.stocks_data[ticker]) > 0:
                df = st.session_state.stocks_data[ticker]
                
                # Get the current price and RSI
                current_price = df['Close'].iloc[-1]
                current_rsi = df['rsi'].iloc[-1]
                
                # Calculate daily change
                if len(df) > 1:
                    daily_change = ((df['Close'].iloc[-1] / df['Close'].iloc[-2]) - 1) * 100
                else:
                    daily_change = 0
                
                # Get additional indicators if available
                macd_line = df['macd'].iloc[-1] if st.session_state.show_macd else None
                macd_signal = df['macd_signal'].iloc[-1] if st.session_state.show_macd else None
                
                bb_upper = df['bb_upper'].iloc[-1] if st.session_state.show_bollinger else None
                bb_middle = df['bb_middle'].iloc[-1] if st.session_state.show_bollinger else None
                bb_lower = df['bb_lower'].iloc[-1] if st.session_state.show_bollinger else None
                
                # Get advanced indicator values if available
                stoch_k = df['stoch_k'].iloc[-1] if st.session_state.show_stochastic and 'stoch_k' in df.columns else None
                stoch_d = df['stoch_d'].iloc[-1] if st.session_state.show_stochastic and 'stoch_d' in df.columns else None
                adx = df['adx'].iloc[-1] if st.session_state.show_adx and 'adx' in df.columns else None
                obv_slope = df['obv_slope'].iloc[-1] if st.session_state.show_obv and 'obv_slope' in df.columns else None
                custom_indicator = df['custom_indicator'].iloc[-1] if st.session_state.show_custom_indicator and 'custom_indicator' in df.columns else None
                
                # Check for buy signal
                buy_signal_text = check_buy_signal(
                    ticker,
                    current_price,
                    current_rsi,
                    daily_change,
                    st.session_state.drop_threshold,
                    macd_line,
                    macd_signal,
                    bb_upper,
                    bb_middle,
                    bb_lower,
                    stoch_k,
                    stoch_d,
                    adx,
                    obv_slope,
                    custom_indicator
                )
                
                # Extract signal strength if available
                signal_strength = None
                if "Signal Strength" in buy_signal_text:
                    try:
                        # Extract signal strength percentage from the text
                        signal_part = buy_signal_text.split("Signal Strength:")[1].strip()
                        signal_strength = int(signal_part.split("%")[0].strip())
                    except:
                        signal_strength = None
                
                # Send SMS alert for strong buy signals if enabled
                if (st.session_state.sms_alerts_enabled and 
                    st.session_state.phone_number and 
                    "BUY SIGNAL" in buy_signal_text and
                    signal_strength is not None and
                    signal_strength >= st.session_state.sms_signal_threshold):
                    
                    validated_number = validate_phone_number(st.session_state.phone_number)
                    if validated_number:
                        send_price_alert(
                            validated_number,
                            ticker,
                            current_price,
                            "BUY",
                            signal_strength
                        )
                
                # Create collapsible stock card with ticker and price in the header
                with st.expander(f"{ticker} - ${current_price:.2f} ({'↑' if daily_change >= 0 else '↓'}{abs(daily_change):.2f}%)", expanded=True):
                    # Create columns for metrics and charts
                    col1, col2 = st.columns([2, 3])
                    
                    with col1:
                        # Display signal message
                        if "BUY SIGNAL" in buy_signal_text:
                            st.success(buy_signal_text)
                            
                            # Create a clickable link to Yahoo Finance for more details
                            st.markdown(f"[View {ticker} on Yahoo Finance](https://finance.yahoo.com/quote/{ticker})")
                        else:
                            if "No clear signal" in buy_signal_text:
                                st.info(buy_signal_text)
                            else:
                                st.warning(buy_signal_text)
                        # Technical metrics
                        st.metric("RSI (14)", f"{current_rsi:.2f}", 
                                 f"{'Oversold' if current_rsi < 30 else 'Overbought' if current_rsi > 70 else 'Neutral'}")
                        
                        if st.session_state.show_macd and 'macd' in df.columns:
                            st.metric("MACD", f"{df['macd'].iloc[-1]:.4f}", 
                                     f"{df['macd'].iloc[-1] - df['macd_signal'].iloc[-1]:.4f} from Signal")
                        
                        # Show advanced indicator values
                        if st.session_state.show_stochastic and 'stoch_k' in df.columns:
                            st.metric("Stochastic %K", f"{df['stoch_k'].iloc[-1]:.2f}", 
                                     f"%D: {df['stoch_d'].iloc[-1]:.2f}")
                        
                        if st.session_state.show_adx and 'adx' in df.columns:
                            st.metric("ADX", f"{df['adx'].iloc[-1]:.2f}", 
                                     f"{'Strong' if df['adx'].iloc[-1] > 25 else 'Weak'} Trend")
                        
                        if st.session_state.show_obv and 'obv_slope' in df.columns:
                            st.metric("OBV Slope", f"{df['obv_slope'].iloc[-1]:.2f}", 
                                     f"{'Positive' if df['obv_slope'].iloc[-1] > 0 else 'Negative'} Volume Trend")
                        
                        # Show pattern recognition results if enabled
                        if st.session_state.show_pattern_recognition:
                            detected_patterns = []
                            for col in df.columns:
                                if col.startswith('pattern_') and df[col].iloc[-1] == True:
                                    pattern_name = col.replace('pattern_', '').replace('_', ' ').title()
                                    detected_patterns.append(pattern_name)
                            
                            if detected_patterns:
                                st.subheader("Detected Patterns:")
                                for pattern in detected_patterns:
                                    st.info(f"✓ {pattern}")
                            
                            # Check for stored patterns in the dataframe attributes
                            if hasattr(df, 'attrs') and 'patterns' in df.attrs:
                                patterns_data = df.attrs['patterns']
                                
                                # Show support/resistance levels if available
                                if "support_levels" in patterns_data and patterns_data["support_levels"]:
                                    st.subheader("Support Levels:")
                                    for level in patterns_data["support_levels"]:
                                        st.text(f"${level:.2f}")
                                
                                if "resistance_levels" in patterns_data and patterns_data["resistance_levels"]:
                                    st.subheader("Resistance Levels:")
                                    for level in patterns_data["resistance_levels"]:
                                        st.text(f"${level:.2f}")
                        
                        # Show custom indicator if enabled
                        if st.session_state.show_custom_indicator and 'custom_indicator' in df.columns:
                            custom_value = df['custom_indicator'].iloc[-1]
                            signal_text = "Strong Buy" if custom_value > 70 else "Moderate Buy" if custom_value > 50 else "Neutral" if custom_value > 30 else "Sell/Avoid"
                            st.metric("Custom Indicator", f"{custom_value:.2f}", signal_text)
                        
                        # Display last signal time
                        if ticker in st.session_state.last_signals:
                            last_signal = st.session_state.last_signals[ticker]
                            if last_signal['days'] is not None and last_signal['date'] is not None:
                                if last_signal['days'] <= 30:  # Only show recent signals
                                    signal_date_str = last_signal['date'].strftime('%Y-%m-%d') if last_signal['date'] else "Unknown"
                                    st.caption(f"Last buy opportunity: {last_signal['days']} days ago ({signal_date_str})")
                    
                    with col2:
                        # Create a price chart with technical indicators
                        fig = go.Figure()
                    
                        # Add price data
                        fig.add_trace(go.Candlestick(
                            x=df.index,
                            open=df['Open'],
                            high=df['High'],
                            low=df['Low'],
                            close=df['Close'],
                            name="Price"
                        ))
                        
                        # Add EMA if enabled
                        if st.session_state.show_ema and 'ema' in df.columns:
                            fig.add_trace(go.Scatter(
                                x=df.index,
                                y=df['ema'],
                                name=f"EMA ({st.session_state.ema_period})",
                                line=dict(color='orange', width=1)
                            ))
                        
                        # Add Bollinger Bands if enabled
                        if st.session_state.show_bollinger and 'bb_upper' in df.columns:
                            fig.add_trace(go.Scatter(
                                x=df.index,
                                y=df['bb_upper'],
                                name=f"Upper Band ({st.session_state.bb_period}, {st.session_state.bb_std}σ)",
                                line=dict(color='rgba(0,128,0,0.3)', width=1)
                            ))
                            
                            fig.add_trace(go.Scatter(
                                x=df.index,
                                y=df['bb_middle'],
                                name=f"Middle Band (SMA {st.session_state.bb_period})",
                                line=dict(color='rgba(0,128,0,0.7)', width=1)
                            ))
                            
                            fig.add_trace(go.Scatter(
                                x=df.index,
                                y=df['bb_lower'],
                                name=f"Lower Band ({st.session_state.bb_period}, {st.session_state.bb_std}σ)",
                                line=dict(color='rgba(0,128,0,0.3)', width=1)
                            ))
                        
                        # Update layout
                        fig.update_layout(
                            title=f"{ticker} Price Chart",
                            xaxis_title="Date",
                            yaxis_title="Price ($)",
                            height=400,
                            xaxis_rangeslider_visible=False,
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1
                            )
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Determine how many technical indicators to show
                        num_indicators = 1  # RSI is always shown
                        if st.session_state.show_macd:
                            num_indicators += 1
                        if st.session_state.show_stochastic and 'stoch_k' in df.columns:
                            num_indicators += 1
                        if st.session_state.show_adx and 'adx' in df.columns:
                            num_indicators += 1
                        if st.session_state.show_obv and 'obv' in df.columns:
                            num_indicators += 1
                        
                        # Create subplots for all enabled indicators
                        subplot_titles = ["RSI"]
                        row_heights = [1/num_indicators] * num_indicators
                        current_row = 1
                        
                        if st.session_state.show_macd:
                            subplot_titles.append("MACD")
                        if st.session_state.show_stochastic and 'stoch_k' in df.columns:
                            subplot_titles.append("Stochastic")
                        if st.session_state.show_adx and 'adx' in df.columns:
                            subplot_titles.append("ADX")
                        if st.session_state.show_obv and 'obv' in df.columns:
                            subplot_titles.append("OBV")
                        
                        # Create the subplot layout
                        fig = make_subplots(rows=num_indicators, cols=1, shared_xaxes=True, 
                                           vertical_spacing=0.08,
                                           subplot_titles=subplot_titles,
                                           row_heights=row_heights)
                        
                        # Add RSI (always shown)
                        current_row = 1
                        fig.add_trace(go.Scatter(
                            x=df.index,
                            y=df['rsi'],
                            name=f"RSI ({st.session_state.rsi_period})",
                            line=dict(color='blue', width=1)
                        ), row=current_row, col=1)
                        
                        # Add RSI threshold lines
                        fig.add_trace(go.Scatter(
                            x=df.index,
                            y=[70] * len(df),
                            name="Overbought (70)",
                            line=dict(color='red', width=1, dash='dash')
                        ), row=current_row, col=1)
                        
                        fig.add_trace(go.Scatter(
                            x=df.index,
                            y=[30] * len(df),
                            name="Oversold (30)",
                            line=dict(color='green', width=1, dash='dash')
                        ), row=current_row, col=1)
                        
                        # Add MACD if enabled
                        if st.session_state.show_macd:
                            current_row += 1
                            fig.add_trace(go.Scatter(
                                x=df.index,
                                y=df['macd'],
                                name="MACD Line",
                                line=dict(color='blue', width=1)
                            ), row=current_row, col=1)
                            
                            fig.add_trace(go.Scatter(
                                x=df.index,
                                y=df['macd_signal'],
                                name="Signal Line",
                                line=dict(color='red', width=1)
                            ), row=current_row, col=1)
                            
                            # Add MACD histogram
                            colors = ['green' if val >= 0 else 'red' for val in df['macd_hist']]
                            fig.add_trace(go.Bar(
                                x=df.index,
                                y=df['macd_hist'],
                                name="Histogram",
                                marker=dict(color=colors)
                            ), row=current_row, col=1)
                        
                        # Add Stochastic if enabled
                        if st.session_state.show_stochastic and 'stoch_k' in df.columns:
                            current_row += 1
                            fig.add_trace(go.Scatter(
                                x=df.index,
                                y=df['stoch_k'],
                                name="%K",
                                line=dict(color='blue', width=1)
                            ), row=current_row, col=1)
                            
                            fig.add_trace(go.Scatter(
                                x=df.index,
                                y=df['stoch_d'],
                                name="%D",
                                line=dict(color='red', width=1)
                            ), row=current_row, col=1)
                            
                            # Add Stochastic threshold lines (80/20 are common)
                            fig.add_trace(go.Scatter(
                                x=df.index,
                                y=[80] * len(df),
                                name="Overbought (80)",
                                line=dict(color='red', width=1, dash='dash')
                            ), row=current_row, col=1)
                            
                            fig.add_trace(go.Scatter(
                                x=df.index,
                                y=[20] * len(df),
                                name="Oversold (20)",
                                line=dict(color='green', width=1, dash='dash')
                            ), row=current_row, col=1)
                        
                        # Add ADX if enabled
                        if st.session_state.show_adx and 'adx' in df.columns:
                            current_row += 1
                            fig.add_trace(go.Scatter(
                                x=df.index,
                                y=df['adx'],
                                name="ADX",
                                line=dict(color='purple', width=1)
                            ), row=current_row, col=1)
                            
                            # Add +DI and -DI lines
                            fig.add_trace(go.Scatter(
                                x=df.index,
                                y=df['plus_di'],
                                name="+DI",
                                line=dict(color='green', width=1)
                            ), row=current_row, col=1)
                            
                            fig.add_trace(go.Scatter(
                                x=df.index,
                                y=df['minus_di'],
                                name="-DI",
                                line=dict(color='red', width=1)
                            ), row=current_row, col=1)
                            
                            # Add ADX threshold line at 25
                            fig.add_trace(go.Scatter(
                                x=df.index,
                                y=[25] * len(df),
                                name="Strong Trend (25)",
                                line=dict(color='gray', width=1, dash='dash')
                            ), row=current_row, col=1)
                        
                        # Add OBV if enabled
                        if st.session_state.show_obv and 'obv' in df.columns:
                            current_row += 1
                            # Normalize OBV to make it visually easier to interpret
                            obv_normalized = (df['obv'] - df['obv'].min()) / (df['obv'].max() - df['obv'].min()) * 100
                            
                            fig.add_trace(go.Scatter(
                                x=df.index,
                                y=obv_normalized,
                                name="OBV (normalized)",
                                line=dict(color='blue', width=1)
                            ), row=current_row, col=1)
                            
                            # Add OBV slope indicator
                            if 'obv_slope' in df.columns:
                                # Create a moving average of the OBV slope for visualization
                                obv_slope_ma = df['obv_slope'].rolling(window=5).mean()
                                fig.add_trace(go.Scatter(
                                    x=df.index,
                                    y=obv_slope_ma * 10 + 50,  # Scale and center for visibility
                                    name="OBV Slope (5-day MA)",
                                    line=dict(color='orange', width=1, dash='dot')
                                ), row=current_row, col=1)
                        
                        # Create axis title dictionaries for the update_layout
                        axis_titles = {
                            "yaxis_title": "RSI"  # Fixed: Using proper format for yaxis_title
                        }
                        
                        # Add axis titles for each indicator with proper dictionary format
                        for i in range(2, num_indicators + 1):
                            axis_titles[f"yaxis{i}_title"] = subplot_titles[i-1]  # Fixed: Using proper format for yaxis titles
                        
                        # Set the height based on the number of indicators
                        chart_height = max(400, 200 * num_indicators)
                        
                        fig.update_layout(
                            height=chart_height,
                            showlegend=True,
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1
                            ),
                            **axis_titles
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                        # Add separator between stocks
                        st.markdown("---")
            else:
                st.warning(f"No data available for {ticker}")
