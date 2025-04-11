import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
import os
import re

# Import our custom modules
import sms_alerts
import email_alerts
from utils import calculate_technical_indicators

# Constants
ALERT_TYPES = ["SMS", "Email", "Both"]
SIGNAL_TYPES = ["Buy Signals Only", "Sell Signals Only", "All Signals"]
STRENGTH_THRESHOLDS = ["Any Strength", "Strong (>70%)", "Very Strong (>85%)"]
MAX_ALERTS_PER_DAY = 5  # Limit alerts to prevent spamming

class SignalAlertManager:
    """
    Manages signal alerts for stocks, including alert preferences, 
    historical alerts, and sending notifications.
    """
    
    def __init__(self):
        # Initialize alert preferences if they don't exist
        if 'alert_preferences' not in st.session_state:
            st.session_state.alert_preferences = {
                'enabled': False,
                'alert_type': "SMS",
                'phone_number': "",
                'email': "",
                'signal_types': "Buy Signals Only",
                'strength_threshold': "Strong (>70%)",
                'max_alerts_per_day': MAX_ALERTS_PER_DAY,
                'cooldown_minutes': 60,
                'watched_tickers': [],
                'custom_indicators': [],
            }
            
        # Initialize alert history if it doesn't exist
        if 'alert_history' not in st.session_state:
            st.session_state.alert_history = []
    
    def save_preferences(self, preferences):
        """Save alert preferences to session state"""
        st.session_state.alert_preferences = preferences
    
    def load_preferences(self):
        """Load alert preferences from session state"""
        return st.session_state.alert_preferences
    
    def save_alert_history(self, alert):
        """Add an alert to the history"""
        if 'alert_history' not in st.session_state:
            st.session_state.alert_history = []
            
        # Add the alert to the beginning of the list
        st.session_state.alert_history.insert(0, alert)
        
        # Limit history size to prevent memory issues
        if len(st.session_state.alert_history) > 100:
            st.session_state.alert_history = st.session_state.alert_history[:100]
    
    def check_signal(self, ticker, price_data, threshold=70):
        """
        Check if the stock has a strong buy or sell signal
        
        Args:
            ticker: Stock symbol
            price_data: DataFrame with price data
            threshold: Signal strength threshold percentage
            
        Returns:
            dict with signal details or None if no signal
        """
        if len(price_data) == 0:
            return None
            
        # Get the latest data point
        latest_data = price_data.iloc[-1]
        
        # Calculate technical indicators
        indicators = calculate_technical_indicators(price_data)
        
        # Count buy and sell signals
        buy_signals = 0
        sell_signals = 0
        total_indicators = 0
        active_indicators = {}
        
        # Check RSI
        if 'RSI' in indicators:
            rsi = indicators['RSI'].iloc[-1]
            total_indicators += 1
            if rsi < 30:  # Oversold - buy signal
                buy_signals += 1
                active_indicators['RSI'] = f"{rsi:.2f} (Oversold)"
            elif rsi > 70:  # Overbought - sell signal
                sell_signals += 1
                active_indicators['RSI'] = f"{rsi:.2f} (Overbought)"
        
        # Check MACD
        if 'MACD' in indicators and 'MACD_Signal' in indicators:
            macd = indicators['MACD'].iloc[-1]
            signal = indicators['MACD_Signal'].iloc[-1]
            macd_prev = indicators['MACD'].iloc[-2]
            signal_prev = indicators['MACD_Signal'].iloc[-2]
            
            total_indicators += 1
            
            # Check for crossover (current and previous values)
            if macd > signal and macd_prev <= signal_prev:  # Bullish crossover
                buy_signals += 1
                active_indicators['MACD'] = f"Bullish crossover ({macd:.2f} > {signal:.2f})"
            elif macd < signal and macd_prev >= signal_prev:  # Bearish crossover
                sell_signals += 1 
                active_indicators['MACD'] = f"Bearish crossover ({macd:.2f} < {signal:.2f})"
        
        # Check Bollinger Bands
        if all(x in indicators for x in ['BB_Upper', 'BB_Middle', 'BB_Lower']):
            upper = indicators['BB_Upper'].iloc[-1]
            middle = indicators['BB_Middle'].iloc[-1]
            lower = indicators['BB_Lower'].iloc[-1]
            close = latest_data['Close']
            
            total_indicators += 1
            
            if close <= lower:  # Price at or below lower band - buy signal
                buy_signals += 1
                active_indicators['Bollinger Bands'] = f"Price at lower band (${close:.2f} <= ${lower:.2f})"
            elif close >= upper:  # Price at or above upper band - sell signal
                sell_signals += 1
                active_indicators['Bollinger Bands'] = f"Price at upper band (${close:.2f} >= ${upper:.2f})"
        
        # Check Moving Average Crossovers
        if 'SMA_50' in indicators and 'SMA_200' in indicators:
            sma_50 = indicators['SMA_50'].iloc[-1]
            sma_200 = indicators['SMA_200'].iloc[-1]
            sma_50_prev = indicators['SMA_50'].iloc[-2]
            sma_200_prev = indicators['SMA_200'].iloc[-2]
            
            total_indicators += 1
            
            # Golden Cross (50-day crosses above 200-day)
            if sma_50 > sma_200 and sma_50_prev <= sma_200_prev:
                buy_signals += 2  # Stronger signal, count it twice
                active_indicators['Golden Cross'] = f"50-day SMA (${sma_50:.2f}) crossed above 200-day SMA (${sma_200:.2f})"
            # Death Cross (50-day crosses below 200-day)
            elif sma_50 < sma_200 and sma_50_prev >= sma_200_prev:
                sell_signals += 2  # Stronger signal, count it twice
                active_indicators['Death Cross'] = f"50-day SMA (${sma_50:.2f}) crossed below 200-day SMA (${sma_200:.2f})"
        
        # Check for price breakouts
        if len(price_data) >= 20:
            # Calculate 20-day high and low
            high_20d = price_data['High'].tail(20).max()
            low_20d = price_data['Low'].tail(20).min()
            prev_high = price_data['High'].iloc[-2]
            prev_close = price_data['Close'].iloc[-2]
            
            close = latest_data['Close']
            total_indicators += 1
            
            # Breakout above 20-day high
            if close > high_20d and prev_close <= high_20d:
                buy_signals += 1
                active_indicators['Breakout'] = f"Price broke above 20-day high (${close:.2f} > ${high_20d:.2f})"
            # Breakdown below 20-day low
            elif close < low_20d and prev_close >= low_20d:
                sell_signals += 1
                active_indicators['Breakdown'] = f"Price broke below 20-day low (${close:.2f} < ${low_20d:.2f})"
        
        # Calculate signal strength as a percentage
        total_possible = total_indicators * 2  # Max 2 points per indicator (for golden/death cross)
        
        if buy_signals > 0:
            buy_strength = (buy_signals / total_possible) * 100
            if buy_strength >= threshold:
                signal_type = "STRONG BUY" if buy_strength >= 85 else "BUY"
                return {
                    'ticker': ticker,
                    'price': latest_data['Close'],
                    'signal_type': signal_type,
                    'signal_strength': buy_strength,
                    'indicators': active_indicators,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
        
        if sell_signals > 0:
            sell_strength = (sell_signals / total_possible) * 100
            if sell_strength >= threshold:
                signal_type = "STRONG SELL" if sell_strength >= 85 else "SELL"
                return {
                    'ticker': ticker,
                    'price': latest_data['Close'],
                    'signal_type': signal_type,
                    'signal_strength': sell_strength,
                    'indicators': active_indicators,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
        
        return None
    
    def should_send_alert(self, ticker, signal_type):
        """
        Check if we should send an alert based on preferences and history
        
        Args:
            ticker: Stock symbol
            signal_type: Type of signal (BUY, SELL, etc.)
            
        Returns:
            Boolean indicating whether to send the alert
        """
        preferences = self.load_preferences()
        
        # Check if alerts are enabled
        if not preferences['enabled']:
            return False
        
        # Check if we have contact info based on alert type
        if preferences['alert_type'] == "SMS" and not preferences['phone_number']:
            return False
        elif preferences['alert_type'] == "Email" and not preferences['email']:
            return False
        elif preferences['alert_type'] == "Both" and (not preferences['phone_number'] or not preferences['email']):
            return False
        
        # Check if ticker is in watched list
        if preferences['watched_tickers'] and ticker not in preferences['watched_tickers']:
            return False
        
        # Check if signal type matches preferences
        if preferences['signal_types'] == "Buy Signals Only" and "SELL" in signal_type:
            return False
        elif preferences['signal_types'] == "Sell Signals Only" and "BUY" in signal_type:
            return False
        
        # Check for alert frequency limits
        alert_history = st.session_state.alert_history
        current_time = datetime.now()
        
        # Check for daily limit
        today_alerts = [
            a for a in alert_history 
            if a.get('timestamp') and current_time - datetime.strptime(a['timestamp'], '%Y-%m-%d %H:%M:%S') < timedelta(days=1)
        ]
        
        if len(today_alerts) >= preferences['max_alerts_per_day']:
            return False
        
        # Check for cooldown period for this specific ticker
        ticker_alerts = [
            a for a in alert_history 
            if a.get('ticker') == ticker and a.get('timestamp') and 
            current_time - datetime.strptime(a['timestamp'], '%Y-%m-%d %H:%M:%S') < timedelta(minutes=preferences['cooldown_minutes'])
        ]
        
        if ticker_alerts:
            return False
        
        return True
    
    def send_alert(self, signal):
        """
        Send an alert based on the detected signal and user preferences
        
        Args:
            signal: Dictionary with signal details
            
        Returns:
            String with the result of the alert attempt
        """
        preferences = self.load_preferences()
        alert_type = preferences['alert_type']
        
        result = ""
        
        # Send SMS alert
        if alert_type in ["SMS", "Both"]:
            phone = preferences['phone_number']
            if phone:
                # Format and validate phone number
                formatted_phone = sms_alerts.validate_phone_number(phone)
                if formatted_phone:
                    sms_result = sms_alerts.send_price_alert(
                        formatted_phone,
                        signal['ticker'],
                        signal['price'],
                        signal['signal_type'],
                        signal['signal_strength']
                    )
                    result += f"SMS: {sms_result}\n"
                else:
                    result += "SMS: Invalid phone number format\n"
            else:
                result += "SMS: No phone number provided\n"
        
        # Send email alert
        if alert_type in ["Email", "Both"]:
            email = preferences['email']
            if email:
                # Validate email
                if email_alerts.validate_email(email):
                    # Create email subject
                    subject = f"{signal['signal_type']} Alert: {signal['ticker']} at ${signal['price']:.2f}"
                    
                    # Create email content
                    html_content = email_alerts.create_buy_signal_email(
                        signal['ticker'],
                        signal['price'],
                        signal['signal_type'],
                        signal['signal_strength'],
                        signal['indicators'],
                        signal['timestamp']
                    )
                    
                    # Send email
                    email_result = email_alerts.send_email_alert(
                        email,
                        subject,
                        html_content
                    )
                    result += f"Email: {email_result}\n"
                else:
                    result += "Email: Invalid email format\n"
            else:
                result += "Email: No email address provided\n"
        
        # Save alert to history
        signal['result'] = result.strip()
        self.save_alert_history(signal)
        
        return result.strip()
    
    def display_alert_setup(self):
        """Display the UI for setting up alerts"""
        st.subheader("⚠️ Signal Alerts Setup")
        
        preferences = self.load_preferences()
        
        # Enable/disable alerts
        enabled = st.toggle("Enable Automated Alerts", preferences['enabled'])
        
        if enabled:
            # Choose alert type
            alert_type = st.selectbox(
                "Alert Method", 
                ALERT_TYPES, 
                index=ALERT_TYPES.index(preferences['alert_type'])
            )
            
            # SMS settings
            if alert_type in ["SMS", "Both"]:
                st.write("📱 SMS Settings")
                phone_number = st.text_input(
                    "Phone Number (e.g., +15551234567)", 
                    value=preferences['phone_number']
                )
                st.caption("Message rates may apply. International numbers should include country code.")
                
                # Test SMS button
                if st.button("Test SMS") and phone_number:
                    formatted_phone = sms_alerts.validate_phone_number(phone_number)
                    if formatted_phone:
                        result = sms_alerts.send_price_alert(
                            formatted_phone,
                            "TEST",
                            100.00,
                            "TEST",
                            75
                        )
                        st.info(f"Test result: {result}")
                    else:
                        st.error("Invalid phone number format. Please include country code (e.g., +15551234567).")
            else:
                phone_number = preferences['phone_number']
            
            # Email settings
            if alert_type in ["Email", "Both"]:
                st.write("📧 Email Settings")
                email = st.text_input(
                    "Email Address", 
                    value=preferences['email']
                )
                
                # Test email button
                if st.button("Test Email") and email:
                    if email_alerts.validate_email(email):
                        # Create test email
                        subject = "Test Alert from Stock Analysis Tool"
                        test_indicators = {
                            "RSI": "28.5 (Oversold)",
                            "MACD": "Bullish crossover (0.75 > 0.25)"
                        }
                        html_content = email_alerts.create_buy_signal_email(
                            "TEST",
                            100.00,
                            "TEST BUY",
                            75,
                            test_indicators,
                            datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        )
                        
                        result = email_alerts.send_email_alert(
                            email,
                            subject,
                            html_content
                        )
                        st.info(f"Test result: {result}")
                    else:
                        st.error("Invalid email format.")
            else:
                email = preferences['email']
            
            # Alert filters
            st.write("🔍 Alert Filters")
            
            # Signal types to alert on
            signal_types = st.selectbox(
                "Signal Types to Alert On", 
                SIGNAL_TYPES, 
                index=SIGNAL_TYPES.index(preferences['signal_types'])
            )
            
            # Signal strength threshold
            strength_threshold = st.selectbox(
                "Minimum Signal Strength", 
                STRENGTH_THRESHOLDS, 
                index=STRENGTH_THRESHOLDS.index(preferences['strength_threshold'])
            )
            
            # Frequency controls
            st.write("⏱️ Alert Frequency")
            
            # Max alerts per day
            max_alerts = st.slider(
                "Maximum Alerts Per Day", 
                1, 20, 
                value=preferences['max_alerts_per_day']
            )
            
            # Cooldown between alerts for the same ticker
            cooldown = st.slider(
                "Minimum Time Between Alerts for Same Stock (minutes)", 
                15, 1440, 
                value=preferences['cooldown_minutes']
            )
            
            # Stock watchlist
            st.write("📋 Stock Watchlist")
            
            # Default stocks to watch
            default_tickers = ["AAPL", "MSFT", "AMZN", "GOOGL", "TSLA"]
            
            # Add custom tickers
            custom_tickers_str = st.text_input(
                "Enter stock symbols to watch (comma separated)", 
                value=",".join(preferences['watched_tickers']) if preferences['watched_tickers'] else ""
            )
            
            # Parse custom tickers
            if custom_tickers_str:
                # Split by commas and clean up
                custom_tickers = [
                    t.strip().upper() for t in custom_tickers_str.split(",") if t.strip()
                ]
            else:
                custom_tickers = []
            
            # Use default tickers if none provided
            watched_tickers = custom_tickers if custom_tickers else default_tickers
            
            # Custom indicators (future enhancement)
            custom_indicators = preferences['custom_indicators']
            
            # Update preferences
            new_preferences = {
                'enabled': enabled,
                'alert_type': alert_type,
                'phone_number': phone_number,
                'email': email,
                'signal_types': signal_types,
                'strength_threshold': strength_threshold,
                'max_alerts_per_day': max_alerts,
                'cooldown_minutes': cooldown,
                'watched_tickers': watched_tickers,
                'custom_indicators': custom_indicators,
            }
            
            # Save when button is clicked
            if st.button("Save Alert Preferences"):
                self.save_preferences(new_preferences)
                st.success("✅ Alert preferences saved successfully!")
                
                # Provide guidance based on settings
                if alert_type in ["SMS", "Both"] and not phone_number:
                    st.warning("⚠️ SMS alerts enabled but no phone number provided.")
                if alert_type in ["Email", "Both"] and not email:
                    st.warning("⚠️ Email alerts enabled but no email address provided.")
                
                # Explanation of what was set up
                threshold_map = {
                    "Any Strength": "any signal strength",
                    "Strong (>70%)": "signals above 70% strength",
                    "Very Strong (>85%)": "signals above 85% strength"
                }
                
                signals_map = {
                    "Buy Signals Only": "BUY signals only",
                    "Sell Signals Only": "SELL signals only",
                    "All Signals": "both BUY and SELL signals"
                }
                
                st.write(f"""
                **Alert Summary:**
                
                You'll receive {alert_type.lower()} alerts for {signals_map[signal_types]} with {threshold_map[strength_threshold]}.
                Maximum {max_alerts} alerts per day, with at least {cooldown} minutes between alerts for the same stock.
                
                Watching {len(watched_tickers)} stocks: {', '.join(watched_tickers[:5])}{' and more...' if len(watched_tickers) > 5 else ''}
                """)
        else:
            # When alerts are disabled, update preferences
            new_preferences = preferences.copy()
            new_preferences['enabled'] = False
            self.save_preferences(new_preferences)
            
            st.info("📴 Automated alerts are currently disabled.")
    
    def display_alert_history(self):
        """Display the historical alerts in a nice UI"""
        st.subheader("📜 Alert History")
        
        alert_history = st.session_state.alert_history
        
        if not alert_history:
            st.info("No alerts have been sent yet.")
            return
        
        # Create a dataframe for nicer display
        alerts_df = pd.DataFrame([
            {
                'Date': alert.get('timestamp', ''),
                'Ticker': alert.get('ticker', ''),
                'Signal': alert.get('signal_type', ''),
                'Price': alert.get('price', 0),
                'Strength': alert.get('signal_strength', 0),
                'Status': 'Success' if 'error' not in alert.get('result', '').lower() else 'Failed'
            }
            for alert in alert_history
        ])
        
        # Show the alert history
        st.dataframe(alerts_df, use_container_width=True)
        
        # Option to view detailed history
        if st.button("Clear Alert History"):
            st.session_state.alert_history = []
            st.success("Alert history cleared.")
            st.rerun()
        
        # Show individual alert details in expanders
        for i, alert in enumerate(alert_history):
            # Format alert details
            signal_type = alert.get('signal_type', '')
            ticker = alert.get('ticker', '')
            price = alert.get('price', 0)
            timestamp = alert.get('timestamp', '')
            
            with st.expander(f"{timestamp} - {signal_type} alert for {ticker} at ${price:.2f}"):
                # Left column for alert details
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write("**Alert Details:**")
                    st.write(f"**Ticker:** {ticker}")
                    st.write(f"**Signal Type:** {signal_type}")
                    st.write(f"**Price:** ${price:.2f}")
                    st.write(f"**Signal Strength:** {alert.get('signal_strength', 0):.1f}%")
                    st.write(f"**Timestamp:** {timestamp}")
                    
                    # Display indicators that triggered the alert
                    st.write("**Technical Indicators:**")
                    indicators = alert.get('indicators', {})
                    for ind, value in indicators.items():
                        st.write(f"- {ind}: {value}")
                
                with col2:
                    # Show delivery status
                    st.write("**Delivery Status:**")
                    result = alert.get('result', 'No delivery information')
                    
                    # Format the delivery status
                    for line in result.split("\n"):
                        if "error" in line.lower():
                            st.error(line)
                        elif "sent" in line.lower():
                            st.success(line)
                        else:
                            st.info(line)
    
    def process_alerts_for_ticker(self, ticker, price_data):
        """
        Process alerts for a specific ticker
        
        Args:
            ticker: Stock symbol
            price_data: DataFrame with price data
            
        Returns:
            Signal information if an alert was triggered, None otherwise
        """
        preferences = self.load_preferences()
        
        # Skip if alerts are disabled
        if not preferences['enabled']:
            return None
        
        # Get threshold based on preference
        if preferences['strength_threshold'] == "Any Strength":
            threshold = 0
        elif preferences['strength_threshold'] == "Very Strong (>85%)":
            threshold = 85
        else:  # "Strong (>70%)"
            threshold = 70
        
        # Check for signals
        signal = self.check_signal(ticker, price_data, threshold)
        
        # If we have a signal and should send an alert, send it
        if signal and self.should_send_alert(ticker, signal['signal_type']):
            self.send_alert(signal)
            return signal
        
        return None


def signal_alerts_section():
    """Main function for the signal alerts section"""
    st.title("📊 Signal Alerts")
    
    # Create the alert manager
    alert_manager = SignalAlertManager()
    
    # Create tabs for setup and history
    setup_tab, history_tab, test_tab = st.tabs(["Alert Setup", "Alert History", "Test Alerts"])
    
    with setup_tab:
        alert_manager.display_alert_setup()
    
    with history_tab:
        alert_manager.display_alert_history()
    
    with test_tab:
        st.subheader("🧪 Test Alert Generation")
        st.write("Test how the alert system would respond to different stock data.")
        
        # Input for ticker to test
        test_ticker = st.text_input("Enter Stock Symbol", "AAPL").upper()
        
        if test_ticker:
            import yfinance as yf
            from datetime import datetime, timedelta
            
            # Fetch recent data for the ticker
            end_date = datetime.now()
            start_date = end_date - timedelta(days=200)  # Need enough data for indicators
            
            try:
                # Show loading spinner
                with st.spinner(f"Fetching data for {test_ticker}..."):
                    stock_data = yf.download(test_ticker, start=start_date, end=end_date)
                
                if len(stock_data) > 0:
                    # Check for signals
                    preferences = alert_manager.load_preferences()
                    
                    # Get threshold based on dropdown
                    thresholds = {
                        "Any Strength": 0,
                        "Strong (>70%)": 70,
                        "Very Strong (>85%)": 85
                    }
                    
                    # Create multiple threshold tests
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write("**Low Threshold (>30%)**")
                        low_signal = alert_manager.check_signal(test_ticker, stock_data, threshold=30)
                        if low_signal:
                            signal_color = "green" if "BUY" in low_signal['signal_type'] else "red"
                            st.markdown(f"<div style='color:{signal_color};font-weight:bold;font-size:1.2em;'>{low_signal['signal_type']}</div>", unsafe_allow_html=True)
                            st.write(f"Strength: {low_signal['signal_strength']:.1f}%")
                            st.write(f"Price: ${low_signal['price']:.2f}")
                            
                            # List active indicators
                            for ind, value in low_signal['indicators'].items():
                                st.write(f"- {ind}: {value}")
                        else:
                            st.write("No signal detected")
                    
                    with col2:
                        st.write("**Medium Threshold (>70%)**")
                        med_signal = alert_manager.check_signal(test_ticker, stock_data, threshold=70)
                        if med_signal:
                            signal_color = "green" if "BUY" in med_signal['signal_type'] else "red"
                            st.markdown(f"<div style='color:{signal_color};font-weight:bold;font-size:1.2em;'>{med_signal['signal_type']}</div>", unsafe_allow_html=True)
                            st.write(f"Strength: {med_signal['signal_strength']:.1f}%")
                            st.write(f"Price: ${med_signal['price']:.2f}")
                            
                            # List active indicators
                            for ind, value in med_signal['indicators'].items():
                                st.write(f"- {ind}: {value}")
                        else:
                            st.write("No signal detected")
                    
                    with col3:
                        st.write("**High Threshold (>85%)**")
                        high_signal = alert_manager.check_signal(test_ticker, stock_data, threshold=85)
                        if high_signal:
                            signal_color = "green" if "BUY" in high_signal['signal_type'] else "red"
                            st.markdown(f"<div style='color:{signal_color};font-weight:bold;font-size:1.2em;'>{high_signal['signal_type']}</div>", unsafe_allow_html=True)
                            st.write(f"Strength: {high_signal['signal_strength']:.1f}%")
                            st.write(f"Price: ${high_signal['price']:.2f}")
                            
                            # List active indicators
                            for ind, value in high_signal['indicators'].items():
                                st.write(f"- {ind}: {value}")
                        else:
                            st.write("No signal detected")
                    
                    # Option to send a test alert
                    # Use proper Python logic to check if any signal exists
                    if med_signal is not None:
                        selected_signal = med_signal
                    elif low_signal is not None:
                        selected_signal = low_signal
                    elif high_signal is not None:
                        selected_signal = high_signal
                    else:
                        selected_signal = None
                        
                    if selected_signal is not None and st.button("Send Test Alert"):
                        preferences = alert_manager.load_preferences()
                        if preferences['enabled']:
                            # Override should_send_alert to always return True for test
                            result = alert_manager.send_alert(selected_signal)
                            st.success(f"Test alert sent. Result: {result}")
                        else:
                            st.warning("Alerts are currently disabled. Please enable them in the Alert Setup tab.")
                else:
                    st.error(f"No data available for {test_ticker}")
            except Exception as e:
                st.error(f"Error fetching data: {str(e)}")

if __name__ == "__main__":
    signal_alerts_section()
