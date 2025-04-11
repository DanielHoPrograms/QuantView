import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from utils import calculate_rsi, calculate_macd, calculate_bollinger_bands

# Signal tracking data structure in session state:
# st.session_state.historical_signals = {
#    ticker: [
#        {
#            'date': datetime,  # When the signal was generated
#            'price': float,    # The price at signal generation
#            'signal_type': str,  # 'BUY' or 'SELL'
#            'signal_strength': int,  # 0-100 percentage 
#            'indicator': str,  # Which indicator generated this (RSI, MACD, BB, COMBINED)
#            'tracked_returns': {  # Performance after signal
#                '1d': float,   # 1-day return percentage
#                '1w': float,   # 1-week return percentage
#                '1m': float,   # 1-month return percentage
#                '3m': float,   # 3-month return percentage
#                '6m': float,   # 6-month return percentage
#                'current': float,  # Current return percentage
#            }
#        },
#        # More signal entries...
#    ],
#    # More tickers...
# }


def initialize_signal_tracker():
    """Initialize signal tracker data structures in session state"""
    if 'historical_signals' not in st.session_state:
        st.session_state.historical_signals = {}
    if 'signal_tracker_loaded' not in st.session_state:
        st.session_state.signal_tracker_loaded = False


def record_signal(ticker, price, signal_type, signal_strength, indicator):
    """
    Record a new signal in the historical signals data
    
    Args:
        ticker: Stock symbol
        price: Current stock price
        signal_type: 'BUY' or 'SELL'
        signal_strength: Strength of the signal (0-100)
        indicator: Which indicator generated this signal (RSI, MACD, BB, COMBINED)
    """
    # Initialize ticker entry if not exists
    if ticker not in st.session_state.historical_signals:
        st.session_state.historical_signals[ticker] = []
    
    # Create new signal entry
    signal_entry = {
        'date': datetime.now(),
        'price': price,
        'signal_type': signal_type,
        'signal_strength': signal_strength,
        'indicator': indicator,
        'tracked_returns': {
            '1d': None,
            '1w': None,
            '1m': None,
            '3m': None,
            '6m': None,
            'current': None
        }
    }
    
    # Check if we already have a similar signal today (avoid duplicates)
    today = datetime.now().date()
    duplicate = False
    
    for existing_signal in st.session_state.historical_signals[ticker]:
        if existing_signal['date'].date() == today and existing_signal['signal_type'] == signal_type:
            duplicate = True
            break
    
    # Add signal if not duplicate
    if not duplicate:
        st.session_state.historical_signals[ticker].append(signal_entry)
        st.success(f"New {signal_type} signal for {ticker} recorded at ${price:.2f}")
        return True
    
    return False


def update_signal_performance():
    """Update performance metrics for all recorded signals"""
    for ticker, signals in st.session_state.historical_signals.items():
        # Skip if no signals
        if not signals:
            continue
        
        try:
            # Get historical data
            end_date = datetime.now()
            # Calculate start date based on the oldest signal or 6 months ago, whichever is earlier
            oldest_signal_date = min([s['date'] for s in signals])
            start_date = min(oldest_signal_date - timedelta(days=1), end_date - timedelta(days=180))
            
            # Convert dates to strings for yfinance to avoid type issues
            start_date_str = start_date.strftime('%Y-%m-%d')
            end_date_str = end_date.strftime('%Y-%m-%d')
            
            # Fetch data
            data = yf.download(ticker, start=start_date_str, end=end_date_str)
            
            if len(data) == 0:
                continue
                
            # Update each signal's performance
            for signal in signals:
                signal_date = signal['date']
                signal_price = signal['price']
                
                # Convert signal date to pandas Timestamp for comparison with data.index
                signal_pd_timestamp = pd.Timestamp(signal_date)
                
                # Find the closest date in data after the signal date
                # Find dates in dataframe that are >= signal date
                future_dates = data.index[data.index >= signal_pd_timestamp]
                
                if len(future_dates) == 0:
                    # No data points after this signal, skip it
                    continue
                    
                closest_date = future_dates[0]
                
                # Get current price (most recent in the data)
                current_price = data['Close'].iloc[-1]
                
                # Calculate current return
                current_return = ((current_price / signal_price) - 1) * 100
                signal['tracked_returns']['current'] = current_return
                
                # Calculate returns for different periods
                periods = {
                    '1d': timedelta(days=1),
                    '1w': timedelta(days=7),
                    '1m': timedelta(days=30),
                    '3m': timedelta(days=90),
                    '6m': timedelta(days=180)
                }
                
                for period_key, period_delta in periods.items():
                    # Skip if signal is newer than the period
                    if signal_date > datetime.now() - period_delta:
                        signal['tracked_returns'][period_key] = None
                        continue
                        
                    # Calculate target date after period
                    target_date = signal_date + period_delta
                    
                    # Convert to pandas Timestamp for comparison
                    target_pd_timestamp = pd.Timestamp(target_date)
                    
                    # Find dates in dataframe that are >= target date
                    future_dates_for_period = data.index[data.index >= target_pd_timestamp]
                    
                    if len(future_dates_for_period) > 0:
                        future_date = future_dates_for_period[0]
                        future_price = data.loc[future_date, 'Close']
                        period_return = ((future_price / signal_price) - 1) * 100
                        signal['tracked_returns'][period_key] = period_return
                    else:
                        signal['tracked_returns'][period_key] = None
        
        except Exception as e:
            st.error(f"Error updating signals for {ticker}: {str(e)}")


def get_signal_success_rate(signals, period='1m', threshold=2.0):
    """
    Calculate success rate of signals based on returns
    
    Args:
        signals: List of signal entries
        period: Return period to evaluate ('1d', '1w', '1m', '3m', '6m', 'current')
        threshold: Minimum return percentage to consider a success
        
    Returns:
        Tuple of (success rate, number of signals evaluated)
    """
    if not signals:
        return 0, 0
    
    # Filter for completed signals (have the period's return calculated)
    valid_signals = [s for s in signals if s['tracked_returns'][period] is not None]
    
    if not valid_signals:
        return 0, 0
    
    # Count successful signals
    successful = sum(1 for s in valid_signals if s['tracked_returns'][period] >= threshold)
    
    # Calculate success rate
    success_rate = (successful / len(valid_signals)) * 100
    
    return success_rate, len(valid_signals)


def detect_and_record_signals(ticker, df, rsi_threshold=30, drop_threshold=5.0):
    """
    Analyze stock data and record signals if conditions are met
    
    Args:
        ticker: Stock symbol
        df: DataFrame with stock data including technical indicators
        rsi_threshold: RSI threshold for buy signals
        drop_threshold: Price drop threshold for buy signals
    """
    if len(df) == 0:
        return
    
    # Get the latest data
    current_price = df['Close'].iloc[-1]
    current_rsi = df['rsi'].iloc[-1]
    
    # Calculate daily change
    if len(df) > 1:
        daily_change = ((df['Close'].iloc[-1] / df['Close'].iloc[-2]) - 1) * 100
    else:
        daily_change = 0
    
    # Get additional indicators if available
    macd_line = df['macd'].iloc[-1] if 'macd' in df else None
    macd_signal = df['macd_signal'].iloc[-1] if 'macd_signal' in df else None
    
    bb_upper = df['bb_upper'].iloc[-1] if 'bb_upper' in df else None
    bb_middle = df['bb_middle'].iloc[-1] if 'bb_middle' in df else None
    bb_lower = df['bb_lower'].iloc[-1] if 'bb_lower' in df else None
    
    # Check for signals
    signal_strength = 0
    signal_indicator = None
    
    # RSI Signal (Oversold Condition)
    if current_rsi <= rsi_threshold:
        signal_strength += 50
        signal_indicator = "RSI"
    
    # Price Drop Signal
    if daily_change <= -drop_threshold:
        signal_strength += 30
        signal_indicator = signal_indicator or "Price Drop"
    
    # MACD Signal (MACD line crosses above signal line)
    if macd_line is not None and macd_signal is not None:
        if macd_line > macd_signal and df['macd'].iloc[-2] <= df['macd_signal'].iloc[-2]:
            signal_strength += 40
            signal_indicator = "MACD" if signal_indicator is None else "COMBINED"
    
    # Bollinger Bands Signal (Price below lower band)
    if current_price is not None and bb_lower is not None:
        if current_price < bb_lower:
            signal_strength += 40
            signal_indicator = "Bollinger Bands" if signal_indicator is None else "COMBINED"
    
    # Record signal if strong enough
    if signal_strength >= 40:  # Threshold for recording a signal
        record_signal(ticker, current_price, "BUY", signal_strength, signal_indicator)


def calculate_signal_volatility(signals, period='current'):
    """
    Calculate volatility of signal returns for a given period
    
    Args:
        signals: List of signal entries
        period: Period to analyze ('1d', '1w', '1m', '3m', '6m', 'current')
        
    Returns:
        Volatility value (standard deviation of returns)
    """
    # Get valid returns for the period
    returns = [s['tracked_returns'][period] for s in signals if s['tracked_returns'][period] is not None]
    
    if len(returns) < 2:  # Need at least 2 points for std dev
        return None
    
    # Calculate standard deviation
    return np.std(returns)


def save_historical_signals():
    """Save historical signals to session state for persistence"""
    if not st.session_state.historical_signals:
        st.warning("No signals to save.")
        return
    
    try:
        # Convert datetime objects to strings
        serializable_signals = {}
        for ticker, signals in st.session_state.historical_signals.items():
            serializable_signals[ticker] = []
            for signal in signals:
                signal_copy = signal.copy()
                signal_copy['date'] = signal_copy['date'].strftime('%Y-%m-%d %H:%M:%S')
                serializable_signals[ticker].append(signal_copy)
        
        # Store in session state
        st.session_state.saved_historical_signals = serializable_signals
        st.success("Historical signals saved successfully!")
    except Exception as e:
        st.error(f"Error saving signals: {str(e)}")


def load_historical_signals():
    """Load historical signals from session state"""
    if 'saved_historical_signals' not in st.session_state:
        st.warning("No saved signals found.")
        return
    
    try:
        # Convert string dates back to datetime
        for ticker, signals in st.session_state.saved_historical_signals.items():
            if ticker not in st.session_state.historical_signals:
                st.session_state.historical_signals[ticker] = []
            
            for signal in signals:
                # Convert date string back to datetime
                signal['date'] = datetime.strptime(signal['date'], '%Y-%m-%d %H:%M:%S')
                
                # Check if this signal already exists to prevent duplicates
                duplicate = False
                for existing_signal in st.session_state.historical_signals[ticker]:
                    if (existing_signal['date'] == signal['date'] and 
                        existing_signal['signal_type'] == signal['signal_type']):
                        duplicate = True
                        break
                
                if not duplicate:
                    st.session_state.historical_signals[ticker].append(signal)
        
        st.success("Historical signals loaded successfully!")
        
    except Exception as e:
        st.error(f"Error loading signals: {str(e)}")


def get_signal_data_over_time():
    """
    Analyze how signal returns change over time
    
    Returns:
        DataFrame with time series data for signal returns
    """
    if not st.session_state.historical_signals:
        return None
    
    time_series_data = []
    
    for ticker, signals in st.session_state.historical_signals.items():
        for signal in signals:
            if not signal['tracked_returns']['current']:
                continue
                
            signal_date = signal['date']
            signal_type = signal['signal_type']
            indicator = signal['indicator']
            
            # For each day since the signal was generated, track the return
            days_since = (datetime.now() - signal_date).days
            
            # Try to estimate return progression over time
            # We'll use the available period returns as checkpoints
            
            # Add signal start point
            time_series_data.append({
                'Ticker': ticker,
                'Signal Date': signal_date.strftime('%Y-%m-%d'),
                'Days After Signal': 0,
                'Return (%)': 0,
                'Signal Type': signal_type,
                'Indicator': indicator
            })
            
            # Add return data points using tracked periods
            if signal['tracked_returns']['1d'] is not None and days_since >= 1:
                time_series_data.append({
                    'Ticker': ticker,
                    'Signal Date': signal_date.strftime('%Y-%m-%d'),
                    'Days After Signal': 1,
                    'Return (%)': signal['tracked_returns']['1d'],
                    'Signal Type': signal_type,
                    'Indicator': indicator
                })
                
            if signal['tracked_returns']['1w'] is not None and days_since >= 7:
                time_series_data.append({
                    'Ticker': ticker,
                    'Signal Date': signal_date.strftime('%Y-%m-%d'),
                    'Days After Signal': 7,
                    'Return (%)': signal['tracked_returns']['1w'],
                    'Signal Type': signal_type,
                    'Indicator': indicator
                })
                
            if signal['tracked_returns']['1m'] is not None and days_since >= 30:
                time_series_data.append({
                    'Ticker': ticker,
                    'Signal Date': signal_date.strftime('%Y-%m-%d'),
                    'Days After Signal': 30,
                    'Return (%)': signal['tracked_returns']['1m'],
                    'Signal Type': signal_type,
                    'Indicator': indicator
                })
                
            if signal['tracked_returns']['3m'] is not None and days_since >= 90:
                time_series_data.append({
                    'Ticker': ticker,
                    'Signal Date': signal_date.strftime('%Y-%m-%d'),
                    'Days After Signal': 90,
                    'Return (%)': signal['tracked_returns']['3m'],
                    'Signal Type': signal_type,
                    'Indicator': indicator
                })
                
            if signal['tracked_returns']['6m'] is not None and days_since >= 180:
                time_series_data.append({
                    'Ticker': ticker,
                    'Signal Date': signal_date.strftime('%Y-%m-%d'),
                    'Days After Signal': 180,
                    'Return (%)': signal['tracked_returns']['6m'],
                    'Signal Type': signal_type,
                    'Indicator': indicator
                })
            
            # Add current return point
            time_series_data.append({
                'Ticker': ticker,
                'Signal Date': signal_date.strftime('%Y-%m-%d'),
                'Days After Signal': days_since,
                'Return (%)': signal['tracked_returns']['current'],
                'Signal Type': signal_type,
                'Indicator': indicator
            })
    
    if not time_series_data:
        return None
        
    return pd.DataFrame(time_series_data)


def create_signal_performance_heatmap(signals):
    """
    Create a heatmap showing the best days/times for signals
    
    Args:
        signals: List of all signals from different tickers
        
    Returns:
        Plotly figure with heatmap
    """
    if not signals:
        return None
    
    # Extract day of week and hour info
    heatmap_data = []
    
    for signal in signals:
        if signal['tracked_returns']['1m'] is not None:
            signal_date = signal['date']
            day_of_week = signal_date.strftime('%A')
            hour = signal_date.hour
            
            # Use 1-month return as the performance metric
            return_value = signal['tracked_returns']['1m']
            
            heatmap_data.append({
                'Day': day_of_week,
                'Hour': hour,
                'Return (%)': return_value
            })
    
    if not heatmap_data:
        return None
        
    df = pd.DataFrame(heatmap_data)
    
    # Create pivot table for heatmap
    pivot_table = df.pivot_table(
        values='Return (%)', 
        index='Day', 
        columns='Hour', 
        aggfunc='mean'
    ).fillna(0)
    
    # Reorder days of week
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    pivot_table = pivot_table.reindex(day_order)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=pivot_table.values,
        x=pivot_table.columns,
        y=pivot_table.index,
        colorscale='RdYlGn',
        colorbar=dict(title='Avg. Return (%)'),
    ))
    
    fig.update_layout(
        title='Signal Performance by Day and Hour',
        xaxis_title='Hour of Day',
        yaxis_title='Day of Week',
        height=400
    )
    
    return fig


def display_signal_summary():
    """Display signal summary view"""
    # Update signal performance
    update_signal_performance()
    
    # Check if we have any signals
    if not st.session_state.historical_signals:
        st.info("No signals have been recorded yet. Signals will be automatically recorded when buy conditions are met.")
        return
    
    st.subheader("Signal Success Rate")
    
    # Select evaluation period
    col1, col2 = st.columns(2)
    with col1:
        period = st.selectbox(
            "Evaluation Period",
            options=['1d', '1w', '1m', '3m', '6m', 'current'],
            format_func=lambda x: {
                '1d': '1 Day',
                '1w': '1 Week',
                '1m': '1 Month',
                '3m': '3 Months',
                '6m': '6 Months',
                'current': 'Current'
            }.get(x, x),
            index=2
        )
    
    with col2:
        threshold = st.slider(
            "Success Threshold (%)",
            min_value=0.0,
            max_value=10.0,
            value=2.0,
            step=0.5,
            help="Minimum return percentage to consider a signal successful"
        )
    
    # Create summary table
    summary_data = []
    
    for ticker, signals in st.session_state.historical_signals.items():
        buy_signals = [s for s in signals if s['signal_type'] == 'BUY']
        success_rate, count = get_signal_success_rate(buy_signals, period, threshold)
        
        if count > 0:
            # Calculate average return
            valid_signals = [s for s in buy_signals if s['tracked_returns'][period] is not None]
            avg_return = sum(s['tracked_returns'][period] for s in valid_signals) / count if count > 0 else 0
            
            summary_data.append({
                "Ticker": ticker,
                "Signal Count": count,
                "Success Rate (%)": f"{success_rate:.1f}%",
                f"Avg. Return ({period})": f"{avg_return:.2f}%",
                "Best Signal": f"{max([s['tracked_returns'][period] for s in valid_signals], default=0):.2f}%",
                "Worst Signal": f"{min([s['tracked_returns'][period] for s in valid_signals], default=0):.2f}%",
            })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        st.table(summary_df)
        
        # Calculate overall success rate
        total_signals = sum(item['Signal Count'] for item in summary_data)
        overall_success = sum(item['Signal Count'] * float(item['Success Rate (%)'].rstrip('%')) / 100 for item in summary_data)
        overall_rate = (overall_success / total_signals) * 100 if total_signals > 0 else 0
        
        st.metric("Overall Signal Success Rate", f"{overall_rate:.1f}%")
    else:
        st.info(f"No completed signals for the selected period ({period}). Signals need time to mature for proper evaluation.")


def display_signal_history():
    """Display detailed signal history view"""
    # Update signal performance
    update_signal_performance()
    
    # Check if we have any signals
    if not st.session_state.historical_signals:
        st.info("No signals have been recorded yet. Signals will be automatically recorded when buy conditions are met.")
        return
    
    st.subheader("Signal History")
    
    # Select ticker for detailed view
    ticker_options = list(st.session_state.historical_signals.keys())
    if ticker_options:
        selected_ticker = st.selectbox("Select Stock", options=ticker_options)
        
        # Get signals for selected ticker
        ticker_signals = st.session_state.historical_signals[selected_ticker]
        
        if ticker_signals:
            # Convert signals to DataFrame for display
            signal_data = []
            
            for signal in ticker_signals:
                signal_data.append({
                    "Date": signal['date'].strftime('%Y-%m-%d %H:%M'),
                    "Type": signal['signal_type'],
                    "Price": f"${signal['price']:.2f}",
                    "Strength": f"{signal['signal_strength']}%",
                    "Indicator": signal['indicator'],
                    "1-Day": f"{signal['tracked_returns']['1d']:.2f}%" if signal['tracked_returns']['1d'] is not None else "N/A",
                    "1-Week": f"{signal['tracked_returns']['1w']:.2f}%" if signal['tracked_returns']['1w'] is not None else "N/A",
                    "1-Month": f"{signal['tracked_returns']['1m']:.2f}%" if signal['tracked_returns']['1m'] is not None else "N/A",
                    "3-Month": f"{signal['tracked_returns']['3m']:.2f}%" if signal['tracked_returns']['3m'] is not None else "N/A",
                    "6-Month": f"{signal['tracked_returns']['6m']:.2f}%" if signal['tracked_returns']['6m'] is not None else "N/A",
                    "Current": f"{signal['tracked_returns']['current']:.2f}%" if signal['tracked_returns']['current'] is not None else "N/A",
                })
            
            signal_df = pd.DataFrame(signal_data)
            st.dataframe(signal_df)
        else:
            st.info(f"No signals recorded for {selected_ticker}")
    else:
        st.info("No signals have been recorded yet")


def display_performance_charts():
    """Display signal performance charts view"""
    # Update signal performance
    update_signal_performance()
    
    # Check if we have any signals
    if not st.session_state.historical_signals:
        st.info("No signals have been recorded yet. Signals will be automatically recorded when buy conditions are met.")
        return
    
    st.subheader("Signal Performance Charts")
    
    # Only show if we have enough data
    all_signals = []
    for ticker, signals in st.session_state.historical_signals.items():
        for signal in signals:
            if signal['tracked_returns']['current'] is not None:
                signal_copy = signal.copy()
                signal_copy['ticker'] = ticker
                all_signals.append(signal_copy)
    
    if all_signals:
        # Create chart data
        chart_data = []
        for signal in all_signals:
            signal_date = signal['date'].strftime('%Y-%m-%d')
            for period, value in signal['tracked_returns'].items():
                if value is not None:
                    chart_data.append({
                        'Ticker': signal['ticker'],
                        'Signal Date': signal_date,
                        'Period': {
                            '1d': '1 Day',
                            '1w': '1 Week',
                            '1m': '1 Month',
                            '3m': '3 Months',
                            '6m': '6 Months',
                            'current': 'Current'
                        }.get(period, period),
                        'Return (%)': value,
                        'Signal Type': signal['signal_type'],
                        'Signal Strength': signal['signal_strength'],
                        'Indicator': signal['indicator']
                    })
        
        if chart_data:
            chart_df = pd.DataFrame(chart_data)
            
            # Create various charts
            st.subheader("Average Returns by Period")
            avg_returns = chart_df.groupby('Period')['Return (%)'].mean().reset_index()
            fig = px.bar(
                avg_returns, 
                x='Period', 
                y='Return (%)',
                color='Return (%)',
                color_continuous_scale='RdYlGn',
                labels={'Return (%)': 'Average Return (%)'},
                title="Average Signal Returns by Period"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Returns by indicator
            st.subheader("Returns by Indicator Type")
            indicator_returns = chart_df.groupby(['Indicator', 'Period'])['Return (%)'].mean().reset_index()
            fig = px.bar(
                indicator_returns,
                x='Period',
                y='Return (%)',
                color='Indicator',
                barmode='group',
                title="Average Returns by Indicator Type"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Signal strength correlation
            st.subheader("Signal Strength vs. Returns")
            fig = px.scatter(
                chart_df[chart_df['Period'] == 'Current'],
                x='Signal Strength',
                y='Return (%)',
                color='Ticker',
                size='Signal Strength',
                hover_data=['Signal Date', 'Indicator'],
                title="Correlation Between Signal Strength and Returns"
            )
            
            # Add trendline
            fig.update_layout(
                xaxis_title="Signal Strength (%)",
                yaxis_title="Return (%)",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("Not enough data for visualizations. Wait for signals to mature.")
    else:
        st.info("No completed signals available for charting. Signals will appear here once they have return data.")


def display_historical_analysis():
    """Display advanced historical signal analysis"""
    # Update signal performance
    update_signal_performance()
    
    # Get all signals in a single list
    all_signals = []
    for ticker, signals in st.session_state.historical_signals.items():
        for signal in signals:
            signal_copy = signal.copy()
            signal_copy['ticker'] = ticker
            all_signals.append(signal_copy)
    
    if not all_signals:
        st.info("No historical signals available for analysis.")
        return
    
    # Filter for buy signals
    all_buy_signals = [s for s in all_signals if s['signal_type'] == 'BUY']
    
    # Generate time series data
    time_series_df = get_signal_data_over_time()
    
    if time_series_df is not None:
        # Create signal performance over time chart
        st.subheader("Signal Performance Over Time")
        
        # Allow filtering by ticker
        tickers = sorted(time_series_df['Ticker'].unique())
        selected_tickers = st.multiselect(
            "Select Tickers to Display",
            options=tickers,
            default=tickers[:min(3, len(tickers))]
        )
        
        if selected_tickers:
            filtered_df = time_series_df[time_series_df['Ticker'].isin(selected_tickers)]
            
            fig = px.line(
                filtered_df, 
                x='Days After Signal', 
                y='Return (%)',
                color='Ticker',
                line_dash='Indicator',
                hover_data=['Signal Date'],
                title="Signal Returns Over Time",
                labels={'Return (%)': 'Return (%)', 'Days After Signal': 'Days After Signal'}
            )
            
            fig.update_layout(
                xaxis_title="Days After Signal",
                yaxis_title="Return (%)",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add description of the chart
            st.markdown("""
            **Chart Explanation:** This chart shows how the returns from buy signals evolve over time.
            Each line represents signals for a stock, with different line styles for different indicators.
            Steeper upward slopes indicate better performing signals.
            """)
    
    # Signal volatility analysis
    st.subheader("Signal Return Volatility")
    
    # Calculate volatility by indicator and ticker
    volatility_data = []
    
    for ticker, signals in st.session_state.historical_signals.items():
        buy_signals = [s for s in signals if s['signal_type'] == 'BUY']
        
        # Skip if no signals
        if not buy_signals:
            continue
        
        # Group by indicator
        indicator_groups = {}
        for signal in buy_signals:
            indicator = signal['indicator']
            if indicator not in indicator_groups:
                indicator_groups[indicator] = []
            indicator_groups[indicator].append(signal)
        
        # Calculate volatility for each group
        for indicator, ind_signals in indicator_groups.items():
            # Calculate for different periods
            for period in ['1d', '1w', '1m', '3m', '6m', 'current']:
                volatility = calculate_signal_volatility(ind_signals, period)
                
                if volatility is not None:
                    period_name = {
                        '1d': '1 Day',
                        '1w': '1 Week',
                        '1m': '1 Month',
                        '3m': '3 Months',
                        '6m': '6 Months',
                        'current': 'Current'
                    }.get(period, period)
                    
                    volatility_data.append({
                        'Ticker': ticker,
                        'Indicator': indicator,
                        'Period': period_name,
                        'Return Volatility': volatility,
                        'Signal Count': len(ind_signals)
                    })
    
    if volatility_data:
        volatility_df = pd.DataFrame(volatility_data)
        
        fig = px.bar(
            volatility_df,
            x='Period',
            y='Return Volatility',
            color='Indicator',
            facet_col='Ticker',
            hover_data=['Signal Count'],
            title="Signal Return Volatility by Period",
            labels={'Return Volatility': 'Return Volatility (Std Dev)'}
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Add explanation
        st.markdown("""
        **Volatility Analysis Explanation:** This chart shows the volatility (standard deviation) of returns for 
        signals by indicator type. Lower volatility means more consistent returns, while higher volatility indicates 
        more variable performance.
        """)
    else:
        st.info("Not enough data to calculate volatility metrics.")
    
    # Create and display heatmap
    st.subheader("Signal Performance Heatmap")
    
    # Filter signals that have 1-month returns
    signals_with_returns = [s for s in all_buy_signals if s['tracked_returns']['1m'] is not None]
    
    if len(signals_with_returns) >= 5:  # Need a minimum number for meaningful heatmap
        heatmap_fig = create_signal_performance_heatmap(signals_with_returns)
        if heatmap_fig:
            st.plotly_chart(heatmap_fig, use_container_width=True)
            
            # Add explanation
            st.markdown("""
            **Heatmap Explanation:** This heatmap shows the average 1-month return for signals generated on 
            different days of the week and hours of the day. Greener cells indicate better average returns,
            helping identify optimal timing for acting on signals.
            """)
    else:
        st.info("Need at least 5 signals with 1-month returns to generate the performance heatmap.")


def signal_tracker_section():
    """Main function for the signal tracking page"""
    st.title("📈 Signal Performance Tracking")
    st.markdown("""
    This page tracks the performance of buy signals generated by the system.
    See how effective different signals have been and which indicators perform best over time.
    """)
    
    initialize_signal_tracker()
    
    # Create buttons for save/load functionality
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 Save Signals"):
            save_historical_signals()
    
    with col2:
        if st.button("📂 Load Saved Signals"):
            load_historical_signals()
    
    # Create tabs for different views
    tabs = st.tabs(["Summary", "Detailed History", "Performance Charts", "Historical Analysis"])
    
    # Display in the appropriate tabs
    with tabs[0]:
        # Summary tab
        display_signal_summary()
        
    with tabs[1]:
        # Detailed History tab
        display_signal_history()
        
    with tabs[2]:
        # Performance Charts tab
        display_performance_charts()
    
    # Display advanced historical analysis in the last tab
    with tabs[3]:
        display_historical_analysis()
