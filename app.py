import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
import yfinance as yf
from utils import (calculate_rsi, check_buy_signal, get_stock_data, 
               calculate_macd, calculate_bollinger_bands, calculate_ema, get_last_signal_time)

# Page config
st.set_page_config(
    page_title="Stock Market Analysis Tool",
    page_icon="📈",
    layout="wide"
)

# Title and description
st.title("📈 Stock Market Analysis Tool")
st.markdown("""
This application monitors selected stocks, calculates technical indicators like RSI (Relative Strength Index),
and provides buy signals based on market conditions.
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
if 'ema_period' not in st.session_state:
    st.session_state.ema_period = 20
if 'bb_period' not in st.session_state:
    st.session_state.bb_period = 20
if 'last_signals' not in st.session_state:
    st.session_state.last_signals = {}

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    
    # Stock selection
    stock_input = st.text_input("Add a stock symbol:", placeholder="e.g., AAPL")
    if st.button("Add Stock") and stock_input:
        stock_input = stock_input.strip().upper()
        if stock_input not in st.session_state.selected_stocks:
            try:
                # Verify the stock exists by fetching its data
                stock = yf.Ticker(stock_input)
                hist = stock.history(period="1d")
                if not hist.empty:
                    st.session_state.selected_stocks.append(stock_input)
                    st.success(f"Added {stock_input} to your watchlist")
                else:
                    st.error(f"Could not find stock with symbol {stock_input}")
            except Exception as e:
                st.error(f"Error adding stock: {str(e)}")
    
    # Display and manage selected stocks
    st.subheader("Your Watchlist")
    for i, stock in enumerate(st.session_state.selected_stocks):
        col1, col2 = st.columns([4, 1])
        with col1:
            st.write(f"{i+1}. {stock}")
        with col2:
            if st.button("Remove", key=f"remove_{stock}"):
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
    tabs = st.tabs(["RSI", "MACD", "Bollinger", "EMA"])
    
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
        st.caption("Default MACD parameters: 12, 26, 9")
    
    with tabs[2]:  # Bollinger Tab
        st.session_state.show_bollinger = st.checkbox("Show Bollinger Bands", value=st.session_state.show_bollinger)
        st.session_state.bb_period = st.slider("Bollinger Period", min_value=5, max_value=50, value=st.session_state.bb_period)
    
    with tabs[3]:  # EMA Tab
        st.session_state.show_ema = st.checkbox("Show EMA", value=st.session_state.show_ema)
        st.session_state.ema_period = st.slider("EMA Period", min_value=5, max_value=50, value=st.session_state.ema_period)
    
    # Auto refresh settings
    st.subheader("Auto Refresh")
    st.session_state.auto_refresh = st.checkbox("Enable 15-minute Auto Refresh", value=st.session_state.auto_refresh)
    
    # Manual refresh button
    if st.button("Refresh Data Now"):
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

# Display last update time
st.caption(f"Last updated: {st.session_state.last_update.strftime('%Y-%m-%d %H:%M:%S')}")

# Main content - Stock analysis
if not st.session_state.selected_stocks:
    st.info("Please add stocks to your watchlist using the sidebar.")
else:
    # Fetch and analyze stock data
    progress_bar = st.progress(0)
    for i, ticker in enumerate(st.session_state.selected_stocks):
        # Update progress
        progress = (i + 1) / len(st.session_state.selected_stocks)
        progress_bar.progress(progress)
        
        try:
            # Get stock data
            data = get_stock_data(ticker, st.session_state.time_period)
            
            if data is not None:
                st.session_state.stocks_data[ticker] = data
            else:
                st.error(f"Could not fetch data for {ticker}")
        except Exception as e:
            st.error(f"Error analyzing {ticker}: {str(e)}")
    
    # Remove progress bar after loading
    progress_bar.empty()
    
    # Display stock analysis cards in a grid
    cols = st.columns(min(3, len(st.session_state.selected_stocks)))
    
    for i, ticker in enumerate(st.session_state.selected_stocks):
        if ticker in st.session_state.stocks_data:
            with cols[i % len(cols)]:
                data = st.session_state.stocks_data[ticker]
                close_data = data['Close']
                
                # Calculate RSI
                rsi = calculate_rsi(close_data, st.session_state.rsi_period)
                current_rsi = rsi.iloc[-1] if not rsi.empty else None
                
                # Calculate daily change
                current_price = close_data.iloc[-1] if not close_data.empty else None
                previous_price = close_data.iloc[-2] if len(close_data) > 1 else None
                
                daily_change = 0
                if current_price is not None and previous_price is not None:
                    daily_change = (current_price - previous_price) / previous_price * 100
                
                # Calculate additional technical indicators
                macd_line, macd_signal, macd_hist = None, None, None
                current_macd_line, current_macd_signal = None, None
                ema = None
                bb_upper, bb_middle, bb_lower = None, None, None
                current_bb_upper, current_bb_middle, current_bb_lower = None, None, None
                
                if st.session_state.show_macd:
                    macd_line, macd_signal, macd_hist = calculate_macd(close_data)
                    if macd_line is not None and not macd_line.empty:
                        current_macd_line = macd_line.iloc[-1] 
                    if macd_signal is not None and not macd_signal.empty:
                        current_macd_signal = macd_signal.iloc[-1]
                
                if st.session_state.show_ema:
                    ema = calculate_ema(close_data, st.session_state.ema_period)
                
                if st.session_state.show_bollinger:
                    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(close_data, st.session_state.bb_period)
                    if bb_upper is not None and not bb_upper.empty:
                        current_bb_upper = bb_upper.iloc[-1]
                    if bb_middle is not None and not bb_middle.empty:
                        current_bb_middle = bb_middle.iloc[-1]
                    if bb_lower is not None and not bb_lower.empty:
                        current_bb_lower = bb_lower.iloc[-1]
                
                # Get last signal time
                days_since_signal, last_signal_date = get_last_signal_time(
                    ticker, 
                    rsi_threshold=30, 
                    drop_threshold=st.session_state.drop_threshold
                )
                
                # Create stock card
                with st.container():
                    st.subheader(f"{ticker}")
                    
                    # Price and change
                    if current_price is not None:
                        price_col, change_col = st.columns(2)
                        with price_col:
                            st.metric("Price", f"${current_price:.2f}")
                        with change_col:
                            st.metric("Daily Change", f"{daily_change:.2f}%", 
                                     delta_color="inverse" if daily_change < 0 else "normal")
                    
                    # Technical indicators
                    metrics_cols = st.columns(3)
                    
                    with metrics_cols[0]:
                        if current_rsi is not None:
                            rsi_color = "red" if current_rsi < 30 else "green" if current_rsi > 70 else "normal"
                            st.metric("RSI", f"{current_rsi:.2f}")
                    
                    with metrics_cols[1]:
                        if st.session_state.show_macd and current_macd_line is not None and current_macd_signal is not None:
                            macd_value = current_macd_line - current_macd_signal
                            st.metric("MACD", f"{macd_value:.2f}", 
                                     delta=f"{current_macd_line:.2f} - {current_macd_signal:.2f}")
                    
                    with metrics_cols[2]:
                        if days_since_signal is not None:
                            st.metric("Days Since Buy Signal", f"{days_since_signal}")
                            last_signal_str = last_signal_date.strftime("%Y-%m-%d") if last_signal_date else "None"
                            st.caption(f"Last signal: {last_signal_str}")
                    
                    # Buy signal with Yahoo Finance link
                    signal = check_buy_signal(
                        ticker, current_price, current_rsi, daily_change, 
                        st.session_state.drop_threshold,
                        current_macd_line, current_macd_signal,
                        current_bb_upper, current_bb_middle, current_bb_lower
                    )
                    
                    # Create Yahoo Finance URL
                    yahoo_finance_url = f"https://finance.yahoo.com/quote/{ticker}"
                    
                    # Display signal text with clickable BUY SIGNAL
                    if "BUY SIGNAL DETECTED" in signal:
                        # Split the signal at the buy signal text
                        parts = signal.split("✅ BUY SIGNAL DETECTED")
                        
                        # Display the first part as plain text
                        st.text(parts[0])
                        
                        # Display the buy signal as a clickable link
                        buy_signal_text = f"✅ BUY SIGNAL DETECTED{parts[1] if len(parts) > 1 else ''}"
                        st.markdown(f"[{buy_signal_text}]({yahoo_finance_url})")
                    else:
                        # Just display as regular text if no strong buy signal
                        st.text(signal)
                        st.markdown(f"[View on Yahoo Finance 🔍]({yahoo_finance_url})")
                    
                    
                    # Create interactive chart with 2 or 3 rows based on if MACD is shown
                    if st.session_state.show_macd:
                        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                                          vertical_spacing=0.05, row_heights=[0.6, 0.2, 0.2])
                    else:
                        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                          vertical_spacing=0.1, row_heights=[0.7, 0.3])
                    
                    # Price chart with candlesticks
                    fig.add_trace(
                        go.Candlestick(
                            x=data.index,
                            open=data['Open'],
                            high=data['High'],
                            low=data['Low'],
                            close=data['Close'],
                            name="Price"
                        ),
                        row=1, col=1
                    )
                    
                    # Add EMA if enabled
                    if st.session_state.show_ema and ema is not None:
                        fig.add_trace(
                            go.Scatter(
                                x=data.index,
                                y=ema,
                                line=dict(color="orange", width=1.5),
                                name=f"EMA ({st.session_state.ema_period})"
                            ),
                            row=1, col=1
                        )
                    
                    # Add Bollinger Bands if enabled
                    if st.session_state.show_bollinger and bb_upper is not None:
                        fig.add_trace(
                            go.Scatter(
                                x=data.index,
                                y=bb_upper,
                                line=dict(color="rgba(0, 0, 255, 0.5)", width=1, dash="dot"),
                                name="Upper Band"
                            ),
                            row=1, col=1
                        )
                        
                        fig.add_trace(
                            go.Scatter(
                                x=data.index,
                                y=bb_middle,
                                line=dict(color="rgba(0, 0, 255, 0.5)", width=1),
                                name="Middle Band"
                            ),
                            row=1, col=1
                        )
                        
                        fig.add_trace(
                            go.Scatter(
                                x=data.index,
                                y=bb_lower,
                                line=dict(color="rgba(0, 0, 255, 0.5)", width=1, dash="dot"),
                                name="Lower Band",
                                fill='tonexty',
                                fillcolor='rgba(0, 0, 255, 0.05)'
                            ),
                            row=1, col=1
                        )
                    
                    # RSI chart
                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=rsi,
                            line=dict(color="purple", width=1.5),
                            name="RSI"
                        ),
                        row=2, col=1
                    )
                    
                    # Add RSI threshold lines
                    fig.add_trace(
                        go.Scatter(
                            x=[data.index[0], data.index[-1]],
                            y=[30, 30],
                            line=dict(color="green", width=1, dash="dash"),
                            name="Oversold",
                            showlegend=False
                        ),
                        row=2, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(
                            x=[data.index[0], data.index[-1]],
                            y=[70, 70],
                            line=dict(color="red", width=1, dash="dash"),
                            name="Overbought",
                            showlegend=False
                        ),
                        row=2, col=1
                    )
                    
                    # Add MACD chart if enabled
                    if st.session_state.show_macd and macd_line is not None:
                        # MACD line
                        fig.add_trace(
                            go.Scatter(
                                x=data.index,
                                y=macd_line,
                                line=dict(color="blue", width=1.5),
                                name="MACD Line"
                            ),
                            row=3, col=1
                        )
                        
                        # Signal line
                        fig.add_trace(
                            go.Scatter(
                                x=data.index,
                                y=macd_signal,
                                line=dict(color="red", width=1.5),
                                name="Signal Line"
                            ),
                            row=3, col=1
                        )
                        
                        # Histogram
                        if macd_hist is not None and not macd_hist.empty and len(macd_hist) > 0:
                            try:
                                colors = ['green' if val >= 0 else 'red' for val in macd_hist]
                                fig.add_trace(
                                    go.Bar(
                                        x=data.index,
                                        y=macd_hist,
                                        marker_color=colors,
                                        name="Histogram"
                                    ),
                                    row=3, col=1
                                )
                            except Exception as e:
                                st.error(f"Error plotting MACD histogram: {str(e)}")
                    
                    # Update layout
                    title_text = f"{ticker} Price and Indicators"
                    
                    fig.update_layout(
                        title=title_text,
                        xaxis_title="Date",
                        yaxis_title="Price ($)",
                        height=700 if st.session_state.show_macd else 500,
                        xaxis_rangeslider_visible=False,
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        )
                    )
                    
                    # Update y-axis labels
                    fig.update_yaxes(title_text="RSI", row=2, col=1)
                    
                    if st.session_state.show_macd:
                        fig.update_yaxes(title_text="MACD", row=3, col=1)
                    
                    st.plotly_chart(fig, use_container_width=True)

# Explanatory section
with st.expander("Understanding Technical Indicators and Buy Signals"):
    st.markdown("""
    ## Relative Strength Index (RSI)
    RSI is a momentum oscillator that measures the speed and change of price movements. It ranges from 0 to 100:
    - **RSI below 30**: The stock is potentially oversold (good buying opportunity)
    - **RSI above 70**: The stock is potentially overbought (might be time to sell)
    
    ## Moving Average Convergence Divergence (MACD)
    MACD is a trend-following momentum indicator that shows the relationship between two moving averages:
    - **MACD Line**: The difference between the 12-period and 26-period EMAs
    - **Signal Line**: 9-period EMA of the MACD Line
    - **Histogram**: Difference between MACD Line and Signal Line
    - **Buy Signal**: When the MACD Line crosses above the Signal Line (bullish)
    - **Sell Signal**: When the MACD Line crosses below the Signal Line (bearish)
    
    ## Bollinger Bands
    Bollinger Bands consist of three lines:
    - **Middle Band**: 20-period simple moving average (SMA)
    - **Upper Band**: Middle Band + (2 × standard deviation)
    - **Lower Band**: Middle Band - (2 × standard deviation)
    - **Buy Signal**: When price falls below the Lower Band (potentially oversold)
    - **Sell Signal**: When price rises above the Upper Band (potentially overbought)
    
    ## Exponential Moving Average (EMA)
    EMA gives more weight to recent prices, making it more responsive to new information:
    - **Buy Signal**: When price crosses above the EMA
    - **Sell Signal**: When price crosses below the EMA
    - Unlike simple moving averages, EMAs react more quickly to price changes
    
    ## Our Buy Signals
    This tool generates buy signals based on a combination of these indicators:
    1. **RSI Values**: When RSI falls below 30 (oversold condition)
    2. **Price Drops**: Sudden drops in price (above threshold)
    3. **MACD**: When MACD Line is above Signal Line (bullish momentum)
    4. **Bollinger Bands**: When price falls below the Lower Band (potentially oversold)
    
    The overall signal strength shows what percentage of the indicators are indicating a buy signal.
    
    ## Best Practices
    - Use this tool as one factor in your investment decision, not the sole basis
    - Consider fundamental analysis alongside technical indicators
    - Always do your own research before investing
    - Past performance is not indicative of future results
    - Different indicators may work better for different market conditions
    """)

st.caption("Note: This application is for informational purposes only and should not be considered financial advice.")
