import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

def calculate_rsi(data, period=14):
    """
    Calculate the Relative Strength Index (RSI) for a given price series
    
    Args:
        data: Pandas Series of price data
        period: RSI calculation period (default: 14)
        
    Returns:
        Pandas Series containing RSI values
    """
    # Ensure we have enough data
    if len(data) < period + 1:
        return pd.Series(index=data.index)
    
    # Calculate price changes
    delta = data.diff()
    
    # Separate gains and losses
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)
    
    # Calculate average gains and losses
    avg_gain = gains.rolling(window=period).mean()
    avg_loss = losses.rolling(window=period).mean()
    
    # Calculate RS (Relative Strength)
    rs = avg_gain / avg_loss
    
    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def get_stock_data(ticker, period="1mo"):
    """
    Fetch stock data for a given ticker
    
    Args:
        ticker: Stock symbol
        period: Time period for data (default: "1mo")
        
    Returns:
        Pandas DataFrame with stock data or None if error
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        
        if hist.empty:
            return None
            
        return hist
    except Exception as e:
        print(f"Error fetching data for {ticker}: {str(e)}")
        return None

def calculate_macd(data, fast_period=12, slow_period=26, signal_period=9):
    """
    Calculate the Moving Average Convergence Divergence (MACD)
    
    Args:
        data: Pandas Series of price data
        fast_period: Fast EMA period (default: 12)
        slow_period: Slow EMA period (default: 26)
        signal_period: Signal EMA period (default: 9)
        
    Returns:
        Tuple of (MACD line, Signal line, Histogram)
    """
    # Calculate EMAs
    ema_fast = data.ewm(span=fast_period, adjust=False).mean()
    ema_slow = data.ewm(span=slow_period, adjust=False).mean()
    
    # Calculate MACD line
    macd_line = ema_fast - ema_slow
    
    # Calculate Signal line
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    
    # Calculate Histogram
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram

def calculate_bollinger_bands(data, period=20, num_std=2):
    """
    Calculate Bollinger Bands
    
    Args:
        data: Pandas Series of price data
        period: Moving average period (default: 20)
        num_std: Number of standard deviations (default: 2)
        
    Returns:
        Tuple of (Upper Band, Middle Band, Lower Band)
    """
    # Calculate middle band (SMA)
    middle_band = data.rolling(window=period).mean()
    
    # Calculate standard deviation
    std = data.rolling(window=period).std()
    
    # Calculate upper and lower bands
    upper_band = middle_band + (std * num_std)
    lower_band = middle_band - (std * num_std)
    
    return upper_band, middle_band, lower_band

def calculate_ema(data, period=20):
    """
    Calculate Exponential Moving Average (EMA)
    
    Args:
        data: Pandas Series of price data
        period: EMA period (default: 20)
        
    Returns:
        Pandas Series with EMA values
    """
    return data.ewm(span=period, adjust=False).mean()

def get_last_signal_time(ticker, rsi_threshold=30, drop_threshold=5.0, lookback_days=30):
    """
    Find the last time a buy signal was generated
    
    Args:
        ticker: Stock symbol
        rsi_threshold: RSI threshold for buy signal (default: 30)
        drop_threshold: Price drop threshold in percent (default: 5.0)
        lookback_days: Number of days to look back (default: 30)
        
    Returns:
        Tuple of (days since last signal, datetime of last signal)
    """
    try:
        # Get historical data for the lookback period (add some buffer)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days + 10)  # Add buffer days
        
        stock = yf.Ticker(ticker)
        hist = stock.history(start=start_date, end=end_date)
        
        if hist.empty or len(hist) < 2:
            return None, None
        
        # Calculate RSI
        rsi_values = calculate_rsi(hist['Close'])
        
        # Calculate daily changes
        daily_changes = hist['Close'].pct_change() * 100
        
        # Find days with buy signals
        signal_days = []
        
        for i in range(1, len(hist)):
            if (rsi_values.iloc[i] < rsi_threshold or 
                daily_changes.iloc[i] < -drop_threshold):
                signal_days.append(hist.index[i])
        
        if not signal_days:
            return None, None
            
        # Get the most recent signal
        last_signal = max(signal_days)
        
        # Convert to timezone-naive for comparison
        last_signal_naive = last_signal.replace(tzinfo=None)
        
        # Calculate days since signal
        days_since_signal = (end_date - last_signal_naive).days
        
        return days_since_signal, last_signal
        
    except Exception as e:
        print(f"Error finding last signal for {ticker}: {str(e)}")
        return None, None

def check_buy_signal(ticker, current_price, rsi, daily_change, drop_threshold=5.0, 
                    macd_line=None, macd_signal=None, bb_upper=None, bb_middle=None, bb_lower=None):
    """
    Determine if a stock has a buy signal based on multiple technical indicators
    
    Args:
        ticker: Stock symbol
        current_price: Current stock price
        rsi: Current RSI value
        daily_change: Percentage change from previous day
        drop_threshold: Threshold for significant price drop (default: 5.0%)
        macd_line: Current MACD line value (optional)
        macd_signal: Current MACD signal line value (optional)
        bb_upper: Current Bollinger Band upper value (optional)
        bb_middle: Current Bollinger Band middle value (optional)
        bb_lower: Current Bollinger Band lower value (optional)
        
    Returns:
        Plain text string with buy signal recommendation
    """
    if current_price is None or rsi is None:
        return f"⚠️ Insufficient data for {ticker}"
        
    signal = ""
    buy_signals = 0
    total_signals = 0
    
    # Check RSI condition (oversold)
    total_signals += 1
    if rsi < 30:
        signal += "👉 RSI below 30 (oversold) - Consider buying\n"
        buy_signals += 1
    elif rsi > 70:
        signal += "🔴 RSI above 70 (overbought) - Consider waiting\n"
    else:
        signal += "RSI in neutral range\n"
    
    # Check for significant price drop
    total_signals += 1
    if daily_change < -drop_threshold:
        signal += f"📉 Price dropped {abs(daily_change):.2f}% - Potential opportunity\n"
        buy_signals += 1
    
    # Check MACD if available
    if macd_line is not None and macd_signal is not None:
        total_signals += 1
        if macd_line > macd_signal:
            signal += "📈 MACD: Bullish signal (MACD line above Signal line)\n"
            buy_signals += 1
        else:
            signal += "📉 MACD: Bearish signal (MACD line below Signal line)\n"
    
    # Check Bollinger Bands if available
    if current_price is not None and bb_lower is not None:
        total_signals += 1
        if current_price < bb_lower:
            signal += "📊 Bollinger Bands: Price below lower band (potential oversold)\n"
            buy_signals += 1
        elif current_price > bb_upper and bb_upper is not None:
            signal += "📊 Bollinger Bands: Price above upper band (potential overbought)\n"
        else:
            signal += "📊 Bollinger Bands: Price within bands (neutral)\n"
    
    # Calculate signal strength
    signal_strength = buy_signals / total_signals if total_signals > 0 else 0
    
    # Final recommendation
    if signal_strength >= 0.5:  # At least half of indicators are showing buy signals
        signal += f"✅ BUY SIGNAL DETECTED (Strength: {signal_strength:.0%})"
        return signal
    elif signal_strength > 0:
        signal += f"⚠️ WEAK BUY SIGNAL (Strength: {signal_strength:.0%})"
        return signal
    else:
        signal += "⚠️ No buy signal at this time"
        return signal
