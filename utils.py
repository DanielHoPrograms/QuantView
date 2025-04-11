import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import scipy.signal as signal
from scipy.stats import linregress
from typing import Tuple, Dict, List, Union, Optional

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

def get_stock_data(ticker, period="1mo", max_retries=3):
    """
    Fetch stock data for a given ticker with robust error handling and retries
    
    Args:
        ticker: Stock symbol
        period: Time period for data (default: "1mo")
        max_retries: Maximum number of retry attempts
        
    Returns:
        Pandas DataFrame with stock data or None if error
    """
    for attempt in range(max_retries):
        try:
            print(f"Fetching data for {ticker}, attempt {attempt+1}/{max_retries}")
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            
            # Check if we got valid data
            if len(hist) > 0 and len(hist) > 3 and 'Close' in hist.columns:
                return hist
            else:
                # If data is empty or missing columns, try again
                if attempt < max_retries - 1:
                    print(f"Empty or invalid data for {ticker}, retrying...")
                    import time
                    time.sleep(1)  # Small delay between attempts
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Error fetching data for {ticker}: {str(e)}, retrying...")
                import time
                time.sleep(1)  # Small delay between attempts
    
    # All attempts failed, try one more approach with a different period
    try:
        print(f"Trying with a different period for {ticker}...")
        alt_period = "3mo" if period != "3mo" else "1mo"
        stock = yf.Ticker(ticker)
        hist = stock.history(period=alt_period)
        if len(hist) > 0 and len(hist) > 3:
            print(f"Successfully fetched {ticker} with alternative period {alt_period}")
            return hist
    except Exception as e:
        print(f"Final attempt failed for {ticker}: {e}")
    
    # If all else failed, try download method instead of history
    try:
        print(f"Trying yf.download for {ticker}...")
        hist = yf.download(ticker, period=period, progress=False)
        if len(hist) > 0 and len(hist) > 3:
            print(f"Successfully fetched {ticker} with yf.download")
            return hist
    except Exception as e:
        print(f"Download attempt failed for {ticker}: {e}")
    
    print(f"All attempts to fetch data for {ticker} failed. Returning None.")
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
    Find the last time a buy signal was generated with robust error handling
    
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
        
        # Use our improved stock data fetching function
        hist = get_stock_data(ticker, period=f"{lookback_days + 10}d")
        
        # If that didn't work, try direct download
        if hist is None or len(hist) == 0 or len(hist) < 2:
            try:
                print(f"Trying direct download for {ticker} signal detection...")
                hist = yf.download(ticker, start=start_date, end=end_date, progress=False)
            except Exception as e:
                print(f"Direct download failed for {ticker} signal detection: {e}")
                return None, None
        
        # If we still don't have data, give up
        if hist is None or len(hist) == 0 or len(hist) < 2:
            print(f"Could not get sufficient data for {ticker} signal detection")
            return None, None
        
        # Calculate RSI
        rsi_values = calculate_rsi(hist['Close'])
        
        # Calculate daily changes
        daily_changes = hist['Close'].pct_change() * 100
        
        # Find days with buy signals
        signal_days = []
        
        for i in range(1, len(hist)):
            # Check for valid RSI value or significant price drop
            if i < len(rsi_values) and not pd.isna(rsi_values.iloc[i]) and not pd.isna(daily_changes.iloc[i]):
                if (rsi_values.iloc[i] < rsi_threshold or 
                    daily_changes.iloc[i] < -drop_threshold):
                    signal_days.append(hist.index[i])
        
        if not signal_days:
            return None, None
            
        # Get the most recent signal
        last_signal = max(signal_days)
        
        # Handle timezone information
        try:
            # Convert to timezone-naive for comparison
            last_signal_naive = last_signal.replace(tzinfo=None)
        except:
            # If replace fails, just use the original
            last_signal_naive = last_signal
        
        # Calculate days since signal
        days_since_signal = (end_date - last_signal_naive).days
        
        return days_since_signal, last_signal
        
    except Exception as e:
        print(f"Error finding last signal for {ticker}: {str(e)}")
        return None, None

def calculate_stochastic_oscillator(high, low, close, k_period=14, d_period=3, smooth_k=3):
    """
    Calculate the Stochastic Oscillator
    
    Args:
        high: Pandas Series of high prices
        low: Pandas Series of low prices
        close: Pandas Series of close prices
        k_period: Period for %K calculation (default: 14)
        d_period: Period for %D calculation (default: 3)
        smooth_k: Smoothing factor for %K (default: 3)
        
    Returns:
        Tuple of (%K, %D)
    """
    # Calculate %K
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    
    # Calculate raw %K
    k_raw = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    
    # Apply smoothing to %K if specified
    if smooth_k > 1:
        k = k_raw.rolling(window=smooth_k).mean()
    else:
        k = k_raw
    
    # Calculate %D (SMA of %K)
    d = k.rolling(window=d_period).mean()
    
    return k, d

def calculate_adx(high, low, close, period=14):
    """
    Calculate the Average Directional Index (ADX)
    
    Args:
        high: Pandas Series of high prices
        low: Pandas Series of low prices
        close: Pandas Series of close prices
        period: Period for ADX calculation (default: 14)
        
    Returns:
        Tuple of (ADX, +DI, -DI)
    """
    # Calculate True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    
    # Calculate +DM and -DM
    plus_dm = high.diff()
    minus_dm = low.diff()
    
    # Ensure +DM is positive and exceeds -DM
    plus_dm = plus_dm.where((plus_dm > 0) & (plus_dm > minus_dm.abs()), 0)
    
    # Ensure -DM is positive and exceeds +DM
    minus_dm = minus_dm.abs().where((minus_dm < 0) & (minus_dm.abs() > plus_dm), 0)
    
    # Calculate +DI and -DI
    plus_di = 100 * (plus_dm.rolling(period).sum() / atr.replace(0, np.nan))
    minus_di = 100 * (minus_dm.rolling(period).sum() / atr.replace(0, np.nan))
    
    # Calculate DX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    
    # Calculate ADX
    adx = dx.rolling(period).mean()
    
    return adx, plus_di, minus_di

def calculate_obv(close, volume):
    """
    Calculate On-Balance Volume (OBV)
    
    Args:
        close: Pandas Series of close prices
        volume: Pandas Series of volume data
        
    Returns:
        Pandas Series with OBV values
    """
    obv = pd.Series(index=close.index)
    obv.iloc[0] = volume.iloc[0]
    
    for i in range(1, len(close)):
        if close.iloc[i] > close.iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
        elif close.iloc[i] < close.iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i-1]
    
    return obv

def calculate_ichimoku(high, low, close, tenkan_period=9, kijun_period=26, senkou_period=52, displacement=26):
    """
    Calculate Ichimoku Cloud indicator
    
    Args:
        high: Pandas Series of high prices
        low: Pandas Series of low prices
        close: Pandas Series of close prices
        tenkan_period: Period for Tenkan-sen (Conversion Line) (default: 9)
        kijun_period: Period for Kijun-sen (Base Line) (default: 26)
        senkou_period: Period for Senkou Span B (Leading Span B) (default: 52)
        displacement: Period for Cloud displacement (default: 26)
        
    Returns:
        Dictionary with Ichimoku components
    """
    # Calculate Tenkan-sen (Conversion Line)
    tenkan_high = high.rolling(window=tenkan_period).max()
    tenkan_low = low.rolling(window=tenkan_period).min()
    tenkan_sen = (tenkan_high + tenkan_low) / 2
    
    # Calculate Kijun-sen (Base Line)
    kijun_high = high.rolling(window=kijun_period).max()
    kijun_low = low.rolling(window=kijun_period).min()
    kijun_sen = (kijun_high + kijun_low) / 2
    
    # Calculate Senkou Span A (Leading Span A)
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(displacement)
    
    # Calculate Senkou Span B (Leading Span B)
    senkou_high = high.rolling(window=senkou_period).max()
    senkou_low = low.rolling(window=senkou_period).min()
    senkou_span_b = ((senkou_high + senkou_low) / 2).shift(displacement)
    
    # Calculate Chikou Span (Lagging Span)
    chikou_span = close.shift(-displacement)
    
    return {
        'tenkan_sen': tenkan_sen,
        'kijun_sen': kijun_sen,
        'senkou_span_a': senkou_span_a,
        'senkou_span_b': senkou_span_b,
        'chikou_span': chikou_span
    }

def calculate_awesome_oscillator(high, low, fast_period=5, slow_period=34):
    """
    Calculate the Awesome Oscillator
    
    Args:
        high: Pandas Series of high prices
        low: Pandas Series of low prices
        fast_period: Period for fast SMA (default: 5)
        slow_period: Period for slow SMA (default: 34)
        
    Returns:
        Pandas Series with Awesome Oscillator values
    """
    # Calculate midpoint prices
    midpoint = (high + low) / 2
    
    # Calculate SMAs
    sma_fast = midpoint.rolling(window=fast_period).mean()
    sma_slow = midpoint.rolling(window=slow_period).mean()
    
    # Calculate Awesome Oscillator
    ao = sma_fast - sma_slow
    
    return ao




def detect_chart_patterns(df, window=20):
    """
    Detect common chart patterns in price data
    
    Args:
        df: DataFrame with OHLC price data
        window: Window size for pattern detection (default: 20)
        
    Returns:
        Dictionary with detected patterns
    """
    close = df['Close']
    high = df['High']
    low = df['Low']
    
    patterns = {
        'head_and_shoulders': False,
        'double_top': False,
        'double_bottom': False,
        'triangle': False,
        'wedge': False,
        'channel': False,
        'support_levels': [],
        'resistance_levels': []
    }
    
    # Not enough data for pattern detection
    if len(close) < window:
        return patterns
    
    # Find local peaks and valleys using signal processing
    close_array = close.values
    peak_indices = signal.find_peaks(close_array, distance=5)[0]
    valley_indices = signal.find_peaks(-close_array, distance=5)[0]
    
    # Extract peak and valley values
    peaks = [close_array[i] for i in peak_indices if i < len(close_array)]
    valleys = [close_array[i] for i in valley_indices if i < len(close_array)]
    
    # Detect double top pattern (two similar peaks with a valley in between)
    if len(peaks) >= 2:
        # Check last two peaks
        peak1, peak2 = peaks[-2], peaks[-1]
        peak_diff_pct = abs(peak1 - peak2) / peak1
        
        # If peaks are within 3% of each other and there's a valley in between
        if peak_diff_pct < 0.03 and len(valleys) > 0:
            # Make sure at least one valley between the peaks
            valley_between = False
            for v in valleys:
                if v < min(peak1, peak2) and peak_indices[-2] < valley_indices[-1] < peak_indices[-1]:
                    valley_between = True
                    break
            
            if valley_between:
                patterns['double_top'] = True
    
    # Detect double bottom pattern (two similar valleys with a peak in between)
    if len(valleys) >= 2:
        # Check last two valleys
        valley1, valley2 = valleys[-2], valleys[-1]
        valley_diff_pct = abs(valley1 - valley2) / valley1
        
        # If valleys are within 3% of each other and there's a peak in between
        if valley_diff_pct < 0.03 and len(peaks) > 0:
            # Make sure at least one peak between the valleys
            peak_between = False
            for p in peaks:
                if p > max(valley1, valley2) and valley_indices[-2] < peak_indices[-1] < valley_indices[-1]:
                    peak_between = True
                    break
            
            if peak_between:
                patterns['double_bottom'] = True
    
    # Detect head and shoulders pattern
    if len(peaks) >= 3 and len(valleys) >= 2:
        # Get the last three peaks and two valleys
        shoulder1, head, shoulder2 = peaks[-3], peaks[-2], peaks[-1]
        valley1, valley2 = valleys[-2], valleys[-1]
        
        # Check if middle peak (head) is higher than shoulders
        if (head > shoulder1 and head > shoulder2 and 
            abs(shoulder1 - shoulder2) / shoulder1 < 0.05 and 
            valley1 < min(shoulder1, shoulder2) and 
            valley2 < min(shoulder1, shoulder2)):
            patterns['head_and_shoulders'] = True
    
    # Detect support and resistance levels
    if len(valleys) >= 2:
        # Find clusters of valleys (support)
        for i in range(len(valleys) - 1):
            for j in range(i + 1, len(valleys)):
                if abs(valleys[i] - valleys[j]) / valleys[i] < 0.02:  # Within 2%
                    support_level = (valleys[i] + valleys[j]) / 2
                    if support_level not in patterns['support_levels']:
                        patterns['support_levels'].append(round(support_level, 2))
    
    if len(peaks) >= 2:
        # Find clusters of peaks (resistance)
        for i in range(len(peaks) - 1):
            for j in range(i + 1, len(peaks)):
                if abs(peaks[i] - peaks[j]) / peaks[i] < 0.02:  # Within 2%
                    resistance_level = (peaks[i] + peaks[j]) / 2
                    if resistance_level not in patterns['resistance_levels']:
                        patterns['resistance_levels'].append(round(resistance_level, 2))
    
    # Detect triangle patterns
    if len(peaks) >= 3 and len(valleys) >= 3:
        # Get the last three peaks and valleys
        peak_x = [peak_indices[-3], peak_indices[-2], peak_indices[-1]]
        peak_y = [peaks[-3], peaks[-2], peaks[-1]]
        
        valley_x = [valley_indices[-3], valley_indices[-2], valley_indices[-1]]
        valley_y = [valleys[-3], valleys[-2], valleys[-1]]
        
        # Check for converging trend lines (triangle)
        peak_slope, _, peak_r, _, _ = linregress(peak_x, peak_y)
        valley_slope, _, valley_r, _, _ = linregress(valley_x, valley_y)
        
        # Triangle: slopes are in opposite directions and good correlation
        if (peak_slope * valley_slope < 0 and 
            abs(peak_r) > 0.7 and abs(valley_r) > 0.7):
            patterns['triangle'] = True
    
    # Detect channels (parallel support and resistance)
    if len(peaks) >= 3 and len(valleys) >= 3:
        # Calculate trend lines
        peak_x = [peak_indices[-3], peak_indices[-2], peak_indices[-1]]
        peak_y = [peaks[-3], peaks[-2], peaks[-1]]
        
        valley_x = [valley_indices[-3], valley_indices[-2], valley_indices[-1]]
        valley_y = [valleys[-3], valleys[-2], valleys[-1]]
        
        peak_slope, peak_intercept, peak_r, _, _ = linregress(peak_x, peak_y)
        valley_slope, valley_intercept, valley_r, _, _ = linregress(valley_x, valley_y)
        
        # Channel: slopes are similar (parallel) and good correlation
        if (abs(peak_slope - valley_slope) / abs(peak_slope) < 0.2 and 
            abs(peak_r) > 0.7 and abs(valley_r) > 0.7):
            patterns['channel'] = True
    
    return patterns

def create_custom_indicator(df, indicators, weights=None):
    """
    Create a custom combined indicator based on multiple technical indicators
    
    Args:
        df: DataFrame with price data and calculated indicators
        indicators: List of indicator column names to combine
        weights: List of weights for each indicator (default: equal weights)
        
    Returns:
        Pandas Series with custom indicator values scaled from 0-100
    """
    if len(indicators) == 0:
        return pd.Series(index=df.index)
    
    # Validate that indicators exist in DataFrame
    available_indicators = [ind for ind in indicators if ind in df.columns]
    if not available_indicators:
        # Return neutral values if no indicators are available
        return pd.Series(50, index=df.index)
    
    # If no weights are provided, use equal weights
    if weights is None:
        weights = [1] * len(available_indicators)
    else:
        # Match weights to available indicators
        weights = [weights[i] for i, ind in enumerate(indicators) if ind in df.columns]
    
    # Normalize weights to sum to 1
    weights = [w / sum(weights) for w in weights]
    
    # Create custom indicator
    custom = pd.Series(0, index=df.index)
    
    # Combine indicators using weights
    for i, indicator in enumerate(available_indicators):
        # Normalize the indicator to 0-100 scale if necessary
        if df[indicator].min() < 0 or df[indicator].max() > 100:
            if df[indicator].max() == df[indicator].min():
                normalized_indicator = df[indicator].copy()
                normalized_indicator[:] = 50  # Neutral value
            else:
                normalized_indicator = (df[indicator] - df[indicator].min()) / (df[indicator].max() - df[indicator].min()) * 100
        else:
            normalized_indicator = df[indicator]
        
        custom += normalized_indicator * weights[i]
    
    # Ensure result is within 0-100 range
    custom = custom.clip(0, 100)
    
    return custom

def check_buy_signal(ticker, current_price, rsi, daily_change, drop_threshold=5.0, 
                    macd_line=None, macd_signal=None, bb_upper=None, bb_middle=None, bb_lower=None,
                    stoch_k=None, stoch_d=None, adx=None, obv_slope=None, custom_indicator=None):
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
        stoch_k: Current Stochastic Oscillator %K value (optional)
        stoch_d: Current Stochastic Oscillator %D value (optional)
        adx: Current ADX value (optional)
        obv_slope: Current On-Balance Volume slope (optional)
        custom_indicator: Value from custom combined indicator (optional)
        
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
    
    # Check Stochastic Oscillator if available
    if stoch_k is not None and stoch_d is not None:
        total_signals += 1
        if stoch_k < 20 and stoch_d < 20:
            signal += "📊 Stochastic: Oversold region (potential buy)\n"
            buy_signals += 1
        elif stoch_k > 80 and stoch_d > 80:
            signal += "📊 Stochastic: Overbought region (potential sell/avoid)\n"
        elif stoch_k > stoch_d:
            signal += "📊 Stochastic: Bullish crossover (%K crossed above %D)\n"
            buy_signals += 0.5  # Half signal for crossover
        else:
            signal += "📊 Stochastic: Neutral or bearish\n"
    
    # Check ADX if available (trend strength)
    if adx is not None:
        total_signals += 1
        if adx > 25:
            signal += f"📏 ADX: Strong trend detected (ADX: {adx:.1f})\n"
            # ADX doesn't indicate direction, just strength, so it's informational
            # We don't add a buy signal here, but it provides context
        else:
            signal += f"📏 ADX: Weak or no trend (ADX: {adx:.1f})\n"
    
    # Check OBV slope if available
    if obv_slope is not None:
        total_signals += 1
        if obv_slope > 0:
            signal += "📈 OBV: Positive volume trend (accumulation)\n"
            buy_signals += 1
        else:
            signal += "📉 OBV: Negative volume trend (distribution)\n"
    
    # Check custom indicator if available
    if custom_indicator is not None:
        total_signals += 1
        if custom_indicator > 70:
            signal += f"🔮 Custom Indicator: Strong buy signal ({custom_indicator:.1f})\n"
            buy_signals += 1
        elif custom_indicator > 50:
            signal += f"🔮 Custom Indicator: Moderate buy signal ({custom_indicator:.1f})\n"
            buy_signals += 0.5
        elif custom_indicator < 30:
            signal += f"🔮 Custom Indicator: Sell/Avoid signal ({custom_indicator:.1f})\n"
        else:
            signal += f"🔮 Custom Indicator: Neutral ({custom_indicator:.1f})\n"
    
    # Calculate signal strength
    signal_strength = buy_signals / total_signals if total_signals > 0 else 0
    
    # Final recommendation
    if signal_strength >= 0.6:  # At least 60% of indicators are showing buy signals
        signal += f"✅ STRONG BUY SIGNAL DETECTED (Strength: {signal_strength:.0%})"
        return signal
    elif signal_strength >= 0.4:  # At least 40% of indicators are showing buy signals
        signal += f"✅ MODERATE BUY SIGNAL DETECTED (Strength: {signal_strength:.0%})"
        return signal
    elif signal_strength > 0:
        signal += f"⚠️ WEAK BUY SIGNAL (Strength: {signal_strength:.0%})"
        return signal
    else:
        signal += "⚠️ No buy signal at this time"
        return signal

def calculate_technical_indicators(price_data):
    """
    Calculate a comprehensive set of technical indicators for a price dataframe
    
    Args:
        price_data: DataFrame with price data (must have OHLCV columns)
        
    Returns:
        Dictionary with calculated technical indicators
    """
    if len(price_data) == 0:
        return {}
    
    indicators = {}
    
    # Calculate RSI
    indicators['RSI'] = calculate_rsi(price_data['Close'])
    
    # Calculate MACD
    macd, signal, hist = calculate_macd(price_data['Close'])
    indicators['MACD'] = macd
    indicators['MACD_Signal'] = signal
    indicators['MACD_Histogram'] = hist
    
    # Calculate Bollinger Bands
    upper, middle, lower = calculate_bollinger_bands(price_data['Close'])
    indicators['BB_Upper'] = upper
    indicators['BB_Middle'] = middle
    indicators['BB_Lower'] = lower
    
    # Calculate EMAs
    indicators['EMA_9'] = calculate_ema(price_data['Close'], 9)
    indicators['EMA_21'] = calculate_ema(price_data['Close'], 21)
    
    # Calculate SMAs
    indicators['SMA_50'] = price_data['Close'].rolling(window=50).mean()
    indicators['SMA_200'] = price_data['Close'].rolling(window=200).mean()
    
    # Calculate Stochastic Oscillator
    indicators['Stoch_K'], indicators['Stoch_D'] = calculate_stochastic_oscillator(
        price_data['High'], price_data['Low'], price_data['Close'])
    
    # Calculate ADX
    indicators['ADX'], indicators['Plus_DI'], indicators['Minus_DI'] = calculate_adx(
        price_data['High'], price_data['Low'], price_data['Close'])
    
    # Calculate OBV
    indicators['OBV'] = calculate_obv(price_data['Close'], price_data['Volume'])
    
    # Calculate OBV slope (using last 5 periods)
    if len(price_data) >= 5:
        obv_recent = indicators['OBV'].iloc[-5:].values
        indicators['OBV_Slope'] = np.polyfit(np.arange(len(obv_recent)), obv_recent, 1)[0]
    else:
        indicators['OBV_Slope'] = 0
    
    return indicators
