import bt
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

def get_historical_data(ticker, start_date, end_date):
    """
    Fetch historical data for backtesting
    
    Args:
        ticker: Stock symbol
        start_date: Start date for historical data
        end_date: End date for historical data
        
    Returns:
        Pandas DataFrame with historical data
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(start=start_date, end=end_date)
        
        if hist.empty:
            return None
            
        return hist
    except Exception as e:
        print(f"Error fetching historical data for {ticker}: {str(e)}")
        return None

def create_rsi_strategy(data, rsi_period=14, rsi_lower=30, rsi_upper=70):
    """
    Create a strategy based on RSI
    
    Args:
        data: DataFrame with price data
        rsi_period: Period for RSI calculation
        rsi_lower: Lower threshold for oversold (buy signal)
        rsi_upper: Upper threshold for overbought (sell signal)
        
    Returns:
        bt.Strategy object
    """
    # Calculate RSI
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=rsi_period).mean()
    avg_loss = loss.rolling(window=rsi_period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    # Create signals: 1 for buy, -1 for sell, 0 for hold
    signals = pd.Series(0, index=data.index)
    signals[rsi < rsi_lower] = 1  # Buy when RSI < lower threshold (oversold)
    signals[rsi > rsi_upper] = -1  # Sell when RSI > upper threshold (overbought)
    
    # Create strategy
    strategy = bt.Strategy('RSI_Strategy', 
                        [bt.algos.SelectWhere(signals == 1),  # Select assets with buy signal
                         bt.algos.WeighEqually(),  # Equal weight for all selected assets
                         bt.algos.Rebalance()])  # Rebalance portfolio
    
    return strategy

def create_macd_strategy(data, fast_period=12, slow_period=26, signal_period=9):
    """
    Create a strategy based on MACD
    
    Args:
        data: DataFrame with price data
        fast_period: Fast EMA period
        slow_period: Slow EMA period
        signal_period: Signal line period
        
    Returns:
        bt.Strategy object
    """
    # Calculate MACD components
    ema_fast = data['Close'].ewm(span=fast_period, adjust=False).mean()
    ema_slow = data['Close'].ewm(span=slow_period, adjust=False).mean()
    
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    
    # Create signals based on MACD crossovers
    # Buy when MACD line crosses above signal line, sell when it crosses below
    signals = pd.Series(0, index=data.index)
    
    # Calculate crossovers
    crossover = (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))
    crossunder = (macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))
    
    signals[crossover] = 1  # Buy signal
    signals[crossunder] = -1  # Sell signal
    
    # Create strategy
    strategy = bt.Strategy('MACD_Strategy', 
                        [bt.algos.SelectWhere(signals == 1),  # Select assets with buy signal
                         bt.algos.WeighEqually(),  # Equal weight for all selected assets
                         bt.algos.Rebalance()])  # Rebalance portfolio
    
    return strategy

def create_bollinger_bands_strategy(data, period=20, num_std=2):
    """
    Create a strategy based on Bollinger Bands
    
    Args:
        data: DataFrame with price data
        period: Period for moving average
        num_std: Number of standard deviations for bands
        
    Returns:
        bt.Strategy object
    """
    # Calculate Bollinger Bands components
    middle_band = data['Close'].rolling(window=period).mean()
    std = data['Close'].rolling(window=period).std()
    
    upper_band = middle_band + (std * num_std)
    lower_band = middle_band - (std * num_std)
    
    # Create signals
    signals = pd.Series(0, index=data.index)
    
    # Buy when price crosses below lower band
    buy_signal = data['Close'] < lower_band
    
    # Sell when price crosses above upper band
    sell_signal = data['Close'] > upper_band
    
    signals[buy_signal] = 1  # Buy signal
    signals[sell_signal] = -1  # Sell signal
    
    # Create strategy
    strategy = bt.Strategy('BBands_Strategy', 
                        [bt.algos.SelectWhere(signals == 1),  # Select assets with buy signal
                         bt.algos.WeighEqually(),  # Equal weight for all selected assets
                         bt.algos.Rebalance()])  # Rebalance portfolio
    
    return strategy

def create_combined_strategy(data, rsi_period=14, rsi_lower=30, rsi_upper=70, 
                           fast_period=12, slow_period=26, signal_period=9,
                           bb_period=20, bb_std=2):
    """
    Create a strategy that combines RSI, MACD, and Bollinger Bands
    
    Args:
        data: DataFrame with price data
        rsi_period, rsi_lower, rsi_upper: RSI parameters
        fast_period, slow_period, signal_period: MACD parameters
        bb_period, bb_std: Bollinger Bands parameters
        
    Returns:
        bt.Strategy object
    """
    # Calculate RSI
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=rsi_period).mean()
    avg_loss = loss.rolling(window=rsi_period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate MACD
    ema_fast = data['Close'].ewm(span=fast_period, adjust=False).mean()
    ema_slow = data['Close'].ewm(span=slow_period, adjust=False).mean()
    
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    
    # Calculate Bollinger Bands
    middle_band = data['Close'].rolling(window=bb_period).mean()
    std = data['Close'].rolling(window=bb_period).std()
    
    upper_band = middle_band + (std * bb_std)
    lower_band = middle_band - (std * bb_std)
    
    # Create combined signals
    rsi_buy = rsi < rsi_lower
    macd_buy = macd_line > signal_line
    bb_buy = data['Close'] < lower_band
    
    # Combined buy signal: at least 2 out of 3 indicators signal buy
    signals = pd.Series(0, index=data.index)
    combined_signal = (rsi_buy.astype(int) + macd_buy.astype(int) + bb_buy.astype(int) >= 2)
    signals[combined_signal] = 1
    
    # Create strategy
    strategy = bt.Strategy('Combined_Strategy', 
                        [bt.algos.SelectWhere(signals == 1),  # Select assets with buy signal
                         bt.algos.WeighEqually(),  # Equal weight for all selected assets
                         bt.algos.Rebalance()])  # Rebalance portfolio
    
    return strategy

def run_backtest(ticker, strategy_type, start_date, end_date, initial_capital=10000, **strategy_params):
    """
    Run backtest for a given strategy
    
    Args:
        ticker: Stock symbol
        strategy_type: Type of strategy ('rsi', 'macd', 'bbands', 'combined')
        start_date: Start date for backtest
        end_date: End date for backtest
        initial_capital: Initial capital for backtest
        **strategy_params: Additional parameters for the specific strategy
        
    Returns:
        Backtest results and performance metrics
    """
    # Get historical data
    data = get_historical_data(ticker, start_date, end_date)
    
    if data is None or data.empty:
        return None, None
    
    # Create a security using the data
    security = bt.Security(ticker, data)
    
    # Choose strategy based on strategy_type
    if strategy_type == 'rsi':
        strategy = create_rsi_strategy(data, **strategy_params)
    elif strategy_type == 'macd':
        strategy = create_macd_strategy(data, **strategy_params)
    elif strategy_type == 'bbands':
        strategy = create_bollinger_bands_strategy(data, **strategy_params)
    elif strategy_type == 'combined':
        strategy = create_combined_strategy(data, **strategy_params)
    else:
        return None, None
    
    # Create backtest
    backtest = bt.Backtest(strategy, security, initial_capital=initial_capital)
    
    # Run backtest
    result = bt.run(backtest)
    
    # Extract performance metrics
    performance = {
        'total_return': result.stats['total_return'],
        'cagr': result.stats['cagr'],
        'sharpe': result.stats['sharpe'],
        'max_drawdown': result.stats['max_drawdown'],
        'volatility': result.stats['daily_vol'],
        'win_rate': result.stats.get('win_rate', 0),
    }
    
    return result, performance


def get_benchmark_performance(ticker, start_date, end_date, initial_capital=10000):
    """
    Get buy and hold performance for benchmark comparison
    
    Args:
        ticker: Stock symbol
        start_date: Start date for benchmark
        end_date: End date for benchmark
        initial_capital: Initial capital
        
    Returns:
        Benchmark performance metrics
    """
    # Get historical data
    data = get_historical_data(ticker, start_date, end_date)
    
    if data is None or data.empty:
        return None
    
    # Create a simple buy and hold strategy
    security = bt.Security(ticker, data)
    strategy = bt.Strategy('Buy_Hold', 
                         [bt.algos.RunOnce(),
                          bt.algos.SelectAll(),
                          bt.algos.WeighEqually(),
                          bt.algos.Rebalance()])
    
    # Create backtest
    backtest = bt.Backtest(strategy, security, initial_capital=initial_capital)
    
    # Run backtest
    result = bt.run(backtest)
    
    # Extract performance metrics
    performance = {
        'total_return': result.stats['total_return'],
        'cagr': result.stats['cagr'],
        'sharpe': result.stats['sharpe'],
        'max_drawdown': result.stats['max_drawdown'],
        'volatility': result.stats['daily_vol'],
    }
    
    return result, performance
