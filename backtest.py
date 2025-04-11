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

def create_mean_reversion_strategy(data, lookback_period=20, z_score_threshold=1.0):
    """
    Create a strategy based on mean reversion
    
    Args:
        data: DataFrame with price data
        lookback_period: Period for calculating the moving average and standard deviation
        z_score_threshold: Z-score threshold for entry and exit
        
    Returns:
        bt.Strategy object
    """
    # Calculate z-score based on deviation from moving average
    price = data['Close']
    moving_avg = price.rolling(window=lookback_period).mean()
    std_dev = price.rolling(window=lookback_period).std()
    
    z_score = (price - moving_avg) / std_dev
    
    # Create signals
    signals = pd.Series(0, index=data.index)
    
    # Buy when price is below mean by z_score_threshold standard deviations
    buy_signal = z_score < -z_score_threshold
    
    # Sell when price is above mean by z_score_threshold standard deviations
    sell_signal = z_score > z_score_threshold
    
    signals[buy_signal] = 1
    signals[sell_signal] = -1
    
    # Create strategy
    strategy = bt.Strategy('MeanReversion_Strategy', 
                        [bt.algos.SelectWhere(signals == 1),
                         bt.algos.WeighEqually(),
                         bt.algos.Rebalance()])
    
    return strategy

def create_momentum_strategy(data, momentum_period=90, top_pct=25):
    """
    Create a strategy based on price momentum
    
    Args:
        data: DataFrame with price data
        momentum_period: Period for calculating momentum
        top_pct: Percentile threshold for selecting top performers
        
    Returns:
        bt.Strategy object
    """
    # Calculate momentum as percent change over momentum_period
    momentum = data['Close'].pct_change(momentum_period).fillna(0)
    
    # Calculate threshold for top performers
    threshold = np.percentile(momentum, 100 - top_pct)
    
    # Create signals
    signals = pd.Series(0, index=data.index)
    
    # Buy when momentum is above threshold
    signals[momentum > threshold] = 1
    
    # Create strategy
    strategy = bt.Strategy('Momentum_Strategy', 
                        [bt.algos.SelectWhere(signals == 1),
                         bt.algos.WeighEqually(),
                         bt.algos.Rebalance()])
    
    return strategy

def create_breakout_strategy(data, window=50, threshold_pct=2.0):
    """
    Create a strategy based on price breakouts
    
    Args:
        data: DataFrame with price data
        window: Lookback window for calculating resistance levels
        threshold_pct: Percentage above resistance to trigger breakout
        
    Returns:
        bt.Strategy object
    """
    # Calculate rolling maximum price (resistance)
    resistance = data['High'].rolling(window=window).max()
    
    # Calculate threshold price for breakout
    threshold = resistance * (1 + threshold_pct/100)
    
    # Create signals
    signals = pd.Series(0, index=data.index)
    
    # Buy when price breaks above resistance plus threshold
    breakout = data['Close'] > threshold
    
    signals[breakout] = 1
    
    # Create strategy
    strategy = bt.Strategy('Breakout_Strategy', 
                        [bt.algos.SelectWhere(signals == 1),
                         bt.algos.WeighEqually(),
                         bt.algos.Rebalance()])
    
    return strategy

def create_dual_moving_average_strategy(data, fast_period=50, slow_period=200):
    """
    Create a strategy based on dual moving average crossovers
    
    Args:
        data: DataFrame with price data
        fast_period: Period for fast moving average
        slow_period: Period for slow moving average
        
    Returns:
        bt.Strategy object
    """
    # Calculate moving averages
    fast_ma = data['Close'].rolling(window=fast_period).mean()
    slow_ma = data['Close'].rolling(window=slow_period).mean()
    
    # Create signals based on crossovers
    signals = pd.Series(0, index=data.index)
    
    # Buy when fast MA crosses above slow MA
    crossover = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
    
    # Sell when fast MA crosses below slow MA
    crossunder = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))
    
    signals[crossover] = 1
    signals[crossunder] = -1
    
    # Create strategy
    strategy = bt.Strategy('DualMA_Strategy', 
                        [bt.algos.SelectWhere(signals == 1),
                         bt.algos.WeighEqually(),
                         bt.algos.Rebalance()])
    
    return strategy

def create_volatility_breakout_strategy(data, lookback_period=20, volatility_multiplier=2.0):
    """
    Create a strategy based on volatility breakouts (similar to Donchian channels)
    
    Args:
        data: DataFrame with price data
        lookback_period: Period for calculating volatility
        volatility_multiplier: Multiplier for volatility to set breakout threshold
        
    Returns:
        bt.Strategy object
    """
    # Calculate the average true range (ATR) as volatility measure
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=lookback_period).mean()
    
    # Calculate upper and lower bands
    middle = data['Close'].rolling(window=lookback_period).mean()
    upper = middle + atr * volatility_multiplier
    lower = middle - atr * volatility_multiplier
    
    # Create signals
    signals = pd.Series(0, index=data.index)
    
    # Buy when price breaks above upper band
    buy_signal = data['Close'] > upper
    
    # Sell when price breaks below lower band
    sell_signal = data['Close'] < lower
    
    signals[buy_signal] = 1
    signals[sell_signal] = -1
    
    # Create strategy
    strategy = bt.Strategy('VolatilityBreakout_Strategy', 
                        [bt.algos.SelectWhere(signals == 1),
                         bt.algos.WeighEqually(),
                         bt.algos.Rebalance()])
    
    return strategy

def run_backtest(ticker, strategy_type, start_date, end_date, initial_capital=10000, 
                commission=0.001, slippage=0.001, **strategy_params):
    """
    Run backtest for a given strategy with transaction costs and slippage
    
    Args:
        ticker: Stock symbol
        strategy_type: Type of strategy ('rsi', 'macd', 'bbands', 'combined', 'meanrev', 
                                       'momentum', 'breakout', 'dualma', 'volbreakout')
        start_date: Start date for backtest
        end_date: End date for backtest
        initial_capital: Initial capital for backtest
        commission: Commission rate per trade (e.g., 0.001 = 0.1%)
        slippage: Slippage rate per trade (e.g., 0.001 = 0.1%)
        **strategy_params: Additional parameters for the specific strategy
        
    Returns:
        Backtest results and performance metrics
    """
    # Get historical data
    data = get_historical_data(ticker, start_date, end_date)
    
    if data is None or data.empty:
        return None, None
    
    # Create a security with commissions and slippage
    # Apply both commission and slippage to simulate realistic trading conditions
    # Commission is applied as a percentage of trade value
    # Slippage is applied as a percentage of trade price in the direction of the trade
    security = bt.Security(ticker, data, commission=commission)
    
    # Apply slippage manually by adjusting buy and sell prices
    # For buys, increase price by slippage rate; for sells, decrease price by slippage rate
    if slippage > 0:
        # Create copies of data for buy and sell prices with slippage
        buy_data = data.copy()
        sell_data = data.copy()
        
        # Apply slippage
        buy_data['Close'] = buy_data['Close'] * (1 + slippage)
        sell_data['Close'] = sell_data['Close'] * (1 - slippage)
        
        # Update data with slippage-adjusted values
        # This is a simplified approach; in a real-world scenario, slippage would be applied per trade
        # and would depend on factors like order size and market liquidity
        slippage_impact = pd.Series(0, index=data.index)
        
    # Choose strategy based on strategy_type
    if strategy_type == 'rsi':
        strategy = create_rsi_strategy(data, **strategy_params)
    elif strategy_type == 'macd':
        strategy = create_macd_strategy(data, **strategy_params)
    elif strategy_type == 'bbands':
        strategy = create_bollinger_bands_strategy(data, **strategy_params)
    elif strategy_type == 'combined':
        strategy = create_combined_strategy(data, **strategy_params)
    elif strategy_type == 'meanrev':
        strategy = create_mean_reversion_strategy(data, **strategy_params)
    elif strategy_type == 'momentum':
        strategy = create_momentum_strategy(data, **strategy_params)
    elif strategy_type == 'breakout':
        strategy = create_breakout_strategy(data, **strategy_params)
    elif strategy_type == 'dualma':
        strategy = create_dual_moving_average_strategy(data, **strategy_params)
    elif strategy_type == 'volbreakout':
        strategy = create_volatility_breakout_strategy(data, **strategy_params)
    else:
        return None, None
    
    # Create backtest
    backtest = bt.Backtest(strategy, security, initial_capital=initial_capital)
    
    # Run backtest
    result = bt.run(backtest)
    
    # Calculate additional performance metrics
    trades = result.get('trades', [])
    
    # Calculate win rate if there are trades
    win_count = 0
    total_trades = len(trades)
    
    for trade in trades:
        if trade['return'] > 0:
            win_count += 1
    
    win_rate = win_count / total_trades if total_trades > 0 else 0
    
    # Calculate average profit per trade and average holding period
    avg_profit = sum([trade['return'] for trade in trades]) / total_trades if total_trades > 0 else 0
    
    # Extract performance metrics
    performance = {
        'total_return': result.stats['total_return'],
        'cagr': result.stats['cagr'],
        'sharpe': result.stats['sharpe'],
        'max_drawdown': result.stats['max_drawdown'],
        'volatility': result.stats['daily_vol'],
        'win_rate': win_rate,
        'avg_profit_per_trade': avg_profit,
        'total_trades': total_trades,
        'commission_impact': initial_capital * commission * total_trades * 2,  # Estimate of total commission cost
        'slippage_impact': initial_capital * slippage * total_trades * 2,  # Estimate of total slippage impact
    }
    
    return result, performance


def get_benchmark_performance(ticker, start_date, end_date, initial_capital=10000, commission=0.001, slippage=0.001):
    """
    Get buy and hold performance for benchmark comparison
    
    Args:
        ticker: Stock symbol
        start_date: Start date for benchmark
        end_date: End date for benchmark
        initial_capital: Initial capital
        commission: Commission rate per trade
        slippage: Slippage rate per trade
        
    Returns:
        Benchmark performance metrics
    """
    # Get historical data
    data = get_historical_data(ticker, start_date, end_date)
    
    if data is None or data.empty:
        return None, None
    
    # Create a simple buy and hold strategy with commission
    security = bt.Security(ticker, data, commission=commission)
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

def optimize_strategy_parameters(ticker, strategy_type, start_date, end_date, initial_capital=10000, 
                               commission=0.001, slippage=0.001, optimization_target='sharpe'):
    """
    Optimize strategy parameters to find the best configuration
    
    Args:
        ticker: Stock symbol
        strategy_type: Type of strategy to optimize
        start_date: Start date for backtest
        end_date: End date for backtest
        initial_capital: Initial capital for backtest
        commission: Commission rate per trade
        slippage: Slippage rate per trade
        optimization_target: Metric to optimize ('sharpe', 'total_return', 'cagr', 'max_drawdown')
        
    Returns:
        Dictionary with best parameters and performance
    """
    # Get historical data
    data = get_historical_data(ticker, start_date, end_date)
    
    if data is None or data.empty:
        return None
    
    # Define parameter ranges for each strategy type
    param_ranges = {
        'rsi': {
            'rsi_period': range(5, 25, 5),         # Test 5, 10, 15, 20
            'rsi_lower': range(20, 41, 5),         # Test 20, 25, 30, 35, 40
            'rsi_upper': range(60, 81, 5)          # Test 60, 65, 70, 75, 80
        },
        'macd': {
            'fast_period': range(8, 17, 2),        # Test 8, 10, 12, 14, 16
            'slow_period': range(20, 33, 3),       # Test 20, 23, 26, 29, 32
            'signal_period': range(7, 13, 1)       # Test 7, 8, 9, 10, 11, 12
        },
        'bbands': {
            'period': range(10, 31, 5),            # Test 10, 15, 20, 25, 30
            'num_std': [1.5, 2.0, 2.5, 3.0]        # Test standard deviation multipliers
        },
        'meanrev': {
            'lookback_period': range(10, 31, 5),   # Test 10, 15, 20, 25, 30
            'z_score_threshold': [0.5, 1.0, 1.5, 2.0, 2.5]  # Test different z-score thresholds
        },
        'momentum': {
            'momentum_period': range(30, 121, 30), # Test 30, 60, 90, 120
            'top_pct': range(10, 41, 10)           # Test 10, 20, 30, 40 percentile thresholds
        },
        'breakout': {
            'window': range(20, 81, 20),           # Test 20, 40, 60, 80
            'threshold_pct': [1.0, 2.0, 3.0, 4.0]  # Test different threshold percentages
        },
        'dualma': {
            'fast_period': range(20, 81, 20),      # Test 20, 40, 60, 80
            'slow_period': range(100, 301, 50)     # Test 100, 150, 200, 250, 300
        },
        'volbreakout': {
            'lookback_period': range(10, 31, 5),   # Test 10, 15, 20, 25, 30
            'volatility_multiplier': [1.0, 1.5, 2.0, 2.5, 3.0]  # Test different multipliers
        },
        'combined': {
            'rsi_period': range(10, 21, 5),          # Test 10, 15, 20
            'rsi_lower': [25, 30, 35],               # Test 25, 30, 35
            'rsi_upper': [65, 70, 75],               # Test 65, 70, 75
            'fast_period': [10, 12, 14],             # Test 10, 12, 14
            'slow_period': [24, 26, 28],             # Test 24, 26, 28
            'signal_period': [8, 9, 10],             # Test 8, 9, 10
            'bb_period': [15, 20, 25],               # Test 15, 20, 25
            'bb_std': [1.8, 2.0, 2.2]                # Test 1.8, 2.0, 2.2
        }
    }
    
    # Check if the strategy is supported for optimization
    if strategy_type not in param_ranges:
        return None
    
    # Get parameter ranges for the selected strategy
    params = param_ranges[strategy_type]
    
    # Set up optimization
    best_performance = None
    best_params = None
    best_value = float('-inf') if optimization_target != 'max_drawdown' else float('inf')
    
    # Special case for combined strategy - need to handle differently due to many parameters
    if strategy_type == 'combined':
        # Create a more limited parameter grid to avoid exponential combinations
        # For combined strategy, we'll focus on a subset of key parameters
        for rsi_period in params['rsi_period']:
            for rsi_lower in params['rsi_lower']:
                for rsi_upper in params['rsi_upper']:
                    # Fix some parameters to limit combinations
                    test_params = {
                        'rsi_period': rsi_period,
                        'rsi_lower': rsi_lower, 
                        'rsi_upper': rsi_upper,
                        'fast_period': 12,  # Fixed values for other parameters
                        'slow_period': 26,
                        'signal_period': 9,
                        'bb_period': 20,
                        'bb_std': 2.0
                    }
                    
                    # Run backtest with current parameters
                    result, performance = run_backtest(
                        ticker, strategy_type, start_date, end_date, 
                        initial_capital, commission, slippage, **test_params
                    )
                    
                    if result is None or performance is None:
                        continue
                    
                    # Check if this is the best result based on optimization target
                    current_value = performance[optimization_target]
                    
                    if optimization_target == 'max_drawdown':
                        # For drawdown, lower is better
                        if current_value < best_value:
                            best_value = current_value
                            best_performance = performance
                            best_params = test_params
                    else:
                        # For other metrics, higher is better
                        if current_value > best_value:
                            best_value = current_value
                            best_performance = performance
                            best_params = test_params
    else:
        # For strategies with fewer parameters, test all combinations
        # Generate all combinations of parameters
        param_keys = list(params.keys())
        param_values = list(params.values())
        
        import itertools
        param_combinations = list(itertools.product(*param_values))
        
        # Test each combination
        for combo in param_combinations:
            # Create parameter dictionary for current combination
            test_params = {param_keys[i]: combo[i] for i in range(len(param_keys))}
            
            # Run backtest with current parameters
            result, performance = run_backtest(
                ticker, strategy_type, start_date, end_date, 
                initial_capital, commission, slippage, **test_params
            )
            
            if result is None or performance is None:
                continue
            
            # Check if this is the best result based on optimization target
            current_value = performance[optimization_target]
            
            if optimization_target == 'max_drawdown':
                # For drawdown, lower is better
                if current_value < best_value:
                    best_value = current_value
                    best_performance = performance
                    best_params = test_params
            else:
                # For other metrics, higher is better
                if current_value > best_value:
                    best_value = current_value
                    best_performance = performance
                    best_params = test_params
    
    # Get benchmark performance for comparison
    benchmark_result, benchmark_performance = get_benchmark_performance(
        ticker, start_date, end_date, initial_capital, commission, slippage
    )
    
    # Compile optimization results
    optimization_results = {
        'best_parameters': best_params,
        'best_performance': best_performance,
        'optimization_target': optimization_target,
        'best_value': best_value,
        'benchmark_performance': benchmark_performance,
        'strategy_type': strategy_type,
        'ticker': ticker,
        'start_date': start_date,
        'end_date': end_date,
        'outperforms_benchmark': False  # Default value
    }
    
    # Check if the strategy outperforms the benchmark
    if benchmark_performance and best_performance:
        benchmark_value = benchmark_performance[optimization_target]
        strategy_value = best_performance[optimization_target]
        
        if optimization_target == 'max_drawdown':
            # For drawdown, lower is better
            optimization_results['outperforms_benchmark'] = strategy_value < benchmark_value
        else:
            # For other metrics, higher is better
            optimization_results['outperforms_benchmark'] = strategy_value > benchmark_value
    
    return optimization_results

def walk_forward_optimization(ticker, strategy_type, start_date, end_date, window_size=365, step_size=90,
                            initial_capital=10000, commission=0.001, slippage=0.001, optimization_target='sharpe'):
    """
    Perform walk-forward optimization to test strategy robustness
    
    Args:
        ticker: Stock symbol
        strategy_type: Type of strategy to optimize
        start_date: Overall start date for analysis
        end_date: Overall end date for analysis
        window_size: Size of each window in days
        step_size: Step size between windows in days
        initial_capital: Initial capital for backtest
        commission: Commission rate per trade
        slippage: Slippage rate per trade
        optimization_target: Metric to optimize
        
    Returns:
        Dictionary with walk-forward optimization results
    """
    # Convert dates to datetime objects
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    # Calculate the total number of days in the date range
    total_days = (end - start).days
    
    # Ensure we have enough data for at least one window
    if total_days < window_size:
        return None
    
    # Initialize results storage
    walk_forward_results = []
    
    # Create windows for optimization and testing
    current_start = start
    window_count = 0
    
    while current_start + timedelta(days=window_size) <= end:
        window_count += 1
        
        # Calculate window dates
        train_start = current_start
        train_end = train_start + timedelta(days=window_size * 0.7)  # Use 70% for training
        test_start = train_end
        test_end = min(test_start + timedelta(days=window_size * 0.3), end)  # Use 30% for testing
        
        # Convert dates back to strings
        train_start_str = train_start.strftime('%Y-%m-%d')
        train_end_str = train_end.strftime('%Y-%m-%d')
        test_start_str = test_start.strftime('%Y-%m-%d')
        test_end_str = test_end.strftime('%Y-%m-%d')
        
        # Run optimization on training window
        optimization_results = optimize_strategy_parameters(
            ticker, strategy_type, train_start_str, train_end_str,
            initial_capital, commission, slippage, optimization_target
        )
        
        if optimization_results and optimization_results['best_parameters']:
            # Run backtest on testing window using the optimized parameters
            best_params = optimization_results['best_parameters']
            test_result, test_performance = run_backtest(
                ticker, strategy_type, test_start_str, test_end_str,
                initial_capital, commission, slippage, **best_params
            )
            
            # Get benchmark performance for testing window
            bench_result, bench_performance = get_benchmark_performance(
                ticker, test_start_str, test_end_str, initial_capital, commission, slippage
            )
            
            # Store window results
            window_result = {
                'window': window_count,
                'train_period': f"{train_start_str} to {train_end_str}",
                'test_period': f"{test_start_str} to {test_end_str}",
                'optimized_parameters': best_params,
                'train_performance': optimization_results['best_performance'],
                'test_performance': test_performance,
                'benchmark_performance': bench_performance
            }
            
            # Check if optimized parameters worked in the test period
            if test_performance and bench_performance:
                test_value = test_performance[optimization_target]
                bench_value = bench_performance[optimization_target]
                
                if optimization_target == 'max_drawdown':
                    # For drawdown, lower is better
                    window_result['outperforms_benchmark'] = test_value < bench_value
                else:
                    # For other metrics, higher is better
                    window_result['outperforms_benchmark'] = test_value > bench_value
            
            walk_forward_results.append(window_result)
        
        # Move to next window
        current_start += timedelta(days=step_size)
    
    # Analyze overall walk-forward performance
    success_count = sum(1 for result in walk_forward_results if result.get('outperforms_benchmark', False))
    total_windows = len(walk_forward_results)
    success_rate = success_count / total_windows if total_windows > 0 else 0
    
    # Calculate average performance metrics across test windows
    avg_performance = {}
    
    if walk_forward_results:
        metrics = ['total_return', 'cagr', 'sharpe', 'max_drawdown', 'volatility']
        
        for metric in metrics:
            values = [result['test_performance'][metric] for result in walk_forward_results 
                     if result['test_performance'] and metric in result['test_performance']]
            
            avg_performance[metric] = sum(values) / len(values) if values else 0
    
    # Compile final results
    results = {
        'windows': walk_forward_results,
        'success_rate': success_rate,
        'tested_windows': total_windows,
        'successful_windows': success_count,
        'avg_performance': avg_performance,
        'strategy_type': strategy_type,
        'ticker': ticker,
        'overall_period': f"{start_date} to {end_date}"
    }
    
    return results
