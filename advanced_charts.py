import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf

def create_renko_chart(df, brick_size=None, auto_brick=True):
    """
    Create a Renko chart from OHLC data
    
    Args:
        df: DataFrame with OHLC data
        brick_size: Size of each brick (fixed value or percentage)
        auto_brick: If True, automatically calculate brick size as ATR
        
    Returns:
        Plotly figure with Renko chart
    """
    # Make a copy of the dataframe to avoid modifying the original
    df_renko = df.copy()
    
    # If auto_brick is True, calculate brick size as the 14-period ATR
    if auto_brick and brick_size is None:
        # Calculate True Range
        high_low = df_renko['High'] - df_renko['Low']
        high_close = np.abs(df_renko['High'] - df_renko['Close'].shift(1))
        low_close = np.abs(df_renko['Low'] - df_renko['Close'].shift(1))
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        # Calculate 14-period ATR
        atr = tr.rolling(14).mean()
        brick_size = atr.iloc[-1]
    
    # If brick_size is not provided, use 2% of the current price
    if brick_size is None:
        brick_size = df_renko['Close'].iloc[-1] * 0.02
        
    # Initialize renko data
    renko_prices = []
    renko_directions = []
    
    # Set the first brick at the close of the first candle
    current_price = df_renko['Close'].iloc[0]
    renko_prices.append(current_price)
    renko_directions.append(0)  # Neutral for the first brick
    
    # Iterate through the DataFrame to create bricks
    for i in range(1, len(df_renko)):
        close = df_renko['Close'].iloc[i]
        
        # Calculate how many bricks to add
        price_change = close - current_price
        num_bricks = int(abs(price_change) / brick_size)
        
        if num_bricks > 0:
            # Add bricks in the appropriate direction
            direction = 1 if price_change > 0 else -1
            for _ in range(num_bricks):
                current_price += direction * brick_size
                renko_prices.append(current_price)
                renko_directions.append(direction)
    
    # Create a new DataFrame for the Renko chart
    renko_df = pd.DataFrame({
        'price': renko_prices,
        'direction': renko_directions,
        'date': [df_renko.index[0]] + [None] * (len(renko_prices) - 1)  # Only include the first date
    })
    
    # Create the Renko chart using Plotly
    fig = go.Figure()
    
    for i in range(len(renko_df)):
        price = renko_df['price'].iloc[i]
        direction = renko_df['direction'].iloc[i]
        
        # Set color based on direction
        color = 'green' if direction > 0 else 'red' if direction < 0 else 'gray'
        
        # Add a rectangle for each brick
        fig.add_shape(
            type="rect",
            x0=i - 0.4, x1=i + 0.4,
            y0=price - brick_size if direction > 0 else price,
            y1=price if direction > 0 else price - brick_size,
            fillcolor=color,
            line=dict(color=color),
            opacity=0.8
        )
    
    # Set layout
    fig.update_layout(
        title="Renko Chart",
        xaxis_title="Brick Number",
        yaxis_title="Price",
        showlegend=False,
        xaxis=dict(
            showgrid=True,
            zeroline=False,
            showticklabels=False
        ),
        yaxis=dict(
            showgrid=True,
            zeroline=False
        )
    )
    
    # Add annotations for current price
    fig.add_annotation(
        x=len(renko_df) - 1,
        y=renko_df['price'].iloc[-1],
        text=f"${renko_df['price'].iloc[-1]:.2f}",
        showarrow=True,
        arrowhead=1
    )
    
    return fig

def create_point_and_figure_chart(df, box_size=None, reversal=3, auto_box=True):
    """
    Create a Point & Figure chart from OHLC data
    
    Args:
        df: DataFrame with OHLC data
        box_size: Size of each box
        reversal: Number of boxes needed for a reversal
        auto_box: If True, automatically calculate box size as a percentage of price
        
    Returns:
        Plotly figure with Point & Figure chart
    """
    # Make a copy of the dataframe to avoid modifying the original
    df_pnf = df.copy()
    
    # If auto_box is True, calculate box size as a percentage of the current price
    if auto_box and box_size is None:
        price = df_pnf['Close'].iloc[-1]
        if price < 10:
            box_size = price * 0.01  # 1% for low priced stocks
        elif price < 100:
            box_size = price * 0.005  # 0.5% for medium priced stocks
        else:
            box_size = price * 0.0025  # 0.25% for high priced stocks
    
    # If box_size is not provided, use 1% of the current price
    if box_size is None:
        box_size = df_pnf['Close'].iloc[-1] * 0.01
    
    # Initialize P&F chart data
    columns = []  # Each column in the P&F chart
    current_column = []
    current_direction = None  # 1 for X (up), -1 for O (down)
    
    # Set the starting price at the close of the first candle
    current_price = df_pnf['Close'].iloc[0]
    
    # Calculate starting price aligned to box size
    current_price = (current_price // box_size) * box_size
    
    # Process each price move
    for i in range(1, len(df_pnf)):
        high = df_pnf['High'].iloc[i]
        low = df_pnf['Low'].iloc[i]
        
        # First determine the initial direction if not set
        if current_direction is None:
            if high >= current_price + box_size:
                current_direction = 1  # X column (up)
            elif low <= current_price - box_size:
                current_direction = -1  # O column (down)
        
        if current_direction == 1:  # X column (up)
            # Check if price moved up by at least one box
            if high >= current_price + box_size:
                # Calculate how many boxes to add
                boxes_to_add = int((high - current_price) / box_size)
                for _ in range(boxes_to_add):
                    current_price += box_size
                    current_column.append([current_price, 'X'])
            
            # Check for reversal
            if low <= current_price - (reversal * box_size):
                # Save the current column and start a new one
                if current_column:
                    columns.append(current_column)
                current_column = []
                
                # Calculate new starting price for the O column
                new_price = current_price - (reversal * box_size)
                boxes_to_add = int((current_price - new_price) / box_size)
                current_price = current_price - (boxes_to_add * box_size)
                
                for _ in range(boxes_to_add):
                    current_column.append([current_price, 'O'])
                    current_price -= box_size
                current_price += box_size  # Adjust back
                
                current_direction = -1  # Switch to O column
                
        elif current_direction == -1:  # O column (down)
            # Check if price moved down by at least one box
            if low <= current_price - box_size:
                # Calculate how many boxes to add
                boxes_to_add = int((current_price - low) / box_size)
                for _ in range(boxes_to_add):
                    current_price -= box_size
                    current_column.append([current_price, 'O'])
            
            # Check for reversal
            if high >= current_price + (reversal * box_size):
                # Save the current column and start a new one
                if current_column:
                    columns.append(current_column)
                current_column = []
                
                # Calculate new starting price for the X column
                new_price = current_price + (reversal * box_size)
                boxes_to_add = int((new_price - current_price) / box_size)
                current_price = current_price + (boxes_to_add * box_size)
                
                for _ in range(boxes_to_add):
                    current_column.append([current_price, 'X'])
                    current_price += box_size
                current_price -= box_size  # Adjust back
                
                current_direction = 1  # Switch to X column
    
    # Add the last column
    if current_column:
        columns.append(current_column)
    
    # Create the Point & Figure chart using Plotly
    fig = go.Figure()
    
    # Calculate all possible price levels across all columns
    all_prices = []
    for column in columns:
        for box in column:
            all_prices.append(box[0])
    
    if not all_prices:
        # If no columns were created, return empty figure with a note
        fig.add_annotation(
            x=0.5, y=0.5,
            text="Not enough price movement for Point & Figure chart",
            showarrow=False,
            font=dict(size=15)
        )
        fig.update_layout(
            title="Point & Figure Chart",
            xaxis_title="Columns",
            yaxis_title="Price",
            showlegend=False
        )
        return fig
    
    price_levels = sorted(list(set(all_prices)))
    
    # Add shapes for each X and O
    for col_idx, column in enumerate(columns):
        for box in column:
            price = box[0]
            box_type = box[1]
            
            if box_type == 'X':
                fig.add_annotation(
                    x=col_idx,
                    y=price,
                    text="X",
                    showarrow=False,
                    font=dict(size=15, color="green")
                )
            else:  # 'O'
                fig.add_annotation(
                    x=col_idx,
                    y=price,
                    text="O",
                    showarrow=False,
                    font=dict(size=15, color="red")
                )
    
    # Set layout
    fig.update_layout(
        title="Point & Figure Chart",
        xaxis_title="Columns",
        yaxis_title="Price",
        showlegend=False,
        xaxis=dict(
            range=[-0.5, len(columns) - 0.5],
            showgrid=True,
            zeroline=False,
            tickvals=list(range(len(columns))),
            ticktext=[f"{i+1}" for i in range(len(columns))]
        ),
        yaxis=dict(
            showgrid=True,
            zeroline=False,
            tickvals=price_levels,
            tickformat="$.2f"
        )
    )
    
    return fig

def create_comparison_chart(tickers, start_date, end_date=None, normalize=True):
    """
    Create a comparison chart for multiple stocks
    
    Args:
        tickers: List of ticker symbols
        start_date: Start date for comparison
        end_date: End date for comparison (defaults to current date)
        normalize: If True, normalize prices to start at 100 for better comparison
        
    Returns:
        Plotly figure with comparison chart
    """
    # Download data for all tickers
    data = {}
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date)
            if len(df) > 0:
                data[ticker] = df
        except Exception as e:
            print(f"Error fetching data for {ticker}: {str(e)}")
    
    if not data:
        # Return empty figure with error message
        fig = go.Figure()
        fig.add_annotation(
            x=0.5, y=0.5,
            text="Could not retrieve data for any of the selected tickers",
            showarrow=False,
            font=dict(size=15)
        )
        return fig
    
    # Create the comparison chart
    fig = go.Figure()
    
    for ticker, df in data.items():
        prices = df['Close']
        
        if normalize:
            # Normalize to start at 100
            prices = (prices / prices.iloc[0]) * 100
        
        fig.add_trace(go.Scatter(
            x=df.index,
            y=prices,
            mode='lines',
            name=ticker
        ))
    
    # Set layout
    y_axis_title = "Normalized Price (Base=100)" if normalize else "Price"
    fig.update_layout(
        title="Stock Price Comparison",
        xaxis_title="Date",
        yaxis_title=y_axis_title,
        legend_title="Tickers",
        hovermode="x unified"
    )
    
    return fig

def create_interactive_chart(df, ticker, events=None):
    """
    Create an interactive chart with annotations for important events
    
    Args:
        df: DataFrame with OHLC data
        ticker: Stock ticker symbol
        events: List of dictionaries with event data (date, description)
        
    Returns:
        Plotly figure with interactive chart
    """
    # Create the base candlestick chart
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name=ticker
    )])
    
    # Add volume bars at the bottom
    fig.add_trace(go.Bar(
        x=df.index,
        y=df['Volume'],
        name='Volume',
        yaxis='y2',
        marker=dict(color='rgba(0,0,0,0.2)')
    ))
    
    # Add events as annotations if provided
    if events:
        for event in events:
            event_date = event.get('date')
            description = event.get('description', '')
            event_type = event.get('type', 'info')  # info, earnings, dividend, split, news
            
            # Set color based on event type
            if event_type == 'earnings':
                color = 'blue'
                symbol = 'circle'
            elif event_type == 'dividend':
                color = 'green'
                symbol = 'triangle-up'
            elif event_type == 'split':
                color = 'purple'
                symbol = 'square'
            elif event_type == 'news':
                color = 'orange'
                symbol = 'star'
            else:  # info
                color = 'gray'
                symbol = 'diamond'
            
            # Find the corresponding price for this date
            price_point = None
            if event_date in df.index:
                price_point = df.loc[event_date, 'High'] * 1.01  # Slightly above the high
            else:
                # Find the nearest date
                for i, date in enumerate(df.index):
                    if date >= event_date:
                        price_point = df.iloc[i]['High'] * 1.01
                        break
            
            if price_point is not None:
                # Add a marker for the event
                fig.add_trace(go.Scatter(
                    x=[event_date],
                    y=[price_point],
                    mode='markers',
                    marker=dict(
                        symbol=symbol,
                        size=10,
                        color=color
                    ),
                    name=event_type.capitalize(),
                    text=description,
                    hoverinfo='text+name'
                ))
    
    # Update layout to include secondary y-axis for volume
    fig.update_layout(
        title=f"{ticker} Interactive Chart",
        xaxis_title="Date",
        yaxis_title="Price",
        yaxis2=dict(
            title="Volume",
            titlefont=dict(color="black"),
            tickfont=dict(color="black"),
            anchor="x",
            overlaying="y",
            side="right",
            showgrid=False
        ),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Add rangeslider and buttons for time ranges
    fig.update_layout(
        xaxis=dict(
            rangeslider=dict(visible=True),
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            )
        )
    )
    
    return fig

def fetch_stock_events(ticker, start_date=None, end_date=None):
    """
    Fetch important events for a stock (earnings, dividends, splits, news)
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date for events
        end_date: End date for events
        
    Returns:
        List of dictionaries with event data
    """
    events = []
    
    try:
        stock = yf.Ticker(ticker)
        
        # Get earnings dates
        earnings = stock.earnings_dates
        if earnings is not None and not earnings.empty:
            for date, row in earnings.iterrows():
                if (start_date is None or date >= start_date) and (end_date is None or date <= end_date):
                    estimated_eps = row.get('EPS Estimate', None)
                    reported_eps = row.get('Reported EPS', None)
                    surprise = row.get('Surprise(%)', None)
                    
                    description = f"Earnings: "
                    if estimated_eps:
                        description += f"Est. EPS: ${estimated_eps:.2f}, "
                    if reported_eps:
                        description += f"Reported EPS: ${reported_eps:.2f}, "
                    if surprise:
                        description += f"Surprise: {surprise:.2f}%"
                    
                    events.append({
                        'date': date,
                        'description': description.strip(', '),
                        'type': 'earnings'
                    })
        
        # Get dividends
        dividends = stock.dividends
        if dividends is not None and not dividends.empty:
            for date, dividend in dividends.items():
                if (start_date is None or date >= start_date) and (end_date is None or date <= end_date):
                    events.append({
                        'date': date,
                        'description': f"Dividend: ${dividend:.4f}",
                        'type': 'dividend'
                    })
        
        # Get stock splits
        splits = stock.splits
        if splits is not None and not splits.empty:
            for date, split in splits.items():
                if (start_date is None or date >= start_date) and (end_date is None or date <= end_date):
                    numerator = int(split) if split >= 1 else 1
                    denominator = 1 if split >= 1 else int(1/split)
                    events.append({
                        'date': date,
                        'description': f"Split: {numerator}:{denominator}",
                        'type': 'split'
                    })
    
    except Exception as e:
        print(f"Error fetching events for {ticker}: {str(e)}")
    
    return events
