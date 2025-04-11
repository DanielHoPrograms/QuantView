import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import math

def generate_wave_effect(strength, x_range, period=5, amplitude=1.0, phase_shift=0.0):
    """
    Generate a wave effect for sentiment visualization
    
    Args:
        strength: Strength of sentiment (0-100)
        x_range: Range of x values to generate the wave for
        period: Wave period
        amplitude: Wave amplitude
        phase_shift: Phase shift to offset the wave
        
    Returns:
        Array of y values representing the wave
    """
    # Scale amplitude by sentiment strength
    scaled_amplitude = amplitude * (strength / 100)
    
    # Generate wave
    return [scaled_amplitude * math.sin((2 * math.pi / period) * x + phase_shift) for x in x_range]

def create_animated_sentiment_wave(ticker, price_data, news_items, period_days=90):
    """
    Create an animated wave visualization showing sentiment impact on stock price
    
    Args:
        ticker: Stock symbol
        price_data: DataFrame with price data
        news_items: List of news items with sentiment and dates
        period_days: Number of days to show in the animation
        
    Returns:
        Plotly figure with animated sentiment wave
    """
    # Validate inputs to avoid Series truth value ambiguity
    if price_data is None or news_items is None:
        return None
        
    if isinstance(price_data, pd.DataFrame):
        if price_data.empty or len(price_data) == 0:
            return None
    else:
        return None
        
    if not isinstance(news_items, list) or len(news_items) == 0:
        return None
    
    # Ensure price_data index is datetime
    if not isinstance(price_data.index, pd.DatetimeIndex):
        st.error("Price data index must be a DatetimeIndex")
        return None
    
    # Filter price data to the period we want to show
    end_date = price_data.index[-1]
    start_date = end_date - timedelta(days=period_days)
    
    # Create boolean mask safely - avoid Series truth value ambiguity
    date_filter = pd.Series(False, index=price_data.index)
    for date_idx in price_data.index:
        if date_idx >= start_date and date_idx <= end_date:
            date_filter.loc[date_idx] = True
    
    filtered_data = price_data[date_filter].copy()
    
    if filtered_data.empty or len(filtered_data) == 0:
        st.warning(f"No price data available for the last {period_days} days")
        return None
    
    # Process news items
    processed_news = []
    for news in news_items:
        try:
            # Try to parse the date
            date_str = news.get('published', '')
            if not date_str:
                continue
                
            try:
                if len(date_str) > 10:  # If it has time component
                    news_date = datetime.strptime(date_str, '%Y-%m-%d %H:%M')
                else:
                    news_date = datetime.strptime(date_str, '%Y-%m-%d')
            except:
                # If parsing fails, try a different format or skip
                try:
                    news_date = datetime.strptime(date_str[:10], '%Y-%m-%d')
                except:
                    continue
            
            # Only include news within our filtered data date range
            if news_date.date() < start_date.date() or news_date.date() > end_date.date():
                continue
                
            # Get sentiment data
            sentiment = news.get('sentiment', {'score': 0.5, 'label': 'NEUTRAL'})
            score = sentiment.get('score', 0.5)
            label = sentiment.get('label', 'NEUTRAL')
            
            # Get impact categories
            impact = news.get('impact', {'categories': ['General News'], 'has_high_impact': False})
            categories = impact.get('categories', ['General News'])
            is_high_impact = impact.get('has_high_impact', False) or any(cat in ['Earnings', 'M&A Activity', 'Legal/Regulatory'] for cat in categories)
            
            # Set news color and sentiment strength based on label
            if isinstance(label, str) and label == 'POSITIVE':
                color = '#4CAF50'  # Green
                strength = score * 100  # Convert to percentage
            elif isinstance(label, str) and label == 'NEGATIVE':
                color = '#F44336'  # Red
                strength = -score * 100  # Negative for bearish sentiment
            else:
                color = '#FFC107'  # Amber
                strength = 0  # Neutral sentiment
            
            # Add to processed news
            processed_news.append({
                'date': news_date,
                'title': news.get('title', 'News'),
                'strength': strength,
                'color': color,
                'impact': 2 if is_high_impact else 1,  # Impact multiplier
                'categories': categories
            })
            
        except Exception as e:
            st.error(f"Error processing news for animation: {str(e)}")
            continue
    
    # If no news after filtering, return None
    if not processed_news:
        st.warning("No news found in the selected date range")
        return None
    
    # Sort news by date
    processed_news.sort(key=lambda x: x['date'])
    
    # Create animated visualization
    # First, create base figure with candlestick chart
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                       vertical_spacing=0.03, row_heights=[0.7, 0.3],
                       subplot_titles=(f"{ticker} Price", "Sentiment Waves"))
    
    # Add price candlestick
    fig.add_trace(
        go.Candlestick(
            x=filtered_data.index,
            open=filtered_data['Open'],
            high=filtered_data['High'],
            low=filtered_data['Low'],
            close=filtered_data['Close'],
            name='Price',
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350'
        ),
        row=1, col=1
    )
    
    # Add volume bars with safe iterrows comparison
    colors = []
    for _, row in filtered_data.iterrows():
        # Use scalar comparison to avoid Series truth value ambiguity
        close_val = row['Close']
        open_val = row['Open']
        if isinstance(close_val, (int, float)) and isinstance(open_val, (int, float)):
            colors.append('#ef5350' if close_val < open_val else '#26a69a')
        else:
            colors.append('#26a69a')  # Default color
    
    fig.add_trace(
        go.Bar(
            x=filtered_data.index,
            y=filtered_data['Volume'],
            name='Volume',
            marker_color=colors,
            opacity=0.5
        ),
        row=2, col=1
    )
    
    # Create a time slider for animation
    # First, convert all dates to datetime objects if they aren't already
    dates = filtered_data.index.to_list()
    
    # Create frames for animation
    frames = []
    
    # Define the number of animation steps
    num_steps = min(50, len(dates))  # Limit to 50 frames for performance
    step_size = max(1, len(dates) // num_steps)
    animation_dates = dates[::step_size] + [dates[-1]]  # Ensure the last date is included
    
    # For each animation step
    for i, current_date in enumerate(animation_dates):
        # Filter news up to this date
        current_news = [n for n in processed_news if n['date'] <= current_date]
        
        # Skip if no news yet
        if not current_news:
            continue
        
        # Create wave data
        x_range = np.linspace(0, len(dates), 500)
        wave_data = np.zeros(len(x_range))
        
        # Add waves for each news item
        for news in current_news:
            # Calculate days since news
            days_ago = (current_date - news['date']).days
            
            # Skip if news is too old (more than 30 days)
            if days_ago > 30:
                continue
            
            # Calculate decay factor based on days ago
            decay = max(0, 1 - (days_ago / 30))
            
            # Generate wave based on sentiment strength and impact
            # Ensure strength is a number to avoid ambiguity
            strength = float(news['strength']) * float(news['impact']) * decay
            wave = generate_wave_effect(
                abs(strength), 
                x_range, 
                period=30, 
                amplitude=0.5, 
                phase_shift=i * 0.2  # Shift increases with each frame
            )
            
            # Make wave negative if sentiment is negative
            if strength < 0:
                wave = [-w for w in wave]
            
            # Add to total wave
            wave_data += wave
        
        # Create a new trace for this frame
        price_trace = go.Candlestick(
            x=filtered_data.index,
            open=filtered_data['Open'],
            high=filtered_data['High'],
            low=filtered_data['Low'],
            close=filtered_data['Close']
        )
        
        # Create sentiment wave trace - convert dates to timestamps for interpolation
        try:
            # Convert datetimes to numeric timestamps for interpolation
            start_ts = pd.Timestamp(start_date).timestamp()
            end_ts = pd.Timestamp(end_date).timestamp()
            # Create evenly spaced timestamps
            time_points = np.linspace(start_ts, end_ts, len(wave_data))
            # Convert back to datetime objects
            date_points = [pd.Timestamp.fromtimestamp(ts) for ts in time_points]
            
            wave_trace = go.Scatter(
                x=date_points,
                y=wave_data,
                mode='lines',
                line=dict(
                    width=3,
                    color='rgba(220, 0, 255, 0.7)',
                    shape='spline'
                ),
                name='Sentiment Wave'
            )
        except Exception as e:
            st.error(f"Error creating wave trace: {str(e)}")
            # Fallback to a simpler trace if needed
            wave_trace = go.Scatter(
                x=filtered_data.index,
                y=[0] * len(filtered_data.index),
                mode='lines',
                line=dict(width=3, color='rgba(220, 0, 255, 0.7)'),
                name='Sentiment Wave'
            )
        
        # Create a frame with these traces
        frame = go.Frame(
            data=[price_trace, go.Bar(
                x=filtered_data.index,
                y=filtered_data['Volume'],
                marker_color=colors
            ), wave_trace],
            traces=[0, 1, 2],
            name=f'frame{i}'
        )
        frames.append(frame)
    
    # Add frames to figure
    fig.frames = frames
    
    # Add initial wave (empty) - safely convert datetime for interpolation
    try:
        # Convert datetimes to numeric timestamps for interpolation
        start_ts = pd.Timestamp(start_date).timestamp()
        end_ts = pd.Timestamp(end_date).timestamp()
        # Create evenly spaced timestamps
        time_points = np.linspace(start_ts, end_ts, 500)
        # Convert back to datetime objects
        date_points = [pd.Timestamp.fromtimestamp(ts) for ts in time_points]
        
        fig.add_trace(
            go.Scatter(
                x=date_points,
                y=np.zeros(500),
                mode='lines',
                line=dict(
                    width=3,
                    color='rgba(220, 0, 255, 0.7)',
                    shape='spline'
                ),
                name='Sentiment Wave'
            ),
            row=2, col=1
        )
    except Exception as e:
        st.error(f"Error creating initial wave: {str(e)}")
        # Fallback to filtered data index
        fig.add_trace(
            go.Scatter(
                x=filtered_data.index,
                y=np.zeros(len(filtered_data.index)),
                mode='lines',
                line=dict(width=3, color='rgba(220, 0, 255, 0.7)'),
                name='Sentiment Wave'
            ),
            row=2, col=1
        )
    
    # Set up animation parameters
    animation_settings = dict(
        frame=dict(duration=200, redraw=True),
        fromcurrent=True,
        transition=dict(duration=100, easing='cubic-in-out')
    )
    
    # Add news events as annotations
    for news in processed_news:
        # Use explicit numeric comparison to avoid Series truth value ambiguity
        impact = float(news.get('impact', 1))
        if impact > 1:  # Only show high-impact news
            fig.add_annotation(
                x=news['date'],
                y=filtered_data['High'].max(),
                text=news['categories'][0] if news['categories'] else "News",
                showarrow=True,
                arrowhead=1,
                arrowcolor=news['color'],
                arrowsize=1,
                arrowwidth=2,
                row=1, col=1
            )
    
    # Add play button and slider
    # This is where we define the animation playback controls
    fig.update_layout(
        updatemenus=[
            dict(
                type='buttons',
                showactive=False,
                buttons=[
                    dict(
                        label='Play',
                        method='animate',
                        args=[None, animation_settings]
                    ),
                    dict(
                        label='Pause',
                        method='animate',
                        args=[[None], dict(
                            frame=dict(duration=0, redraw=False),
                            mode='immediate',
                            transition=dict(duration=0)
                        )]
                    )
                ],
                x=0.1,
                y=0,
                xanchor='right',
                yanchor='top'
            )
        ],
        sliders=[
            dict(
                active=0,
                steps=[
                    dict(
                        method='animate',
                        args=[
                            [f'frame{k}'],
                            dict(
                                mode='immediate',
                                frame=dict(duration=0, redraw=True),
                                transition=dict(duration=0)
                            )
                        ],
                        label=f'{animation_dates[k].strftime("%Y-%m-%d")}'
                    )
                    for k in range(len(frames))
                ],
                x=0.1,
                y=0,
                currentvalue=dict(
                    font=dict(size=12),
                    prefix='Date: ',
                    visible=True,
                    xanchor='center'
                ),
                len=0.9,
                pad=dict(b=10, t=50),
                ticklen=5
            )
        ]
    )
    
    # Update layout for better aesthetics
    fig.update_layout(
        title=f"{ticker} Price with Animated Sentiment Wave",
        xaxis_title="Date",
        yaxis_title="Price",
        height=800,
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h", 
            yanchor="bottom", 
            y=1.02, 
            xanchor="right", 
            x=1
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=60, r=60, t=80, b=120),  # Increase bottom margin for slider
        hovermode='closest'
    )
    
    # Add grid lines for better readability
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(211, 211, 211, 0.5)',
        zeroline=False,
        row=1, col=1
    )
    
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(211, 211, 211, 0.5)',
        row=1, col=1
    )
    
    # Update second subplot
    fig.update_yaxes(
        title="Sentiment & Volume",
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(211, 211, 211, 0.5)',
        zeroline=True,
        zerolinecolor='black',
        zerolinewidth=1,
        row=2, col=1
    )
    
    return fig

def display_animated_sentiment(ticker, price_data, news_items):
    """
    Display animated sentiment visualization in the Streamlit app
    
    Args:
        ticker: Stock symbol
        price_data: DataFrame with price data
        news_items: List of news items with sentiment and dates
    """
    st.subheader("Sentiment Wave Animation")
    st.write("This visualization shows how news sentiment creates waves of market impact over time.")
    
    # Add detailed debug information
    with st.expander("Debug Information"):
        st.write(f"Price data shape: {price_data.shape if hasattr(price_data, 'shape') else 'N/A'}")
        st.write(f"Price data columns: {list(price_data.columns) if hasattr(price_data, 'columns') else 'N/A'}")
        st.write(f"Number of news items: {len(news_items) if isinstance(news_items, list) else 'N/A'}")
        
        if isinstance(news_items, list) and len(news_items) > 0:
            st.write("First news item sample:")
            st.write(news_items[0])
    
    # Create a simple visualization 
    st.subheader("Price Chart")
    
    # Create a basic price chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=price_data.index,
        y=price_data['Close'],
        mode='lines',
        name='Close Price'
    ))
    
    # Add news markers if available
    if isinstance(news_items, list) and len(news_items) > 0:
        news_dates = []
        news_annotations = []
        
        for news in news_items:
            try:
                # Extract date
                date_str = news.get('published', '')
                if not date_str:
                    continue
                    
                try:
                    if isinstance(date_str, str) and len(date_str) > 10:  # If it has time component
                        news_date = datetime.strptime(date_str, '%Y-%m-%d %H:%M')
                    else:
                        news_date = datetime.strptime(date_str, '%Y-%m-%d')
                except:
                    try:
                        # Try alternative format
                        news_date = datetime.strptime(date_str[:10], '%Y-%m-%d')
                    except:
                        continue
                
                # Extract sentiment
                sentiment = news.get('sentiment', {})
                sentiment_label = sentiment.get('label', 'NEUTRAL')
                
                # Set color based on sentiment
                if sentiment_label == 'POSITIVE':
                    color = 'green'
                elif sentiment_label == 'NEGATIVE':
                    color = 'red'
                else:
                    color = 'orange'
                
                # Only add if the news date falls within the price data range
                if news_date.date() >= price_data.index[0].date() and news_date.date() <= price_data.index[-1].date():
                    news_dates.append(news_date)
                    
                    # Get y-value (price) for annotation at this date
                    # Find closest date
                    closest_idx = price_data.index.get_indexer([news_date], method='nearest')[0]
                    price_val = price_data['Close'].iloc[closest_idx]
                    
                    # Add annotation
                    fig.add_annotation(
                        x=news_date,
                        y=price_val,
                        text=sentiment_label,
                        showarrow=True,
                        arrowhead=2,
                        arrowcolor=color,
                        arrowsize=1,
                        arrowwidth=2
                    )
            except Exception as e:
                st.error(f"Error processing news for chart: {str(e)}")
    
    # Configure layout
    fig.update_layout(
        title=f"{ticker} Price with News Sentiment Indicators",
        xaxis_title="Date",
        yaxis_title="Price",
        height=500,
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Create animation - show warning if it fails
    try:
        with st.spinner("Creating sentiment wave animation..."):
            animation_fig = create_animated_sentiment_wave(ticker, price_data, news_items)
        
        if animation_fig:
            st.subheader("Sentiment Wave Animation")
            st.plotly_chart(animation_fig, use_container_width=True)
            
            # Add explanation
            st.markdown("""
            **How to Read This Chart:**
            
            - **Top panel**: Shows the stock price with candlesticks
            - **Bottom panel**: Displays volume bars and the sentiment wave
            - **Sentiment wave**: Purple waves show the cumulative impact of news sentiment
              - Upward waves indicate positive sentiment
              - Downward waves indicate negative sentiment
              - Wave amplitude shows sentiment strength
              - Waves dissipate over time as news impact fades
            
            **Tip**: Use the play button to animate how sentiment ripples through the market over time.
            """)
        else:
            st.warning("Could not create the animated sentiment wave visualization. A simpler visualization is shown above.")
    except Exception as e:
        st.error(f"Error creating sentiment wave animation: {str(e)}")
        st.warning("Using basic visualization instead of animation due to error.")
        
def create_3d_sentiment_surface(ticker, price_data, news_items, period_days=90):
    """
    Create a 3D surface visualization of sentiment impact landscape
    
    Args:
        ticker: Stock symbol
        price_data: DataFrame with price data
        news_items: List of news items with sentiment and dates
        period_days: Number of days to show
        
    Returns:
        Plotly figure with 3D sentiment surface
    """
    # Validate inputs to avoid Series truth value ambiguity
    if price_data is None or news_items is None:
        return None
        
    if isinstance(price_data, pd.DataFrame):
        if price_data.empty or len(price_data) == 0:
            return None
    else:
        return None
        
    if not isinstance(news_items, list) or len(news_items) == 0:
        return None
    
    # Ensure price_data index is datetime
    if not isinstance(price_data.index, pd.DatetimeIndex):
        st.error("Price data index must be a DatetimeIndex")
        return None
    
    # Filter price data to the period we want to show
    end_date = price_data.index[-1]
    start_date = end_date - timedelta(days=period_days)
    
    # Create boolean mask safely - avoid Series truth value ambiguity
    date_filter = pd.Series(False, index=price_data.index)
    for date_idx in price_data.index:
        if date_idx >= start_date and date_idx <= end_date:
            date_filter.loc[date_idx] = True
    
    filtered_data = price_data[date_filter].copy()
    
    if filtered_data.empty or len(filtered_data) == 0:
        st.warning(f"No price data available for the last {period_days} days")
        return None
    
    # Process news items (similar to previous function)
    processed_news = []
    for news in news_items:
        try:
            # Try to parse the date
            date_str = news.get('published', '')
            if not date_str:
                continue
                
            try:
                if len(date_str) > 10:  # If it has time component
                    news_date = datetime.strptime(date_str, '%Y-%m-%d %H:%M')
                else:
                    news_date = datetime.strptime(date_str, '%Y-%m-%d')
            except:
                # If parsing fails, try a different format or skip
                try:
                    news_date = datetime.strptime(date_str[:10], '%Y-%m-%d')
                except:
                    continue
            
            # Only include news within our filtered data date range
            if news_date.date() < start_date.date() or news_date.date() > end_date.date():
                continue
                
            # Get sentiment data
            sentiment = news.get('sentiment', {'score': 0.5, 'label': 'NEUTRAL'})
            score = sentiment.get('score', 0.5)
            label = sentiment.get('label', 'NEUTRAL')
            
            # Get impact categories
            impact = news.get('impact', {'categories': ['General News'], 'has_high_impact': False})
            categories = impact.get('categories', ['General News'])
            is_high_impact = impact.get('has_high_impact', False)
            
            # Set news color and sentiment strength
            if isinstance(label, str) and label == 'POSITIVE':
                color = 'green'  # Green
                strength = score * 100  # Convert to percentage
            elif isinstance(label, str) and label == 'NEGATIVE':
                color = 'red'  # Red
                strength = -score * 100  # Negative for bearish sentiment
            else:
                color = 'gold'  # Amber
                strength = 0  # Neutral sentiment
            
            # Add to processed news
            processed_news.append({
                'date': news_date,
                'title': news.get('title', 'News'),
                'label': label,
                'strength': strength,
                'color': color,
                'impact': 2 if is_high_impact else 1,  # Impact multiplier
                'categories': categories
            })
            
        except Exception as e:
            st.error(f"Error processing news for 3D visualization: {str(e)}")
            continue
    
    # If no news after filtering, return None
    if not processed_news:
        st.warning("No news found in the selected date range")
        return None
    
    # Create the 3D landscape
    # Extract price data - ensure we get it as a list
    try:
        # Convert to list correctly based on what we have
        if isinstance(filtered_data['Close'], pd.Series):
            prices = filtered_data['Close'].tolist()
        else:
            prices = filtered_data['Close'].values.tolist() if hasattr(filtered_data['Close'], 'values') else list(filtered_data['Close'])
    except Exception as e:
        st.error(f"Error converting prices to list: {str(e)}")
        # Create fallback prices
        prices = list(range(len(filtered_data)))
    
    # Calculate price range for normalization safely
    try:
        # Convert prices to float if they're not already
        float_prices = [float(p) if isinstance(p, (int, float)) else 0.0 for p in prices]
        
        # Get min and max as numeric values
        min_price = min(float_prices) if float_prices else 0.0
        max_price = max(float_prices) if float_prices else 1.0
        
        # Calculate range 
        price_range = max_price - min_price
        
        if price_range > 0:
            # Normalize prices to 0-1 range
            normalized_prices = [(float(p) - min_price) / price_range if isinstance(p, (int, float)) else 0.5 
                                for p in prices]
        else:
            normalized_prices = [0.5] * len(prices)
    except Exception as e:
        st.error(f"Error normalizing prices: {str(e)}")
        normalized_prices = [0.5] * len(prices)
    
    # Create a grid for the surface
    # X axis: days from start date
    price_x = [(date - start_date).days for date in filtered_data.index]
    x_range = np.linspace(min(price_x), max(price_x), 50) if price_x else np.linspace(0, 1, 50)
    
    # Y axis: sentiment strength (-100 to 100)
    y_range = np.linspace(-100, 100, 50)
    
    # Create meshgrid for surface
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.zeros_like(X)
    
    # Calculate sentiment impact on the landscape
    for i in range(len(x_range)):
        x_val = x_range[i]
        for j in range(len(y_range)):
            y_val = y_range[j]
            
            # Find the price at or before this x point
            closest_price_idx = max(0, np.searchsorted(price_x, x_val) - 1) if price_x else 0
            if closest_price_idx < len(normalized_prices):
                price_factor = normalized_prices[closest_price_idx]
            else:
                price_factor = 0.5
            
            # Calculate base height (price)
            base_height = price_factor * 0.5
            
            # Add a sentiment ridge when y matches news sentiment
            sentiment_factor = 0
            for news in processed_news:
                news_day = (news['date'] - start_date).days
                sentiment_val = float(news['strength'])
                
                # Distance in days and sentiment value
                day_dist = abs(x_val - news_day)
                sentiment_dist = abs(y_val - sentiment_val)
                
                # If close to this news point, add to surface height
                if day_dist < 10 and sentiment_dist < 20:
                    impact = float(news['impact'])
                    # Gaussian falloff with distance
                    day_factor = np.exp(-0.1 * day_dist)
                    sentiment_factor = np.exp(-0.01 * sentiment_dist)
                    news_contribution = 0.2 * impact * day_factor * sentiment_factor
                    
                    # Add to total sentiment factor
                    base_height += news_contribution
            
            Z[j, i] = base_height
    
    # Create the 3D surface
    surface_fig = go.Figure(data=[go.Surface(z=Z, x=x_range, y=y_range, 
                                colorscale='Viridis', opacity=0.8)])
    
    # Add news points
    news_x = [(news['date'] - start_date).days for news in processed_news]
    news_y = [float(news['strength']) for news in processed_news]
    news_z = [0.7] * len(processed_news)  # Constant height for news markers
    
    news_colors = ['green' if isinstance(news['label'], str) and news['label'] == 'POSITIVE' 
                  else 'red' if isinstance(news['label'], str) and news['label'] == 'NEGATIVE' 
                  else 'gold' for news in processed_news]
                  
    news_sizes = [12 if float(news['impact']) > 1 else 8 for news in processed_news]
    news_texts = [f"{news['title']}<br>Date: {news['date'].strftime('%Y-%m-%d')}<br>Impact: {'High' if float(news['impact']) > 1 else 'Normal'}" for news in processed_news]
    
    surface_fig.add_trace(go.Scatter3d(
        x=news_x,
        y=news_y,
        z=news_z,
        mode='markers',
        marker=dict(
            size=news_sizes,
            color=news_colors,
            symbol='circle'
        ),
        text=news_texts,
        hoverinfo='text'
    ))
    
    # Add price line
    surface_fig.add_trace(go.Scatter3d(
        x=price_x,
        y=[0] * len(price_x),  # Center line
        z=normalized_prices,
        mode='lines',
        line=dict(
            color='blue',
            width=5
        ),
        name='Price'
    ))
    
    # Update layout
    surface_fig.update_layout(
        title=f"{ticker} Sentiment Landscape",
        scene=dict(
            xaxis_title="Days from Start",
            yaxis_title="Sentiment Value",
            zaxis_title="Market Impact",
            xaxis=dict(
                gridcolor='rgb(200, 200, 200)',
                showbackground=True,
                backgroundcolor='rgb(240, 240, 240)'
            ),
            yaxis=dict(
                gridcolor='rgb(200, 200, 200)',
                showbackground=True,
                backgroundcolor='rgb(240, 240, 240)',
                range=[-100, 100]
            ),
            zaxis=dict(
                gridcolor='rgb(200, 200, 200)',
                showbackground=True,
                backgroundcolor='rgb(240, 240, 240)'
            ),
        ),
        height=700,
        margin=dict(l=0, r=0, b=0, t=30),
        plot_bgcolor='white',
    )
    
    return surface_fig

def display_3d_sentiment_landscape(ticker, price_data, news_items):
    """
    Display a 3D sentiment landscape in the Streamlit app
    
    Args:
        ticker: Stock symbol
        price_data: DataFrame with price data
        news_items: List of news items with sentiment and dates
    """
    st.subheader("3D Sentiment Impact Landscape")
    st.write("This visualization shows how news sentiment shapes the market landscape over time.")
    
    # Create 3D surface
    surface_fig = create_3d_sentiment_surface(ticker, price_data, news_items)
    
    if surface_fig:
        st.plotly_chart(surface_fig, use_container_width=True)
        
        # Add explanation
        st.markdown("""
        **How to Read This Chart:**
        
        - **X-axis**: Time (days from start)
        - **Y-axis**: Sentiment value (positive/negative)
        - **Z-axis**: Market impact (height)
        - **Colored points**: News events
          - Green: Positive sentiment
          - Red: Negative sentiment
          - Gold: Neutral sentiment
          - Larger points: Higher impact news
        - **Blue line**: Price movement over time
        
        **Tip**: Click and drag to rotate the 3D view. Scroll to zoom in/out.
        """)
    else:
        st.warning("Could not create the 3D sentiment landscape. Make sure you have price data and news items for the selected period.")

def sentiment_animation_section():
    """Main function for the sentiment animation section"""
    st.title("📈 Sentiment Wave Visualization")
    
    # Ticker selection
    ticker = st.text_input("Enter Stock Symbol", "AAPL").upper()
    
    if ticker:
        # Fetch data
        import yfinance as yf
        from datetime import datetime, timedelta
        
        # Fetch recent data for the ticker
        end_date = datetime.now()
        start_date = end_date - timedelta(days=180)  # 6 months for a good amount of data
        
        try:
            # Show loading spinner
            with st.spinner(f"Fetching data for {ticker}..."):
                stock_data = yf.download(ticker, start=start_date, end=end_date)
            
            if isinstance(stock_data, pd.DataFrame) and len(stock_data) > 0:
                # Get news for this ticker from sentiment_tracker module
                from sentiment_tracker import analyze_news_sentiment
                
                # Number of news articles to analyze
                num_articles = st.slider("Number of news articles to analyze", 5, 20, 10)
                
                with st.spinner("Fetching and analyzing news sentiment..."):
                    news_items = analyze_news_sentiment(ticker, limit=num_articles)
                
                if news_items and isinstance(news_items, list) and len(news_items) > 0:
                    # Create tabs for different visualizations
                    viz_tabs = st.tabs(["Wave Animation", "3D Landscape"])
                    
                    with viz_tabs[0]:
                        # Display animated wave
                        display_animated_sentiment(ticker, stock_data, news_items)
                    
                    with viz_tabs[1]:
                        # Display 3D surface
                        display_3d_sentiment_landscape(ticker, stock_data, news_items)
                else:
                    st.warning(f"Could not find news for {ticker}. Try a different ticker symbol.")
            else:
                st.error(f"No price data available for {ticker}")
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")
    else:
        st.info("Enter a stock symbol to visualize sentiment waves.")

if __name__ == "__main__":
    sentiment_animation_section()
