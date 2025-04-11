import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import math
from datetime import datetime, timedelta
import yfinance as yf

def get_sentiment_color(sentiment_score):
    """
    Get a color based on a sentiment score (-1 to 1)
    
    Args:
        sentiment_score: Score from -1 (negative) to 1 (positive)
        
    Returns:
        Color in RGB format
    """
    # Normalize the score to 0-1 range
    norm_score = (sentiment_score + 1) / 2
    
    # Define color transitions:
    # Deep red (-1) -> Orange (-0.5) -> Yellow (0) -> Light green (0.5) -> Deep green (1)
    if norm_score < 0.25:
        # Red to orange
        r = 255
        g = int(255 * (norm_score * 4))
        b = 0
    elif norm_score < 0.5:
        # Orange to yellow
        r = 255
        g = 255
        b = 0
    elif norm_score < 0.75:
        # Yellow to light green
        r = int(255 * (1 - (norm_score - 0.5) * 4))
        g = 255
        b = 0
    else:
        # Light green to deep green
        r = 0
        g = 255
        b = int(255 * (norm_score - 0.75) * 4)
    
    return f'rgb({r}, {g}, {b})'

def get_mood_description(sentiment_score):
    """
    Get a text description of the mood based on sentiment score
    
    Args:
        sentiment_score: Score from -1 (negative) to 1 (positive)
        
    Returns:
        Mood description
    """
    if sentiment_score < -0.8:
        return "Extremely Bearish", "Markets are in extreme fear, pessimism is prevalent"
    elif sentiment_score < -0.6:
        return "Very Bearish", "Strong negative sentiment, high pessimism"
    elif sentiment_score < -0.4:
        return "Bearish", "Negative sentiment, caution advised"
    elif sentiment_score < -0.2:
        return "Slightly Bearish", "Mildly negative sentiment, slight pessimism" 
    elif sentiment_score < 0.2:
        return "Neutral", "Mixed sentiment, no clear direction"
    elif sentiment_score < 0.4:
        return "Slightly Bullish", "Mildly positive sentiment, cautious optimism"
    elif sentiment_score < 0.6:
        return "Bullish", "Positive sentiment, optimism prevails"
    elif sentiment_score < 0.8:
        return "Very Bullish", "Strong positive sentiment, high optimism"
    else:
        return "Extremely Bullish", "Markets are extremely optimistic, possibly euphoric"

def create_mood_ring_visualization(ticker, price_data, sentiment_score, volatility, trading_volume, news_sentiment=None):
    """
    Create an animated mood ring visualization for a stock
    
    Args:
        ticker: Stock symbol
        price_data: DataFrame with price data
        sentiment_score: Overall market sentiment score (-1 to 1)
        volatility: Volatility measure (0 to 1)
        trading_volume: Normalized trading volume (0 to 1)
        news_sentiment: Optional news sentiment scores
        
    Returns:
        Plotly figure with animated mood ring
    """
    # Create a container for the visualization
    container = st.empty()
    
    # Get the sentiment color
    color = get_sentiment_color(sentiment_score)
    mood_text, mood_desc = get_mood_description(sentiment_score)
    
    # Animate the mood ring appearance
    for i in range(21):
        progress = i / 20
        
        # Create the figure
        fig = go.Figure()
        
        # Create concentric rings with pulsating animation
        ring_sizes = [0.8, 0.65, 0.5, 0.35, 0.2]
        opacities = [0.9, 0.7, 0.5, 0.3, 0.15]
        
        for j, (size, opacity_base) in enumerate(zip(ring_sizes, opacities)):
            # Adjust opacity for animation
            opacity = opacity_base * progress
            
            # Add each ring as a circle
            fig.add_shape(
                type="circle",
                xref="x", yref="y",
                x0=-size * progress, y0=-size * progress,
                x1=size * progress, y1=size * progress,
                fillcolor=color,
                line_color=color,
                opacity=opacity
            )
        
        # Add ticker and sentiment text (appears after rings)
        if progress > 0.5:
            text_opacity = (progress - 0.5) * 2
            
            fig.add_annotation(
                x=0, y=0,
                text=f"{ticker}",
                font=dict(
                    color="white",
                    size=24 * progress
                ),
                showarrow=False,
                opacity=text_opacity
            )
            
            fig.add_annotation(
                x=0, y=-0.15,
                text=f"{mood_text}",
                font=dict(
                    color="white",
                    size=18 * progress
                ),
                showarrow=False,
                opacity=text_opacity
            )
        
        # Set up layout
        fig.update_layout(
            width=400,
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(
                showgrid=False,
                showticklabels=False,
                zeroline=False,
                range=[-1, 1]
            ),
            yaxis=dict(
                showgrid=False,
                showticklabels=False,
                zeroline=False,
                range=[-1, 1]
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=False
        )
        
        # Display the updated figure
        container.plotly_chart(fig)
        time.sleep(0.05)
    
    # Display mood description
    st.markdown(f"""
    <div style="text-align: center; margin: 10px 0; padding: 10px; border-radius: 5px; 
              background-color: rgba({color.replace('rgb(', '').replace(')', '')}, 0.2);">
        <div style="font-weight: bold; font-size: 16px;">{mood_desc}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Display metrics with animations
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Price change animation
        if price_data is not None and not price_data.empty:
            last_price = price_data['Close'].iloc[-1]
            prev_price = price_data['Close'].iloc[-2] if len(price_data) > 1 else last_price
            price_change = ((last_price - prev_price) / prev_price) * 100
            
            price_container = st.empty()
            for i in range(11):
                progress = i / 10
                current_price = prev_price + (last_price - prev_price) * progress
                current_change = price_change * progress
                
                arrow = "↑" if price_change >= 0 else "↓"
                color = "#4CAF50" if price_change >= 0 else "#F44336"
                
                price_container.markdown(f"""
                <div style="text-align: center;">
                    <div style="font-size: 12px; color: #999;">Price</div>
                    <div style="font-size: 18px; font-weight: bold;">${current_price:.2f}</div>
                    <div style="color: {color};">{arrow} {abs(current_change):.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
                
                time.sleep(0.05)
    
    with col2:
        # Volatility animation
        volatility_container = st.empty()
        for i in range(11):
            progress = i / 10
            current_volatility = volatility * progress
            
            # Volatility color: Green (low) to Red (high)
            color = get_sentiment_color(1 - 2 * current_volatility)  # Invert scale for volatility
            
            volatility_container.markdown(f"""
            <div style="text-align: center;">
                <div style="font-size: 12px; color: #999;">Volatility</div>
                <div style="font-size: 18px; font-weight: bold;">{current_volatility * 100:.1f}%</div>
                <div style="background: linear-gradient(90deg, #4CAF50, {color}); 
                          height: 5px; border-radius: 5px; width: {current_volatility * 100}%;
                          margin: 0 auto;"></div>
            </div>
            """, unsafe_allow_html=True)
            
            time.sleep(0.05)
    
    with col3:
        # Volume animation
        volume_container = st.empty()
        for i in range(11):
            progress = i / 10
            current_volume = trading_volume * progress
            
            # Volume bars
            # Generate random heights for volume bars
            bar_heights = []
            for _ in range(10):
                r = random.random()
                height = int(20 * min(1, 0.3 + current_volume * r * 1.4))
                bar_heights.append(f"<div style='width: 3px; margin: 0 1px; background-color: #0C7BDC; height: {height}px;'></div>")
            
            volume_bars_html = "".join(bar_heights)
            
            volume_container.markdown(f"""
            <div style="text-align: center;">
                <div style="font-size: 12px; color: #999;">Volume</div>
                <div style="font-size: 18px; font-weight: bold;">{int(current_volume * 100)}%</div>
                <div style="display: flex; justify-content: center; align-items: flex-end; height: 20px;">
                    {volume_bars_html}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            time.sleep(0.05)
    
    # Display sentiment trend if news sentiment is provided
    if news_sentiment is not None and len(news_sentiment) > 1:
        st.subheader("Sentiment Trend")
        
        # Create the sentiment trend chart
        fig = go.Figure()
        
        dates = [item['date'] for item in news_sentiment]
        scores = [item['score'] for item in news_sentiment]
        
        # Add gradual animation building up the sentiment trend
        trend_container = st.empty()
        
        for i in range(1, len(dates) + 1):
            # Create a color gradient based on sentiment
            colors = [get_sentiment_color(score) for score in scores[:i]]
            
            # Add the line with a gradient
            fig.add_trace(go.Scatter(
                x=dates[:i],
                y=scores[:i],
                mode='lines+markers',
                line=dict(
                    color='white', 
                    width=3,
                    shape='spline'
                ),
                marker=dict(
                    size=8,
                    color=colors,
                    line=dict(width=1, color='white')
                ),
                name='Sentiment'
            ))
            
            # Add gradient area under the line
            fig.add_trace(go.Scatter(
                x=dates[:i],
                y=[min(min(scores), -0.2) - 0.1] * i,  # Bottom of the gradient area
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='none'
            ))
            
            # Create a filled area with gradient
            for j in range(i):
                fig.add_shape(
                    type='rect',
                    xref='x',
                    yref='y',
                    x0=dates[j] - pd.Timedelta(hours=12) if j > 0 else dates[j] - pd.Timedelta(days=1),
                    y0=min(min(scores), -0.2) - 0.1,
                    x1=dates[j] + pd.Timedelta(hours=12) if j < i-1 else dates[j] + pd.Timedelta(days=1),
                    y1=scores[j],
                    fillcolor=colors[j],
                    opacity=0.3,
                    line=dict(width=0)
                )
            
            # Add a horizontal line at neutral sentiment
            fig.add_shape(
                type='line',
                xref='x',
                yref='y',
                x0=dates[0] - pd.Timedelta(days=1),
                y0=0,
                x1=dates[-1] + pd.Timedelta(days=1),
                y1=0,
                line=dict(
                    color='gray',
                    width=1,
                    dash='dash'
                )
            )
            
            fig.update_layout(
                height=300,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(
                    showgrid=False,
                    zeroline=False,
                    title='Date'
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor='rgba(255,255,255,0.1)',
                    zeroline=False,
                    range=[min(min(scores), -0.2) - 0.1, max(max(scores), 0.2) + 0.1],
                    title='Sentiment Score'
                ),
                margin=dict(l=0, r=0, t=10, b=0),
                showlegend=False,
                hovermode='closest'
            )
            
            trend_container.plotly_chart(fig, use_container_width=True)
            time.sleep(0.2)
    
    return container

def generate_sentiment_data(ticker, days=30):
    """
    Generate realistic sentiment data for a stock based on its price movements
    
    Args:
        ticker: Stock symbol
        days: Number of days of data to generate
        
    Returns:
        Tuple of (overall_sentiment, news_sentiment_items)
    """
    try:
        # Get real stock data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days*2)  # Get extra data for calculations
        
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        
        if stock_data.empty:
            return 0, []
        
        # Calculate price changes to influence sentiment
        stock_data['PriceChange'] = stock_data['Close'].pct_change()
        stock_data['Return5d'] = stock_data['Close'].pct_change(5)
        stock_data['Return10d'] = stock_data['Close'].pct_change(10)
        
        # Filter to the requested number of days
        filtered_data = stock_data.tail(days)
        
        # Generate sentiment for the period based on stock performance
        sentiment_items = []
        
        # Use actual returns to influence sentiment
        for idx, row in filtered_data.iterrows():
            # Base sentiment on recent price movements
            daily_change = row['PriceChange'] if not pd.isna(row['PriceChange']) else 0
            short_trend = row['Return5d'] if not pd.isna(row['Return5d']) else 0
            long_trend = row['Return10d'] if not pd.isna(row['Return10d']) else 0
            
            # Calculate sentiment with some randomness
            base_sentiment = (daily_change * 3 + short_trend * 5 + long_trend * 2) / 10
            randomness = np.random.normal(0, 0.2)  # Add random noise
            
            # Clamp between -1 and 1
            sentiment_score = max(-1, min(1, base_sentiment * 10 + randomness))
            
            sentiment_items.append({
                'date': idx,
                'score': sentiment_score
            })
        
        # Calculate overall sentiment as weighted average of recent sentiment
        # Give more weight to recent days
        weights = np.linspace(0.5, 1.0, len(sentiment_items))
        overall_sentiment = sum(item['score'] * w for item, w in zip(sentiment_items, weights)) / sum(weights)
        
        # Calculate volatility from price data
        volatility = filtered_data['Close'].pct_change().std() * np.sqrt(252)  # Annualized
        volatility = min(1.0, volatility)  # Cap at 1.0
        
        # Calculate normalized volume
        avg_volume = filtered_data['Volume'].mean()
        recent_volume = filtered_data['Volume'].iloc[-5:].mean()
        volume_ratio = min(1.0, recent_volume / avg_volume if avg_volume > 0 else 0.5)
        
        return filtered_data, overall_sentiment, volatility, volume_ratio, sentiment_items
        
    except Exception as e:
        st.error(f"Error generating sentiment data: {str(e)}")
        return None, 0, 0, 0, []

def mood_ring_section():
    """Main function for the mood ring visualization section"""
    st.title("Stock Sentiment Mood Ring")
    st.markdown("""
    This interactive visualization shows the market sentiment for a stock as a color-changing mood ring,
    along with key metrics and sentiment trends.
    """)
    
    # Set up the form
    with st.form("mood_ring_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            ticker = st.text_input("Stock Symbol", value="AAPL")
        
        with col2:
            days = st.slider("Days of Sentiment History", min_value=7, max_value=60, value=30)
        
        submit_button = st.form_submit_button("Generate Mood Ring")
    
    if submit_button or 'mood_ring_ticker' in st.session_state:
        # Store ticker in session state to prevent regeneration on page refresh
        if submit_button:
            st.session_state.mood_ring_ticker = ticker
            st.session_state.mood_ring_days = days
        else:
            ticker = st.session_state.mood_ring_ticker
            days = st.session_state.mood_ring_days
        
        # Show loading animation
        with st.spinner(f"Generating mood ring for {ticker}..."):
            # Get data
            price_data, sentiment, volatility, volume, sentiment_items = generate_sentiment_data(ticker, days)
            
            if price_data is None or price_data.empty:
                st.error(f"Could not retrieve data for {ticker}. Please try a different symbol.")
                return
            
            # Display the mood ring visualization
            create_mood_ring_visualization(
                ticker, 
                price_data, 
                sentiment, 
                volatility, 
                volume, 
                sentiment_items
            )

import random  # Used by the code above for randomness in visualizations
