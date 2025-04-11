import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import os
import re
from urllib.request import urlopen
import base64

# Import function from sentiment_animation_simple for integrated visualization
from sentiment_animation_simple import create_simple_sentiment_visualization

# Initialize global variables
# Completely disable OpenAI to avoid any error messages
HAS_OPENAI = False
OPENAI_QUOTA_EXCEEDED = True

# No OpenAI API calls to prevent quota errors
try:
    from openai import OpenAI
except ImportError:
    pass

# Functions for news retrieval and sentiment analysis
def fetch_recent_news(ticker, limit=5):
    """
    Fetch recent news for a given ticker.
    Using Yahoo Finance data through the yfinance library.
    
    Args:
        ticker: Stock symbol
        limit: Maximum number of news articles to return
        
    Returns:
        List of news dictionary items
    """
    try:
        # Get the ticker object
        ticker_obj = yf.Ticker(ticker)
        
        # Get news - handle potential API changes in yfinance
        try:
            news_data = ticker_obj.news
            if not news_data:
                # If news data is empty, return sample data
                return generate_placeholder_news(ticker)
        except AttributeError:
            # If news attribute doesn't exist, use placeholder data
            return generate_placeholder_news(ticker)
        
        # Prepare news items with required fields
        news_items = []
        for item in news_data[:limit]:
            try:
                # Convert timestamp to datetime (safely)
                timestamp = item.get('providerPublishTime', 0)
                if timestamp:
                    published_date = datetime.fromtimestamp(timestamp)
                else:
                    published_date = datetime.now() - timedelta(days=1)
                
                news_items.append({
                    'title': item.get('title', f"{ticker} Recent News"),
                    'publisher': item.get('publisher', 'Financial News'),
                    'link': item.get('link', '#'),
                    'published': published_date.strftime('%Y-%m-%d %H:%M'),
                    'summary': item.get('summary', 'Details not available')
                })
            except (KeyError, TypeError) as e:
                # Skip malformed items
                continue
            
        if not news_items:
            # If we couldn't get news items, use placeholder data
            return generate_placeholder_news(ticker)
            
        return news_items
    except Exception as e:
        st.error(f"Error fetching news for {ticker}: {e}")
        return generate_placeholder_news(ticker)

def generate_placeholder_news(ticker):
    """
    Generate placeholder news items when real news isn't available
    This creates basic news items for analysis
    
    Args:
        ticker: Stock symbol
        
    Returns:
        List of placeholder news items
    """
    today = datetime.now()
    
    # Create placeholder news items
    return [
        {
            'title': f"{ticker} Stock Analysis Update",
            'publisher': "Market Analysis",
            'link': '#',
            'published': (today - timedelta(days=1)).strftime('%Y-%m-%d %H:%M'),
            'summary': f"{ticker} continues to trade based on market conditions and company fundamentals."
        },
        {
            'title': f"Industry Outlook Affecting {ticker}",
            'publisher': "Sector Reports",
            'link': '#',
            'published': (today - timedelta(days=2)).strftime('%Y-%m-%d %H:%M'),
            'summary': f"Industry trends show evolving conditions that may impact {ticker}'s business operations."
        },
        {
            'title': f"Market Trends and {ticker} Position",
            'publisher': "Financial Times",
            'link': '#',
            'published': (today - timedelta(days=3)).strftime('%Y-%m-%d %H:%M'),
            'summary': f"Market analysts reviewing the position of {ticker} in current economic conditions."
        }
    ]

def analyze_sentiment_with_openai(text):
    """
    Basic sentiment analysis function (OpenAI API disabled)
    
    Args:
        text: Text to analyze
    
    Returns:
        Dictionary with sentiment analysis results
    """
    # Always use basic sentiment analysis without OpenAI API
    return analyze_sentiment_basic(text)

def analyze_sentiment_basic(text):
    """
    A basic rule-based sentiment analysis when OpenAI is not available.
    
    Args:
        text: Text to analyze
    
    Returns:
        Dictionary with sentiment analysis results
    """
    # Lists of positive and negative words for finance
    positive_words = [
        'gain', 'gains', 'up', 'rise', 'risen', 'rising', 'rose', 'bull', 'bullish',
        'outperform', 'outperformed', 'beat', 'beats', 'positive', 'profit', 'profits',
        'growth', 'grew', 'grow', 'increase', 'increased', 'increasing', 'higher',
        'upgrade', 'upgraded', 'strong', 'strength', 'opportunity', 'opportunities',
        'success', 'successful', 'improve', 'improved', 'improving', 'recovery',
        'rally', 'surge', 'jump', 'jumped', 'boost', 'boosted'
    ]
    
    negative_words = [
        'loss', 'losses', 'down', 'fall', 'fell', 'falling', 'drop', 'dropped',
        'bear', 'bearish', 'underperform', 'underperformed', 'miss', 'missed',
        'negative', 'decline', 'declined', 'declining', 'decrease', 'decreased',
        'lower', 'downgrade', 'downgraded', 'weak', 'weakness', 'risk', 'risks',
        'risky', 'fail', 'failed', 'failure', 'concern', 'concerns', 'worried',
        'worry', 'fears', 'fear', 'sell-off', 'crash', 'crisis', 'trouble',
        'troubled', 'recession', 'slump', 'tumble', 'tumbled', 'plunge', 'plunged'
    ]
    
    # Convert to lowercase and split into words
    text_lower = text.lower()
    words = re.findall(r'\b\w+\b', text_lower)
    
    # Count positive and negative words
    positive_count = sum(1 for word in words if word in positive_words)
    negative_count = sum(1 for word in words if word in negative_words)
    
    # Calculate sentiment score
    total_words = len(words)
    if total_words > 0:
        positive_ratio = positive_count / total_words
        negative_ratio = negative_count / total_words
        sentiment_score = 0.5 + (positive_ratio - negative_ratio)
        # Clamp between 0 and 1
        sentiment_score = max(0, min(1, sentiment_score))
    else:
        sentiment_score = 0.5  # Neutral if no words
    
    # Determine label
    if sentiment_score > 0.6:
        label = "POSITIVE"
        explanation = f"Detected {positive_count} positive financial terms vs {negative_count} negative terms."
    elif sentiment_score < 0.4:
        label = "NEGATIVE"
        explanation = f"Detected {negative_count} negative financial terms vs {positive_count} positive terms."
    else:
        label = "NEUTRAL"
        explanation = f"Balance of positive ({positive_count}) and negative ({negative_count}) financial terms."
    
    # Set confidence based on word count and the distance from neutral
    confidence = min(1.0, (total_words / 100) * abs(sentiment_score - 0.5) * 2)
    
    return {
        "score": sentiment_score,
        "label": label,
        "explanation": explanation,
        "confidence": confidence
    }

def get_market_sentiment_data(period="1mo"):
    """
    Get sentiment data for the overall market (using S&P 500 as proxy).
    
    Args:
        period: Time period for historical data
    
    Returns:
        DataFrame with sentiment data over time
    """
    # First try S&P 500 (^GSPC) then fall back to SPY ETF
    market_data = None
    error_details = None
    
    # Try multiple data sources with fallbacks
    for ticker in ["^GSPC", "SPY", "QQQ", "DIA"]:
        try:
            print(f"Attempting to fetch market data using {ticker}...")
            
            # Method 1: Use yf.download directly
            market_data = yf.download(ticker, period=period, progress=False)
            if market_data is not None and len(market_data) > 5:
                print(f"Successfully fetched market data using {ticker}")
                break
            else:
                market_data = None
                
        except Exception as e:
            error_details = str(e)
            market_data = None
            print(f"Error fetching {ticker} data: {error_details}")
            continue
            
        # Method 2: Use different period if first attempt failed
        try:
            if market_data is None:
                print(f"Trying with longer period for {ticker}...")
                market_data = yf.download(ticker, period="3mo", progress=False)
                if market_data is not None and len(market_data) > 5:
                    print(f"Successfully fetched market data using {ticker} with longer period")
                    break
                else:
                    market_data = None
        except Exception as e:
            error_details = str(e)
            market_data = None
            print(f"Error fetching {ticker} data with longer period: {error_details}")
            continue
            
        # Method 3: Try using Ticker object
        try:
            if market_data is None:
                print(f"Trying Ticker.history for {ticker}...")
                ticker_obj = yf.Ticker(ticker)
                market_data = ticker_obj.history(period=period)
                if market_data is not None and len(market_data) > 5:
                    print(f"Successfully fetched market data using {ticker}.history()")
                    break
                else:
                    market_data = None
        except Exception as e:
            error_details = str(e)
            market_data = None
            print(f"Error fetching {ticker} data with Ticker object: {error_details}")
            continue
    
    # If we still don't have data after trying all options
    if market_data is None or market_data is None:
        st.error(f"Failed to fetch market data: {error_details}")
        return pd.DataFrame()
    
    try:
        # Calculate daily returns
        market_data['Return'] = market_data['Close'].pct_change()
        
        # Generate sentiment scores based on price action
        market_data['Sentiment'] = ((market_data['Return'] * 10) + 0.5).clip(0, 1)
        market_data['Sentiment'] = market_data['Sentiment'].fillna(0.5)  # Fill NaN with neutral sentiment
        
        # Label the sentiment (POSITIVE, NEUTRAL, NEGATIVE)
        conditions = [
            (market_data['Sentiment'] >= 0.6),
            (market_data['Sentiment'] <= 0.4),
        ]
        choices = ['POSITIVE', 'NEGATIVE']
        market_data['Sentiment_Label'] = np.select(conditions, choices, default='NEUTRAL')
        
        # Ensure we have a date column (sometimes needed for plotting)
        market_data = market_data.reset_index()
        
        return market_data
        
    except Exception as e:
        st.error(f"Error processing market sentiment data: {e}")
        return pd.DataFrame()

def get_sentiment_for_multiple_tickers(tickers, days=30):
    """
    Calculate sentiment scores for multiple tickers over the specified time period.
    
    Args:
        tickers: List of ticker symbols
        days: Number of days to look back
    
    Returns:
        DataFrame with sentiment scores for each ticker
    """
    # Access the global variable
    global OPENAI_QUOTA_EXCEEDED
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    result_data = []
    
    for ticker in tickers:
        try:
            # Use a simplified direct approach
            ticker_obj = yf.Ticker(ticker)
            
            # Create a placeholder with minimal required data even if actual fetch fails
            # This ensures we always have some data to display
            minimal_data = {
                'Ticker': ticker,
                'Company': ticker,  # Will try to update with actual name if available
                'Sentiment_Score': 0.5,  # Neutral sentiment
                'Sentiment_Label': 'NEUTRAL',
                'Price_Change_Pct': 0.0, 
                'Volatility': 0.0,
                'Recent_Return': 0.0
            }
            
            # Try to get basic info
            try:
                info = ticker_obj.info
                if info and 'shortName' in info:
                    minimal_data['Company'] = info.get('shortName', info.get('longName', ticker))
            except Exception as e:
                print(f"Unable to get info for {ticker}: {str(e)}")
            
            # Try to get price data with multiple fallback methods
            stock_data = None
            
            # Method 1: Ticker history
            try:
                stock_data = ticker_obj.history(period=f"{days}d")
                if stock_data is not None and len(stock_data) > 5:
                    print(f"Successfully fetched {ticker} data using Ticker.history")
                else:
                    stock_data = None
            except Exception as e:
                print(f"Ticker.history failed for {ticker}: {str(e)}")
                stock_data = None
            
            # Method 2: yf.download with period
            if stock_data is None:
                try:
                    stock_data = yf.download(ticker, period="3mo", progress=False)
                    if stock_data is not None and len(stock_data) > 5:
                        print(f"Successfully fetched {ticker} data using yf.download with period")
                    else:
                        stock_data = None
                except Exception as e:
                    print(f"yf.download with period failed for {ticker}: {str(e)}")
                    stock_data = None
            
            # Method 3: yf.download with date range
            if stock_data is None:
                try:
                    stock_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                    if stock_data is not None and len(stock_data) > 5:
                        print(f"Successfully fetched {ticker} data using yf.download with dates")
                    else:
                        stock_data = None
                except Exception as e:
                    print(f"yf.download with dates failed for {ticker}: {str(e)}")
                    stock_data = None
            
            # If we got stock data, calculate metrics
            if stock_data is not None and stock_data is not None and len(stock_data) > 5:
                # Calculate metrics with careful error handling
                try:
                    # Calculate returns
                    stock_data['Return'] = stock_data['Close'].pct_change().fillna(0)
                    
                    # Get price change percentage
                    first_price = stock_data['Close'].iloc[0] if stock_data is not None else 0
                    last_price = stock_data['Close'].iloc[-1] if stock_data is not None else 0
                    
                    if first_price > 0:
                        price_trend = (last_price / first_price) - 1
                    else:
                        price_trend = 0
                        
                    # Calculate volatility  
                    volatility = stock_data['Return'].std() * (252 ** 0.5) if len(stock_data) > 1 else 0
                    
                    # Calculate sentiment score based on price action
                    price_sentiment = 0.5 + (stock_data['Return'].mean() * 50)  # Scale returns to have more impact
                    price_sentiment = max(0, min(1, price_sentiment))  # Clamp between 0 and 1
                    
                    # Update the minimal data with actual calculated values
                    minimal_data['Price_Change_Pct'] = price_trend * 100
                    minimal_data['Volatility'] = volatility * 100 if not np.isnan(volatility) else 0
                    minimal_data['Recent_Return'] = stock_data['Return'].iloc[-1] * 100 if len(stock_data) > 1 else 0
                    minimal_data['Sentiment_Score'] = price_sentiment
                    
                    # Determine sentiment label
                    if price_sentiment >= 0.6:
                        minimal_data['Sentiment_Label'] = "POSITIVE"
                    elif price_sentiment <= 0.4:
                        minimal_data['Sentiment_Label'] = "NEGATIVE"
                    else:
                        minimal_data['Sentiment_Label'] = "NEUTRAL"
                    
                    # Try to enhance with news sentiment
                    try:
                        # News sentiment - get recent news and analyze
                        news_items = fetch_recent_news(ticker, limit=3)
                        news_sentiment_scores = []
                        
                        for news in news_items:
                            try:
                                # Use OpenAI only if we have a key AND haven't exceeded quota
                                if HAS_OPENAI and not OPENAI_QUOTA_EXCEEDED:
                                    try:
                                        sentiment = analyze_sentiment_with_openai(news['title'] + ". " + news['summary'])
                                    except Exception as e:
                                        # If OpenAI fails, use basic analysis and disable OpenAI
                                        if "quota" in str(e).lower() or "429" in str(e):
                                            # Global already declared at function level
                                            OPENAI_QUOTA_EXCEEDED = True
                                        sentiment = analyze_sentiment_basic(news['title'] + ". " + news['summary'])
                                else:
                                    # Always use basic analysis if OpenAI not available or quota exceeded
                                    sentiment = analyze_sentiment_basic(news['title'] + ". " + news['summary'])
                                
                                if isinstance(sentiment, dict) and 'score' in sentiment:
                                    news_sentiment_scores.append(sentiment['score'])
                                else:
                                    # Use a neutral score if sentiment analysis returned invalid result
                                    news_sentiment_scores.append(0.5)
                            except Exception as e:
                                # If sentiment analysis fails, use a neutral score
                                news_sentiment_scores.append(0.5)
                        
                        # If we got news sentiment, blend it with price sentiment
                        if news_sentiment_scores:
                            news_sentiment = sum(news_sentiment_scores) / len(news_sentiment_scores)
                            # Weight: 70% price action, 30% news
                            overall_sentiment = 0.7 * price_sentiment + 0.3 * news_sentiment
                            
                            # Update minimal data with the combined sentiment
                            minimal_data['Sentiment_Score'] = overall_sentiment
                            
                            # Determine sentiment label
                            if overall_sentiment >= 0.6:
                                minimal_data['Sentiment_Label'] = "POSITIVE"
                            elif overall_sentiment <= 0.4:
                                minimal_data['Sentiment_Label'] = "NEGATIVE"
                            else:
                                minimal_data['Sentiment_Label'] = "NEUTRAL"
                    except Exception as e:
                        print(f"Error processing news for {ticker}: {str(e)}")
                        # Keep using the price-based sentiment (already set)
                        
                except Exception as e:
                    print(f"Error calculating metrics for {ticker}: {str(e)}")
            else:
                # Data fetch failed for all methods
                st.error(f"Error processing {ticker}: Unable to fetch valid data")
                
            # Always add the result to ensure we have something to display
            result_data.append(minimal_data)
            
        except Exception as e:
            st.error(f"Error processing {ticker}: {e}")
            # Still add a minimal placeholder for this ticker
            result_data.append({
                'Ticker': ticker,
                'Company': ticker,
                'Sentiment_Score': 0.5,
                'Sentiment_Label': 'NEUTRAL',
                'Price_Change_Pct': 0.0,
                'Volatility': 0.0,
                'Recent_Return': 0.0
            })
    
    if result_data:
        return pd.DataFrame(result_data)
    else:
        return pd.DataFrame(columns=['Ticker', 'Company', 'Sentiment_Score', 'Sentiment_Label', 
                                    'Price_Change_Pct', 'Volatility', 'Recent_Return'])

def categorize_news_impact(news_item):
    """
    Categorize news based on potential market impact.
    
    Args:
        news_item: News item dictionary
        
    Returns:
        Dictionary with impact category and probability
    """
    title = news_item.get('title', '').lower()
    summary = news_item.get('summary', '').lower()
    text = title + " " + summary
    
    # Keywords for high-impact news events
    earnings_keywords = ['earnings', 'quarterly results', 'financial results', 'q1', 'q2', 'q3', 'q4', 
                         'beat expectations', 'missed expectations', 'eps', 'revenue', 'profit', 'loss']
    
    product_keywords = ['product launch', 'new product', 'announced', 'unveils', 'releases', 'introducing']
    
    management_keywords = ['ceo', 'executive', 'appointed', 'resigned', 'steps down', 'board', 'leadership']
    
    legal_keywords = ['lawsuit', 'litigation', 'settlement', 'regulatory', 'investigation', 'sec', 'fine', 'penalty']
    
    merger_keywords = ['acquisition', 'merger', 'buyout', 'takeover', 'acquiring', 'purchased', 'sold']
    
    stock_action_keywords = ['stock split', 'buyback', 'repurchase', 'dividend', 'offering', 'dilution']
    
    # Check for matches in each category
    impact_categories = []
    
    if any(keyword in text for keyword in earnings_keywords):
        impact_categories.append('Earnings')
    
    if any(keyword in text for keyword in product_keywords):
        impact_categories.append('Product News')
    
    if any(keyword in text for keyword in management_keywords):
        impact_categories.append('Management Change')
    
    if any(keyword in text for keyword in legal_keywords):
        impact_categories.append('Legal/Regulatory')
    
    if any(keyword in text for keyword in merger_keywords):
        impact_categories.append('M&A Activity')
    
    if any(keyword in text for keyword in stock_action_keywords):
        impact_categories.append('Stock Action')
    
    # Return the categorization
    if impact_categories:
        return {
            'categories': impact_categories,
            'has_high_impact': True
        }
    else:
        return {
            'categories': ['General News'],
            'has_high_impact': False
        }

def analyze_news_sentiment(ticker, limit=5):
    """
    Analyze sentiment for recent news articles for a given ticker.
    
    Args:
        ticker: Stock symbol
        limit: Maximum number of news articles to analyze
    
    Returns:
        List of news items with sentiment analysis and impact categorization
    """
    # Access the global variable
    global OPENAI_QUOTA_EXCEEDED
    
    news_items = fetch_recent_news(ticker, limit=limit)
    
    analyzed_news = []
    for news in news_items:
        try:
            # Use OpenAI only if we have API key and haven't exceeded quota
            if HAS_OPENAI and not OPENAI_QUOTA_EXCEEDED:
                # Use OpenAI for sentiment analysis if available
                try:
                    sentiment = analyze_sentiment_with_openai(news['title'] + ". " + news['summary'])
                except Exception as e:
                    # If OpenAI fails (quota exceeded, etc.), fallback to basic
                    if "quota" in str(e).lower() or "429" in str(e):
                        OPENAI_QUOTA_EXCEEDED = True
                        print(f"OpenAI quota exceeded, using fallback analysis: {str(e)}")
                        # We will show a warning at the top of the page instead of on every news item
                    else:
                        print(f"OpenAI analysis failed: {str(e)}")
                        # Log the error but don't show in UI for every item
                    
                    sentiment = analyze_sentiment_basic(news['title'] + ". " + news['summary'])
            else:
                # Fallback to basic sentiment analysis
                sentiment = analyze_sentiment_basic(news['title'] + ". " + news['summary'])
        except Exception as e:
            # If all sentiment analysis fails, use a neutral placeholder
            st.error(f"Sentiment analysis failed: {str(e)}")
            sentiment = {
                "score": 0.5,
                "label": "NEUTRAL",
                "explanation": "Error in sentiment analysis",
                "confidence": 0.0
            }
        
        # Add sentiment to news item
        news['sentiment'] = sentiment
        
        # Add impact categorization
        try:
            news['impact'] = categorize_news_impact(news)
        except Exception as e:
            news['impact'] = {'categories': ['General News'], 'has_high_impact': False}
            print(f"Error in impact categorization: {str(e)}")
            
        analyzed_news.append(news)
    
    return analyzed_news

def get_user_sentiment_preferences():
    """Get user preferences for sentiment tracking from session state"""
    if 'sentiment_preferences' not in st.session_state:
        st.session_state.sentiment_preferences = {
            'tracked_tickers': [],
            'sentiment_alerts': False,
            'alert_threshold': 0.7,
            'news_sources_priority': ['Yahoo Finance'],
            'display_mode': 'Detailed'
        }
    
    return st.session_state.sentiment_preferences

def save_user_sentiment_preferences(preferences):
    """Save user sentiment tracking preferences to session state"""
    st.session_state.sentiment_preferences = preferences

def plot_sentiment_gauge(sentiment_score, title="Sentiment Score"):
    """
    Create a gauge chart to visualize sentiment score.
    
    Args:
        sentiment_score: Sentiment score (0-1)
        title: Title for the gauge
    
    Returns:
        Plotly gauge figure
    """
    # Determine the color based on the sentiment
    if sentiment_score >= 0.6:
        color = "green"
    elif sentiment_score <= 0.4:
        color = "red"
    else:
        color = "gold"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=sentiment_score * 100,  # Convert to percentage
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 40], 'color': 'rgba(255, 0, 0, 0.2)'},  # Negative - light red
                {'range': [40, 60], 'color': 'rgba(255, 215, 0, 0.2)'},  # Neutral - light yellow
                {'range': [60, 100], 'color': 'rgba(0, 128, 0, 0.2)'}  # Positive - light green
            ],
        }
    ))
    
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    
    return fig

def plot_sentiment_timeline(sentiment_data):
    """
    Create a timeline visualization of sentiment data.
    
    Args:
        sentiment_data: DataFrame with sentiment data over time
    
    Returns:
        Plotly figure with sentiment timeline
    """
    if len(sentiment_data) == 0:
        return None
    
    # Ensure we have a Date column
    if 'Date' not in sentiment_data.columns:
        return None
    
    # Create a figure with secondary y-axis
    fig = go.Figure()
    
    # Add price line
    fig.add_trace(
        go.Scatter(
            x=sentiment_data['Date'],
            y=sentiment_data['Close'],
            name='Price',
            line=dict(color='blue')
        )
    )
    
    # Create discrete color map for sentiment labels
    color_map = {
        'POSITIVE': 'green',
        'NEUTRAL': 'gold',
        'NEGATIVE': 'red'
    }
    
    # Map colors to sentiment labels
    colors = sentiment_data['Sentiment_Label'].map(color_map)
    
    # Add sentiment score as markers with varying colors
    fig.add_trace(
        go.Scatter(
            x=sentiment_data['Date'],
            y=sentiment_data['Sentiment'] * 100,  # Scale to percentage
            mode='markers',
            name='Sentiment',
            marker=dict(
                size=8,
                color=colors,
                symbol='circle',
                line=dict(width=1, color='DarkSlateGrey')
            ),
            yaxis='y2'
        )
    )
    
    # Set up the layout with dual y-axes
    fig.update_layout(
        title='Price and Sentiment Over Time',
        xaxis=dict(title='Date'),
        yaxis=dict(
            title=dict(text='Price', font=dict(color='blue')),
            tickfont=dict(color='blue')
        ),
        yaxis2=dict(
            title=dict(text='Sentiment Score (%)', font=dict(color='darkgreen')),
            tickfont=dict(color='darkgreen'),
            anchor='x',
            overlaying='y',
            side='right',
            range=[0, 100]
        ),
        legend=dict(x=0, y=1.1, orientation='h'),
        height=400
    )
    
    return fig

def plot_sentiment_news_overlay(price_data, news_items, ticker):
    """
    Create an enhanced stock chart with news sentiment overlay.
    
    Args:
        price_data: DataFrame with price data
        news_items: List of news items with sentiment and dates
        ticker: Ticker symbol
    
    Returns:
        Plotly figure with price chart and news sentiment overlay
    """
    if len(price_data) == 0 or not news_items:
        return None
    
    # Create figure with secondary y-axis
    fig = go.Figure()
    
    # Add price candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=price_data.index,
            open=price_data['Open'],
            high=price_data['High'],
            low=price_data['Low'],
            close=price_data['Close'],
            name='Price',
            increasing_line_color='#26a69a',  # Nicer green
            decreasing_line_color='#ef5350'   # Nicer red
        )
    )
    
    # Add volume bars at the bottom
    colors = ['#ef5350' if row['Close'] < row['Open'] else '#26a69a' for _, row in price_data.iterrows()]
    
    # Calculate volume and position it at the bottom of the chart
    volume_scale = 0.2  # Use 20% of the chart for volume
    max_volume = price_data['Volume'].max()
    price_min = price_data['Low'].min()
    price_range = price_data['High'].max() - price_min
    volume_base = price_min - (price_range * 0.25)  # Position below price with 25% of price range as gap
    
    fig.add_trace(
        go.Bar(
            x=price_data.index,
            y=price_data['Volume'] * (price_range * volume_scale) / max_volume,  # Scale volume to fit
            base=volume_base,  # Start from the bottom position
            name='Volume',
            marker_color=colors,
            opacity=0.5
        )
    )
    
    # Add moving averages
    fig.add_trace(
        go.Scatter(
            x=price_data.index,
            y=price_data['Close'].rolling(window=20).mean(),
            name='20-day MA',
            line=dict(color='rgba(75, 192, 192, 0.7)', width=1.5)
        )
    )
    
    # Prepare news data for plotting
    news_dates = []
    news_scores = []
    news_labels = []
    news_titles = []
    news_categories = []
    news_links = []
    news_colors = []
    news_impact_levels = []
    news_impact_categories = []
    
    # Process each news item
    for news in news_items:
        try:
            # Try to parse the date
            date_str = news.get('published', '')
            if not date_str:
                continue
                
            # Try different date formats
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
            
            # Only include news within the price data date range
            if news_date.date() < price_data.index[0].date() or news_date.date() > price_data.index[-1].date():
                continue
                
            # Get sentiment data
            sentiment = news.get('sentiment', {'score': 0.5, 'label': 'NEUTRAL'})
            score = sentiment.get('score', 0.5)
            label = sentiment.get('label', 'NEUTRAL')
            
            # Get impact categories
            impact = news.get('impact', {'categories': ['General News'], 'has_high_impact': False})
            categories = impact.get('categories', ['General News'])
            is_high_impact = impact.get('has_high_impact', False) or any(cat in ['Earnings', 'M&A Activity', 'Legal/Regulatory'] for cat in categories)
            
            # Determine impact level
            if is_high_impact:
                impact_level = 'High'
                impact_size = 16
            elif 'Stock Action' in categories:
                impact_level = 'Medium'
                impact_size = 12
            else:
                impact_level = 'Low'
                impact_size = 8
            
            # Set marker color based on sentiment
            if label == 'POSITIVE':
                color = '#4CAF50'  # Vibrant green
                symbol = 'triangle-up'
            elif label == 'NEGATIVE':
                color = '#F44336'  # Vibrant red
                symbol = 'triangle-down'
            else:
                color = '#FFC107'  # Amber yellow
                symbol = 'circle'
                
            # Add to lists for plotting
            news_dates.append(news_date)
            news_scores.append(score * 100)  # Scale to percentage
            news_labels.append(label)
            news_titles.append(news.get('title', 'News'))
            news_categories.append(', '.join(categories))
            news_links.append(news.get('link', '#'))
            news_colors.append(color)
            news_impact_levels.append(impact_level)
            news_impact_categories.append(categories)
            
        except Exception as e:
            print(f"Error processing news item for chart: {str(e)}")
            continue
    
    # Add news sentiment as markers if we have any
    if news_dates:
        # Find price range for vertical positioning
        price_min = price_data['Low'].min()
        price_max = price_data['High'].max()
        price_range = price_max - price_min
        
        # Create high, medium, and low impact news groups for more organized display
        high_impact_dates = []
        high_impact_colors = []
        high_impact_texts = []
        high_impact_categories = []
        
        medium_impact_dates = []
        medium_impact_colors = []
        medium_impact_texts = []
        
        low_impact_dates = []
        low_impact_colors = []
        low_impact_texts = []
        
        # Organize news by impact
        for i, (date, score, label, title, categories, link, color, impact) in enumerate(
            zip(news_dates, news_scores, news_labels, news_titles, news_categories, 
                news_links, news_colors, news_impact_levels)
        ):
            hover_text = f"<b>{title}</b><br>Date: {date.strftime('%Y-%m-%d')}<br>Sentiment: {label} ({score:.1f}%)<br>Categories: {categories}<br>Click for details"
            
            if impact == 'High':
                high_impact_dates.append(date)
                high_impact_colors.append(color)
                high_impact_texts.append(hover_text)
                high_impact_categories.append(categories)
            elif impact == 'Medium':
                medium_impact_dates.append(date)
                medium_impact_colors.append(color)
                medium_impact_texts.append(hover_text)
            else:
                low_impact_dates.append(date)
                low_impact_colors.append(color)
                low_impact_texts.append(hover_text)
                
        # Add vertical regions for earnings dates and other high-impact events
        for i, (date, color, text, categories) in enumerate(
            zip(high_impact_dates, high_impact_colors, high_impact_texts, high_impact_categories)
        ):
            # Determine which impact category to show
            if 'Earnings' in categories:
                event_type = "📊 Earnings"
                line_width = 2
            elif 'M&A Activity' in categories:
                event_type = "🤝 M&A"
                line_width = 2
            elif 'Legal/Regulatory' in categories:
                event_type = "⚖️ Regulatory"
                line_width = 2
            else:
                event_type = "📣 News"
                line_width = 1
                
            # Add vertical line with improved annotation
            fig.add_vline(
                x=date, 
                line_width=line_width, 
                line_dash="dash", 
                line_color=color,
                annotation_text=event_type,
                annotation_position="top right",
                annotation_font=dict(size=10, color=color),
                annotation_bgcolor="white",
                annotation_bordercolor=color,
                annotation_borderwidth=1,
            )
            
            # Add a label at the bottom of the chart
            fig.add_annotation(
                x=date,
                y=price_min - (price_range * 0.12),
                text=event_type,
                showarrow=False,
                font=dict(size=8, color="white"),
                bgcolor=color,
                borderpad=3,
                borderwidth=1,
                bordercolor=color,
                opacity=0.8
            )
        
        # Add high-impact news markers
        if high_impact_dates:
            fig.add_trace(
                go.Scatter(
                    x=high_impact_dates,
                    y=[price_min - (price_range * 0.05)] * len(high_impact_dates),
                    mode='markers',
                    name='High Impact News',
                    marker=dict(
                        size=16,
                        color=high_impact_colors,
                        symbol='star',
                        line=dict(width=1, color='white')
                    ),
                    text=high_impact_texts,
                    hoverinfo='text'
                )
            )
        
        # Add medium-impact news markers
        if medium_impact_dates:
            fig.add_trace(
                go.Scatter(
                    x=medium_impact_dates,
                    y=[price_min - (price_range * 0.05)] * len(medium_impact_dates),
                    mode='markers',
                    name='Medium Impact News',
                    marker=dict(
                        size=12,
                        color=medium_impact_colors,
                        symbol='triangle-up',
                        line=dict(width=1, color='white')
                    ),
                    text=medium_impact_texts,
                    hoverinfo='text'
                )
            )
        
        # Add low-impact news markers
        if low_impact_dates:
            fig.add_trace(
                go.Scatter(
                    x=low_impact_dates,
                    y=[price_min - (price_range * 0.05)] * len(low_impact_dates),
                    mode='markers',
                    name='Low Impact News',
                    marker=dict(
                        size=8,
                        color=low_impact_colors,
                        symbol='circle',
                        line=dict(width=1, color='white')
                    ),
                    text=low_impact_texts,
                    hoverinfo='text'
                )
            )
        
        # Add a sentiment trend line
        if len(news_dates) > 1:
            # Sort dates and scores together
            sorted_data = sorted(zip(news_dates, news_scores))
            sorted_dates = [item[0] for item in sorted_data]
            sorted_scores = [item[1] for item in sorted_data]
            
            # Calculate the normalized position for the sentiment line (in the top 15% of the chart)
            sentiment_base = price_max - (price_range * 0.05)
            sentiment_scale = price_range * 0.10  # Use 10% of the price range
            
            # Calculate normalized scores (0-1 range to use with the scale)
            norm_scores = [(score - 0) / 100 for score in sorted_scores]
            scaled_scores = [sentiment_base + (score * sentiment_scale) for score in norm_scores]
            
            # Add the sentiment trend line
            fig.add_trace(
                go.Scatter(
                    x=sorted_dates,
                    y=scaled_scores,
                    mode='lines',
                    name='Sentiment Trend',
                    line=dict(
                        color='rgba(153, 102, 255, 0.7)',
                        width=2,
                        dash='dot'
                    ),
                    hoverinfo='none'
                )
            )
    
    # Set up layout
    fig.update_layout(
        title=dict(
            text=f'{ticker} Price with News Sentiment Overlay',
            font=dict(size=18)
        ),
        xaxis_title='Date',
        yaxis_title='Price',
        height=700,  # Taller chart for better visibility
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h", 
            yanchor="bottom", 
            y=1.02, 
            xanchor="right", 
            x=1,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='lightgrey',
            borderwidth=1
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=60, r=60, t=60, b=80),
        hovermode='closest'
    )
    
    # Add grid lines for better readability
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(211, 211, 211, 0.5)',
        zeroline=False
    )
    
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(211, 211, 211, 0.5)'
    )
    
    # Update y-axis to include space for news markers and annotations
    if news_dates:
        fig.update_yaxes(
            range=[price_min - (price_range * 0.2), price_max + (price_range * 0.15)]
        )
    
    # Add a background shading for the news sentiment area
    fig.add_shape(
        type="rect",
        x0=price_data.index[0],
        x1=price_data.index[-1],
        y0=price_min - (price_range * 0.15),
        y1=price_min,
        fillcolor="rgba(200, 200, 200, 0.2)",
        line=dict(width=0),
        layer="below"
    )
    
    # Add an explanatory annotation for the news sentiment area
    fig.add_annotation(
        x=price_data.index[0],
        y=price_min - (price_range * 0.15),
        xanchor="left",
        yanchor="bottom",
        text="News Sentiment Indicators",
        showarrow=False,
        font=dict(size=10, color="gray"),
        borderpad=4
    )
    
    return fig

def plot_sentiment_heatmap(sentiment_data):
    """
    Create a heatmap of sentiment across different tickers.
    
    Args:
        sentiment_data: DataFrame with sentiment data for multiple tickers
    
    Returns:
        Plotly figure with sentiment heatmap
    """
    if len(sentiment_data) == 0 or len(sentiment_data) < 2:
        return None
    
    # Sort by sentiment score
    sentiment_data = sentiment_data.sort_values('Sentiment_Score', ascending=False)
    
    # Create a custom colorscale
    colorscale = [
        [0, 'red'],         # Very negative
        [0.4, 'lightcoral'],  # Negative
        [0.5, 'gold'],      # Neutral
        [0.6, 'lightgreen'],  # Positive
        [1, 'green']        # Very positive
    ]
    
    # Create heatmap
    fig = go.Figure()
    
    # Add the heatmap trace
    fig.add_trace(go.Heatmap(
        z=[sentiment_data['Sentiment_Score'] * 100],  # Convert to percentage
        x=sentiment_data['Ticker'],
        y=['Sentiment'],
        colorscale=colorscale,
        zmin=0,
        zmax=100,
        showscale=True,
        colorbar=dict(
            title=dict(
                text='Sentiment Score'
                # Removed 'side' property which was causing errors
            )
        ),
        text=[[f"{score:.1f}%" for score in sentiment_data['Sentiment_Score'] * 100]],
        hoverinfo='text'
    ))
    
    # Add ticker labels
    fig.update_layout(
        title='Sentiment Comparison Across Stocks',
        xaxis=dict(
            title='',
            tickangle=-45
        ),
        yaxis=dict(title=''),
        height=200,
        margin=dict(l=50, r=50, t=50, b=100)
    )
    
    return fig

def plot_correlation_chart(sentiment_data, price_data):
    """
    Create a chart showing correlation between sentiment and price movement.
    
    Args:
        sentiment_data: DataFrame with sentiment scores
        price_data: DataFrame with price data
    
    Returns:
        Plotly figure with correlation chart
    """
    if len(sentiment_data) == 0 or len(price_data) == 0:
        return None
    
    # Create a scatter plot
    fig = px.scatter(
        sentiment_data,
        x='Sentiment_Score',
        y='Price_Change_Pct',
        color='Sentiment_Label',
        size='Volatility',
        hover_name='Ticker',
        color_discrete_map={
            'POSITIVE': 'green',
            'NEUTRAL': 'gold',
            'NEGATIVE': 'red'
        },
        title='Sentiment vs. Price Change',
        labels={
            'Sentiment_Score': 'Sentiment Score',
            'Price_Change_Pct': 'Price Change (%)',
            'Volatility': 'Volatility (%)'
        }
    )
    
    # Add a horizontal line at y=0 to show positive/negative price change
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    
    # Add a vertical line at x=0.5 to show positive/negative sentiment
    fig.add_vline(x=0.5, line_dash="dash", line_color="gray")
    
    # Add quadrant labels
    fig.add_annotation(x=0.25, y=5, text="Negative Sentiment, Positive Return", showarrow=False, font=dict(size=10))
    fig.add_annotation(x=0.75, y=5, text="Positive Sentiment, Positive Return", showarrow=False, font=dict(size=10))
    fig.add_annotation(x=0.25, y=-5, text="Negative Sentiment, Negative Return", showarrow=False, font=dict(size=10))
    fig.add_annotation(x=0.75, y=-5, text="Positive Sentiment, Negative Return", showarrow=False, font=dict(size=10))
    
    # Update layout
    fig.update_layout(
        height=500,
        xaxis=dict(range=[0, 1]),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def sentiment_tracker_section():
    """Main function for the sentiment tracking page"""
    # Access global variables at the beginning of the function
    global OPENAI_QUOTA_EXCEEDED
    
    st.title("📊 Market Sentiment Tracker")
    
    # Sentiment analysis is handled without OpenAI API warnings
    
    # Get user preferences
    preferences = get_user_sentiment_preferences()
    
    # Sidebar for sentiment tracker settings
    with st.sidebar:
        st.subheader("Sentiment Tracker Settings")
        
        # Add ticker tracking
        new_ticker = st.text_input("Add ticker to track:", key="sentiment_new_ticker").upper()
        if st.button("Add Ticker"):
            if new_ticker and new_ticker not in preferences['tracked_tickers']:
                preferences['tracked_tickers'].append(new_ticker)
                save_user_sentiment_preferences(preferences)
                st.success(f"Added {new_ticker} to tracked tickers")
        
        # Show tracked tickers with option to remove
        st.write("Currently tracking:")
        for i, ticker in enumerate(preferences['tracked_tickers']):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(ticker)
            with col2:
                if st.button("❌", key=f"remove_{ticker}"):
                    preferences['tracked_tickers'].pop(i)
                    save_user_sentiment_preferences(preferences)
                    st.rerun()
        
        # Sentiment alerts
        st.subheader("Sentiment Alerts")
        preferences['sentiment_alerts'] = st.toggle("Enable Sentiment Alerts", preferences['sentiment_alerts'])
        
        if preferences['sentiment_alerts']:
            preferences['alert_threshold'] = st.slider("Alert Threshold", 0.0, 1.0, preferences['alert_threshold'], 0.05)
            st.info(f"You'll be alerted when sentiment score exceeds {preferences['alert_threshold']:.2f}")
        
        # Display preferences
        st.subheader("Display Preferences")
        preferences['display_mode'] = st.radio("Display Mode", ["Detailed", "Summary"], index=0 if preferences['display_mode'] == "Detailed" else 1)
        
        # Save preferences
        save_user_sentiment_preferences(preferences)
    
    # Main content area
    # Create tabs for different views including the new Sentiment Waves tab
    tab1, tab2, tab3, tab4 = st.tabs(["Market Overview", "Stock Sentiment", "News Sentiment", "Sentiment Waves"])
    
    with tab1:  # Market Overview tab
        st.subheader("Market Sentiment Overview")
        
        # Time period selection
        period_options = {
            "1 Week": "1wk", 
            "1 Month": "1mo", 
            "3 Months": "3mo", 
            "6 Months": "6mo", 
            "1 Year": "1y"
        }
        selected_period = st.selectbox("Time Period", list(period_options.keys()), index=1)
        period = period_options[selected_period]
        
        # Get market sentiment data
        market_data = get_market_sentiment_data(period=period)
        
        if market_data is not None:
            # Calculate current market sentiment
            current_sentiment = market_data['Sentiment'].iloc[-1]
            sentiment_label = market_data['Sentiment_Label'].iloc[-1]
            
            # Display sentiment gauge
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # Show current market sentiment gauge
                st.plotly_chart(plot_sentiment_gauge(current_sentiment, "Current Market Sentiment"), use_container_width=True)
                
                # Add textual description
                if sentiment_label == "POSITIVE":
                    st.success("Market sentiment is currently positive")
                elif sentiment_label == "NEGATIVE":
                    st.error("Market sentiment is currently negative")
                else:
                    st.info("Market sentiment is currently neutral")
            
            with col2:
                # Show sentiment trend over time
                st.plotly_chart(plot_sentiment_timeline(market_data), use_container_width=True)
            
            # Display market stats
            recent_return = market_data['Return'].iloc[-1] * 100
            avg_sentiment = market_data['Sentiment'].mean()
            volatility = market_data['Return'].std() * 100 * (252 ** 0.5)  # Annualized volatility
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Recent Return", f"{recent_return:.2f}%")
            with col2:
                st.metric("Average Sentiment", f"{avg_sentiment*100:.1f}%")
            with col3:
                st.metric("Market Volatility", f"{volatility:.2f}%")
            
            # Show sentiment distribution
            sentiment_counts = market_data['Sentiment_Label'].value_counts().reset_index()
            sentiment_counts.columns = ['Sentiment', 'Count']
            
            # Create horizontal bar chart for sentiment distribution
            fig = px.bar(
                sentiment_counts,
                x='Count',
                y='Sentiment',
                color='Sentiment',
                orientation='h',
                color_discrete_map={
                    'POSITIVE': 'green',
                    'NEUTRAL': 'gold',
                    'NEGATIVE': 'red'
                },
                title="Sentiment Distribution"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.error("Unable to fetch market sentiment data")
    
    with tab2:  # Stock Sentiment tab
        st.subheader("Individual Stock Sentiment")
        
        # Get tracked tickers or use default major stocks
        tickers = preferences['tracked_tickers']
        if not tickers:
            tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
            st.info("Using default tickers. Add your own tickers in the sidebar.")
        
        # Try to get ticker data, with robust error handling
        sentiment_data = pd.DataFrame()  # Initialize with empty DataFrame
        
        try:
            # Get sentiment data for each ticker
            sentiment_data = get_sentiment_for_multiple_tickers(tickers)
        except Exception as e:
            st.error(f"Error fetching sentiment data: {str(e)}")
            
        # Check if we have valid data
        if not len(sentiment_data) == 0 and len(sentiment_data) > 0:
            try:
                # Show sentiment heatmap (with protection against None)
                heatmap = plot_sentiment_heatmap(sentiment_data)
                if heatmap is not None:
                    st.plotly_chart(heatmap, use_container_width=True)
                else:
                    st.warning("Unable to create sentiment heatmap due to insufficient data")
            except Exception as e:
                st.error(f"Error generating heatmap: {str(e)}")
            
            try:
                # Show correlation between sentiment and price change (with protection against None)
                correlation = plot_correlation_chart(sentiment_data, sentiment_data)
                if correlation is not None:
                    st.plotly_chart(correlation, use_container_width=True)
                else:
                    st.warning("Unable to create correlation chart due to insufficient data")
            except Exception as e:
                st.error(f"Error generating correlation chart: {str(e)}")
            
            # Show detailed sentiment by ticker
            st.subheader("Detailed Stock Sentiment")
            
            # Process individual ticker data with try-except for each ticker
            for _, row in sentiment_data.iterrows():
                try:
                    ticker = row['Ticker']
                    sentiment_score = row['Sentiment_Score']
                    sentiment_label = row['Sentiment_Label']
                    price_change = row['Price_Change_Pct']
                    
                    # Create an expander for each ticker
                    with st.expander(f"{ticker} - {row['Company']} - Sentiment: {sentiment_label}"):
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            # Show sentiment gauge
                            gauge = plot_sentiment_gauge(sentiment_score, f"{ticker} Sentiment")
                            st.plotly_chart(gauge, use_container_width=True)
                        
                        with col2:
                            # Show key metrics
                            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                            with metrics_col1:
                                st.metric("Price Change", f"{price_change:.2f}%")
                            with metrics_col2:
                                st.metric("Volatility", f"{row['Volatility']:.2f}%")
                            with metrics_col3:
                                st.metric("Recent Return", f"{row['Recent_Return']:.2f}%")
                            
                            # Add explanation based on data
                            if sentiment_label == "POSITIVE":
                                st.success(f"Positive sentiment detected for {ticker}")
                                if price_change > 0:
                                    st.write("✅ Price movement confirms positive sentiment")
                                else:
                                    st.write("⚠️ Price movement contradicts positive sentiment - potential buying opportunity?")
                            
                            elif sentiment_label == "NEGATIVE":
                                st.error(f"Negative sentiment detected for {ticker}")
                                if price_change < 0:
                                    st.write("✅ Price movement confirms negative sentiment")
                                else:
                                    st.write("⚠️ Price movement contradicts negative sentiment - potential selling opportunity?")
                            
                            else:
                                st.info(f"Neutral sentiment detected for {ticker}")
                                st.write("Market appears undecided on this stock")
                                
                            if row['Volatility'] > 30:
                                st.warning(f"High volatility detected ({row['Volatility']:.2f}%) - use caution")
                        
                        # Show recent news for this ticker
                        st.write("#### Recent News & Market Events")
                        try:
                            news_items = analyze_news_sentiment(ticker, limit=5)
                            
                            if news_items:
                                # Sort news with high-impact first
                                news_items = sorted(news_items, key=lambda x: (0 if x.get('impact', {}).get('has_high_impact', False) else 1,
                                                                             x.get('published', ''), 
                                                                             abs(x.get('sentiment', {}).get('score', 0.5) - 0.5)), 
                                                   reverse=True)
                                
                                # Count high-impact news
                                high_impact_count = sum(1 for news in news_items if news.get('impact', {}).get('has_high_impact', False))
                                if high_impact_count > 0:
                                    st.info(f"📢 {high_impact_count} high-impact news items found that may affect stock performance")
                                
                                # Count earnings announcements
                                earnings_count = sum(1 for news in news_items 
                                                 if 'Earnings' in news.get('impact', {}).get('categories', []))
                                if earnings_count > 0:
                                    st.warning(f"📊 {earnings_count} earnings-related news items detected")
                                
                                for news in news_items:
                                    try:
                                        sentiment = news['sentiment']
                                        label = sentiment['label']
                                        score = sentiment['score']
                                        
                                        # Create news card with sentiment color
                                        if label == "POSITIVE":
                                            card_color = "#d4edda"  # Light green
                                            emoji = "🔼"
                                        elif label == "NEGATIVE":
                                            card_color = "#f8d7da"  # Light red
                                            emoji = "🔽"
                                        else:
                                            card_color = "#fff3cd"  # Light yellow
                                            emoji = "◀▶"
                                        
                                        # Get news impact categories
                                        impact = news.get('impact', {'categories': ['General News'], 'has_high_impact': False})
                                        categories = impact.get('categories', ['General News'])
                                        is_high_impact = impact.get('has_high_impact', False)
                                        
                                        # Create a news card with full Streamlit components and visual enhancements
                                        with st.container():
                                            # Create a highlighted card for high-impact news
                                            if is_high_impact:
                                                # Draw an attention-grabbing container for high-impact news
                                                st.markdown("---")
                                                
                                                # Use a custom HTML container with red border for high impact news
                                                st.markdown(f"""
                                                <div style="border-left: 5px solid #ff5722; border-radius: 5px; 
                                                         background-color: #fff8f6; padding: 10px; margin: 10px 0px;
                                                         border-top: 1px solid #ff5722; border-right: 1px solid #ff5722; 
                                                         border-bottom: 1px solid #ff5722;">
                                                    <div style="display: flex; justify-content: space-between; align-items: center;">
                                                        <span style="font-weight: bold; color: #ff5722; font-size: 1.1em;">
                                                            {emoji} HIGH IMPACT MARKET NEWS
                                                        </span>
                                                        <span style="background-color: #ff5722; color: white; padding: 2px 8px; 
                                                              border-radius: 10px; font-size: 0.8em;">
                                                            IMPORTANT
                                                        </span>
                                                    </div>
                                                </div>
                                                """, unsafe_allow_html=True)
                                            else:
                                                st.markdown("---")
                                            
                                            # Headline with improved styling
                                            st.markdown(f"**{news['title']}**")
                                            
                                            # Publisher info with better formatting
                                            publish_date = news['published']
                                            # Check if the news is recent (last 24 hours) and add a "New" tag
                                            is_recent = "NEW" if "hour" in publish_date.lower() or "minute" in publish_date.lower() else ""
                                            if is_recent:
                                                st.caption(f"{news['publisher']} • {publish_date} • <span style='color:red;font-weight:bold;'>{is_recent}</span>", unsafe_allow_html=True)
                                            else:
                                                st.caption(f"{news['publisher']} • {publish_date}")
                                            
                                            # Categories as a horizontal collection of colored labels with improved styling
                                            st.write("**Categories:**")
                                            cat_cols = st.columns(len(categories))
                                            for i, cat in enumerate(categories):
                                                with cat_cols[i]:
                                                    if cat == "Earnings":
                                                        st.error(f"📊 {cat}")
                                                    elif cat == "Legal/Regulatory":
                                                        st.info(f"⚖️ {cat}")
                                                    elif cat == "M&A Activity":
                                                        st.warning(f"🤝 {cat}")
                                                    elif cat == "Stock Action":
                                                        st.success(f"📈 {cat}")
                                                    else:
                                                        st.write(f"📰 {cat}")
                                            
                                            # Sentiment display with more visual cues
                                            if label == "POSITIVE":
                                                st.success(f"⬆️ Sentiment: {label} ({score*100:.1f}%)")
                                            elif label == "NEGATIVE":
                                                st.error(f"⬇️ Sentiment: {label} ({score*100:.1f}%)")
                                            else:
                                                st.info(f"↔️ Sentiment: {label} ({score*100:.1f}%)")
                                                
                                            # Explanation
                                            st.caption(sentiment.get('explanation', 'No explanation available'))
                                            
                                            # Action buttons row
                                            action_cols = st.columns([3, 1])
                                            with action_cols[0]:
                                                # Link to article with improved styling
                                                st.markdown(f"[Read Full Article]({news['link']})")
                                            
                                            # Add potential price impact indicator for high-impact news
                                            if is_high_impact:
                                                with action_cols[1]:
                                                    if label == "POSITIVE":
                                                        impact_icon = "📈"
                                                        impact_text = "Bullish"
                                                    elif label == "NEGATIVE":
                                                        impact_icon = "📉" 
                                                        impact_text = "Bearish"
                                                    else:
                                                        impact_icon = "📊"
                                                        impact_text = "Neutral"
                                                    st.markdown(f"<span style='color:gray;font-size:0.9em;'>{impact_icon} {impact_text}</span>", unsafe_allow_html=True)
                                    except Exception as e:
                                        st.warning(f"Error displaying news item: {str(e)}")
                            else:
                                st.write("No recent news found")
                        except Exception as e:
                            st.warning(f"Error fetching news for {ticker}: {str(e)}")
                except Exception as e:
                    st.error(f"Error displaying data for a ticker: {str(e)}")
                    continue
        else:
            st.error("No sentiment data available. Please try different tickers or try again later.")
    
    with tab3:  # News Sentiment tab
        st.subheader("News Sentiment Analysis")
        
        # Enhanced earnings calendar and market events section
        with st.expander("📅 Market Events & Earnings Calendar", expanded=True):
            # Create tabs for different types of market events
            event_tabs = st.tabs(["📊 Earnings Announcements", "🏛️ Economic Events", "📢 Company Events"])
            
            with event_tabs[0]:  # Earnings tab
                st.subheader("Upcoming Earnings Announcements")
                st.info("Important earnings announcements to watch in the coming days")
                
                # Add search and filter options
                col1, col2 = st.columns([1, 1])
                with col1:
                    earnings_search = st.text_input("🔍 Filter by company/ticker:", "")
                
                with col2:
                    importance_filter = st.multiselect("Filter by importance:", 
                                                     ["All", "High Impact", "Medium Impact", "Low Impact"], 
                                                     default=["All"])
                
                # Create a more visually rich earnings table
                earnings_data = {
                    'Date': ['2025-04-15', '2025-04-16', '2025-04-17', '2025-04-18', '2025-04-21'],
                    'Ticker': ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META'],
                    'Company': ['Apple Inc.', 'Microsoft Corp.', 'Alphabet Inc.', 'Amazon.com Inc.', 'Meta Platforms Inc.'],
                    'Expected EPS': ['$1.45', '$2.63', '$1.89', '$0.72', '$4.10'],
                    'Previous EPS': ['$1.32', '$2.45', '$1.78', '$0.65', '$3.85'],
                    'Market Cap': ['$2.8T', '$3.1T', '$1.9T', '$1.8T', '$1.2T'],
                    'Importance': ['High', 'High', 'High', 'High', 'Medium']
                }
                
                # Apply filters if any
                earnings_df = pd.DataFrame(earnings_data)
                if earnings_search:
                    search_term = earnings_search.lower()
                    earnings_df = earnings_df[
                        earnings_df['Ticker'].str.lower().str.contains(search_term) | 
                        earnings_df['Company'].str.lower().str.contains(search_term)
                    ]
                
                if importance_filter and "All" not in importance_filter:
                    importance_map = {"High Impact": "High", "Medium Impact": "Medium", "Low Impact": "Low"}
                    filter_values = [importance_map[imp] for imp in importance_filter if imp in importance_map]
                    if filter_values:
                        earnings_df = earnings_df[earnings_df['Importance'].isin(filter_values)]
                
                # Display the data with visual styling
                if not len(earnings_df) == 0:
                    # Custom render for each row
                    for i, row in earnings_df.iterrows():
                        with st.container():
                            # Set border color based on importance
                            if row['Importance'] == 'High':
                                border_color = "#dc3545"  # Red
                                importance_badge = "🔴 High Impact"
                            elif row['Importance'] == 'Medium':
                                border_color = "#fd7e14"  # Orange
                                importance_badge = "🟠 Medium Impact"
                            else:
                                border_color = "#6c757d"  # Gray
                                importance_badge = "⚪ Low Impact"
                            
                            st.markdown(f"""
                            <div style="border-left: 4px solid {border_color}; padding-left: 10px; margin-bottom: 15px;">
                                <div style="display: flex; justify-content: space-between;">
                                    <h4 style="margin:0;">{row['Ticker']} - {row['Company']}</h4>
                                    <span style="color:white; background-color:{border_color}; padding:2px 8px; border-radius:10px; font-size:0.8em;">{importance_badge}</span>
                                </div>
                                <p style="margin:5px 0;"><strong>Date:</strong> {row['Date']} <span style="color:gray; font-size:0.9em;">(Estimated)</span></p>
                                <div style="display: flex; justify-content: space-between; margin-top:5px;">
                                    <div>
                                        <strong>Expected EPS:</strong> {row['Expected EPS']}
                                    </div>
                                    <div>
                                        <strong>Previous EPS:</strong> {row['Previous EPS']}
                                    </div>
                                    <div>
                                        <strong>Market Cap:</strong> {row['Market Cap']}
                                    </div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                
                        # Add action buttons for each company
                        cols = st.columns([1, 1, 1])
                        with cols[0]:
                            st.button(f"📈 View {row['Ticker']} Chart", key=f"chart_{row['Ticker']}_{i}")
                        with cols[1]:
                            st.button(f"📰 Recent News", key=f"news_{row['Ticker']}_{i}")
                        with cols[2]:
                            st.button(f"📊 Set Alert", key=f"alert_{row['Ticker']}_{i}")
                    
                    st.markdown("---")
                    st.caption("Data is for demonstration purposes. Connect to a financial data API for real-time earnings data.")
                else:
                    st.warning("No earnings announcements match your filters.")
            
            with event_tabs[1]:  # Economic Events tab
                st.subheader("Upcoming Economic Events")
                
                # Create a calendar of economic events
                econ_events = {
                    'Date': ['2025-04-10', '2025-04-12', '2025-04-15', '2025-04-18', '2025-04-25'],
                    'Time': ['8:30 AM ET', '2:00 PM ET', '8:30 AM ET', '10:00 AM ET', '8:30 AM ET'],
                    'Event': ['CPI Data Release', 'FOMC Minutes', 'Retail Sales', 'Housing Starts', 'GDP Preliminary'],
                    'Importance': ['High', 'High', 'Medium', 'Medium', 'High'],
                    'Previous': ['+0.4%', 'No Change', '+0.6%', '1.46M', '+3.2%'],
                    'Forecast': ['+0.3%', 'No Change', '+0.3%', '1.44M', '+2.8%']
                }
                
                econ_df = pd.DataFrame(econ_events)
                
                # Display economic events
                for i, row in econ_df.iterrows():
                    with st.container():
                        # Importance coloring
                        if row['Importance'] == 'High':
                            badge_color = "#dc3545"  # Red
                        elif row['Importance'] == 'Medium':
                            badge_color = "#fd7e14"  # Orange
                        else:
                            badge_color = "#6c757d"  # Gray
                            
                        st.markdown(f"""
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom:10px;">
                            <div>
                                <strong>{row['Date']} - {row['Time']}</strong>
                                <h4 style="margin:0;">{row['Event']}</h4>
                            </div>
                            <span style="color:white; background-color:{badge_color}; padding:2px 8px; border-radius:10px; font-size:0.8em;">
                                {row['Importance']} Impact
                            </span>
                        </div>
                        <div style="display: flex; margin-bottom:15px;">
                            <div style="margin-right:20px;"><strong>Previous:</strong> {row['Previous']}</div>
                            <div><strong>Forecast:</strong> {row['Forecast']}</div>
                        </div>
                        <hr>
                        """, unsafe_allow_html=True)
                
                st.caption("For demonstration purposes. Data would be updated from economic calendar services.")
            
            with event_tabs[2]:  # Company Events tab
                st.subheader("Company-Specific Events")
                
                # Create sample company events
                company_events = {
                    'Date': ['2025-04-11', '2025-04-14', '2025-04-16', '2025-04-22', '2025-04-30'],
                    'Ticker': ['TSLA', 'AAPL', 'NVDA', 'MSFT', 'AMZN'],
                    'Event Type': ['Product Launch', 'Shareholder Meeting', 'Conference', 'Product Launch', 'Investor Day'],
                    'Description': [
                        'New vehicle announcement', 
                        'Annual shareholder meeting', 
                        'AI Technology Conference keynote',
                        'New Surface device lineup', 
                        'AWS strategic announcements'
                    ]
                }
                
                events_df = pd.DataFrame(company_events)
                
                # Display events
                for i, row in events_df.iterrows():
                    # Set icon based on event type
                    if row['Event Type'] == 'Product Launch':
                        icon = "🚀"
                        color = "#0d6efd"  # Blue
                    elif row['Event Type'] == 'Shareholder Meeting':
                        icon = "👥"
                        color = "#6c757d"  # Gray
                    elif row['Event Type'] == 'Conference':
                        icon = "🎤"
                        color = "#6f42c1"  # Purple
                    elif row['Event Type'] == 'Investor Day':
                        icon = "💼"
                        color = "#198754"  # Green
                    else:
                        icon = "📆"
                        color = "#0dcaf0"  # Cyan
                        
                    st.markdown(f"""
                    <div style="border-left: 3px solid {color}; padding-left: 10px; margin-bottom: 15px;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <strong>{row['Date']}</strong> - {row['Ticker']}
                                <h4 style="margin:0;">{icon} {row['Event Type']}</h4>
                            </div>
                        </div>
                        <p>{row['Description']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Add reminder button
                    st.button(f"🔔 Set Reminder for {row['Ticker']} {row['Event Type']}", key=f"remind_{row['Ticker']}_{i}")
                    st.markdown("---")
                
                st.caption("These events can significantly impact stock prices and investor sentiment.")
        
        # Allow user to enter a ticker for news analysis
        news_ticker = st.text_input("Enter ticker for news analysis:", "SPY")
        
        if news_ticker:
            # Number of news articles to analyze
            num_articles = st.slider("Number of articles to analyze", 3, 10, 5)
            
            # Analyze news for this ticker
            news_items = analyze_news_sentiment(news_ticker, limit=num_articles)
            
            if news_items:
                # Calculate average sentiment safely
                try:
                    scores = []
                    for news in news_items:
                        if isinstance(news.get('sentiment'), dict) and 'score' in news['sentiment']:
                            scores.append(news['sentiment']['score'])
                        else:
                            scores.append(0.5)  # Neutral score for problematic items
                    
                    if scores:
                        avg_sentiment = sum(scores) / len(scores)
                    else:
                        avg_sentiment = 0.5  # Default to neutral if no valid scores
                except Exception as e:
                    st.warning(f"Error calculating average sentiment: {str(e)}")
                    avg_sentiment = 0.5  # Default to neutral on error
                
                # Show gauge with average sentiment
                st.plotly_chart(plot_sentiment_gauge(avg_sentiment, f"{news_ticker} News Sentiment"), use_container_width=True)
                
                # Add a sentiment overlay chart option
                with st.expander("📈 View News Sentiment Overlay on Price Chart"):
                    st.info("This chart overlays news sentiment on stock price data, highlighting high-impact events.")
                    
                    # Get historical price data for this ticker
                    try:
                        # Default to last 90 days of data
                        end_date = datetime.now()
                        start_date = end_date - timedelta(days=90)
                        
                        # Fetch data
                        ticker_data = yf.download(news_ticker, start=start_date, end=end_date)
                        
                        if not len(ticker_data) == 0:
                            # Create the overlay chart
                            overlay_chart = plot_sentiment_news_overlay(ticker_data, news_items, news_ticker)
                            
                            if overlay_chart:
                                st.plotly_chart(overlay_chart, use_container_width=True)
                                
                                # Add explanation of the chart
                                st.write("""
                                **Chart Explanation:**
                                - **Candlesticks** show daily price movements
                                - **Vertical lines** indicate high-impact news events
                                - **Triangle markers** below the chart indicate news items, colored by sentiment
                                - **Hover** over markers to see details about the news
                                """)
                            else:
                                st.warning("Unable to generate sentiment overlay chart.")
                        else:
                            st.warning(f"Could not fetch historical price data for {news_ticker}.")
                    except Exception as e:
                        st.error(f"Error generating sentiment overlay chart: {str(e)}")
                
                # Display individual news items
                for news in news_items:
                    try:
                        sentiment = news['sentiment']
                        if isinstance(sentiment, dict) and 'label' in sentiment and 'score' in sentiment:
                            label = sentiment['label']
                            score = sentiment['score']
                        else:
                            # Use neutral values if sentiment is not properly formatted
                            label = "NEUTRAL"
                            score = 0.5
                        
                        # Create news card with appropriate sentiment styling
                        if label == "POSITIVE":
                            card_color = "#d4edda"  # Light green
                            emoji = "🔼"
                        elif label == "NEGATIVE":
                            card_color = "#f8d7da"  # Light red
                            emoji = "🔽"
                        else:
                            card_color = "#fff3cd"  # Light yellow
                            emoji = "◀▶"
                    except Exception as e:
                        # Fallback for any errors in sentiment data
                        st.warning(f"Error processing sentiment data: {str(e)}")
                        label = "NEUTRAL"
                        score = 0.5
                        card_color = "#fff3cd"  # Light yellow
                        emoji = "◀▶"
                    
                    # Get news impact categories
                    impact = news.get('impact', {'categories': ['General News'], 'has_high_impact': False})
                    categories = impact.get('categories', ['General News'])
                    is_high_impact = impact.get('has_high_impact', False)
                    
                    # Create a news card with full Streamlit components and visual enhancements
                    with st.container():
                        # Create a highlighted card for high-impact news
                        if is_high_impact:
                            # Draw an attention-grabbing container for high-impact news
                            st.markdown("---")
                            
                            # Use a custom HTML container with red border for high impact news
                            st.markdown(f"""
                            <div style="border-left: 5px solid #ff5722; border-radius: 5px; 
                                     background-color: #fff8f6; padding: 10px; margin: 10px 0px;
                                     border-top: 1px solid #ff5722; border-right: 1px solid #ff5722; 
                                     border-bottom: 1px solid #ff5722;">
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <span style="font-weight: bold; color: #ff5722; font-size: 1.1em;">
                                        {emoji} HIGH IMPACT MARKET NEWS
                                    </span>
                                    <span style="background-color: #ff5722; color: white; padding: 2px 8px; 
                                          border-radius: 10px; font-size: 0.8em;">
                                        IMPORTANT
                                    </span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown("---")
                        
                        # Headline with improved styling
                        st.subheader(f"{news['title']}")
                        
                        # Publisher info with better formatting
                        publish_date = news['published']
                        # Check if the news is recent (last 24 hours) and add a "New" tag
                        is_recent = "NEW" if "hour" in publish_date.lower() or "minute" in publish_date.lower() else ""
                        if is_recent:
                            st.caption(f"{news['publisher']} • {publish_date} • <span style='color:red;font-weight:bold;'>{is_recent}</span>", unsafe_allow_html=True)
                        else:
                            st.caption(f"{news['publisher']} • {publish_date}")
                        
                        # Categories as a horizontal collection of colored labels with improved styling
                        st.write("**Categories:**")
                        cat_cols = st.columns(len(categories))
                        for i, cat in enumerate(categories):
                            with cat_cols[i]:
                                if cat == "Earnings":
                                    st.error(f"📊 {cat}")
                                elif cat == "Legal/Regulatory":
                                    st.info(f"⚖️ {cat}")
                                elif cat == "M&A Activity":
                                    st.warning(f"🤝 {cat}")
                                elif cat == "Stock Action":
                                    st.success(f"📈 {cat}")
                                else:
                                    st.write(f"📰 {cat}")
                        
                        # News summary with improved visibility for high-impact news
                        if 'summary' in news and news['summary']:
                            if is_high_impact:
                                st.markdown(f"<div style='background-color:#fff8f6;padding:10px;border-radius:5px;'>{news['summary']}</div>", unsafe_allow_html=True)
                            else:
                                st.write(news['summary'])
                        
                        # Sentiment display with more visual cues
                        sentiment_cols = st.columns([1, 3])
                        with sentiment_cols[0]:
                            # Enhanced sentiment indicators with emojis
                            if label == "POSITIVE":
                                st.success(f"⬆️ {label}")
                            elif label == "NEGATIVE":
                                st.error(f"⬇️ {label}")
                            else:
                                st.info(f"↔️ {label}")
                                
                        with sentiment_cols[1]:
                            # Score and explanation with improved formatting
                            sentiment_score = score*100
                            # Visual representation of sentiment strength
                            st.write(f"Score: {sentiment_score:.1f}%")
                            st.caption(sentiment.get('explanation', 'No explanation available'))
                        
                        # Action buttons row
                        action_cols = st.columns([3, 1])
                        with action_cols[0]:
                            # Link to article with improved styling
                            st.markdown(f"[Read Full Article]({news['link']})")
                        
                        # Add potential price impact indicator for high-impact news
                        if is_high_impact:
                            with action_cols[1]:
                                if label == "POSITIVE":
                                    impact_icon = "📈"
                                    impact_text = "Bullish"
                                elif label == "NEGATIVE":
                                    impact_icon = "📉" 
                                    impact_text = "Bearish"
                                else:
                                    impact_icon = "📊"
                                    impact_text = "Neutral"
                                st.markdown(f"<span style='color:gray;font-size:0.9em;'>{impact_icon} {impact_text}</span>", unsafe_allow_html=True)
                
                # Add option to track this ticker
                if news_ticker not in preferences['tracked_tickers'] and news_ticker != "SPY":
                    if st.button(f"Track {news_ticker} Sentiment"):
                        preferences['tracked_tickers'].append(news_ticker)
                        save_user_sentiment_preferences(preferences)
                        st.success(f"Added {news_ticker} to tracked tickers")
                        st.rerun()
            
            else:
                st.warning(f"No news found for {news_ticker}")
        
        # Option to analyze custom text for sentiment
        st.subheader("Analyze Custom Text")
        custom_text = st.text_area("Enter text to analyze for financial sentiment:", height=150)
        
        if custom_text and st.button("Analyze Sentiment"):
            with st.spinner("Analyzing sentiment..."):
                # We already declared the global variable at the function level
                # Using OPENAI_QUOTA_EXCEEDED variable from outer scope
                
                # Check if we have OpenAI key AND haven't exceeded quota
                if HAS_OPENAI and not OPENAI_QUOTA_EXCEEDED:
                    try:
                        sentiment = analyze_sentiment_with_openai(custom_text)
                    except Exception as e:
                        # If OpenAI fails, check if quota error and set the flag
                        if "quota" in str(e).lower() or "429" in str(e):
                            OPENAI_QUOTA_EXCEEDED = True
                        # No warnings shown for OpenAI API failures
                        sentiment = analyze_sentiment_basic(custom_text)
                else:
                    sentiment = analyze_sentiment_basic(custom_text)
                
                # Show results
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.plotly_chart(plot_sentiment_gauge(sentiment.get('score', 0.5), "Text Sentiment"), use_container_width=True)
                
                with col2:
                    st.markdown(f"### Sentiment: {sentiment.get('label', 'NEUTRAL')}")
                    st.markdown(f"**Score:** {sentiment.get('score', 0.5)*100:.1f}%")
                    st.markdown(f"**Confidence:** {sentiment.get('confidence', 0.0)*100:.1f}%")
                    st.markdown(f"**Explanation:** {sentiment.get('explanation', 'No explanation available')}")
                    
    # Tab 4: Sentiment Waves Visualization
    with tab4:
        st.subheader("Sentiment Wave Visualization")
        
        # Ticker selection
        ticker = st.text_input("Enter Stock Symbol for Sentiment Waves", "AAPL").upper()
        
        if ticker:
            # Fetch data
            # Fetch recent data for the ticker
            end_date = datetime.now()
            start_date = end_date - timedelta(days=180)  # 6 months for a good amount of data
            
            try:
                # Show loading spinner
                with st.spinner(f"Fetching data for {ticker}..."):
                    stock_data = yf.download(ticker, start=start_date, end=end_date)
                
                if len(stock_data) > 0:
                    # Number of news articles to analyze
                    num_articles = st.slider("Number of news articles to analyze", 5, 20, 10, key="sentiment_wave_num_articles")
                    
                    with st.spinner("Fetching and analyzing news sentiment..."):
                        news_items = analyze_news_sentiment(ticker, limit=num_articles)
                    
                    # Create a simple visualization with news sentiment indicators
                    create_simple_sentiment_visualization(ticker, stock_data, news_items)
                else:
                    st.error(f"No price data available for {ticker}")
            except Exception as e:
                st.error(f"Error fetching data: {str(e)}")
        else:
            st.info("Enter a stock symbol to visualize sentiment waves.")
