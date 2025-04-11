import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json
import os
import base64
from datetime import datetime
import uuid
from io import BytesIO

# Import our new community feed module
from community_feed import display_community_feed

# Initialize social feature state
def initialize_social_features():
    """Initialize social features state variables"""
    if 'user_watchlists' not in st.session_state:
        st.session_state.user_watchlists = {}
    
    if 'shared_watchlists' not in st.session_state:
        st.session_state.shared_watchlists = []
    
    if 'user_strategies' not in st.session_state:
        st.session_state.user_strategies = []
    
    if 'shared_strategies' not in st.session_state:
        st.session_state.shared_strategies = []
    
    if 'stock_discussions' not in st.session_state:
        st.session_state.stock_discussions = {}
    
    if 'trending_stocks' not in st.session_state:
        st.session_state.trending_stocks = {}
    
    if 'user_achievements' not in st.session_state:
        st.session_state.user_achievements = []
    
    if 'leaderboard' not in st.session_state:
        st.session_state.leaderboard = []
    
    if 'user_profile' not in st.session_state:
        st.session_state.user_profile = {
            'username': f'Investor_{uuid.uuid4().hex[:6]}',
            'avatar': None,
            'about': '',
            'join_date': datetime.now().strftime('%Y-%m-%d'),
            'shared_items': 0,
            'reputation': 0,
            'achievement_points': 0
        }
    
    # Stock ratings and recommendations
    if 'stock_ratings' not in st.session_state:
        st.session_state.stock_ratings = {}
    
    if 'user_recommendations' not in st.session_state:
        st.session_state.user_recommendations = []
    
    if 'recommendation_categories' not in st.session_state:
        st.session_state.recommendation_categories = [
            "Strong Buy", "Buy", "Hold", "Sell", "Strong Sell"
        ]
    
    if 'investment_horizons' not in st.session_state:
        st.session_state.investment_horizons = [
            "Day Trade", "Swing Trade", "Short-Term (< 3 months)", 
            "Medium-Term (3-12 months)", "Long-Term (> 1 year)"
        ]
    
    if 'investor_styles' not in st.session_state:
        st.session_state.investor_styles = [
            "Value Investor", "Growth Investor", "Income Investor", 
            "Momentum Trader", "Technical Trader", "Passive Investor"
        ]
    
    # Achievement system
    if 'available_achievements' not in st.session_state:
        st.session_state.available_achievements = [
            {
                'id': 'first_rating',
                'name': 'Stock Analyst',
                'description': 'Rate your first stock',
                'points': 10,
                'icon': '⭐'
            },
            {
                'id': 'five_ratings',
                'name': 'Senior Analyst',
                'description': 'Rate 5 different stocks',
                'points': 25,
                'icon': '⭐⭐'
            },
            {
                'id': 'first_recommendation',
                'name': 'Market Advisor',
                'description': 'Create your first stock recommendation',
                'points': 15,
                'icon': '📊'
            },
            {
                'id': 'three_recommendations',
                'name': 'Investment Guru',
                'description': 'Create 3 detailed stock recommendations',
                'points': 30,
                'icon': '📈'
            },
            {
                'id': 'top_rated',
                'name': 'Community Leader',
                'description': 'Have one of your recommendations receive 5+ upvotes',
                'points': 50,
                'icon': '🏆'
            }
        ]

# Export watchlist to a shareable format
def export_watchlist(watchlist_name, stocks, notes=None):
    """
    Export a watchlist to a shareable format
    
    Args:
        watchlist_name: Name of the watchlist
        stocks: List of stocks in the watchlist
        notes: Optional notes about the watchlist
        
    Returns:
        Encoded watchlist data
    """
    watchlist_data = {
        'name': watchlist_name,
        'stocks': stocks,
        'notes': notes or '',
        'created_by': st.session_state.user_profile['username'],
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    watchlist_json = json.dumps(watchlist_data)
    return base64.b64encode(watchlist_json.encode()).decode()

# Import a shared watchlist
def import_watchlist(encoded_data):
    """
    Import a shared watchlist
    
    Args:
        encoded_data: Base64 encoded watchlist data
        
    Returns:
        Decoded watchlist data or None if invalid
    """
    try:
        decoded_json = base64.b64decode(encoded_data).decode()
        watchlist_data = json.loads(decoded_json)
        return watchlist_data
    except:
        return None

# Share a strategy
def share_strategy(strategy_name, strategy_type, parameters, performance=None, notes=None):
    """
    Share a trading strategy
    
    Args:
        strategy_name: Name of the strategy
        strategy_type: Type of strategy (RSI, MACD, etc.)
        parameters: Dictionary of strategy parameters
        performance: Optional performance metrics
        notes: Optional notes about the strategy
        
    Returns:
        Strategy ID
    """
    strategy_id = str(uuid.uuid4())
    
    strategy_data = {
        'id': strategy_id,
        'name': strategy_name,
        'type': strategy_type,
        'parameters': parameters,
        'performance': performance or {},
        'notes': notes or '',
        'created_by': st.session_state.user_profile['username'],
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'upvotes': 0,
        'downvotes': 0,
        'comments': []
    }
    
    st.session_state.shared_strategies.append(strategy_data)
    st.session_state.user_strategies.append(strategy_data)
    st.session_state.user_profile['shared_items'] += 1
    
    return strategy_id

# Add a comment to a stock discussion
def add_stock_discussion(ticker, comment, sentiment=None, image=None):
    """
    Add a comment to a stock discussion
    
    Args:
        ticker: Stock symbol
        comment: User comment
        sentiment: Optional sentiment (bullish, bearish, neutral)
        image: Optional base64 encoded image to attach
        
    Returns:
        Comment ID
    """
    comment_id = str(uuid.uuid4())
    
    if ticker not in st.session_state.stock_discussions:
        st.session_state.stock_discussions[ticker] = []
    
    comment_data = {
        'id': comment_id,
        'ticker': ticker,
        'comment': comment,
        'sentiment': sentiment or 'neutral',
        'created_by': st.session_state.user_profile['username'],
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'upvotes': 0,
        'downvotes': 0,
        'replies': [],
        'image': image,  # Store attached image if provided
        'has_image': image is not None
    }
    
    st.session_state.stock_discussions[ticker].append(comment_data)
    
    # Update trending stocks
    if ticker in st.session_state.trending_stocks:
        st.session_state.trending_stocks[ticker] += 1
    else:
        st.session_state.trending_stocks[ticker] = 1
    
    # Award achievement points
    st.session_state.user_profile['reputation'] += 2
    
    return comment_id

# Get screenshot of current analysis
def get_chart_screenshot(fig):
    """
    Generate a screenshot of a plotly chart
    
    Args:
        fig: Plotly figure
        
    Returns:
        Base64 encoded image
    """
    img_bytes = fig.to_image(format="png")
    img_base64 = base64.b64encode(img_bytes).decode()
    return img_base64

# Share chart on social media
def get_social_share_links(title, img_base64=None, text=None):
    """
    Generate social media share links
    
    Args:
        title: Title of the share
        img_base64: Optional base64 encoded image
        text: Optional text content
        
    Returns:
        Dictionary of social media links
    """
    # Note: In a real implementation, this would utilize proper APIs
    # For demo purposes, we just generate some example links
    
    # Placeholder URLs (would need proper API implementation)
    share_text = f"{title} - Stock Analysis Insight"
    if text:
        share_text += f": {text}"
    
    links = {
        'twitter': f"https://twitter.com/intent/tweet?text={share_text}",
        'linkedin': f"https://www.linkedin.com/sharing/share-offsite/?url=https://example.com&title={share_text}",
        'email': f"mailto:?subject={title}&body={share_text}"
    }
    
    return links

# Get trending stocks
def get_trending_stocks(limit=5):
    """
    Get the most discussed stocks
    
    Args:
        limit: Maximum number of stocks to return
        
    Returns:
        List of trending stocks with counts
    """
    trending = sorted(st.session_state.trending_stocks.items(), key=lambda x: x[1], reverse=True)
    return trending[:limit]

# Update leaderboard
# Rate a stock
def add_stock_rating(ticker, rating, comment=""):
    """
    Add a rating for a stock
    
    Args:
        ticker: Stock symbol
        rating: Rating value (1-5)
        comment: Optional comment
        
    Returns:
        Rating ID
    """
    rating_id = str(uuid.uuid4())
    
    if ticker not in st.session_state.stock_ratings:
        st.session_state.stock_ratings[ticker] = []
    
    rating_data = {
        'id': rating_id,
        'ticker': ticker,
        'rating': rating,
        'comment': comment,
        'created_by': st.session_state.user_profile['username'],
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'upvotes': 0
    }
    
    st.session_state.stock_ratings[ticker].append(rating_data)
    
    # Update trending stocks
    if ticker in st.session_state.trending_stocks:
        st.session_state.trending_stocks[ticker] += 1
    else:
        st.session_state.trending_stocks[ticker] = 1
    
    # Check for achievements
    check_rating_achievements()
    
    return rating_id

# Create a stock recommendation
def create_recommendation(ticker, category, price_target, horizon, investment_style, thesis, risk_factors=None):
    """
    Create a detailed stock recommendation
    
    Args:
        ticker: Stock symbol
        category: Recommendation category (Buy, Sell, etc.)
        price_target: Target price
        horizon: Investment horizon
        investment_style: Investment style
        thesis: Investment thesis
        risk_factors: Optional risk factors
        
    Returns:
        Recommendation ID
    """
    recommendation_id = str(uuid.uuid4())
    
    recommendation_data = {
        'id': recommendation_id,
        'ticker': ticker,
        'category': category,
        'price_target': price_target,
        'horizon': horizon,
        'investment_style': investment_style,
        'thesis': thesis,
        'risk_factors': risk_factors or '',
        'created_by': st.session_state.user_profile['username'],
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'upvotes': 0,
        'downvotes': 0,
        'comments': []
    }
    
    st.session_state.user_recommendations.append(recommendation_data)
    
    # Update trending stocks
    if ticker in st.session_state.trending_stocks:
        st.session_state.trending_stocks[ticker] += 1
    else:
        st.session_state.trending_stocks[ticker] = 1
    
    # Check for achievements
    check_recommendation_achievements()
    
    return recommendation_id

# Upvote a recommendation
def upvote_recommendation(recommendation_id):
    """
    Upvote a recommendation
    
    Args:
        recommendation_id: ID of the recommendation to upvote
        
    Returns:
        True if successful, False otherwise
    """
    for rec in st.session_state.user_recommendations:
        if rec['id'] == recommendation_id:
            rec['upvotes'] += 1
            
            # Check for top-rated achievement
            if rec['upvotes'] >= 5:
                award_achievement('top_rated')
            
            return True
    
    return False

# Check for rating achievements
def check_rating_achievements():
    """Check if user has earned any rating achievements"""
    # Count unique tickers the user has rated
    rated_tickers = set()
    
    for ticker, ratings in st.session_state.stock_ratings.items():
        for rating in ratings:
            if rating['created_by'] == st.session_state.user_profile['username']:
                rated_tickers.add(ticker)
    
    # Check for first rating achievement
    if rated_tickers and 'first_rating' not in st.session_state.user_achievements:
        award_achievement('first_rating')
    
    # Check for 5 ratings achievement
    if len(rated_tickers) >= 5 and 'five_ratings' not in st.session_state.user_achievements:
        award_achievement('five_ratings')

# Check for recommendation achievements
def check_recommendation_achievements():
    """Check if user has earned any recommendation achievements"""
    # Count recommendations by the user
    user_recs = [rec for rec in st.session_state.user_recommendations 
                 if rec['created_by'] == st.session_state.user_profile['username']]
    
    # Check for first recommendation achievement
    if user_recs and 'first_recommendation' not in st.session_state.user_achievements:
        award_achievement('first_recommendation')
    
    # Check for 3 recommendations achievement
    if len(user_recs) >= 3 and 'three_recommendations' not in st.session_state.user_achievements:
        award_achievement('three_recommendations')

# Award an achievement
def award_achievement(achievement_id):
    """
    Award an achievement to the user
    
    Args:
        achievement_id: ID of the achievement to award
        
    Returns:
        True if awarded, False otherwise
    """
    # Don't award if already earned
    if achievement_id in st.session_state.user_achievements:
        return False
    
    # Find achievement details
    achievement = None
    for ach in st.session_state.available_achievements:
        if ach['id'] == achievement_id:
            achievement = ach
            break
    
    if not achievement:
        return False
    
    # Award achievement
    st.session_state.user_achievements.append(achievement_id)
    st.session_state.user_profile['achievement_points'] += achievement['points']
    st.session_state.user_profile['reputation'] += achievement['points']
    
    return True

# Calculate consensus rating for a stock
def get_stock_consensus(ticker):
    """
    Calculate the consensus rating for a stock
    
    Args:
        ticker: Stock symbol
        
    Returns:
        Dictionary with consensus information
    """
    if ticker not in st.session_state.stock_ratings or not st.session_state.stock_ratings[ticker]:
        return {
            'average_rating': None,
            'num_ratings': 0,
            'sentiment': None
        }
    
    ratings = st.session_state.stock_ratings[ticker]
    avg_rating = sum(r['rating'] for r in ratings) / len(ratings)
    
    # Determine sentiment based on average rating
    sentiment = None
    if avg_rating >= 4.0:
        sentiment = "Bullish"
    elif avg_rating >= 3.0:
        sentiment = "Neutral"
    else:
        sentiment = "Bearish"
    
    return {
        'average_rating': avg_rating,
        'num_ratings': len(ratings),
        'sentiment': sentiment
    }

def update_leaderboard():
    """Update the social trading leaderboard"""
    # In a real implementation, this would calculate actual portfolio performance
    leaderboard = []
    
    # Current user
    current_user = {
        'username': st.session_state.user_profile['username'],
        'strategies': len(st.session_state.user_strategies),
        'recommendations': len([rec for rec in st.session_state.user_recommendations 
                              if rec['created_by'] == st.session_state.user_profile['username']]),
        'shared_items': st.session_state.user_profile['shared_items'],
        'reputation': st.session_state.user_profile['reputation'],
        'achievements': len(st.session_state.user_achievements),
        'achievement_points': st.session_state.user_profile['achievement_points'],
        'rank': 1
    }
    leaderboard.append(current_user)
    
    # Sort leaderboard by reputation
    st.session_state.leaderboard = sorted(leaderboard, key=lambda x: x['reputation'], reverse=True)
    
    # Update ranks
    for i, entry in enumerate(st.session_state.leaderboard):
        entry['rank'] = i + 1

# Display user profile
def display_user_profile():
    """Display the user profile section"""
    st.subheader("Your Profile")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image("https://www.gravatar.com/avatar/00000000000000000000000000000000?d=mp&f=y", width=150)
        st.button("Change Avatar", disabled=True)
    
    with col2:
        username = st.text_input("Username", value=st.session_state.user_profile['username'])
        if username != st.session_state.user_profile['username']:
            st.session_state.user_profile['username'] = username
            st.success("Username updated!")
        
        about = st.text_area("About Me", value=st.session_state.user_profile['about'])
        if about != st.session_state.user_profile['about']:
            st.session_state.user_profile['about'] = about
            st.success("Profile updated!")
    
    st.divider()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Joined", st.session_state.user_profile['join_date'])
    with col2:
        st.metric("Shared Items", st.session_state.user_profile['shared_items'])
    with col3:
        st.metric("Reputation", st.session_state.user_profile['reputation'])

# Display shareable watchlists
def display_watchlist_sharing():
    """Display the watchlist sharing section with enhanced browsing"""
    st.subheader("Watchlist Community")
    
    # Create tabs for different views
    watchlist_tabs = st.tabs(["Share Watchlist", "Browse Shared Watchlists", "Import Watchlist", "Popular Watchlists"])
    
    with watchlist_tabs[0]:
        st.subheader("Share Your Watchlists")
        
        # Get user's watchlists
        watchlist_names = list(st.session_state.user_watchlists.keys())
        
        if not watchlist_names:
            st.info("You haven't created any watchlists yet. Create a watchlist in the Watchlists section.")
        else:
            selected_watchlist = st.selectbox("Select Watchlist to Share", watchlist_names, key="share_watchlist_select")
            
            if selected_watchlist:
                watchlist = st.session_state.user_watchlists[selected_watchlist]
                
                # Display preview of the watchlist
                with st.expander("Watchlist Preview", expanded=True):
                    st.write(f"**Stocks in this watchlist ({len(watchlist)}):**")
                    watchlist_preview = ", ".join(watchlist)
                    st.markdown(f"```{watchlist_preview}```")
                
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    notes = st.text_area("Add notes to your shared watchlist (optional)", 
                                         placeholder="Describe your watchlist strategy or theme...")
                    
                    # Add tags for better discoverability
                    tags = st.text_input("Add tags (comma separated)", 
                                        placeholder="tech, dividend, growth, etc.")
                
                with col2:
                    # Add visibility options
                    visibility = st.radio(
                        "Visibility",
                        ["Public", "Community Only"],
                        index=0
                    )
                    
                    st.write("")
                    if st.button("Generate Shareable Link", use_container_width=True):
                        encoded_data = export_watchlist(selected_watchlist, watchlist, notes)
                        st.code(f"https://example.com/shared-watchlist?data={encoded_data[:20]}...", language=None)
                        
                        # Add to shared watchlists
                        watchlist_data = {
                            'id': str(uuid.uuid4()),
                            'name': selected_watchlist,
                            'stocks': watchlist,
                            'notes': notes or '',
                            'tags': [tag.strip() for tag in tags.split(',')] if tags else [],
                            'visibility': visibility,
                            'created_by': st.session_state.user_profile['username'],
                            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'upvotes': 0,
                            'views': 0
                        }
                        st.session_state.shared_watchlists.append(watchlist_data)
                        st.session_state.user_profile['shared_items'] += 1
                        
                        st.success("Watchlist shared successfully! Copy the link above to share with others.")
                        
                        # Social share options
                        st.markdown("### Share on social media")
                        social_col1, social_col2, social_col3 = st.columns(3)
                        
                        with social_col1:
                            share_text = f"Check out my '{selected_watchlist}' stock watchlist! #investing #stocks"
                            st.markdown(f"<a href='https://twitter.com/intent/tweet?text={share_text}' target='_blank' style='text-decoration:none;'><div style='background-color:#1DA1F2;color:white;padding:10px;border-radius:5px;text-align:center;'>Twitter</div></a>", unsafe_allow_html=True)
                        
                        with social_col2:
                            st.markdown(f"<a href='https://www.linkedin.com/sharing/share-offsite/?url=https://example.com' target='_blank' style='text-decoration:none;'><div style='background-color:#0077B5;color:white;padding:10px;border-radius:5px;text-align:center;'>LinkedIn</div></a>", unsafe_allow_html=True)
                        
                        with social_col3:
                            st.markdown(f"<a href='mailto:?subject=Check out my stock watchlist&body={share_text}' target='_blank' style='text-decoration:none;'><div style='background-color:#5b5b5b;color:white;padding:10px;border-radius:5px;text-align:center;'>Email</div></a>", unsafe_allow_html=True)
    
    with watchlist_tabs[1]:
        st.subheader("Browse Community Watchlists")
        
        if not st.session_state.shared_watchlists:
            st.info("No shared watchlists available yet. Be the first to share your watchlist!")
        else:
            # Create filtering options
            filter_col1, filter_col2 = st.columns(2)
            
            # Get all unique tags
            all_tags = []
            for wl in st.session_state.shared_watchlists:
                if 'tags' in wl and wl['tags']:
                    all_tags.extend(wl['tags'])
            unique_tags = list(set([tag for tag in all_tags if tag]))
            
            with filter_col1:
                selected_tag = st.selectbox(
                    "Filter by Tag",
                    ["All"] + unique_tags,
                    key="browse_watchlist_tag"
                )
            
            with filter_col2:
                sort_by = st.selectbox(
                    "Sort By",
                    ["Most Recent", "Most Popular", "Most Stocks", "Alphabetical"],
                    key="browse_watchlist_sort"
                )
            
            # Search by name or creator
            search_query = st.text_input("Search by name, creator, or stock symbol", key="watchlist_search")
            
            # Apply filters
            filtered_watchlists = st.session_state.shared_watchlists
            
            # Apply tag filter
            if selected_tag != "All":
                filtered_watchlists = [wl for wl in filtered_watchlists 
                                     if 'tags' in wl and selected_tag in wl['tags']]
            
            # Apply search filter
            if search_query:
                search_query = search_query.lower()
                filtered_watchlists = [
                    wl for wl in filtered_watchlists
                    if (search_query in wl['name'].lower() or 
                        search_query in wl.get('created_by', '').lower() or
                        any(search_query in stock.lower() for stock in wl['stocks']))
                ]
            
            # Apply sorting
            if sort_by == "Most Recent":
                sorted_watchlists = sorted(filtered_watchlists, key=lambda x: x.get('created_at', ''), reverse=True)
            elif sort_by == "Most Popular":
                sorted_watchlists = sorted(filtered_watchlists, key=lambda x: x.get('upvotes', 0), reverse=True)
            elif sort_by == "Most Stocks":
                sorted_watchlists = sorted(filtered_watchlists, key=lambda x: len(x['stocks']), reverse=True)
            else:  # Alphabetical
                sorted_watchlists = sorted(filtered_watchlists, key=lambda x: x['name'])
            
            # Display watchlists
            if sorted_watchlists:
                st.success(f"Found {len(sorted_watchlists)} watchlists")
                
                for wl in sorted_watchlists:
                    with st.container():
                        # Header with name and creator
                        header_col1, header_col2 = st.columns([3, 1])
                        
                        with header_col1:
                            st.markdown(f"### {wl['name']}")
                            st.markdown(f"by **{wl.get('created_by', 'Anonymous')}** on {wl.get('created_at', 'Unknown date')}")
                        
                        with header_col2:
                            # Display stats
                            st.metric("Stocks", len(wl['stocks']))
                            
                            # Upvote button
                            if st.button(f"👍 {wl.get('upvotes', 0)}", key=f"upvote_wl_{wl.get('id', hash(wl['name']))}"):
                                wl['upvotes'] = wl.get('upvotes', 0) + 1
                                st.success("Watchlist upvoted!")
                        
                        # Display stocks in a clean format
                        st.markdown("**Stocks in this watchlist:**")
                        stock_chips = " ".join([f"<span style='background-color:#f0f2f6;padding:5px 10px;border-radius:15px;margin:2px;'>{stock}</span>" for stock in wl['stocks']])
                        st.markdown(f"<div style='line-height:2.5;'>{stock_chips}</div>", unsafe_allow_html=True)
                        
                        # Display tags if available
                        if 'tags' in wl and wl['tags']:
                            st.markdown("**Tags:**")
                            tag_chips = " ".join([f"<span style='background-color:#e6f2ff;padding:4px 8px;border-radius:10px;margin:2px;font-size:0.9em;'>#{tag}</span>" for tag in wl['tags'] if tag])
                            st.markdown(f"<div>{tag_chips}</div>", unsafe_allow_html=True)
                        
                        # Display notes if available
                        if wl.get('notes'):
                            with st.expander("Notes"):
                                st.markdown(wl['notes'])
                        
                        # Action buttons
                        action_col1, action_col2, _ = st.columns([1, 1, 4])
                        
                        with action_col1:
                            if st.button("Clone Watchlist", key=f"clone_wl_{wl.get('id', hash(wl['name']))}", use_container_width=True):
                                # Clone watchlist to user's watchlists
                                new_name = f"{wl['name']} (Cloned)"
                                st.session_state.user_watchlists[new_name] = wl['stocks']
                                st.success(f"Watchlist '{wl['name']}' cloned to your watchlists!")
                        
                        with action_col2:
                            if st.button("Analyze", key=f"analyze_wl_{wl.get('id', hash(wl['name']))}", use_container_width=True):
                                st.info("Redirecting to analysis page... (would navigate in a real app)")
                        
                        st.divider()
            else:
                st.info("No watchlists match your filters. Try adjusting your criteria.")
    
    with watchlist_tabs[2]:
        st.subheader("Import a Shared Watchlist")
        
        import_data = st.text_input("Paste the shared watchlist link or encoded data", key="import_watchlist_data")
        
        if import_data:
            st.info("This will import the watchlist into your personal collection.")
            
            if st.button("Import Watchlist", use_container_width=True):
                # Extract encoded data from URL if needed
                if "?data=" in import_data:
                    import_data = import_data.split("?data=")[1]
                
                watchlist_data = import_watchlist(import_data)
                
                if watchlist_data:
                    # Display import preview
                    st.success(f"Successfully imported watchlist: {watchlist_data['name']}")
                    
                    # Show details in a nice card format
                    st.markdown(f"""
                    <div style="background-color:#f0f2f6;padding:15px;border-radius:5px;margin-top:10px;">
                        <h3>{watchlist_data['name']}</h3>
                        <p><b>Created by:</b> {watchlist_data['created_by']}</p>
                        <p><b>Created on:</b> {watchlist_data['created_at']}</p>
                        <p><b>Stocks ({len(watchlist_data['stocks'])}):</b> {', '.join(watchlist_data['stocks'])}</p>
                        {f"<p><b>Notes:</b> {watchlist_data['notes']}</p>" if watchlist_data['notes'] else ""}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Customize name before importing
                    custom_name = st.text_input("Customize watchlist name (optional)", 
                                               value=f"{watchlist_data['name']} (Imported)")
                    
                    if st.button("Confirm Import with Custom Name", use_container_width=True):
                        # Add to user's watchlists
                        st.session_state.user_watchlists[custom_name] = watchlist_data['stocks']
                        st.success(f"Watchlist imported as '{custom_name}'!")
                    
                else:
                    st.error("Invalid watchlist data. Please check the link or encoded data.")
        
        # QR code scan option (simulated)
        st.divider()
        st.subheader("Scan QR Code")
        st.info("In a mobile app, you could scan a QR code to import a watchlist directly.")
        
        # File uploader for QR code (simulated)
        qr_file = st.file_uploader("Upload QR Code Image", type=["jpg", "jpeg", "png"])
        if qr_file:
            st.image(qr_file, width=300)
            st.info("QR code scanning would be implemented in a production app.")
    
    with watchlist_tabs[3]:
        st.subheader("Popular Watchlists")
        
        if not st.session_state.shared_watchlists:
            st.info("No shared watchlists available yet.")
        else:
            # Get top watchlists by upvotes
            top_watchlists = sorted(st.session_state.shared_watchlists, 
                                   key=lambda x: x.get('upvotes', 0), 
                                   reverse=True)[:5]  # Top 5
            
            # Display in a nice card layout
            for i, wl in enumerate(top_watchlists):
                with st.container():
                    st.markdown(f"### #{i+1}: {wl['name']}")
                    
                    # Stats in columns
                    stat_col1, stat_col2, stat_col3 = st.columns(3)
                    
                    with stat_col1:
                        st.metric("Upvotes", wl.get('upvotes', 0))
                    
                    with stat_col2:
                        st.metric("Stocks", len(wl['stocks']))
                    
                    with stat_col3:
                        st.write(f"**Creator:** {wl.get('created_by', 'Anonymous')}")
                    
                    # Display sample of stocks
                    stocks_preview = ", ".join(wl['stocks'][:10])
                    if len(wl['stocks']) > 10:
                        stocks_preview += f" and {len(wl['stocks']) - 10} more..."
                    st.markdown(f"**Preview:** {stocks_preview}")
                    
                    # Action buttons
                    action_col1, action_col2, _ = st.columns([1, 1, 2])
                    
                    with action_col1:
                        if st.button("Clone", key=f"popular_clone_{i}", use_container_width=True):
                            new_name = f"{wl['name']} (Popular)"
                            st.session_state.user_watchlists[new_name] = wl['stocks']
                            st.success(f"Popular watchlist '{wl['name']}' cloned!")
                    
                    with action_col2:
                        if st.button("View Details", key=f"popular_view_{i}", use_container_width=True):
                            st.session_state.selected_watchlist = wl['id']
                            st.rerun()
                    
                    st.divider()
            
            # Category-based watchlists
            st.subheader("Watchlists by Category")
            
            # Get all unique tags
            tag_counts = {}
            for wl in st.session_state.shared_watchlists:
                if 'tags' in wl and wl['tags']:
                    for tag in wl['tags']:
                        if tag:
                            tag_counts[tag] = tag_counts.get(tag, 0) + 1
            
            if tag_counts:
                # Show top tags
                top_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:6]
                
                # Create a grid of tag buttons
                tag_cols = st.columns(3)
                
                for i, (tag, count) in enumerate(top_tags):
                    with tag_cols[i % 3]:
                        st.markdown(f"""
                        <div style="background-color:#f0f2f6;padding:10px;border-radius:5px;margin-bottom:10px;text-align:center;">
                            <b>#{tag}</b><br>
                            {count} watchlists
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if st.button(f"View #{tag} Watchlists", key=f"tag_{tag}", use_container_width=True):
                            st.session_state.selected_tag = tag
                            st.rerun()
                
                # Display tag cloud visualization
                if len(tag_counts) > 6:
                    st.subheader("All Categories")
                    tag_data = pd.DataFrame([
                        {"tag": tag, "count": count}
                        for tag, count in tag_counts.items()
                    ])
                    
                    fig = px.treemap(
                        tag_data,
                        path=['tag'],
                        values='count',
                        color='count',
                        color_continuous_scale=px.colors.sequential.Blues,
                        title="Watchlist Categories"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No categorized watchlists available yet. Add tags when sharing your watchlists.")
    
    # Display recently shared watchlists at the bottom
    st.subheader("Recently Shared Watchlists")
    
    recent_watchlists = sorted(
        st.session_state.shared_watchlists, 
        key=lambda x: x.get('created_at', ''), 
        reverse=True
    )[:5]  # Get 5 most recent
    
    if recent_watchlists:
        recent_cols = st.columns(min(len(recent_watchlists), 5))
        
        for i, wl in enumerate(recent_watchlists):
            with recent_cols[i]:
                st.markdown(f"**{wl['name']}**")
                st.caption(f"{len(wl['stocks'])} stocks")
                st.caption(f"by {wl.get('created_by', 'Anonymous')}")
                
                if st.button(f"View", key=f"recent_wl_{i}", use_container_width=True):
                    # In a real app, this would navigate to the specific watchlist
                    st.session_state.selected_watchlist = wl.get('id')
                    st.rerun()
    else:
        st.info("No watchlists available yet. Be the first to share a watchlist!")

# Display strategy sharing
def display_strategy_sharing():
    """Display the strategy sharing section with enhanced browsing"""
    st.subheader("Trading Strategies Community")
    
    # Create tabs for different views
    strategy_tabs = st.tabs(["Create Strategy", "Browse Strategies", "Top Strategies", "Search Strategies"])
    
    with strategy_tabs[0]:
        st.subheader("Share Your Trading Strategy")
        
        with st.form("share_strategy_form"):
            strategy_name = st.text_input("Strategy Name")
            
            col1, col2 = st.columns(2)
            
            with col1:
                strategy_type = st.selectbox(
                    "Strategy Type",
                    ["RSI", "MACD", "Bollinger Bands", "Combined", "Custom"]
                )
            
            with col2:
                timeframe = st.selectbox(
                    "Timeframe",
                    ["1 Day", "1 Week", "1 Month", "1 Year"]
                )
            
            st.subheader("Strategy Parameters")
            
            param_cols = st.columns(3)
            
            parameters = {}
            
            # RSI parameters
            if strategy_type in ["RSI", "Combined"]:
                with param_cols[0]:
                    rsi_period = st.number_input("RSI Period", min_value=2, max_value=30, value=14)
                    rsi_oversold = st.number_input("RSI Oversold", min_value=1, max_value=40, value=30)
                    rsi_overbought = st.number_input("RSI Overbought", min_value=60, max_value=99, value=70)
                    
                    parameters.update({
                        "rsi_period": rsi_period,
                        "rsi_oversold": rsi_oversold,
                        "rsi_overbought": rsi_overbought
                    })
            
            # MACD parameters
            if strategy_type in ["MACD", "Combined"]:
                with param_cols[1]:
                    macd_fast = st.number_input("MACD Fast Period", min_value=5, max_value=20, value=12)
                    macd_slow = st.number_input("MACD Slow Period", min_value=15, max_value=40, value=26)
                    macd_signal = st.number_input("MACD Signal Period", min_value=5, max_value=15, value=9)
                    
                    parameters.update({
                        "macd_fast": macd_fast,
                        "macd_slow": macd_slow,
                        "macd_signal": macd_signal
                    })
            
            # Bollinger Bands parameters
            if strategy_type in ["Bollinger Bands", "Combined"]:
                with param_cols[2]:
                    bb_period = st.number_input("BB Period", min_value=5, max_value=50, value=20)
                    bb_std = st.number_input("BB Standard Deviations", min_value=1.0, max_value=3.0, value=2.0, step=0.1)
                    
                    parameters.update({
                        "bb_period": bb_period,
                        "bb_std": bb_std
                    })
            
            # Performance metrics (optional)
            st.subheader("Performance Metrics (Optional)")
            
            perf_cols = st.columns(4)
            
            with perf_cols[0]:
                total_return = st.number_input("Total Return (%)", min_value=-100.0, max_value=1000.0, value=0.0)
            
            with perf_cols[1]:
                annual_return = st.number_input("Annual Return (%)", min_value=-100.0, max_value=500.0, value=0.0)
            
            with perf_cols[2]:
                sharpe_ratio = st.number_input("Sharpe Ratio", min_value=-10.0, max_value=10.0, value=0.0, step=0.01)
            
            with perf_cols[3]:
                max_drawdown = st.number_input("Max Drawdown (%)", min_value=0.0, max_value=100.0, value=0.0)
            
            performance = {
                "total_return": total_return,
                "annual_return": annual_return,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown
            }
            
            # Strategy notes
            notes = st.text_area("Strategy Notes", placeholder="Describe your strategy, entry/exit rules, and any other important information...")
            
            # Associate with tickers
            ticker_options = st.session_state.selected_stocks if st.session_state.selected_stocks else ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
            associated_tickers = st.multiselect(
                "Associated Stocks (Optional)",
                options=ticker_options,
                help="Select stocks this strategy works well with"
            )
            
            submitted = st.form_submit_button("Share Strategy")
            
            if submitted and strategy_name:
                strategy_id = share_strategy(strategy_name, strategy_type, parameters, performance, notes)
                st.success(f"Strategy shared successfully! Strategy ID: {strategy_id}")
                # Increment reputation
                st.session_state.user_profile['reputation'] += 5
    
    with strategy_tabs[1]:
        st.subheader("Browse Trading Strategies")
        
        if not st.session_state.user_strategies:
            st.info("No strategies available yet. Be the first to share a strategy!")
        else:
            # Create a summary of strategies by type
            strategy_counts = {}
            for strategy in st.session_state.user_strategies:
                s_type = strategy['strategy_type']
                if s_type in strategy_counts:
                    strategy_counts[s_type] += 1
                else:
                    strategy_counts[s_type] = 1
            
            # Display summary as a chart
            if strategy_counts:
                summary_data = pd.DataFrame([
                    {"strategy_type": s_type, "count": count}
                    for s_type, count in strategy_counts.items()
                ])
                
                fig = px.pie(
                    summary_data,
                    values="count",
                    names="strategy_type",
                    title="Strategies by Type",
                    color_discrete_sequence=px.colors.qualitative.Pastel,
                    hole=0.4
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Filter options
            filter_col1, filter_col2 = st.columns(2)
            
            with filter_col1:
                strategy_types = ["All"] + list(set(s['strategy_type'] for s in st.session_state.user_strategies))
                selected_type = st.selectbox("Filter by Strategy Type", strategy_types, key="browse_strategy_type")
            
            with filter_col2:
                sort_options = ["Most Recent", "Highest Returns", "Best Sharpe Ratio", "Lowest Drawdown"]
                sort_by = st.selectbox("Sort By", sort_options, key="browse_strategy_sort")
            
            # Apply filters
            filtered_strategies = st.session_state.user_strategies
            
            if selected_type != "All":
                filtered_strategies = [s for s in filtered_strategies 
                                      if s['strategy_type'] == selected_type]
            
            # Apply sorting
            if sort_by == "Most Recent":
                sorted_strategies = sorted(filtered_strategies, key=lambda x: x.get('created_at', ''), reverse=True)
            elif sort_by == "Highest Returns":
                sorted_strategies = sorted(filtered_strategies, key=lambda x: x.get('performance', {}).get('annual_return', 0), reverse=True)
            elif sort_by == "Best Sharpe Ratio":
                sorted_strategies = sorted(filtered_strategies, key=lambda x: x.get('performance', {}).get('sharpe_ratio', 0), reverse=True)
            else:  # "Lowest Drawdown"
                sorted_strategies = sorted(filtered_strategies, key=lambda x: x.get('performance', {}).get('max_drawdown', 100))
            
            # Display strategies in cards
            for strategy in sorted_strategies:
                with st.container():
                    st.markdown(f"### {strategy['name']}")
                    st.markdown(f"**Type**: {strategy['strategy_type']} | **Timeframe**: {strategy.get('timeframe', 'N/A')} | **Created by**: {strategy.get('created_by', 'Anonymous')}")
                    
                    # Display parameters in a clean format
                    if 'parameters' in strategy:
                        param_expander = st.expander("Strategy Parameters")
                        with param_expander:
                            params = strategy['parameters']
                            
                            # Create a nice parameter display based on strategy type
                            if strategy['strategy_type'] in ["RSI", "Combined"]:
                                st.markdown("#### RSI Parameters")
                                param_col1, param_col2, param_col3 = st.columns(3)
                                with param_col1:
                                    st.metric("Period", params.get('rsi_period', 'N/A'))
                                with param_col2:
                                    st.metric("Oversold", params.get('rsi_oversold', 'N/A'))
                                with param_col3:
                                    st.metric("Overbought", params.get('rsi_overbought', 'N/A'))
                            
                            if strategy['strategy_type'] in ["MACD", "Combined"]:
                                st.markdown("#### MACD Parameters")
                                param_col1, param_col2, param_col3 = st.columns(3)
                                with param_col1:
                                    st.metric("Fast Period", params.get('macd_fast', 'N/A'))
                                with param_col2:
                                    st.metric("Slow Period", params.get('macd_slow', 'N/A'))
                                with param_col3:
                                    st.metric("Signal Period", params.get('macd_signal', 'N/A'))
                            
                            if strategy['strategy_type'] in ["Bollinger Bands", "Combined"]:
                                st.markdown("#### Bollinger Bands Parameters")
                                param_col1, param_col2 = st.columns(2)
                                with param_col1:
                                    st.metric("Period", params.get('bb_period', 'N/A'))
                                with param_col2:
                                    st.metric("Standard Deviations", params.get('bb_std', 'N/A'))
                    
                    # Display performance metrics if available
                    if 'performance' in strategy and any(strategy['performance'].values()):
                        perf = strategy['performance']
                        st.markdown("#### Performance Metrics")
                        
                        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                        
                        with metric_col1:
                            st.metric("Total Return", f"{perf.get('total_return', 0):.2f}%")
                        
                        with metric_col2:
                            annual = perf.get('annual_return', 0)
                            st.metric("Annual Return", f"{annual:.2f}%")
                        
                        with metric_col3:
                            sharpe = perf.get('sharpe_ratio', 0)
                            st.metric("Sharpe Ratio", f"{sharpe:.2f}")
                        
                        with metric_col4:
                            drawdown = perf.get('max_drawdown', 0)
                            st.metric("Max Drawdown", f"{drawdown:.2f}%")
                    
                    # Display notes if available
                    if 'notes' in strategy and strategy['notes']:
                        with st.expander("Strategy Notes"):
                            st.markdown(strategy['notes'])
                    
                    # Buttons for actions
                    action_col1, action_col2, action_col3, _ = st.columns([1, 1, 1, 3])
                    
                    with action_col1:
                        if st.button("👍 Upvote", key=f"upvote_strategy_{strategy.get('id', hash(strategy['name']))}"):
                            # Increment upvotes (would update in a real app)
                            st.session_state.user_profile['reputation'] += 1
                            st.success("Strategy upvoted!")
                    
                    with action_col2:
                        if st.button("Clone Strategy", key=f"clone_{strategy.get('id', hash(strategy['name']))}"):
                            st.session_state.clone_strategy = strategy
                            st.success("Strategy cloned! Go to Create Strategy tab to edit.")
                    
                    with action_col3:
                        if st.button("Backtest", key=f"backtest_{strategy.get('id', hash(strategy['name']))}"):
                            st.info("Redirecting to backtest page... (would navigate in a real app)")
                    
                    st.divider()
    
    with strategy_tabs[2]:
        st.subheader("Top Performing Strategies")
        
        if not st.session_state.user_strategies:
            st.info("No strategies available yet.")
        else:
            # Find strategies with performance data
            strategies_with_performance = [s for s in st.session_state.user_strategies 
                                          if 'performance' in s 
                                          and s['performance'].get('annual_return', 0) != 0]
            
            if not strategies_with_performance:
                st.warning("No strategies with performance data available.")
            else:
                # Sort by annual return
                top_strategies = sorted(strategies_with_performance, 
                                       key=lambda x: x['performance'].get('annual_return', 0), 
                                       reverse=True)
                
                # Display top 3 strategies in metrics
                if len(top_strategies) >= 3:
                    st.markdown("### Top 3 Strategies by Annual Return")
                    top_cols = st.columns(3)
                    
                    for i, strat in enumerate(top_strategies[:3]):
                        with top_cols[i]:
                            annual_return = strat['performance'].get('annual_return', 0)
                            st.metric(
                                f"#{i+1}: {strat['name']}",
                                f"{annual_return:.2f}%",
                                f"Sharpe: {strat['performance'].get('sharpe_ratio', 0):.2f}"
                            )
                            st.caption(f"Type: {strat['strategy_type']}")
                
                # Create a comparison table
                st.markdown("### Strategy Performance Comparison")
                comparison_data = []
                
                for strat in top_strategies[:10]:  # Show top 10
                    comparison_data.append({
                        'name': strat['name'],
                        'type': strat['strategy_type'],
                        'annual_return': strat['performance'].get('annual_return', 0),
                        'total_return': strat['performance'].get('total_return', 0),
                        'sharpe_ratio': strat['performance'].get('sharpe_ratio', 0),
                        'max_drawdown': strat['performance'].get('max_drawdown', 0),
                        'created_by': strat.get('created_by', 'Anonymous')
                    })
                
                if comparison_data:
                    df = pd.DataFrame(comparison_data)
                    
                    st.dataframe(
                        df,
                        column_config={
                            'name': 'Strategy Name',
                            'type': 'Type',
                            'annual_return': st.column_config.NumberColumn(
                                'Annual Return (%)',
                                format="%.2f%%"
                            ),
                            'total_return': st.column_config.NumberColumn(
                                'Total Return (%)',
                                format="%.2f%%"
                            ),
                            'sharpe_ratio': st.column_config.NumberColumn(
                                'Sharpe Ratio',
                                format="%.2f"
                            ),
                            'max_drawdown': st.column_config.NumberColumn(
                                'Max Drawdown (%)',
                                format="%.2f%%"
                            ),
                            'created_by': 'Creator'
                        },
                        hide_index=True,
                        use_container_width=True
                    )
                
                # Create a comparison chart
                st.markdown("### Visual Performance Comparison")
                
                if len(comparison_data) >= 2:
                    chart_df = pd.DataFrame([
                        {'Strategy': row['name'], 'Annual Return (%)': row['annual_return']}
                        for row in comparison_data[:8]  # Top 8 for readability
                    ])
                    
                    fig = px.bar(
                        chart_df,
                        x='Strategy',
                        y='Annual Return (%)',
                        color='Annual Return (%)',
                        title="Annual Returns by Strategy",
                        color_continuous_scale=px.colors.sequential.Greens
                    )
                    
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
    
    with strategy_tabs[3]:
        st.subheader("Search Strategies")
        
        search_col1, search_col2 = st.columns([3, 1])
        
        with search_col1:
            search_query = st.text_input("Search by keyword", placeholder="Enter strategy name, type, or keywords in notes...")
        
        with search_col2:
            min_return = st.number_input("Min Annual Return (%)", min_value=0.0, value=0.0)
        
        if st.button("Search Strategies", use_container_width=True):
            search_results = []
            
            for strategy in st.session_state.user_strategies:
                # Check performance filter
                if min_return > 0:
                    annual_return = strategy.get('performance', {}).get('annual_return', 0)
                    if annual_return < min_return:
                        continue
                
                # Check keyword match if provided
                if search_query:
                    name_match = search_query.lower() in strategy['name'].lower()
                    type_match = search_query.lower() in strategy['strategy_type'].lower()
                    notes_match = 'notes' in strategy and strategy['notes'] and search_query.lower() in strategy['notes'].lower()
                    
                    if not (name_match or type_match or notes_match):
                        continue
                
                # Add to results if passed all filters
                search_results.append(strategy)
            
            # Display results
            if search_results:
                st.success(f"Found {len(search_results)} matching strategies")
                
                for strategy in search_results:
                    with st.container():
                        st.markdown(f"### {strategy['name']}")
                        st.markdown(f"**Type**: {strategy['strategy_type']} | **Created by**: {strategy.get('created_by', 'Anonymous')}")
                        
                        # Performance metrics in a clean format
                        if 'performance' in strategy and any(strategy['performance'].values()):
                            perf = strategy['performance']
                            metrics_cols = st.columns(4)
                            
                            with metrics_cols[0]:
                                st.metric("Annual Return", f"{perf.get('annual_return', 0):.2f}%")
                            
                            with metrics_cols[1]:
                                st.metric("Total Return", f"{perf.get('total_return', 0):.2f}%")
                            
                            with metrics_cols[2]:
                                st.metric("Sharpe Ratio", f"{perf.get('sharpe_ratio', 0):.2f}")
                            
                            with metrics_cols[3]:
                                st.metric("Max Drawdown", f"{perf.get('max_drawdown', 0):.2f}%")
                        
                        # Strategy notes
                        if 'notes' in strategy and strategy['notes']:
                            with st.expander("Strategy Notes"):
                                st.markdown(strategy['notes'])
                        
                        # Action buttons
                        col1, col2, _ = st.columns([1, 1, 4])
                        with col1:
                            if st.button("View Details", key=f"view_search_{strategy.get('id', hash(strategy['name']))}", use_container_width=True):
                                st.session_state.selected_strategy = strategy
                                st.rerun()
                        
                        with col2:
                            if st.button("Backtest", key=f"backtest_search_{strategy.get('id', hash(strategy['name']))}", use_container_width=True):
                                st.info("Redirecting to backtest page... (would navigate in a real app)")
                        
                        st.divider()
            else:
                st.info("No matching strategies found. Try different search terms or lower the minimum return.")
    
    # Show trending strategies at the bottom
    st.subheader("Popular Strategy Types")
    
    # Count strategy types
    type_counts = {}
    for strategy in st.session_state.user_strategies:
        s_type = strategy['strategy_type']
        if s_type in type_counts:
            type_counts[s_type] += 1
        else:
            type_counts[s_type] = 1
    
    # Display as horizontal bar
    if type_counts:
        type_cols = st.columns(len(type_counts))
        
        for i, (s_type, count) in enumerate(sorted(type_counts.items(), key=lambda x: x[1], reverse=True)):
            with type_cols[i]:
                st.metric(s_type, count)
                if st.button(f"Browse {s_type}", key=f"browse_{s_type}", use_container_width=True):
                    # Navigate to browse tab with filter in a real app
                    st.session_state.strategy_filter = s_type
                    st.rerun()
    else:
        st.info("No strategy data available yet. Be the first to share a strategy!")

# Display stock discussions
def display_stock_discussions():
    """Display the stock discussions section with image uploads and community browsing"""
    st.subheader("Stock Discussions Community")
    
    # Create tabs for different views - removed "Post New Discussion" as requested
    discussion_tabs = st.tabs(["Latest Discussions", "Browse by Stock", "Search Posts"])
    
    with discussion_tabs[0]:
        st.subheader("Recent Community Discussions")
        
        # Collect all discussions across stocks
        all_discussions = []
        for ticker, discussions in st.session_state.stock_discussions.items():
            for discussion in discussions:
                discussion_copy = discussion.copy()
                discussion_copy['ticker'] = ticker
                all_discussions.append(discussion_copy)
        
        # Sort by most recent
        sorted_discussions = sorted(all_discussions, key=lambda x: x['created_at'], reverse=True)
        
        if sorted_discussions:
            # Show the most recent discussions
            for comment in sorted_discussions[:10]:  # Show the 10 most recent
                with st.container():
                    header_col1, header_col2 = st.columns([5, 1])
                    
                    with header_col1:
                        st.markdown(f"**{comment['ticker']}** - by **{comment['created_by']}** - {comment['created_at']}")
                    
                    with header_col2:
                        sentiment_color = {
                            "Very Bullish": "green",
                            "Bullish": "lightgreen", 
                            "Neutral": "gray",
                            "Bearish": "orange",
                            "Very Bearish": "red"
                        }.get(comment['sentiment'], "gray")
                        
                        st.markdown(f"<span style='color:{sentiment_color};'>{comment['sentiment']}</span>", unsafe_allow_html=True)
                    
                    # Display comment text
                    st.markdown(comment['comment'])
                    
                    # Display attached image if present
                    if 'has_image' in comment and comment['has_image'] and comment['image']:
                        st.markdown("**Attached Image:**")
                        st.markdown(f"<img src='data:image/png;base64,{comment['image']}' style='max-width:100%; max-height:300px;'>", unsafe_allow_html=True)
                    
                    # Voting and interaction buttons
                    vote_col1, vote_col2, reply_col, _ = st.columns([1, 1, 2, 4])
                    
                    with vote_col1:
                        if st.button(f"👍 {comment['upvotes']}", key=f"upvote_latest_{comment['id']}"):
                            comment['upvotes'] += 1
                    
                    with vote_col2:
                        if st.button(f"👎 {comment['downvotes']}", key=f"downvote_latest_{comment['id']}"):
                            comment['downvotes'] += 1
                            
                    with reply_col:
                        st.button(f"💬 Reply", key=f"reply_latest_{comment['id']}", disabled=True)
                    
                    st.divider()
        else:
            st.info("No discussions yet. Be the first to start a discussion!")
        
    with discussion_tabs[1]:
        # Browse discussions by stock
        ticker_options = list(st.session_state.stock_discussions.keys())
        if not ticker_options:
            ticker_options = st.session_state.selected_stocks if st.session_state.selected_stocks else ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
        
        selected_ticker = st.selectbox("Select a stock to view discussions", ticker_options, key="browse_ticker_select")
        
        if selected_ticker and selected_ticker in st.session_state.stock_discussions and st.session_state.stock_discussions[selected_ticker]:
            # Get consensus sentiment for this stock
            sentiment_counts = {"Very Bullish": 0, "Bullish": 0, "Neutral": 0, "Bearish": 0, "Very Bearish": 0}
            for comment in st.session_state.stock_discussions[selected_ticker]:
                if comment['sentiment'] in sentiment_counts:
                    sentiment_counts[comment['sentiment']] += 1
            
            total_comments = len(st.session_state.stock_discussions[selected_ticker])
            dominant_sentiment = max(sentiment_counts, key=sentiment_counts.get)
            
            # Display sentiment summary
            st.info(f"Community sentiment for {selected_ticker}: {dominant_sentiment} ({sentiment_counts[dominant_sentiment]}/{total_comments} posts)")
            
            # Display stock discussions in chronological order (newest first)
            for comment in reversed(st.session_state.stock_discussions[selected_ticker]):
                with st.container():
                    col1, col2 = st.columns([5, 1])
                    
                    with col1:
                        st.markdown(f"**{comment['created_by']}** - {comment['created_at']}")
                    
                    with col2:
                        sentiment_color = {
                            "Very Bullish": "green",
                            "Bullish": "lightgreen",
                            "Neutral": "gray",
                            "Bearish": "orange",
                            "Very Bearish": "red"
                        }.get(comment['sentiment'], "gray")
                        
                        st.markdown(f"<span style='color:{sentiment_color};'>{comment['sentiment']}</span>", unsafe_allow_html=True)
                    
                    # Display comment text
                    st.markdown(comment['comment'])
                    
                    # Display attached image if present
                    if 'has_image' in comment and comment['has_image'] and comment['image']:
                        st.markdown("**Attached Image:**")
                        st.markdown(f"<img src='data:image/png;base64,{comment['image']}' style='max-width:100%; max-height:300px;'>", unsafe_allow_html=True)
                    
                    # Voting buttons
                    vote_col1, vote_col2, reply_col, _ = st.columns([1, 1, 2, 4])
                    
                    with vote_col1:
                        if st.button(f"👍 {comment['upvotes']}", key=f"upvote_browse_{comment['id']}"):
                            comment['upvotes'] += 1
                    
                    with vote_col2:
                        if st.button(f"👎 {comment['downvotes']}", key=f"downvote_browse_{comment['id']}"):
                            comment['downvotes'] += 1
                            
                    with reply_col:
                        st.button(f"💬 Reply", key=f"reply_browse_{comment['id']}", disabled=True)
                    
                    st.divider()
        else:
            st.info(f"No discussions yet for {selected_ticker}. Be the first to comment!")
    
    with discussion_tabs[2]:
        # Search discussions section
        st.subheader("Search Discussions")
        
        # Select a stock to discuss
        ticker_options = st.session_state.selected_stocks if st.session_state.selected_stocks else ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
        selected_ticker = st.selectbox("Select a stock to discuss", ticker_options, key="post_ticker_select")
        
        # Add a new comment with image upload
        with st.form("add_discussion_comment"):
            comment = st.text_area("Your comment", height=150, placeholder="Share your thoughts, analysis or questions about this stock...")
            
            sentiment = st.select_slider(
                "Your sentiment",
                options=["Very Bearish", "Bearish", "Neutral", "Bullish", "Very Bullish"],
                value="Neutral"
            )
            
            # Image upload options
            image_option = st.radio("Attach an image?", ["No image", "Upload image", "Take screenshot"])
            
            uploaded_image = None
            screenshot_image = None
            
            if image_option == "Upload image":
                uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="discussion_img_upload")
                if uploaded_file is not None:
                    # Display preview
                    st.image(uploaded_file, caption="Image Preview", width=300)
                    # Convert to base64
                    img_bytes = uploaded_file.getvalue()
                    uploaded_image = base64.b64encode(img_bytes).decode()
            
            elif image_option == "Take screenshot":
                st.info("This feature would capture your current chart or analysis in a real app.")
                if st.checkbox("Simulate screenshot for demo"):
                    # Use a placeholder image for demo
                    screenshot_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+P+/HgAFDQIYQk4GvgAAAABJRU5ErkJggg=="
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Add hashtags
                hashtags = st.text_input("Add hashtags (optional)", placeholder="#investing #technical")
            
            with col2:
                st.write("")
                st.write("")
                submit_comment = st.form_submit_button("Post Discussion", use_container_width=True)
            
            if submit_comment and comment:
                # Determine which image to use
                image_data = None
                if image_option == "Upload image" and uploaded_image:
                    image_data = uploaded_image
                elif image_option == "Take screenshot" and screenshot_image:
                    image_data = screenshot_image
                
                # Add the comment with image if provided
                comment_id = add_stock_discussion(selected_ticker, comment, sentiment, image_data)
                st.success("Discussion posted successfully!")
                
                # Update user reputation
                st.session_state.user_profile['shared_items'] += 1
    
# Removed duplicate search tab (merged into tab index 2)
        
        if st.button("Search Discussions", use_container_width=True):
            # Collect matching discussions
            search_results = []
            for ticker, discussions in st.session_state.stock_discussions.items():
                for discussion in discussions:
                    # Match by keyword (case insensitive)
                    keyword_match = not search_query or search_query.lower() in discussion['comment'].lower()
                    # Match by sentiment
                    sentiment_match = search_sentiment == "Any" or discussion['sentiment'] == search_sentiment
                    
                    if keyword_match and sentiment_match:
                        discussion_copy = discussion.copy()
                        discussion_copy['ticker'] = ticker
                        search_results.append(discussion_copy)
            
            # Display search results
            if search_results:
                st.success(f"Found {len(search_results)} matching discussions")
                
                for idx, result in enumerate(search_results):
                    with st.container():
                        st.markdown(f"**{result['ticker']}** - by **{result['created_by']}** - {result['created_at']}")
                        st.markdown(f"Sentiment: {result['sentiment']}")
                        st.markdown(result['comment'])
                        
                        # Display attached image if present
                        if 'has_image' in result and result['has_image'] and result['image']:
                            st.markdown("**Attached Image:**")
                            st.markdown(f"<img src='data:image/png;base64,{result['image']}' style='max-width:100%; max-height:300px;'>", unsafe_allow_html=True)
                            
                        st.divider()
            else:
                st.info("No matching discussions found. Try different search terms.")
                
    # Add a quick search by trending stocks at the bottom
    st.subheader("Trending Stocks in Discussions")
    trending = get_trending_stocks(limit=5)
    
    if trending:
        cols = st.columns(len(trending))
        for i, (ticker, count) in enumerate(trending):
            with cols[i]:
                st.metric(ticker, f"{count} posts")
                if st.button(f"View {ticker} Discussions", key=f"trend_{ticker}", use_container_width=True):
                    # This would navigate to the ticker's discussion in a real app
                    st.session_state.stock_discussion_selected = ticker
                    st.rerun()

# Display the leaderboard
def display_leaderboard():
    """Display the social trading leaderboard"""
    st.subheader("Social Trading Leaderboard")
    
    update_leaderboard()
    
    if not st.session_state.leaderboard:
        st.info("The leaderboard is empty. Share your strategies to appear on the leaderboard!")
        return
    
    # Create a dataframe from the leaderboard data
    leaderboard_df = pd.DataFrame(st.session_state.leaderboard)
    
    # Add visual ranking with emojis
    def format_rank(rank):
        if rank == 1:
            return "🥇 1st"
        elif rank == 2:
            return "🥈 2nd"
        elif rank == 3:
            return "🥉 3rd"
        else:
            return f"{rank}th"
    
    leaderboard_df['visual_rank'] = leaderboard_df['rank'].apply(format_rank)
    
    # Display the leaderboard
    st.dataframe(
        leaderboard_df[['visual_rank', 'username', 'strategies', 'shared_items', 'reputation']],
        column_config={
            "visual_rank": "Rank",
            "username": "User",
            "strategies": "Strategies",
            "shared_items": "Shared Items",
            "reputation": "Reputation"
        },
        hide_index=True,
        use_container_width=True
    )

# Display trending stocks
def display_trending_stocks():
    """Display trending stocks based on discussion activity"""
    st.subheader("Trending Stocks")
    
    trending = get_trending_stocks()
    
    if not trending:
        st.info("No trending stocks yet. Join the discussion to see stocks trend!")
        return
    
    # Create a bar chart of trending stocks
    trending_df = pd.DataFrame(trending, columns=['ticker', 'count'])
    
    fig = px.bar(
        trending_df,
        x='ticker',
        y='count',
        title="Most Discussed Stocks",
        labels={'ticker': 'Stock', 'count': 'Discussion Count'},
        color='count',
        color_continuous_scale=px.colors.sequential.Blues
    )
    
    fig.update_layout(xaxis_title="Stock Symbol", yaxis_title="Number of Comments")
    
    st.plotly_chart(fig, use_container_width=True)

# Display one-click sharing options
def display_one_click_share(chart_fig=None):
    """Display enhanced one-click sharing options for insights with screenshot support"""
    st.subheader("📸 Share Your Insights")
    
    # Add tabs for different sharing options
    share_tabs = st.tabs(["Create Post", "Take Screenshot", "Upload Image"])
    
    with share_tabs[0]:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            share_title = st.text_input("Title", "My Stock Analysis Insight")
            share_text = st.text_area("Description", "Check out this interesting pattern I found...")
            
            # Add stock tag selector
            stock_tags = st.multiselect(
                "Tag stocks",
                options=st.session_state.selected_stocks if hasattr(st.session_state, 'selected_stocks') else ["AAPL", "MSFT", "GOOGL"],
                default=[],
                help="Tag relevant stocks in your post"
            )
        
        with col2:
            st.write("")
            st.write("")
            share_type = st.radio(
                "Share as",
                ["Insight", "Question", "News", "Educational"],
                index=0
            )
            
            # Add visibility options
            visibility = st.radio(
                "Visibility",
                ["Public", "Community Only"],
                index=0
            )
        
        # Generate share links with proper metadata
        if st.button("Generate Social Media Share Links", use_container_width=True):
            img_base64 = None
            if chart_fig:
                img_base64 = get_chart_screenshot(chart_fig)
            
            # Create hashtags from stock tags
            hashtags = " ".join([f"#{stock}" for stock in stock_tags]) if stock_tags else ""
            formatted_text = f"{share_title}\n\n{share_text}\n\n{hashtags}"
            
            links = get_social_share_links(share_title, img_base64, formatted_text)
            
            st.subheader("Share on Social Media")
            for platform, link in links.items():
                st.markdown(f"<a href='{link}' target='_blank' style='text-decoration:none;'><div style='background-color:#f0f2f6;padding:10px;border-radius:5px;margin:5px 0;'><b>Share on {platform.capitalize()}</b></div></a>", unsafe_allow_html=True)
            
            st.success("Share links generated! Click any platform to share your insight.")
            
            # Add to user shared items count
            st.session_state.user_profile['shared_items'] += 1
            st.session_state.user_profile['reputation'] += 5
    
    with share_tabs[1]:
        st.subheader("Capture Current Chart")
        st.write("Take a screenshot of the current analysis chart to share")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            chart_title = st.text_input("Screenshot Title", "My Technical Analysis")
        
        with col2:
            st.write("")
            st.write("")
            if st.button("Capture Screenshot", use_container_width=True):
                if chart_fig:
                    img_base64 = get_chart_screenshot(chart_fig)
                    st.success("Screenshot captured!")
                    
                    # Display the captured screenshot
                    st.markdown(f"### Preview: {chart_title}")
                    st.markdown(f"<img src='data:image/png;base64,{img_base64}' style='width:100%;'>", unsafe_allow_html=True)
                    
                    # Store in session state
                    if 'captured_screenshots' not in st.session_state:
                        st.session_state.captured_screenshots = []
                        
                    st.session_state.captured_screenshots.append({
                        'title': chart_title,
                        'image': img_base64,
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    })
                    
                    # Generate share links for the screenshot
                    links = get_social_share_links(chart_title, img_base64, "Check out my technical analysis!")
                    
                    st.subheader("Share Screenshot")
                    for platform, link in links.items():
                        st.markdown(f"<a href='{link}' target='_blank' style='text-decoration:none;'><div style='background-color:#f0f2f6;padding:10px;border-radius:5px;margin:5px 0;'><b>Share on {platform.capitalize()}</b></div></a>", unsafe_allow_html=True)
                else:
                    st.warning("No active chart to capture. Please navigate to a page with charts first.")
    
    with share_tabs[2]:
        st.subheader("Upload Image to Share")
        
        # File uploader
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Display the uploaded image
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
            
            # Convert to base64 for sharing
            img_bytes = uploaded_file.getvalue()
            img_base64 = base64.b64encode(img_bytes).decode()
            
            # Image description
            img_title = st.text_input("Image Title", "My Investment Chart")
            img_desc = st.text_area("Image Description", "A chart showing my investment analysis...")
            
            # Generate share links for the uploaded image
            if st.button("Share Uploaded Image", use_container_width=True):
                links = get_social_share_links(img_title, img_base64, img_desc)
                
                st.subheader("Share Your Image")
                for platform, link in links.items():
                    st.markdown(f"<a href='{link}' target='_blank' style='text-decoration:none;'><div style='background-color:#f0f2f6;padding:10px;border-radius:5px;margin:5px 0;'><b>Share on {platform.capitalize()}</b></div></a>", unsafe_allow_html=True)
                
                st.success("Ready to share! Click any platform to share your image.")
                
                # Add to user shared items count
                st.session_state.user_profile['shared_items'] += 1
                st.session_state.user_profile['reputation'] += 5

# Main social features page
# Display stock ratings
def display_stock_ratings():
    """Display the stock ratings section with enhanced browsing"""
    st.subheader("Stock Ratings Community")
    
    # Create tabs for different views
    rating_tabs = st.tabs(["Rate a Stock", "Browse Ratings", "Top Rated Stocks", "Search Ratings"])
    
    with rating_tabs[0]:
        st.subheader("Rate & Review Stocks")
        
        # Select a stock to rate
        ticker_options = st.session_state.selected_stocks if st.session_state.selected_stocks else ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
        selected_ticker = st.selectbox("Select a stock to rate", ticker_options, key="rating_stock_select")
        
        if selected_ticker:
            # Show current consensus if available
            consensus = get_stock_consensus(selected_ticker)
            
            if consensus['average_rating']:
                # Display star rating with colored indicator
                rating_color = "green" if consensus['average_rating'] >= 4.0 else "orange" if consensus['average_rating'] >= 3.0 else "red"
                st.markdown(f"""
                <div style="background-color:#f0f2f6;padding:10px;border-radius:5px;">
                    <h3>Community Consensus for {selected_ticker}</h3>
                    <p style="font-size:24px;color:{rating_color};">{"⭐" * round(consensus['average_rating'])}</p>
                    <p><b>{consensus['average_rating']:.1f}/5.0</b> from {consensus['num_ratings']} ratings - <b>{consensus['sentiment']}</b></p>
                </div>
                """, unsafe_allow_html=True)
            
            # Add a new rating
            with st.form("add_stock_rating"):
                st.subheader(f"Rate {selected_ticker}")
                
                rating = st.slider("Your Rating", 1, 5, 3, 1, 
                                help="1=Strong Sell, 2=Sell, 3=Hold, 4=Buy, 5=Strong Buy")
                
                comment = st.text_area("Your Review (Optional)", 
                                    placeholder="Share your thoughts on this stock...")
                
                submitted = st.form_submit_button("Submit Rating")
                
                if submitted:
                    rating_id = add_stock_rating(selected_ticker, rating, comment)
                    st.success(f"Rating submitted successfully for {selected_ticker}!")
                    # Increment reputation
                    st.session_state.user_profile['reputation'] += 2
            
            # Display existing ratings
            if selected_ticker in st.session_state.stock_ratings and st.session_state.stock_ratings[selected_ticker]:
                st.subheader(f"Ratings for {selected_ticker}")
                
                for rating_data in reversed(st.session_state.stock_ratings[selected_ticker]):
                    with st.container():
                        col1, col2 = st.columns([3, 2])
                        
                        with col1:
                            st.markdown(f"**{rating_data['created_by']}** - {rating_data['created_at']}")
                        
                        with col2:
                            # Display star rating
                            stars = "⭐" * rating_data['rating']
                            st.markdown(f"{stars} ({rating_data['rating']}/5)")
                        
                        if rating_data['comment']:
                            st.markdown(rating_data['comment'])
                        
                        # Upvote button for helpful ratings
                        if st.button(f"👍 Helpful ({rating_data['upvotes']})", key=f"upvote_rating_{rating_data['id']}"):
                            rating_data['upvotes'] += 1
                            st.session_state.user_profile['reputation'] += 1
                        
                        st.divider()
            else:
                st.info(f"No ratings yet for {selected_ticker}. Be the first to rate!")
    
    with rating_tabs[1]:
        st.subheader("Browse Stock Ratings")
        
        # Get all stocks with ratings
        rated_stocks = list(st.session_state.stock_ratings.keys())
        
        if not rated_stocks:
            st.info("No stocks have been rated yet. Be the first to rate a stock!")
        else:
            # Collect ratings stats for all stocks
            ratings_summary = []
            for ticker in rated_stocks:
                consensus = get_stock_consensus(ticker)
                if consensus['average_rating']:
                    ratings_summary.append({
                        'ticker': ticker,
                        'average_rating': consensus['average_rating'],
                        'num_ratings': consensus['num_ratings'],
                        'sentiment': consensus['sentiment']
                    })
            
            # Convert to DataFrame for display
            if ratings_summary:
                summary_df = pd.DataFrame(ratings_summary)
                
                # Add star rating
                summary_df['stars'] = summary_df['average_rating'].apply(lambda x: "⭐" * round(x))
                
                # Display as interactive table
                st.dataframe(
                    summary_df,
                    column_config={
                        "ticker": "Stock",
                        "stars": "Rating",
                        "average_rating": st.column_config.NumberColumn(
                            "Average Rating", 
                            format="%.1f",
                            help="Average community rating out of 5"
                        ),
                        "num_ratings": "# of Ratings",
                        "sentiment": "Sentiment"
                    },
                    hide_index=True,
                    use_container_width=True
                )
            
            # Select a stock to view detailed ratings
            selected_browse_ticker = st.selectbox(
                "Select a stock to view detailed ratings", 
                rated_stocks,
                key="browse_ratings_ticker"
            )
            
            if selected_browse_ticker:
                # Display detailed ratings for selected stock
                if selected_browse_ticker in st.session_state.stock_ratings:
                    ratings = st.session_state.stock_ratings[selected_browse_ticker]
                    
                    # Sort options
                    sort_by = st.radio(
                        "Sort by",
                        ["Most Recent", "Highest Rating", "Lowest Rating", "Most Helpful"],
                        horizontal=True
                    )
                    
                    # Sort the ratings
                    if sort_by == "Most Recent":
                        sorted_ratings = sorted(ratings, key=lambda x: x['created_at'], reverse=True)
                    elif sort_by == "Highest Rating":
                        sorted_ratings = sorted(ratings, key=lambda x: x['rating'], reverse=True)
                    elif sort_by == "Lowest Rating":
                        sorted_ratings = sorted(ratings, key=lambda x: x['rating'])
                    else:  # Most Helpful
                        sorted_ratings = sorted(ratings, key=lambda x: x['upvotes'], reverse=True)
                    
                    # Display ratings
                    for rating_data in sorted_ratings:
                        with st.container():
                            header_col1, header_col2 = st.columns([3, 2])
                            
                            with header_col1:
                                st.markdown(f"**{rating_data['created_by']}** - {rating_data['created_at']}")
                            
                            with header_col2:
                                # Display star rating
                                rating_color = "green" if rating_data['rating'] >= 4 else "orange" if rating_data['rating'] >= 3 else "red"
                                st.markdown(f"<span style='color:{rating_color};font-size:18px;'>{'⭐' * rating_data['rating']}</span> ({rating_data['rating']}/5)", unsafe_allow_html=True)
                            
                            if rating_data['comment']:
                                st.markdown(f"<div style='background-color:#f9f9f9;padding:10px;border-radius:5px;'>{rating_data['comment']}</div>", unsafe_allow_html=True)
                            
                            # Helpful button and counter
                            col1, col2, _ = st.columns([1, 1, 4])
                            with col1:
                                if st.button(f"👍 Helpful ({rating_data['upvotes']})", key=f"browse_helpful_{rating_data['id']}"):
                                    rating_data['upvotes'] += 1
                            
                            with col2:
                                st.button("💬 Reply", key=f"browse_reply_{rating_data['id']}", disabled=True)
                            
                            st.divider()
    
    with rating_tabs[2]:
        st.subheader("Top Rated Stocks")
        
        # Collect all ratings data
        all_ratings = []
        for ticker, ratings in st.session_state.stock_ratings.items():
            consensus = get_stock_consensus(ticker)
            if consensus['average_rating'] and consensus['num_ratings'] >= 2:  # Only include stocks with at least 2 ratings
                all_ratings.append({
                    'ticker': ticker,
                    'average_rating': consensus['average_rating'], 
                    'num_ratings': consensus['num_ratings'],
                    'sentiment': consensus['sentiment']
                })
        
        if not all_ratings:
            st.info("Not enough ratings data to show top rated stocks yet. Contribute ratings to help build this list!")
        else:
            # Sort by average rating
            top_rated = sorted(all_ratings, key=lambda x: x['average_rating'], reverse=True)
            
            # Create a visual metric display
            col_count = min(len(top_rated), 3)  # Show at most 3 columns
            if col_count > 0:
                cols = st.columns(col_count)
                
                for i, stock_data in enumerate(top_rated[:col_count]):
                    with cols[i]:
                        st.metric(
                            f"#{i+1}: {stock_data['ticker']}", 
                            f"{stock_data['average_rating']:.1f}/5.0",
                            f"{stock_data['num_ratings']} ratings"
                        )
                        st.markdown(f"{'⭐' * round(stock_data['average_rating'])}")
                        st.markdown(f"<b>{stock_data['sentiment']}</b>", unsafe_allow_html=True)
            
            # Show all top rated stocks in a table
            if len(top_rated) > 3:
                st.markdown("### All Top Rated Stocks")
                top_df = pd.DataFrame(top_rated)
                top_df['stars'] = top_df['average_rating'].apply(lambda x: "⭐" * round(x))
                
                st.dataframe(
                    top_df,
                    column_config={
                        "ticker": "Stock",
                        "stars": "Rating",
                        "average_rating": st.column_config.NumberColumn(
                            "Score", 
                            format="%.1f",
                            help="Average community rating out of 5"
                        ),
                        "num_ratings": "# Ratings",
                        "sentiment": "Consensus"
                    },
                    hide_index=True,
                    use_container_width=True
                )
            
            # Visualization of top rated stocks
            if len(top_rated) >= 3:  # Only show chart if we have enough data
                chart_data = pd.DataFrame(top_rated[:10])  # Top 10 stocks
                
                fig = px.bar(
                    chart_data,
                    x='ticker',
                    y='average_rating',
                    color='average_rating',
                    text='average_rating',
                    color_continuous_scale=px.colors.sequential.Greens,
                    title="Top Rated Stocks by Community"
                )
                
                fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
                fig.update_layout(
                    xaxis_title="Stock",
                    yaxis_title="Rating (out of 5)",
                    yaxis_range=[0, 5.5]
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    with rating_tabs[3]:
        st.subheader("Search Stock Ratings")
        
        search_col1, search_col2 = st.columns([3, 1])
        
        with search_col1:
            search_query = st.text_input("Search by keyword in reviews", placeholder="dividend, growth, CEO, etc.")
        
        with search_col2:
            min_rating = st.selectbox("Minimum Rating", [1, 2, 3, 4, 5], index=0)
        
        if st.button("Search Ratings", use_container_width=True):
            if not search_query and min_rating == 1:
                st.warning("Please enter a search term or select a minimum rating to filter results.")
            else:
                # Collect all matching ratings
                search_results = []
                
                for ticker, ratings in st.session_state.stock_ratings.items():
                    for rating in ratings:
                        # Check minimum rating filter
                        if rating['rating'] < min_rating:
                            continue
                            
                        # Check keyword match if provided
                        if search_query and rating['comment'] and search_query.lower() not in rating['comment'].lower():
                            continue
                            
                        # Add to results if it passed all filters
                        result_copy = rating.copy()
                        result_copy['ticker'] = ticker
                        search_results.append(result_copy)
                
                # Display results
                if search_results:
                    st.success(f"Found {len(search_results)} matching ratings")
                    
                    # Sort by most recent
                    sorted_results = sorted(search_results, key=lambda x: x['created_at'], reverse=True)
                    
                    for result in sorted_results:
                        with st.container():
                            st.markdown(f"**{result['ticker']}** - Rated by **{result['created_by']}** on {result['created_at']}")
                            st.markdown(f"{'⭐' * result['rating']} ({result['rating']}/5)")
                            
                            if result['comment']:
                                st.markdown(f"<div style='background-color:#f9f9f9;padding:10px;border-radius:5px;'>{result['comment']}</div>", unsafe_allow_html=True)
                            
                            col1, col2, _ = st.columns([1, 1, 4])
                            with col1:
                                if st.button(f"👍 Helpful ({result['upvotes']})", key=f"search_helpful_{result['id']}"):
                                    result['upvotes'] += 1
                            
                            with col2:
                                if st.button(f"View {result['ticker']} Ratings", key=f"search_view_{result['id']}"):
                                    # This would navigate to the stock's ratings in a real app
                                    st.session_state.ratings_ticker_selected = result['ticker']
                                    st.rerun()
                            
                            st.divider()
                else:
                    st.info("No matching ratings found. Try different search terms or lower the minimum rating.")
                
    # Display recent or featured ratings at the bottom
    st.subheader("Recently Added Ratings")
    
    # Collect recent ratings across all stocks
    recent_ratings = []
    for ticker, ratings in st.session_state.stock_ratings.items():
        for rating in ratings:
            rating_copy = rating.copy()
            rating_copy['ticker'] = ticker
            recent_ratings.append(rating_copy)
    
    # Sort by most recent
    recent_ratings = sorted(recent_ratings, key=lambda x: x['created_at'], reverse=True)
    
    if recent_ratings:
        # Show the 5 most recent ratings
        recent_cols = st.columns(min(len(recent_ratings[:5]), 5))
        
        for i, rating in enumerate(recent_ratings[:5]):
            with recent_cols[i]:
                st.markdown(f"**{rating['ticker']}**")
                st.markdown(f"{'⭐' * rating['rating']}")
                st.markdown(f"by {rating['created_by']}")
                
                if st.button(f"View Details", key=f"recent_{rating['id']}", use_container_width=True):
                    # This would navigate to the specific rating in a real app
                    st.session_state.ratings_ticker_selected = rating['ticker']
                    st.rerun()
    else:
        st.info("No ratings available yet. Be the first to rate a stock!")

# Display user recommendations
def display_recommendations():
    """Display the stock recommendations section with browsing capabilities"""
    st.subheader("Stock Recommendations Community")
    
    # Create tabs for different recommendation views
    rec_tabs = st.tabs(["Create Recommendation", "Browse Recommendations", "Top Recommendations", "Search Recommendations"])
    
    with rec_tabs[0]:
        st.subheader("Create a New Stock Recommendation")
        
        # Create new recommendation form
        with st.form("create_recommendation"):
            st.markdown("### Create a Detailed Stock Recommendation")
            
            # Select stock
            ticker_options = st.session_state.selected_stocks if st.session_state.selected_stocks else ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
            ticker = st.selectbox("Stock Symbol", ticker_options, key="rec_ticker_select")
            
            col1, col2 = st.columns(2)
            
            with col1:
                category = st.selectbox("Recommendation", st.session_state.recommendation_categories)
                price_target = st.number_input("Price Target ($)", min_value=0.01, step=0.01, value=100.00)
            
            with col2:
                horizon = st.selectbox("Investment Horizon", st.session_state.investment_horizons)
                investment_style = st.selectbox("Investment Style", st.session_state.investor_styles)
            
            thesis = st.text_area("Investment Thesis", 
                                placeholder="Explain your reasoning for this recommendation...")
            
            risk_factors = st.text_area("Risk Factors (Optional)",
                                    placeholder="Outline potential risks to your recommendation...")
            
            submitted = st.form_submit_button("Publish Recommendation")
            
            if submitted and ticker and thesis:
                recommendation_id = create_recommendation(
                    ticker, category, price_target, horizon, investment_style, thesis, risk_factors
                )
                st.success(f"Recommendation for {ticker} published successfully!")
                # Increment reputation
                st.session_state.user_profile['reputation'] += 5
    
    with rec_tabs[1]:
        st.subheader("Browse Stock Recommendations")
        
        if not st.session_state.user_recommendations:
            st.info("No recommendations available yet. Be the first to create a recommendation!")
        else:
            # Create a summary of recommendations by ticker and category
            rec_summary = {}
            
            for rec in st.session_state.user_recommendations:
                ticker = rec['ticker']
                category = rec['category']
                
                if ticker not in rec_summary:
                    rec_summary[ticker] = {"Strong Buy": 0, "Buy": 0, "Hold": 0, "Sell": 0, "Strong Sell": 0, "total": 0}
                
                rec_summary[ticker][category] += 1
                rec_summary[ticker]["total"] += 1
            
            # Display summary as a table
            summary_data = []
            for ticker, counts in rec_summary.items():
                sentiment_score = (counts["Strong Buy"] * 2 + counts["Buy"] - counts["Sell"] - counts["Strong Sell"] * 2) / counts["total"]
                consensus = "Strong Buy" if sentiment_score > 1.5 else "Buy" if sentiment_score > 0.5 else "Hold" if sentiment_score > -0.5 else "Sell" if sentiment_score > -1.5 else "Strong Sell"
                
                summary_data.append({
                    "ticker": ticker,
                    "total": counts["total"], 
                    "strong_buy": counts["Strong Buy"],
                    "buy": counts["Buy"],
                    "hold": counts["Hold"],
                    "sell": counts["Sell"],
                    "strong_sell": counts["Strong Sell"],
                    "sentiment_score": sentiment_score,
                    "consensus": consensus
                })
            
            # Convert to DataFrame and display
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                
                # Get consensus color
                def get_consensus_color(consensus):
                    colors = {
                        "Strong Buy": "green",
                        "Buy": "lightgreen",
                        "Hold": "gray",
                        "Sell": "orange",
                        "Strong Sell": "red"
                    }
                    return colors.get(consensus, "gray")
                
                # Add formatted consensus column
                summary_df["formatted_consensus"] = summary_df["consensus"].apply(
                    lambda x: f"<span style='color:{get_consensus_color(x)};font-weight:bold;'>{x}</span>"
                )
                
                # Display as interactive table
                st.dataframe(
                    summary_df,
                    column_config={
                        "ticker": "Stock",
                        "total": "Total Recs",
                        "strong_buy": "Strong Buy",
                        "buy": "Buy",
                        "hold": "Hold",
                        "sell": "Sell",
                        "strong_sell": "Strong Sell",
                        "formatted_consensus": st.column_config.Column(
                            "Consensus",
                            help="Weighted consensus based on all recommendations",
                            width="medium"
                        )
                    },
                    hide_index=True,
                    use_container_width=True
                )
            
            # Get all tickers with recommendations
            tickers = set(rec['ticker'] for rec in st.session_state.user_recommendations)
            
            # Filter recommendations by ticker
            filter_col1, filter_col2, filter_col3 = st.columns([2, 2, 1])
            
            with filter_col1:
                selected_ticker = st.selectbox("Filter by Stock", ["All"] + list(tickers), key="browse_rec_ticker")
            
            with filter_col2:
                selected_category = st.selectbox(
                    "Filter by Recommendation Type",
                    ["All"] + st.session_state.recommendation_categories,
                    key="browse_rec_category"
                )
            
            with filter_col3:
                sort_by = st.selectbox(
                    "Sort By",
                    ["Most Recent", "Most Upvoted", "Price Target (High to Low)"],
                    key="browse_rec_sort"
                )
            
            # Apply filters
            filtered_recommendations = st.session_state.user_recommendations
            
            if selected_ticker != "All":
                filtered_recommendations = [rec for rec in filtered_recommendations 
                                           if rec['ticker'] == selected_ticker]
            
            if selected_category != "All":
                filtered_recommendations = [rec for rec in filtered_recommendations 
                                           if rec['category'] == selected_category]
            
            # Apply sorting
            if sort_by == "Most Recent":
                sorted_recommendations = sorted(filtered_recommendations, key=lambda x: x['created_at'], reverse=True)
            elif sort_by == "Most Upvoted":
                sorted_recommendations = sorted(filtered_recommendations, key=lambda x: x['upvotes'], reverse=True)
            else:  # Price Target (High to Low)
                sorted_recommendations = sorted(filtered_recommendations, key=lambda x: x['price_target'], reverse=True)
            
            # Display recommendations
            if sorted_recommendations:
                for rec in sorted_recommendations:
                    with st.container():
                        # Header with ticker and recommendation type
                        header_col1, header_col2 = st.columns([3, 2])
                        
                        with header_col1:
                            st.markdown(f"### {rec['ticker']} - {rec['category']}")
                            st.markdown(f"by **{rec['created_by']}** on {rec['created_at']}")
                        
                        with header_col2:
                            # Color-code the recommendation
                            rec_color = {
                                "Strong Buy": "green",
                                "Buy": "lightgreen",
                                "Hold": "gray",
                                "Sell": "orange",
                                "Strong Sell": "red"
                            }.get(rec['category'], "gray")
                            
                            st.markdown(f"""
                            <div style="background-color:{rec_color}; padding:10px; border-radius:5px;">
                                <h3 style="color:white; text-align:center; margin:0;">{rec['category']}</h3>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Price target and details
                        details_col1, details_col2, details_col3 = st.columns(3)
                        
                        with details_col1:
                            st.metric("Price Target", f"${rec['price_target']:.2f}")
                        
                        with details_col2:
                            st.write("Horizon:", rec['horizon'])
                        
                        with details_col3:
                            st.write("Style:", rec['investment_style'])
                        
                        # Thesis and risk factors in expanders
                        with st.expander("Investment Thesis", expanded=True):
                            st.write(rec['thesis'])
                        
                        if rec['risk_factors']:
                            with st.expander("Risk Factors"):
                                st.write(rec['risk_factors'])
                        
                        # Voting buttons
                        vote_col1, vote_col2, _ = st.columns([1, 1, 4])
                        
                        with vote_col1:
                            if st.button(f"👍 Agree ({rec['upvotes']})", key=f"browse_upvote_rec_{rec['id']}"):
                                upvote_recommendation(rec['id'])
                                st.session_state.user_profile['reputation'] += 1
                        
                        with vote_col2:
                            if st.button(f"👎 Disagree ({rec['downvotes']})", key=f"browse_downvote_rec_{rec['id']}"):
                                rec['downvotes'] += 1
                        
                        st.divider()
            else:
                st.info("No recommendations match your filters. Try adjusting your criteria.")
    
    with rec_tabs[2]:
        st.subheader("Top Stock Recommendations")
        
        if not st.session_state.user_recommendations:
            st.info("No recommendations available yet.")
        else:
            # Create a list of top buy and sell recommendations
            buy_recommendations = []
            sell_recommendations = []
            
            for rec in st.session_state.user_recommendations:
                if rec['category'] in ["Strong Buy", "Buy"]:
                    buy_recommendations.append(rec)
                elif rec['category'] in ["Strong Sell", "Sell"]:
                    sell_recommendations.append(rec)
            
            # Sort by upvotes
            buy_recommendations = sorted(buy_recommendations, key=lambda x: x['upvotes'], reverse=True)
            sell_recommendations = sorted(sell_recommendations, key=lambda x: x['upvotes'], reverse=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Top Buy Recommendations")
                if buy_recommendations:
                    for i, rec in enumerate(buy_recommendations[:5]):  # Show top 5
                        with st.container():
                            rec_color = "green" if rec['category'] == "Strong Buy" else "lightgreen"
                            st.markdown(f"""
                            <div style="background-color:#f0f2f6;padding:10px;border-radius:5px;margin-bottom:10px;">
                                <div style="display:flex;justify-content:space-between;">
                                    <span style="font-weight:bold;">{rec['ticker']}</span>
                                    <span style="color:{rec_color};font-weight:bold;">{rec['category']}</span>
                                </div>
                                <div>Target: ${rec['price_target']:.2f}</div>
                                <div>👍 {rec['upvotes']} upvotes</div>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.info("No buy recommendations available yet.")
            
            with col2:
                st.subheader("Top Sell Recommendations")
                if sell_recommendations:
                    for i, rec in enumerate(sell_recommendations[:5]):  # Show top 5
                        with st.container():
                            rec_color = "red" if rec['category'] == "Strong Sell" else "orange"
                            st.markdown(f"""
                            <div style="background-color:#f0f2f6;padding:10px;border-radius:5px;margin-bottom:10px;">
                                <div style="display:flex;justify-content:space-between;">
                                    <span style="font-weight:bold;">{rec['ticker']}</span>
                                    <span style="color:{rec_color};font-weight:bold;">{rec['category']}</span>
                                </div>
                                <div>Target: ${rec['price_target']:.2f}</div>
                                <div>👍 {rec['upvotes']} upvotes</div>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.info("No sell recommendations available yet.")
            
            # Most Active Stocks (with most recommendations)
            st.subheader("Most Discussed Stocks")
            
            # Count recommendations per ticker
            ticker_counts = {}
            for rec in st.session_state.user_recommendations:
                ticker = rec['ticker']
                if ticker in ticker_counts:
                    ticker_counts[ticker] += 1
                else:
                    ticker_counts[ticker] = 1
            
            # Convert to dataframe for visualization
            if ticker_counts:
                ticker_data = pd.DataFrame([
                    {"ticker": ticker, "count": count}
                    for ticker, count in ticker_counts.items()
                ])
                
                ticker_data = ticker_data.sort_values("count", ascending=False).head(10)
                
                fig = px.bar(
                    ticker_data,
                    x="ticker",
                    y="count",
                    title="Stocks with Most Recommendations",
                    labels={"ticker": "Stock", "count": "Number of Recommendations"},
                    color="count",
                    color_continuous_scale=px.colors.sequential.Blues
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    with rec_tabs[3]:
        st.subheader("Search Recommendations")
        
        search_col1, search_col2 = st.columns([3, 1])
        
        with search_col1:
            search_query = st.text_input("Search by keyword in thesis or risk factors", placeholder="earnings, growth, revenue, competition, etc.")
        
        with search_col2:
            search_category = st.selectbox(
                "Filter by Type",
                ["All"] + st.session_state.recommendation_categories,
                key="search_rec_category"
            )
        
        if st.button("Search Recommendations", use_container_width=True):
            # Collect matching recommendations
            search_results = []
            
            for rec in st.session_state.user_recommendations:
                # Check category filter
                if search_category != "All" and rec['category'] != search_category:
                    continue
                
                # Check keyword match if provided
                if search_query:
                    thesis_match = search_query.lower() in rec['thesis'].lower()
                    risk_match = rec['risk_factors'] and search_query.lower() in rec['risk_factors'].lower()
                    
                    if not (thesis_match or risk_match):
                        continue
                
                # Add to results if it passed all filters
                search_results.append(rec)
            
            # Display results
            if search_results:
                st.success(f"Found {len(search_results)} matching recommendations")
                
                # Sort by most recent
                sorted_results = sorted(search_results, key=lambda x: x['created_at'], reverse=True)
                
                for rec in sorted_results:
                    with st.container():
                        # Header with ticker and recommendation type
                        st.markdown(f"### {rec['ticker']} - {rec['category']}")
                        st.markdown(f"by **{rec['created_by']}** on {rec['created_at']}")
                        
                        # Color-code the recommendation
                        rec_color = {
                            "Strong Buy": "green",
                            "Buy": "lightgreen",
                            "Hold": "gray",
                            "Sell": "orange",
                            "Strong Sell": "red"
                        }.get(rec['category'], "gray")
                        
                        # Main content
                        st.markdown(f"""
                        <div style="background-color:#f9f9f9;padding:15px;border-radius:5px;margin-bottom:10px;border-left:5px solid {rec_color};">
                            <p><b>Price Target:</b> ${rec['price_target']:.2f} | <b>Horizon:</b> {rec['horizon']} | <b>Style:</b> {rec['investment_style']}</p>
                            <h4>Investment Thesis</h4>
                            <p>{rec['thesis']}</p>
                            {f'<h4>Risk Factors</h4><p>{rec["risk_factors"]}</p>' if rec['risk_factors'] else ''}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Voting buttons
                        col1, col2, _ = st.columns([1, 1, 4])
                        with col1:
                            if st.button(f"👍 Agree ({rec['upvotes']})", key=f"search_upvote_rec_{rec['id']}"):
                                upvote_recommendation(rec['id'])
                        
                        with col2:
                            if st.button(f"View All {rec['ticker']} Recs", key=f"search_view_rec_{rec['id']}"):
                                # This would navigate to the ticker's recommendations in a real app
                                st.session_state.rec_ticker_selected = rec['ticker']
                                st.rerun()
                        
                        st.divider()
            else:
                st.info("No matching recommendations found. Try different search terms.")
                
    # Display recent recommendations at the bottom
    st.subheader("Recently Published Recommendations")
    
    # Sort by most recent
    recent_recommendations = sorted(
        st.session_state.user_recommendations, 
        key=lambda x: x['created_at'], 
        reverse=True
    )[:5]  # Get 5 most recent
    
    if recent_recommendations:
        recent_cols = st.columns(min(len(recent_recommendations), 5))
        
        for i, rec in enumerate(recent_recommendations):
            with recent_cols[i]:
                rec_color = {
                    "Strong Buy": "green",
                    "Buy": "lightgreen",
                    "Hold": "gray",
                    "Sell": "orange",
                    "Strong Sell": "red"
                }.get(rec['category'], "gray")
                
                st.markdown(f"**{rec['ticker']}**")
                st.markdown(f"<span style='color:{rec_color};font-weight:bold;'>{rec['category']}</span>", unsafe_allow_html=True)
                st.markdown(f"Target: ${rec['price_target']:.2f}")
                
                if st.button(f"View Details", key=f"recent_rec_{rec['id']}", use_container_width=True):
                    # In a real app, this would navigate to the specific recommendation
                    st.session_state.rec_selected = rec['id']
                    st.rerun()
    else:
        st.info("No recommendations available yet. Be the first to publish a recommendation!")

# Display achievements
def display_achievements():
    """Display user achievements section"""
    st.subheader("Your Achievements")
    
    # Display earned achievements
    earned = []
    unearned = []
    
    # Categorize achievements
    for achievement in st.session_state.available_achievements:
        if achievement['id'] in st.session_state.user_achievements:
            earned.append(achievement)
        else:
            unearned.append(achievement)
    
    # Display achievement stats
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Achievements Earned", f"{len(earned)}/{len(st.session_state.available_achievements)}")
    
    with col2:
        total_points = sum(a['points'] for a in earned)
        max_points = sum(a['points'] for a in st.session_state.available_achievements)
        st.metric("Achievement Points", f"{total_points}/{max_points}")
    
    with col3:
        st.metric("Reputation", st.session_state.user_profile['reputation'])
    
    # Earned achievements
    if earned:
        st.subheader("Earned Achievements")
        for achievement in earned:
            with st.container():
                col1, col2 = st.columns([1, 5])
                
                with col1:
                    st.markdown(f"<h1 style='text-align: center;'>{achievement['icon']}</h1>", unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"**{achievement['name']}** (+{achievement['points']} points)")
                    st.write(achievement['description'])
                
                st.divider()
    
    # Unearned achievements
    if unearned:
        st.subheader("Available Achievements")
        for achievement in unearned:
            with st.container():
                col1, col2 = st.columns([1, 5])
                
                with col1:
                    st.markdown(f"<h1 style='text-align: center; opacity: 0.5;'>{achievement['icon']}</h1>", unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"**{achievement['name']}** (+{achievement['points']} points)")
                    st.write(achievement['description'])
                
                st.divider()

def social_features_section():
    """Main function for the social features page"""
    st.title("📱 Social Hub")
    
    # Initialize social features
    initialize_social_features()
    
    # Create tabs for different social features - simplified structure
    tabs = st.tabs([
        "📁 Profile",           # Profile including Achievements
        "💬 Community"          # Discussions & Leaderboard
    ])
    
    # 📁 Profile Tab (including Achievements)
    with tabs[0]:
        st.header("📁 Profile & Achievements")
        
        # Create Profile and Achievements sections
        profile_tab, achievements_tab = st.tabs(["Profile", "Achievements"])
        
        with profile_tab:
            display_user_profile()
        
        with achievements_tab:
            display_achievements()
    
    # 💬 Community Tab (Discussions + Leaderboard)
    with tabs[1]:
        st.header("💬 Community")
        
        # Create Feed, Discussions, Leaderboard, and Trending sections
        feed_tab, discussions_tab, leaderboard_tab = st.tabs(["Community Feed", "Stock Discussions", "Leaderboard & Trending"])
        
        with feed_tab:
            display_community_feed()
            
        with discussions_tab:
            display_stock_discussions()
        
        with leaderboard_tab:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                display_leaderboard()
            
            with col2:
                display_trending_stocks()
    
    # One-click share section (would use the active chart in a real implementation)
    st.divider()
    display_one_click_share()
