import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import time
from datetime import datetime
import uuid

# Define the post types
POST_TYPES = {
    "recommendation": {"icon": "📊", "label": "Stock Recommendation"},
    "strategy": {"icon": "🔄", "label": "Trading Strategy"},
    "watchlist": {"icon": "📋", "label": "Watchlist"},
    "discussion": {"icon": "💬", "label": "Discussion"}
}

# Initialize community feed state
def initialize_community_feed():
    """Initialize community feed state variables"""
    if 'community_posts' not in st.session_state:
        st.session_state.community_posts = []
    
    # Import existing recommendations if they exist
    if 'user_recommendations' in st.session_state and 'imported_recommendations' not in st.session_state:
        import_existing_recommendations()
        st.session_state.imported_recommendations = True
    
    # Add some example posts if none exist
    if len(st.session_state.community_posts) == 0:
        # Add example posts for demonstration
        add_example_posts()
        
def import_existing_recommendations():
    """Import existing recommendations from user_recommendations to community feed"""
    if not st.session_state.user_recommendations:
        return
        
    for rec in st.session_state.user_recommendations:
        # Convert recommendation to post format
        post_id = str(uuid.uuid4())
        
        # Create a formatted title based on recommendation category
        title = f"{rec['ticker']} - {rec['category']} recommendation (Target: ${rec['price_target']})"
        
        # Format the content with all the recommendation details
        content = f"""
**Investment Thesis:** {rec['thesis']}

**Price Target:** ${rec['price_target']}  
**Investment Horizon:** {rec['horizon']}  
**Investment Style:** {rec['investment_style']}
        """
        
        if rec['risk_factors']:
            content += f"\n\n**Risk Factors:** {rec['risk_factors']}"
            
        # Create post
        post = {
            "id": post_id,
            "title": title,
            "content": content,
            "post_type": "recommendation",
            "tickers": [rec['ticker']],
            "tags": [rec['category'].lower(), rec['horizon'].lower(), rec['investment_style'].lower()],
            "author": rec['created_by'],
            "timestamp": datetime.fromisoformat(rec['created_at'].replace(' ', 'T')).isoformat(),
            "likes": rec['upvotes'],
            "comments": [],
            "is_example": False,
            "recommendation_data": rec  # Store original data for reference
        }
        
        # Add to community posts
        st.session_state.community_posts.append(post)

def add_example_posts():
    """Add example posts for demonstration"""
    # Example 1: Stock Recommendation
    create_post(
        title="NVDA looks bullish ahead of earnings",
        content="Based on my technical analysis, NVDA is showing a strong bullish pattern with increasing volume. RSI is at 62, indicating momentum without being overbought yet.",
        post_type="recommendation",
        tickers=["NVDA"],
        tags=["tech", "semiconductor", "earnings"],
        author="TechTrader",
        is_example=True
    )
    
    # Example 2: Trading Strategy
    create_post(
        title="My RSI + MACD strategy that's been working",
        content="I've been using a combined RSI and MACD strategy that's yielding good results. Buy when RSI crosses above 30 AND MACD line crosses above signal line. Sell when RSI crosses above 70.",
        post_type="strategy",
        tickers=[],
        tags=["RSI", "MACD", "technical analysis"],
        author="IndicatorExpert",
        is_example=True
    )
    
    # Example 3: Watchlist
    create_post(
        title="My AI & Chip Stocks Watchlist",
        content="Here's my watchlist of AI and semiconductor companies that I'm tracking: NVDA, AMD, INTC, TSM, AVGO, ARM",
        post_type="watchlist",
        tickers=["NVDA", "AMD", "INTC", "TSM", "AVGO", "ARM"],
        tags=["AI", "semiconductor", "tech"],
        author="ChipWatcher",
        is_example=True
    )
    
    # Example 4: Discussion
    create_post(
        title="How are you planning for upcoming Fed rate decision?",
        content="With the Fed meeting next week, I'm wondering how everyone is positioning their portfolios? Are you hedging or staying the course?",
        post_type="discussion",
        tickers=[],
        tags=["Fed", "interest rates", "market events"],
        author="MacroViewer",
        is_example=True
    )

# Create a post
def create_post(title, content, post_type, tickers=None, tags=None, author=None, is_example=False, recommendation_data=None):
    """
    Create a new post in the community feed
    
    Args:
        title: Post title
        content: Post content
        post_type: Type of post (recommendation, strategy, watchlist, discussion)
        tickers: List of stock tickers related to the post
        tags: List of tags for the post
        author: Author of the post (defaults to username or 'Anonymous')
        is_example: Whether this is an example post
        recommendation_data: Optional detailed recommendation data for recommendation posts
        
    Returns:
        Post ID
    """
    # Initialize tickers and tags if None
    tickers = tickers or []
    tags = tags or []
    
    # Generate post ID
    post_id = str(uuid.uuid4())
    
    # Set author (would normally get from user account)
    if author is None:
        if 'username' in st.session_state:
            author = st.session_state.username
        else:
            author = "Anonymous"
    
    # Create post data
    post = {
        "id": post_id,
        "title": title,
        "content": content,
        "post_type": post_type,
        "tickers": tickers,
        "tags": tags,
        "author": author,
        "timestamp": datetime.now().isoformat(),
        "likes": 0,
        "comments": [],
        "is_example": is_example
    }
    
    # Add recommendation data if provided
    if recommendation_data and post_type == "recommendation":
        post["recommendation_data"] = recommendation_data
        
        # If this is a valid recommendation, also create a traditional recommendation
        if 'user_recommendations' in st.session_state and len(tickers) > 0:
            recommendation_id = str(uuid.uuid4())
            
            # Create the recommendation data structure
            rec_data = {
                'id': recommendation_id,
                'ticker': tickers[0],  # First ticker
                'category': recommendation_data.get('category', 'Hold'),
                'price_target': recommendation_data.get('price_target', 0.0),
                'horizon': recommendation_data.get('horizon', 'Medium-Term'),
                'investment_style': recommendation_data.get('investment_style', 'Value Investor'),
                'thesis': content,
                'risk_factors': recommendation_data.get('risk_factors', ''),
                'created_by': author,
                'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'upvotes': 0,
                'downvotes': 0,
                'comments': []
            }
            
            # Add to user recommendations
            st.session_state.user_recommendations.append(rec_data)
            
            # Update trending stocks
            if tickers[0] in st.session_state.trending_stocks:
                st.session_state.trending_stocks[tickers[0]] += 1
            else:
                st.session_state.trending_stocks[tickers[0]] = 1
            
            # Check for recommendation achievements if the function exists
            if 'user_profile' in st.session_state:
                # We'll just increment the reputation here since we don't have direct access
                # to the check_recommendation_achievements function
                st.session_state.user_profile['reputation'] += 5
    
    # Add post to session state
    st.session_state.community_posts.append(post)
    
    # Increment reputation points if available
    if 'user_profile' in st.session_state:
        st.session_state.user_profile['reputation'] = st.session_state.user_profile.get('reputation', 0) + 5
        st.session_state.user_profile['shared_items'] = st.session_state.user_profile.get('shared_items', 0) + 1
    
    return post_id

# Get all posts with optional filtering
def get_posts(post_type=None, ticker=None, tag=None, author=None, recommendation_type=None, sort_by="latest"):
    """
    Get all posts with optional filtering
    
    Args:
        post_type: Filter by post type (recommendation, strategy, watchlist, discussion)
        ticker: Filter by stock ticker
        tag: Filter by tag
        author: Filter by author
        recommendation_type: Filter by recommendation type (Buy, Sell, Hold, etc.)
        sort_by: How to sort posts (latest, oldest, most_liked)
        
    Returns:
        List of filtered posts
    """
    # Start with all posts
    posts = st.session_state.community_posts
    
    # Apply filters
    if post_type:
        posts = [p for p in posts if p["post_type"] == post_type]
    
    if ticker:
        posts = [p for p in posts if any(ticker.upper() in t.upper() for t in p["tickers"])]
    
    if tag:
        posts = [p for p in posts if any(tag.lower() in t.lower() for t in p["tags"])]
    
    if author:
        posts = [p for p in posts if p["author"].lower() == author.lower()]
    
    # Filter by recommendation type (if applicable)
    if recommendation_type and post_type == "recommendation":
        # First check in tags
        recommendation_posts = []
        for p in posts:
            if any(recommendation_type.lower() in t.lower() for t in p["tags"]):
                recommendation_posts.append(p)
            # Also check in recommendation_data if available
            elif "recommendation_data" in p and p["recommendation_data"].get("category", "").lower() == recommendation_type.lower():
                recommendation_posts.append(p)
        posts = recommendation_posts
    
    # Sort posts
    if sort_by == "latest":
        posts = sorted(posts, key=lambda p: p["timestamp"], reverse=True)
    elif sort_by == "oldest":
        posts = sorted(posts, key=lambda p: p["timestamp"])
    elif sort_by == "most_liked":
        posts = sorted(posts, key=lambda p: p["likes"], reverse=True)
    
    return posts

# Like a post
def like_post(post_id):
    """
    Like a post
    
    Args:
        post_id: ID of post to like
        
    Returns:
        True if successful, False otherwise
    """
    for post in st.session_state.community_posts:
        if post["id"] == post_id:
            post["likes"] += 1
            return True
    
    return False

# Add a comment to a post
def add_comment(post_id, comment_text, author=None):
    """
    Add a comment to a post
    
    Args:
        post_id: ID of post to comment on
        comment_text: Comment text
        author: Author of the comment (defaults to username or 'Anonymous')
        
    Returns:
        True if successful, False otherwise
    """
    # Set author (would normally get from user account)
    if author is None:
        if 'username' in st.session_state:
            author = st.session_state.username
        else:
            author = "Anonymous"
            
    comment = {
        "id": str(uuid.uuid4()),
        "text": comment_text,
        "author": author,
        "timestamp": datetime.now().isoformat()
    }
    
    for post in st.session_state.community_posts:
        if post["id"] == post_id:
            post["comments"].append(comment)
            return True
    
    return False

# Display the community feed
def display_community_feed():
    """Display the community feed with filtering options"""
    # Initialize community feed
    initialize_community_feed()
    
    # Add title and description
    st.header("🌐 Community Feed")
    st.write("Share and discover stock insights, trading strategies, and market discussions.")
    
    # Create tabs for feed and creating new posts
    feed_tab, create_tab = st.tabs(["Browse Feed", "Create Post"])
    
    with feed_tab:
        # Filtering options
        with st.expander("Filter posts", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                filter_type = st.selectbox(
                    "Post Type",
                    options=["All"] + list(POST_TYPES.keys()),
                    format_func=lambda x: "All Types" if x == "All" else POST_TYPES.get(x, {}).get("label", x)
                )
            
            with col2:
                filter_ticker = st.text_input("Stock Ticker", placeholder="e.g., AAPL")
            
            # Additional filters based on selected post type
            if filter_type == "recommendation" or filter_type == "All":
                col3, col4 = st.columns(2)
                
                with col3:
                    # Only show if we're filtering recommendations
                    recommendation_categories = ["All", "Strong Buy", "Buy", "Hold", "Sell", "Strong Sell"]
                    filter_rec_type = st.selectbox(
                        "Recommendation Type",
                        options=recommendation_categories,
                        index=0
                    )
                
                with col4:
                    sort_option = st.selectbox(
                        "Sort By",
                        options=["latest", "oldest", "most_liked"],
                        format_func=lambda x: "Latest" if x == "latest" else "Oldest" if x == "oldest" else "Most Liked"
                    )
            else:
                # For non-recommendation filters, just show sort option in full width
                sort_option = st.selectbox(
                    "Sort By",
                    options=["latest", "oldest", "most_liked"],
                    format_func=lambda x: "Latest" if x == "latest" else "Oldest" if x == "oldest" else "Most Liked"
                )
                filter_rec_type = "All"  # Default value
            
            # Add tag and author filters
            col5, col6 = st.columns(2)
            
            with col5:
                filter_tag = st.text_input("Filter by Tag", placeholder="e.g., earnings, tech")
            
            with col6:
                filter_author = st.text_input("Filter by Author", placeholder="Enter username")
        
        # Convert "All" to None for filtering
        filter_type = None if filter_type == "All" else filter_type
        filter_ticker = None if filter_ticker == "" else filter_ticker
        filter_rec_type = None if filter_rec_type == "All" else filter_rec_type
        filter_tag = None if filter_tag == "" else filter_tag
        filter_author = None if filter_author == "" else filter_author
        
        # Get filtered posts
        posts = get_posts(
            post_type=filter_type, 
            ticker=filter_ticker, 
            tag=filter_tag,
            author=filter_author,
            recommendation_type=filter_rec_type,
            sort_by=sort_option
        )
        
        if not posts:
            st.info("No posts found. Be the first to share something with the community!")
        
        # Display posts
        for post in posts:
            with st.container():
                # Post header
                header_col1, header_col2 = st.columns([3, 1])
                
                with header_col1:
                    # Post type and title
                    post_type_info = POST_TYPES.get(post["post_type"], {})
                    post_icon = post_type_info.get("icon", "📄")
                    post_type_label = post_type_info.get("label", post["post_type"].capitalize())
                    
                    st.markdown(f"### {post_icon} {post['title']}")
                    st.caption(f"{post_type_label} by **{post['author']}** • {format_timestamp(post['timestamp'])}")
                
                with header_col2:
                    # Likes
                    like_btn = st.button(f"👍 {post['likes']}", key=f"like_{post['id']}")
                    if like_btn:
                        like_post(post["id"])
                        st.rerun()
                
                # Post content
                st.markdown(post["content"])
                
                # Tickers and tags
                if post["tickers"]:
                    st.markdown(f"**Tickers:** {', '.join([f'`{ticker}`' for ticker in post['tickers']])}")
                
                if post["tags"]:
                    st.markdown(f"**Tags:** {', '.join([f'`{tag}`' for tag in post['tags']])}")
                
                # Comments
                if post["comments"]:
                    with st.expander(f"Show {len(post['comments'])} comments"):
                        for comment in post["comments"]:
                            st.markdown(f"**{comment['author']}** ({format_timestamp(comment['timestamp'])}): {comment['text']}")
                
                # Add comment
                with st.expander("Add a comment"):
                    comment_text = st.text_area("Your comment", key=f"comment_text_{post['id']}", max_chars=500)
                    if st.button("Post Comment", key=f"post_comment_{post['id']}"):
                        if comment_text.strip():
                            add_comment(post["id"], comment_text)
                            st.success("Comment added!")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("Comment cannot be empty.")
                
                st.divider()
    
    with create_tab:
        # Form to create a new post
        st.subheader("Create a New Post")
        
        # Post type selection
        post_type = st.selectbox(
            "Post Type",
            options=list(POST_TYPES.keys()),
            format_func=lambda x: POST_TYPES.get(x, {}).get("label", x)
        )
        
        # Different form based on post type
        if post_type == "recommendation":
            # For recommendation posts, use a more detailed form similar to the original
            # Get recommendation categories from session state if available
            recommendation_categories = ["Strong Buy", "Buy", "Hold", "Sell", "Strong Sell"]
            if 'recommendation_categories' in st.session_state:
                recommendation_categories = st.session_state.recommendation_categories
                
            investment_horizons = ["Day Trade", "Swing Trade", "Short-Term (< 3 months)", 
                                "Medium-Term (3-12 months)", "Long-Term (> 1 year)"]
            if 'investment_horizons' in st.session_state:
                investment_horizons = st.session_state.investment_horizons
                
            investor_styles = ["Value Investor", "Growth Investor", "Income Investor", 
                            "Momentum Trader", "Technical Trader", "Passive Investor"]
            if 'investor_styles' in st.session_state:
                investor_styles = st.session_state.investor_styles
            
            # Stock ticker (single for recommendations)
            ticker_options = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META"]
            if 'selected_stocks' in st.session_state and st.session_state.selected_stocks:
                ticker_options = st.session_state.selected_stocks
                
            ticker = st.selectbox("Stock Symbol", ticker_options, key="cf_rec_ticker_select")
            tickers = [ticker] if ticker else []
            
            col1, col2 = st.columns(2)
            
            with col1:
                category = st.selectbox("Recommendation", recommendation_categories, key="cf_recommendation_category")
                price_target = st.number_input("Price Target ($)", min_value=0.01, step=0.01, value=100.00, key="cf_price_target")
            
            with col2:
                horizon = st.selectbox("Investment Horizon", investment_horizons, key="cf_investment_horizon")
                investment_style = st.selectbox("Investment Style", investor_styles, key="cf_investment_style")
            
            # Investment thesis becomes the content
            content = st.text_area("Investment Thesis", 
                            placeholder="Explain your reasoning for this recommendation...",
                            height=150, max_chars=2000,
                            key="cf_investment_thesis")
            
            risk_factors = st.text_area("Risk Factors (Optional)",
                                placeholder="Outline potential risks to your recommendation...",
                                key="cf_risk_factors")
            
            # Auto-generate title based on inputs
            title = f"{ticker} - {category} recommendation (Target: ${price_target})"
            
            # Auto-generate tags from recommendation fields
            tags = [category.lower(), horizon.lower().split(' ')[0], investment_style.lower().split(' ')[0]]
            
            # Store additional recommendation data
            recommendation_data = {
                "category": category,
                "price_target": price_target,
                "horizon": horizon,
                "investment_style": investment_style,
                "risk_factors": risk_factors
            }
            
        else:
            # Standard post creation form for other post types
            # Post title
            title = st.text_input("Title", max_chars=100, key="cf_other_title")
            
            # Post content
            content = st.text_area("Content", height=150, max_chars=2000, key="cf_other_content")
            
            # Post tickers (if applicable)
            if post_type in ["watchlist"]:
                tickers_input = st.text_input(
                    "Stock Tickers (comma-separated)", 
                    placeholder="e.g., AAPL, MSFT, GOOGL",
                    key="cf_tickers_input"
                )
                tickers = [t.strip().upper() for t in tickers_input.split(",")] if tickers_input else []
            else:
                tickers = []
            
            # Post tags
            tags_input = st.text_input(
                "Tags (comma-separated)", 
                placeholder="e.g., tech, earnings, analysis",
                key="cf_tags_input"
            )
            tags = [t.strip().lower() for t in tags_input.split(",")] if tags_input else []
            
            # Reset recommendation data
            recommendation_data = None
        
        # Author (would normally be tied to user account)
        if 'username' not in st.session_state:
            st.session_state.username = "User" + str(int(time.time()))[-4:]
            
        author = st.text_input("Author", value=st.session_state.username, key="cf_author")
        
        # Submit button
        if st.button("Create Post", key="cf_submit_button"):
            if not title:
                st.error("Title is required.")
            elif not content:
                st.error("Content is required.")
            elif post_type in ["recommendation", "watchlist"] and not tickers:
                st.error("At least one ticker is required for this post type.")
            else:
                # Pass recommendation data if it's a recommendation post
                if post_type == "recommendation" and 'recommendation_data' in locals():
                    post_id = create_post(
                        title=title,
                        content=content,
                        post_type=post_type,
                        tickers=tickers,
                        tags=tags,
                        author=author,
                        recommendation_data=recommendation_data
                    )
                else:
                    post_id = create_post(
                        title=title,
                        content=content,
                        post_type=post_type,
                        tickers=tickers,
                        tags=tags,
                        author=author
                    )
                st.success("Post created successfully!")
                time.sleep(1)
                st.rerun()

# Format timestamp for display
def format_timestamp(iso_timestamp):
    """
    Format ISO timestamp to a readable format
    
    Args:
        iso_timestamp: ISO format timestamp
        
    Returns:
        Formatted timestamp string
    """
    try:
        dt = datetime.fromisoformat(iso_timestamp)
        now = datetime.now()
        delta = now - dt
        
        if delta.days > 30:
            return dt.strftime("%b %d, %Y")
        elif delta.days > 0:
            return f"{delta.days} day{'s' if delta.days > 1 else ''} ago"
        elif delta.seconds >= 3600:
            hours = delta.seconds // 3600
            return f"{hours} hour{'s' if hours > 1 else ''} ago"
        elif delta.seconds >= 60:
            minutes = delta.seconds // 60
            return f"{minutes} minute{'s' if minutes > 1 else ''} ago"
        else:
            return "just now"
    except:
        return iso_timestamp
