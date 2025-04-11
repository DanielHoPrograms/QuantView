import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import time
import requests
from datetime import datetime
import uuid

# Import from auth UI
from user_auth_ui import display_auth_status, init_auth_session_state

# Helper function for API authentication
def get_auth_headers():
    """
    Get authentication headers with JWT token if available
    
    Returns:
        Dictionary with authorization headers
    """
    headers = {}
    if 'jwt_token' in st.session_state and st.session_state.jwt_token:
        headers["Authorization"] = f"Bearer {st.session_state.jwt_token}"
    return headers
    
def get_user_id():
    """
    Get user ID from session state if available
    
    Returns:
        User ID or None
    """
    if 'user' in st.session_state and st.session_state.user and 'id' in st.session_state.user:
        return st.session_state.user['id']
    return None

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
        # Check if the ticker is in the post's tickers list (case insensitive)
        posts = [p for p in posts if p["tickers"] and any(ticker.upper() == t.upper() for t in p["tickers"])]
    
    if tag:
        posts = [p for p in posts if any(tag.lower() in t.lower() for t in p["tags"])]
    
    if author:
        posts = [p for p in posts if p["author"].lower() == author.lower()]
    
    # Filter by recommendation type (if applicable)
    if recommendation_type:
        # First check in tags
        recommendation_posts = []
        for p in posts:
            # Check in post tags
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
    elif sort_by == "most_liked" or sort_by == "likes":
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
    
    # Display API server status if in developer mode
    if st.session_state.get('developer_mode', False):
        if st.session_state.get('api_server_running', False):
            st.success(f"API Server running on port {st.session_state.get('api_server_port', 'unknown')}")
        else:
            st.error(f"API Server not running: {st.session_state.get('api_server_error', 'Unknown error')}")
            
    # Add JavaScript for API interaction
    # Use a consistent port (5001) for API connections
    api_url = st.session_state.get('api_server_url', 'http://0.0.0.0:5001')
    st.markdown(f"""
    <script>
    // Function to call the posts API with filters
    async function fetchPosts(filters) {{
        try {{
            // Build query string from filters
            const queryParams = new URLSearchParams();
            if (filters.type) queryParams.append('type', filters.type);
            if (filters.ticker) queryParams.append('ticker', filters.ticker);
            if (filters.sort) queryParams.append('sort', filters.sort);
            if (filters.tag) queryParams.append('tag', filters.tag);
            if (filters.author) queryParams.append('author', filters.author);
            if (filters.recommendation_type) queryParams.append('recommendation_type', filters.recommendation_type);
            
            // Call API
            const response = await fetch(`{api_url}/api/posts?${{queryParams.toString()}}`);
            const data = await response.json();
            return data;
        }} catch (error) {{
            console.error('Error fetching posts:', error);
            return {{ success: false, error: error.message }};
        }}
    }}
    
    // Function to like a post
    async function likePost(postId) {{
        try {{
            const response = await fetch(`{api_url}/api/posts/${{postId}}/like`, {{
                method: 'POST',
                headers: {{ 'Content-Type': 'application/json' }}
            }});
            const data = await response.json();
            return data;
        }} catch (error) {{
            console.error('Error liking post:', error);
            return {{ success: false, error: error.message }};
        }}
    }}
    
    // Function to add a comment
    async function addComment(postId, commentText, author) {{
        try {{
            const response = await fetch(`{api_url}/api/posts/${{postId}}/comments`, {{
                method: 'POST',
                headers: {{ 'Content-Type': 'application/json' }},
                body: JSON.stringify({{ text: commentText, author: author }})
            }});
            const data = await response.json();
            return data;
        }} catch (error) {{
            console.error('Error adding comment:', error);
            return {{ success: false, error: error.message }};
        }}
    }}
    
    // Make functions available globally
    window.communityAPI = {{ fetchPosts, likePost, addComment }};
    </script>
    """, unsafe_allow_html=True)
    
    # Create tabs for feed and creating new posts
    feed_tab, create_tab = st.tabs(["Browse Feed", "Create Post"])
    
    with feed_tab:
        # Enhanced Filtering UI
        with st.expander("Filter and Sort Posts", expanded=True):
            # Main Filter Categories
            st.subheader("Filter Options")
            
            # First row: Post Type and Stock Ticker (most common filters)
            filter_col1, filter_col2 = st.columns(2)
            
            with filter_col1:
                # Post Type filter with icons
                filter_type = st.selectbox(
                    "Post Type",
                    options=["All"] + list(POST_TYPES.keys()),
                    format_func=lambda x: "All Types" if x == "All" else f"{POST_TYPES.get(x, {}).get('icon', '')} {POST_TYPES.get(x, {}).get('label', x)}",
                    key="filter_post_type"
                )
            
            with filter_col2:
                # Stock Ticker filter with autocomplete
                ticker_options = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META"]
                # Add tickers from existing posts
                for post in st.session_state.community_posts:
                    if post["tickers"]:
                        ticker_options.extend(post["tickers"])
                # Remove duplicates and sort
                ticker_options = sorted(list(set(ticker_options)))
                
                filter_ticker = st.selectbox(
                    "Stock Ticker",
                    options=[""] + ticker_options,
                    format_func=lambda x: "All Stocks" if x == "" else x,
                    key="filter_ticker"
                )
            
            # Second row: More specific filters
            st.divider()
            st.subheader("Advanced Filters")
            
            filter_col3, filter_col4, filter_col5 = st.columns(3)
            
            with filter_col3:
                # Recommendation Type filter
                recommendation_categories = ["All", "Strong Buy", "Buy", "Hold", "Sell", "Strong Sell"]
                filter_rec_type = st.selectbox(
                    "Recommendation Type",
                    options=recommendation_categories,
                    index=0,
                    key="filter_rec_type"
                )
            
            with filter_col4:
                # Tag filter with common options
                common_tags = ["tech", "earnings", "AI", "semiconductor", "crypto", "market events", "Fed"]
                # Add tags from existing posts (up to a reasonable limit)
                tag_set = set(common_tags)
                for post in st.session_state.community_posts:
                    if post["tags"] and len(tag_set) < 20:  # Limit to 20 options
                        tag_set.update(post["tags"][:2])  # Add up to 2 tags from each post
                tag_options = sorted(list(tag_set))
                
                filter_tag = st.selectbox(
                    "Filter by Tag",
                    options=[""] + tag_options,
                    format_func=lambda x: "All Tags" if x == "" else x,
                    key="filter_tag"
                )
            
            with filter_col5:
                # Author filter with autocomplete from existing authors
                author_set = set()
                for post in st.session_state.community_posts:
                    author_set.add(post["author"])
                author_options = sorted(list(author_set))
                
                filter_author = st.selectbox(
                    "Filter by Author",
                    options=[""] + author_options,
                    format_func=lambda x: "All Authors" if x == "" else x,
                    key="filter_author"
                )
            
            # Third row: Sorting and display options
            st.divider()
            st.subheader("Sort and Display Options")
            
            sort_col1, sort_col2 = st.columns(2)
            
            with sort_col1:
                # Sorting options with icons
                sort_option = st.selectbox(
                    "Sort By",
                    options=["latest", "oldest", "most_liked"],
                    format_func=lambda x: "⏱️ Latest First" if x == "latest" else 
                                         "🕰️ Oldest First" if x == "oldest" else 
                                         "❤️ Most Liked First",
                    key="sort_option"
                )
            
            with sort_col2:
                # Results count
                posts_count = len(st.session_state.community_posts)
                st.write(f"Total Posts: **{posts_count}**")
                
                # Reset filters button
                if st.button("Reset All Filters", use_container_width=True):
                    st.rerun()
        
        # Convert "All" to None for filtering
        filter_type = None if filter_type == "All" else filter_type
        filter_ticker = None if filter_ticker == "" else filter_ticker
        filter_rec_type = None if filter_rec_type == "All" else filter_rec_type
        filter_tag = None if filter_tag == "" else filter_tag
        filter_author = None if filter_author == "" else filter_author
        
        # Add an API Integration section for developer mode
        if st.session_state.get('developer_mode', False):
            with st.expander("API Integration (Developer Mode)", expanded=False):
                st.code("""
# This section uses the API to fetch posts instead of session state
# API endpoint: {api_url}/api/posts

# API query parameters:
# - type: Filter by post_type (recommendation, strategy, watchlist, discussion)
# - ticker: Filter by stock ticker
# - sort: Sort by 'latest' or 'likes' 
# - tag: Filter by tag
# - author: Filter by author
# - recommendation_type: Filter by recommendation type
                """)
        
        # Check if we should use API or not
        use_api = True
        
        # Get filtered posts (either from API or local storage)
        if use_api:
            # Import requests and handle errors
            import requests
            try:
                # Build API URL with query parameters
                api_endpoint = f"http://0.0.0.0:5001/api/posts"
                params = {}
                if filter_type:
                    params['type'] = filter_type
                if filter_ticker:
                    params['ticker'] = filter_ticker
                if filter_tag:
                    params['tag'] = filter_tag
                if filter_author:
                    params['author'] = filter_author
                if filter_rec_type:
                    params['recommendation_type'] = filter_rec_type
                params['sort'] = sort_option
                
                # Make API request
                response = requests.get(api_endpoint, params=params)
                
                # Check if request was successful
                if response.status_code == 200:
                    data = response.json()
                    posts = data.get('posts', [])
                    # Add status indicator for developer mode
                    if st.session_state.get('developer_mode', False):
                        st.success(f"API Request Successful: {len(posts)} posts found")
                else:
                    st.error(f"API Request Failed: {response.status_code} - {response.text}")
                    # Fall back to local data
                    posts = get_posts(
                        post_type=filter_type, 
                        ticker=filter_ticker, 
                        tag=filter_tag,
                        author=filter_author,
                        recommendation_type=filter_rec_type,
                        sort_by=sort_option
                    )
            except Exception as e:
                st.error(f"API Connection Error: {str(e)}. Falling back to local data.")
                # Fall back to local data
                posts = get_posts(
                    post_type=filter_type, 
                    ticker=filter_ticker, 
                    tag=filter_tag,
                    author=filter_author,
                    recommendation_type=filter_rec_type,
                    sort_by=sort_option
                )
        else:
            # Use local data only
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
                    # Get authentication status
                    init_auth_session_state()
                    username, is_logged_in = display_auth_status()
                    
                    # Show like button with login requirement
                    like_btn = st.button(f"👍 {post['likes']}", key=f"like_{post['id']}")
                    if like_btn:
                        if not is_logged_in:
                            # Create a modal-like UI for login prompt
                            with st.container():
                                st.warning("Login to like and comment")
                                col1, col2, col3 = st.columns([1, 1, 1])
                                with col2:
                                    if st.button("Go to Login", key=f"goto_login_{post['id']}"):
                                        # Set session state to navigate to profile/login tab
                                        st.session_state.active_tab = "Profile"
                                        st.session_state.show_login = True
                                        st.rerun()
                        elif use_api:
                            # Use the API to like the post with authentication
                            try:
                                # Get authentication headers
                                headers = get_auth_headers()
                                    
                                response = requests.post(
                                    f"http://0.0.0.0:5001/api/posts/{post['id']}/like",
                                    headers=headers,
                                    json={"user_id": get_user_id()}
                                )
                                
                                if response.status_code == 200:
                                    data = response.json()
                                    st.toast(f"Post liked successfully! Total likes: {data.get('new_likes', post['likes'] + 1)}")
                                else:
                                    st.error(f"Error liking post: {response.status_code}")
                            except Exception as e:
                                st.error(f"API Error: {str(e)}")
                                # Fall back to local like function
                                like_post(post["id"])
                        else:
                            # Use local function
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
                    # Get authentication status
                    init_auth_session_state()
                    username, is_logged_in = display_auth_status()
                    
                    if not is_logged_in:
                        # Create a modal-like UI for login prompt
                        st.warning("Login to like and comment")
                        col1, col2, col3 = st.columns([1, 1, 1])
                        with col2:
                            if st.button("Go to Login", key=f"goto_login_comment_{post['id']}"):
                                # Set session state to navigate to profile/login tab
                                st.session_state.active_tab = "Profile"
                                st.session_state.show_login = True
                                st.rerun()
                    else:
                        comment_text = st.text_area("Your comment", key=f"comment_text_{post['id']}", max_chars=500)
                        if st.button("Post Comment", key=f"post_comment_{post['id']}"):
                            if comment_text.strip():
                                if use_api:
                                    # Use the API to add a comment with authentication
                                    try:
                                        # Use authenticated username
                                        author = username
                                        
                                        # Get authentication headers
                                        headers = get_auth_headers()
                                        
                                        # Make the API request
                                        response = requests.post(
                                            f"http://0.0.0.0:5001/api/posts/{post['id']}/comments",
                                            headers=headers,
                                            json={"text": comment_text, "author": author, "user_id": get_user_id()}
                                        )
                                        
                                        # Check response
                                        if response.status_code == 200:
                                            data = response.json()
                                            st.success("Comment added successfully!")
                                        else:
                                            st.error(f"Error adding comment: {response.status_code}")
                                            # Fall back to local comment function
                                            add_comment(post["id"], comment_text)
                                    except Exception as e:
                                        st.error(f"API Error: {str(e)}")
                                        # Fall back to local comment function
                                        add_comment(post["id"], comment_text)
                            else:
                                # Use local function
                                add_comment(post["id"], comment_text)
                                st.success("Comment added!")
                            
                            time.sleep(1)
                            st.rerun()
                        elif comment_text.strip() == "":
                            st.error("Comment cannot be empty.")
                
                st.divider()
    
    with create_tab:
        # Form to create a new post
        st.subheader("Create a New Post")
        
        # Add Share Your Insights section for creating posts from charts and analyses
        from social_features import display_one_click_share
        with st.expander("📸 Share Your Insights", expanded=False):
            display_one_click_share()
        
        # Get authentication status
        init_auth_session_state()
        username, is_logged_in = display_auth_status()
        
        # Require login to create posts
        if not is_logged_in:
            st.warning("You need to be logged in to create posts.")
            st.info("Please go to the Profile tab and log in or register.")
            return
            
        # Post type selection (only show if logged in)
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
        
        # Use the authenticated username for the author field
        author = username
        st.info(f"You will post as: {author}")
        
        # Hidden field to keep the value in the form
        st.text_input("Author", value=author, key="cf_author_hidden", disabled=True, label_visibility="collapsed")
        
        # Submit button
        if st.button("Create Post", key="cf_submit_button"):
            if not title:
                st.error("Title is required.")
            elif not content:
                st.error("Content is required.")
            elif post_type in ["recommendation", "watchlist"] and not tickers:
                st.error("At least one ticker is required for this post type.")
            else:
                # Create post data for API
                post_data = {
                    "title": title,
                    "content": content,
                    "post_type": post_type,
                    "tickers": tickers,
                    "tags": tags,
                    "author": author
                }
                
                # Add recommendation data if it's a recommendation post
                if post_type == "recommendation" and 'recommendation_data' in locals():
                    post_data["recommendation_data"] = recommendation_data
                
                # Create the post (either through API or locally)
                if use_api:
                    # Use the API to create a post
                    try:
                        # Get authentication headers
                        headers = get_auth_headers()
                        
                        # Add user_id to the post data if available
                        user_id = get_user_id()
                        if user_id:
                            post_data["user_id"] = user_id
                            
                        response = requests.post(
                            f"http://0.0.0.0:5001/api/posts",
                            headers=headers,
                            json=post_data
                        )
                        
                        if response.status_code == 200:
                            data = response.json()
                            post_id = data.get('post_id')
                            st.success("Post created successfully via API!")
                        else:
                            st.error(f"Error creating post via API: {response.status_code} - {response.text}")
                            # Fall back to local function
                            post_id = create_post(
                                title=title,
                                content=content,
                                post_type=post_type,
                                tickers=tickers,
                                tags=tags,
                                author=author,
                                recommendation_data=recommendation_data if post_type == "recommendation" and 'recommendation_data' in locals() else None
                            )
                    except Exception as e:
                        st.error(f"API Error: {str(e)}. Using local data.")
                        # Fall back to local function
                        post_id = create_post(
                            title=title,
                            content=content,
                            post_type=post_type,
                            tickers=tickers,
                            tags=tags,
                            author=author,
                            recommendation_data=recommendation_data if post_type == "recommendation" and 'recommendation_data' in locals() else None
                        )
                else:
                    # Use local function
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
        
