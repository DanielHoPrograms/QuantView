import os
import json
from flask import Flask, request, jsonify, session
from datetime import datetime, timedelta
import uuid
import streamlit as st
from community_feed import get_posts, create_post, like_post, add_comment
from threading import Thread
import time
import secrets

from models import db, User, Post, Comment, init_db
from auth import init_auth

# Create Flask app
app = Flask(__name__)

# Configure app
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', secrets.token_hex(16))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///stockmarket.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['JWT_SECRET_KEY'] = os.environ.get('JWT_SECRET_KEY', secrets.token_hex(16))
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(days=7)
app.config['SESSION_TYPE'] = 'filesystem'

# Initialize database and auth
init_db(app)
init_auth(app)

# Make sure Flask server can access Streamlit's session state
def initialize_session_state():
    """
    Initialize session state if running outside of Streamlit context
    This is necessary because the Flask server runs in a different thread
    """
    # Initialize example posts for the API server
    example_posts = [
        {
            "id": "1",
            "title": "NVDA looks bullish ahead of earnings",
            "content": "Based on my technical analysis, NVDA is showing a strong bullish pattern with increasing volume. RSI is at 62, indicating momentum without being overbought yet.",
            "post_type": "recommendation",
            "tickers": ["NVDA"],
            "tags": ["tech", "semiconductor", "earnings"],
            "author": "TechTrader",
            "timestamp": "2025-04-11T03:30:00",
            "likes": 15,
            "comments": []
        },
        {
            "id": "2",
            "title": "My RSI + MACD strategy that's been working",
            "content": "I've been using a combined RSI and MACD strategy that's yielding good results. Buy when RSI crosses above 30 AND MACD line crosses above signal line. Sell when RSI crosses above 70.",
            "post_type": "strategy",
            "tickers": [],
            "tags": ["RSI", "MACD", "technical analysis"],
            "author": "IndicatorExpert",
            "timestamp": "2025-04-10T14:15:00",
            "likes": 8,
            "comments": []
        },
        {
            "id": "3",
            "title": "My AI & Chip Stocks Watchlist",
            "content": "Here's my watchlist of AI and semiconductor companies that I'm tracking: NVDA, AMD, INTC, TSM, AVGO, ARM",
            "post_type": "watchlist",
            "tickers": ["NVDA", "AMD", "INTC", "TSM", "AVGO", "ARM"],
            "tags": ["AI", "semiconductor", "tech"],
            "author": "ChipWatcher",
            "timestamp": "2025-04-09T09:45:00",
            "likes": 12,
            "comments": []
        }
    ]
    
    # Use an in-memory variable for the Flask server
    global api_posts
    api_posts = example_posts

# Initialize posts for API
api_posts = []
initialize_session_state()

@app.route('/api/posts', methods=['GET'])
def get_post_list():
    """
    API endpoint to get posts with filtering options
    Query parameters:
    - type: Filter by post_type (recommendation, strategy, watchlist, discussion)
    - ticker: Filter by stock ticker
    - sort: Sort by 'latest' or 'likes'
    - tag: Filter by tag
    - author: Filter by author
    - recommendation_type: Filter by recommendation type (Buy, Sell, etc.)
    - user_id: Filter by user ID
    """
    # Get query parameters
    post_type = request.args.get('type')
    ticker = request.args.get('ticker')
    sort_by = request.args.get('sort', 'latest')
    tag = request.args.get('tag')
    author = request.args.get('author')
    recommendation_type = request.args.get('recommendation_type')
    user_id = request.args.get('user_id')
    
    # Validate sort parameter
    if sort_by not in ['latest', 'oldest', 'likes', 'most_liked']:
        sort_by = 'latest'
    
    # Map 'likes' to 'most_liked' for compatibility
    if sort_by == 'likes':
        sort_by = 'most_liked'
        
    try:
        # Start with a base query
        query = Post.query
        
        # Apply filters on database model
        if post_type:
            query = query.filter_by(post_type=post_type)
            
        if ticker:
            # Filter for posts containing the ticker in their tickers list
            query = query.filter(Post._tickers.like(f'%"{ticker.upper()}"%'))
            
        if tag:
            # Filter for posts containing the tag in their tags list
            query = query.filter(Post._tags.like(f'%"{tag.lower()}"%'))
            
        if author:
            # Join with User model to filter by author's username
            query = query.join(User, Post.user_id == User.id)
            query = query.filter(User.username.ilike(f'%{author}%'))
            
        if user_id:
            # Filter by user ID
            query = query.filter_by(user_id=user_id)
            
        if recommendation_type:
            # This is more complex as it requires checking JSON fields
            query = query.filter(Post._recommendation_data.like(f'%"category":"{recommendation_type}"%'))
            
        # Apply sorting
        if sort_by == "latest":
            query = query.order_by(Post.timestamp.desc())
        elif sort_by == "oldest":
            query = query.order_by(Post.timestamp.asc())
        elif sort_by in ["most_liked", "likes"]:
            query = query.order_by(Post.likes.desc())
            
        # Execute query
        posts = query.all()
        
        # Convert to dictionaries for JSON response
        post_dicts = [post.to_dict() for post in posts]
        
        return jsonify({
            'success': True,
            'posts': post_dicts,
            'count': len(post_dicts),
            'filters': {
                'post_type': post_type,
                'ticker': ticker,
                'sort_by': sort_by,
                'tag': tag,
                'author': author,
                'recommendation_type': recommendation_type,
                'user_id': user_id
            }
        })
        
    except Exception as e:
        print(f"Database error: {str(e)}")
        
        # Fall back to in-memory filtering if database query fails
        filtered_posts = api_posts.copy()
        
        # Apply filters
        if post_type:
            filtered_posts = [p for p in filtered_posts if p["post_type"] == post_type]
        
        if ticker:
            filtered_posts = [p for p in filtered_posts if "tickers" in p and ticker.upper() in [t.upper() for t in p["tickers"]]]
        
        if tag:
            filtered_posts = [p for p in filtered_posts if "tags" in p and any(tag.lower() in t.lower() for t in p["tags"])]
        
        if author:
            filtered_posts = [p for p in filtered_posts if "author" in p and p["author"].lower() == author.lower()]
        
        if recommendation_type:
            rec_posts = []
            for p in filtered_posts:
                if "tags" in p and any(recommendation_type.lower() in t.lower() for t in p["tags"]):
                    rec_posts.append(p)
                elif "recommendation_data" in p and p["recommendation_data"].get("category", "").lower() == recommendation_type.lower():
                    rec_posts.append(p)
            filtered_posts = rec_posts
        
        # Sort posts
        if sort_by == "latest":
            filtered_posts = sorted(filtered_posts, key=lambda p: p["timestamp"], reverse=True)
        elif sort_by == "oldest":
            filtered_posts = sorted(filtered_posts, key=lambda p: p["timestamp"])
        elif sort_by in ["most_liked", "likes"]:
            filtered_posts = sorted(filtered_posts, key=lambda p: p["likes"], reverse=True)
        
        posts = filtered_posts
        
        # Convert datetime objects to strings
        for post in posts:
            # Ensure timestamp is a string
            if isinstance(post.get('timestamp'), datetime):
                post['timestamp'] = post['timestamp'].isoformat()
        
        return jsonify({
            'success': True,
            'posts': posts,
            'count': len(posts),
            'filters': {
                'post_type': post_type,
                'ticker': ticker,
                'sort_by': sort_by,
                'tag': tag,
                'author': author,
                'recommendation_type': recommendation_type,
                'user_id': user_id
            },
            'using_fallback': True
        })

@app.route('/api/posts', methods=['POST'])
def create_new_post():
    """
    API endpoint to create a new post
    Required fields:
    - title: Post title
    - content: Post content
    - post_type: Type of post (recommendation, strategy, watchlist, discussion)
    Optional fields:
    - tickers: List of stock tickers related to the post
    - tags: List of tags for the post
    - user_id: ID of the user creating the post
    - jwt_token: JWT token for authentication (alternative to user_id)
    - recommendation_data: Optional detailed recommendation data for recommendation posts
    """
    # Get request data
    data = request.json
    
    # Check required fields
    if not data.get('title'):
        return jsonify({'success': False, 'error': 'Title is required'})
    
    if not data.get('content'):
        return jsonify({'success': False, 'error': 'Content is required'})
    
    if not data.get('post_type'):
        return jsonify({'success': False, 'error': 'Post type is required'})
    
    # Generate a new post ID
    post_id = str(uuid.uuid4())
    
    # Set default values
    tickers = data.get('tickers', [])
    tags = data.get('tags', [])
    user_id = data.get('user_id')
    recommendation_data = data.get('recommendation_data')
    
    # Get user if provided
    user = None
    if user_id:
        user = User.query.get(user_id)
    
    # Try to extract user from JWT token
    jwt_token = data.get('jwt_token')
    if not user and jwt_token:
        from flask_jwt_extended import decode_token
        try:
            decoded = decode_token(jwt_token)
            user_identity = decoded['sub']
            user = User.query.get(user_identity['user_id'])
        except Exception as e:
            print(f"JWT Token error: {str(e)}")
    
    try:
        # Create database model instance
        new_post = Post(
            id=post_id,
            title=data.get('title'),
            content=data.get('content'),
            post_type=data.get('post_type'),
            user_id=user.id if user else None,
            likes=0
        )
        
        # Set JSON fields
        new_post.tickers = tickers
        new_post.tags = tags
        
        # Add recommendation data if provided
        if recommendation_data and data.get('post_type') == 'recommendation':
            new_post.recommendation_data = recommendation_data
        
        # Save to database
        db.session.add(new_post)
        db.session.commit()
        
        # Return the post data
        return jsonify({
            'success': True,
            'post_id': post_id,
            'message': 'Post created successfully in database',
            'post': new_post.to_dict()
        })
        
    except Exception as e:
        print(f"Database error when creating post: {str(e)}")
        db.session.rollback()
        
        # Fall back to in-memory storage
        # Create the post object for in-memory fallback
        author = user.username if user else data.get('author', 'Anonymous')
        fallback_post = {
            "id": post_id,
            "title": data.get('title'),
            "content": data.get('content'),
            "post_type": data.get('post_type'),
            "tickers": tickers,
            "tags": tags,
            "author": author,
            "timestamp": datetime.now().isoformat(),
            "likes": 0,
            "comments": [],
            "user_id": user.id if user else None
        }
        
        # Add recommendation data if provided and it's a recommendation post
        if recommendation_data and data.get('post_type') == 'recommendation':
            fallback_post["recommendation_data"] = recommendation_data
        
        # Add the post to our in-memory data store
        api_posts.append(fallback_post)
        
        return jsonify({
            'success': True,
            'post_id': post_id,
            'message': 'Post created successfully (using fallback)',
            'post': fallback_post,
            'using_fallback': True
        })

@app.route('/api/posts/<post_id>/like', methods=['POST'])
def like_post_endpoint(post_id):
    """
    API endpoint to like a post
    Optional fields:
    - user_id: ID of the user liking the post (for tracking who liked what)
    - jwt_token: JWT token for authentication (alternative to user_id)
    """
    # Get user info if provided
    data = request.json or {}
    user_id = data.get('user_id')
    
    # Get user if provided
    user = None
    if user_id:
        user = User.query.get(user_id)
    
    # Try to extract user from JWT token
    jwt_token = data.get('jwt_token')
    if not user and jwt_token:
        from flask_jwt_extended import decode_token
        try:
            decoded = decode_token(jwt_token)
            user_identity = decoded['sub']
            user = User.query.get(user_identity['user_id'])
        except Exception as e:
            print(f"JWT Token error: {str(e)}")
    
    try:
        # Find post in database
        post = Post.query.get(post_id)
        
        if post:
            # Increment likes count
            post.likes += 1
            
            # Save changes
            db.session.commit()
            
            return jsonify({
                'success': True,
                'message': 'Post liked successfully in database',
                'new_likes': post.likes,
                'post': post.to_dict()
            })
        else:
            # Try to find in the in-memory store as fallback
            success = False
            for post in api_posts:
                if post["id"] == post_id:
                    post["likes"] += 1
                    success = True
                    new_likes = post["likes"]
                    break
            
            if success:
                return jsonify({
                    'success': True,
                    'message': 'Post liked successfully (using fallback)',
                    'new_likes': new_likes,
                    'using_fallback': True
                })
            else:
                return jsonify({
                    'success': False,
                    'error': 'Post not found'
                })
    
    except Exception as e:
        print(f"Database error when liking post: {str(e)}")
        db.session.rollback()
        
        # Try the in-memory fallback
        success = False
        for post in api_posts:
            if post["id"] == post_id:
                post["likes"] += 1
                success = True
                new_likes = post["likes"]
                break
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Post liked successfully (using fallback)',
                'new_likes': new_likes,
                'using_fallback': True
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Post not found'
            })

@app.route('/api/posts/<post_id>/comments', methods=['POST'])
def add_comment_endpoint(post_id):
    """
    API endpoint to add a comment to a post
    Required fields:
    - text: Comment text
    Optional fields:
    - user_id: ID of the user creating the comment
    - jwt_token: JWT token for authentication (alternative to user_id)
    - author: Author name if not registered user
    """
    # Get request data
    data = request.json
    
    # Check required fields
    if not data.get('text'):
        return jsonify({'success': False, 'error': 'Comment text is required'})
    
    # Generate a comment ID
    comment_id = str(uuid.uuid4())
    
    # Get user info if provided
    user_id = data.get('user_id')
    
    # Get user if provided
    user = None
    if user_id:
        user = User.query.get(user_id)
    
    # Try to extract user from JWT token
    jwt_token = data.get('jwt_token')
    if not user and jwt_token:
        from flask_jwt_extended import decode_token
        try:
            decoded = decode_token(jwt_token)
            user_identity = decoded['sub']
            user = User.query.get(user_identity['user_id'])
        except Exception as e:
            print(f"JWT Token error: {str(e)}")
    
    # Set author name (username if user exists, otherwise from request or default)
    author_name = user.username if user else data.get('author', 'Anonymous')
    
    try:
        # Find post in database
        post = Post.query.get(post_id)
        
        if post:
            # Create a new comment in the database
            new_comment = Comment(
                id=comment_id,
                text=data.get('text'),
                post_id=post_id,
                user_id=user.id if user else None
            )
            
            # Add and commit to database
            db.session.add(new_comment)
            db.session.commit()
            
            # Return the comment data
            return jsonify({
                'success': True,
                'message': 'Comment added successfully in database',
                'comment': new_comment.to_dict()
            })
        else:
            # Try to find in the in-memory store as fallback
            success = False
            for post in api_posts:
                if post["id"] == post_id:
                    # Create new comment for in-memory fallback
                    comment = {
                        "id": comment_id,
                        "text": data.get('text'),
                        "author": author_name,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    # Add comment to post
                    if "comments" not in post:
                        post["comments"] = []
                    post["comments"].append(comment)
                    success = True
                    break
            
            if success:
                return jsonify({
                    'success': True,
                    'message': 'Comment added successfully (using fallback)',
                    'comment': comment,
                    'using_fallback': True
                })
            else:
                return jsonify({
                    'success': False,
                    'error': 'Post not found'
                })
    
    except Exception as e:
        print(f"Database error when adding comment: {str(e)}")
        db.session.rollback()
        
        # Try to find in the in-memory store as fallback
        success = False
        for post in api_posts:
            if post["id"] == post_id:
                # Create new comment for in-memory fallback
                comment = {
                    "id": comment_id,
                    "text": data.get('text'),
                    "author": author_name,
                    "timestamp": datetime.now().isoformat()
                }
                
                # Add comment to post
                if "comments" not in post:
                    post["comments"] = []
                post["comments"].append(comment)
                success = True
                break
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Comment added successfully (using fallback)',
                'comment': comment,
                'using_fallback': True
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Post not found or database error'
            })

def find_available_port(start_port=5001, max_port=5100):
    """
    Find an available port to use for the Flask server
    """
    # For consistency, always try to use port 5001 first
    # This helps the client code know which port to connect to
    fixed_port = 5001
    
    import socket
    
    # First try the fixed port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('0.0.0.0', fixed_port))
            return fixed_port  # Fixed port is available
        except socket.error:
            # If fixed port is not available, find another port
            pass
    
    # If fixed port is not available, find another port
    for port in range(start_port, max_port):
        if port == fixed_port:
            continue  # Already tried this port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('0.0.0.0', port))
                return port  # Port is available
            except socket.error:
                continue  # Port is in use, try the next one
    
    # If we get here, no ports were available
    raise RuntimeError(f"Could not find an available port in range {start_port}-{max_port}")

def start_flask_server(port=None):
    """
    Start Flask server on a separate thread
    """
    if port is None:
        try:
            port = find_available_port()
        except RuntimeError as e:
            print(f"Warning: {str(e)}")
            return
    
    try:
        app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
    except Exception as e:
        print(f"Flask server error: {str(e)}")

def run_flask_server():
    """
    Start Flask server in a separate thread
    """
    try:
        port = find_available_port()
        Thread(target=lambda: start_flask_server(port), daemon=True).start()
        print(f"Flask server started on http://0.0.0.0:{port}")
        return port
    except Exception as e:
        print(f"Could not start Flask server: {str(e)}")
        return None

# Run the Flask server when this module is imported
if __name__ == '__main__':
    run_flask_server()
