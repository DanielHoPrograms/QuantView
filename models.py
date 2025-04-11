import os
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from flask_bcrypt import Bcrypt
from datetime import datetime
import json

# Initialize extensions
db = SQLAlchemy()
bcrypt = Bcrypt()

# User model for authentication
class User(db.Model, UserMixin):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    reputation = db.Column(db.Integer, default=0)
    
    # New fields for enhanced user profile
    avatar = db.Column(db.Text, nullable=True)  # Base64 encoded image
    reset_token = db.Column(db.String(100), nullable=True)
    reset_token_expires = db.Column(db.DateTime, nullable=True)
    email_verified = db.Column(db.Boolean, default=False)
    verification_code = db.Column(db.String(10), nullable=True)
    level = db.Column(db.Integer, default=1)
    
    # User statistics
    posts_count = db.Column(db.Integer, default=0)
    comments_count = db.Column(db.Integer, default=0)
    likes_given = db.Column(db.Integer, default=0)
    likes_received = db.Column(db.Integer, default=0)
    
    # Relationship with posts
    posts = db.relationship('Post', backref='author_user', lazy=True)
    
    # Relationship with comments
    comments = db.relationship('Comment', backref='author_user', lazy=True)
    
    def set_password(self, password):
        self.password_hash = bcrypt.generate_password_hash(password).decode('utf-8')
    
    def check_password(self, password):
        return bcrypt.check_password_hash(self.password_hash, password)
    
    def calculate_level(self):
        """Calculate user level based on reputation and activity"""
        base_points = self.reputation + (self.posts_count * 5) + (self.comments_count * 2)
        
        # Level thresholds
        level_thresholds = [0, 20, 50, 100, 200, 350, 550, 800, 1200, 1700]
        
        # Find the highest threshold the user has passed
        level = 1
        for i, threshold in enumerate(level_thresholds):
            if base_points >= threshold:
                level = i + 1
        
        return level
    
    def update_statistics(self):
        """Update user statistics and level"""
        self.posts_count = len(self.posts)
        self.comments_count = len(self.comments)
        
        # Count likes received across all posts
        likes_received = 0
        for post in self.posts:
            likes_received += post.likes
        
        self.likes_received = likes_received
        self.level = self.calculate_level()
        
    def to_dict(self):
        return {
            'id': self.id,
            'email': self.email,
            'username': self.username,
            'created_at': self.created_at.isoformat(),
            'reputation': self.reputation,
            'avatar': self.avatar,
            'email_verified': self.email_verified,
            'level': self.level,
            'posts_count': self.posts_count,
            'comments_count': self.comments_count,
            'likes_received': self.likes_received,
            'likes_given': self.likes_given
        }

# Post model for community content
class Post(db.Model):
    __tablename__ = 'posts'
    
    id = db.Column(db.String(36), primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    content = db.Column(db.Text, nullable=False)
    post_type = db.Column(db.String(50), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    likes = db.Column(db.Integer, default=0)
    
    # Foreign key to user
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    
    # JSON fields (stored as text)
    _tickers = db.Column(db.Text, default='[]')
    _tags = db.Column(db.Text, default='[]')
    _recommendation_data = db.Column(db.Text, nullable=True)
    
    # Relationship with comments
    comments = db.relationship('Comment', backref='post', lazy=True, cascade='all, delete-orphan')
    
    @property
    def tickers(self):
        return json.loads(self._tickers)
    
    @tickers.setter
    def tickers(self, value):
        self._tickers = json.dumps(value)
    
    @property
    def tags(self):
        return json.loads(self._tags)
    
    @tags.setter
    def tags(self, value):
        self._tags = json.dumps(value)
    
    @property
    def recommendation_data(self):
        if self._recommendation_data:
            return json.loads(self._recommendation_data)
        return None
    
    @recommendation_data.setter
    def recommendation_data(self, value):
        if value:
            self._recommendation_data = json.dumps(value)
        else:
            self._recommendation_data = None
    
    def to_dict(self):
        result = {
            'id': self.id,
            'title': self.title,
            'content': self.content,
            'post_type': self.post_type,
            'timestamp': self.timestamp.isoformat(),
            'likes': self.likes,
            'tickers': self.tickers,
            'tags': self.tags,
            'author': self.author_user.username if self.author_user else "Anonymous",
            'user_id': self.user_id,
            'comments': [comment.to_dict() for comment in self.comments]
        }
        
        if self.recommendation_data:
            result['recommendation_data'] = self.recommendation_data
            
        return result

# Comment model for post discussions
class Comment(db.Model):
    __tablename__ = 'comments'
    
    id = db.Column(db.String(36), primary_key=True)
    text = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Foreign keys
    post_id = db.Column(db.String(36), db.ForeignKey('posts.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    
    def to_dict(self):
        return {
            'id': self.id,
            'text': self.text,
            'timestamp': self.timestamp.isoformat(),
            'post_id': self.post_id,
            'author': self.author_user.username if self.author_user else "Anonymous"
        }

# Initialize database
def init_db(app):
    db.init_app(app)
    bcrypt.init_app(app)
    
    # Create tables if they don't exist
    with app.app_context():
        db.create_all()
