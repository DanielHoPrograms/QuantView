from flask import Blueprint, request, jsonify, current_app
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from flask_jwt_extended import JWTManager, create_access_token, get_jwt_identity, jwt_required
import json
import uuid
from datetime import datetime, timedelta

from models import db, User, Post, Comment

# Create Blueprint for authentication routes
auth_bp = Blueprint('auth', __name__)

# Initialize login manager for session-based auth
login_manager = LoginManager()

# Initialize JWT manager for token-based auth
jwt = JWTManager()

# User loader for Flask-Login
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Register a new user
@auth_bp.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    
    # Validate input
    if not data or not data.get('email') or not data.get('password') or not data.get('username'):
        return jsonify({'success': False, 'message': 'Missing required fields'}), 400
    
    # Check if user already exists
    if User.query.filter_by(email=data['email']).first():
        return jsonify({'success': False, 'message': 'Email already registered'}), 400
    
    if User.query.filter_by(username=data['username']).first():
        return jsonify({'success': False, 'message': 'Username already taken'}), 400
    
    # Create new user
    user = User(
        email=data['email'],
        username=data['username']
    )
    user.set_password(data['password'])
    
    # Save to database
    db.session.add(user)
    db.session.commit()
    
    # Generate JWT token
    access_token = create_access_token(
        identity={'user_id': user.id, 'username': user.username, 'email': user.email},
        expires_delta=timedelta(days=7)
    )
    
    return jsonify({
        'success': True,
        'message': 'User registered successfully',
        'user': user.to_dict(),
        'access_token': access_token
    }), 201

# Login user
@auth_bp.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    
    # Validate input
    if not data or not data.get('email') or not data.get('password'):
        return jsonify({'success': False, 'message': 'Missing email or password'}), 400
    
    # Find user by email
    user = User.query.filter_by(email=data['email']).first()
    
    # Check password
    if not user or not user.check_password(data['password']):
        return jsonify({'success': False, 'message': 'Invalid email or password'}), 401
    
    # Login user (session-based)
    login_user(user)
    
    # Generate JWT token
    access_token = create_access_token(
        identity={'user_id': user.id, 'username': user.username, 'email': user.email},
        expires_delta=timedelta(days=7)
    )
    
    return jsonify({
        'success': True,
        'message': 'Login successful',
        'user': user.to_dict(),
        'access_token': access_token
    }), 200

# Logout user
@auth_bp.route('/logout', methods=['POST'])
@login_required
def logout():
    logout_user()
    return jsonify({'success': True, 'message': 'Logout successful'}), 200

# Get current user profile
@auth_bp.route('/profile', methods=['GET'])
@jwt_required()
def get_profile():
    # Get user identity from JWT
    current_identity = get_jwt_identity()
    
    # Find user
    user = User.query.get(current_identity['user_id'])
    
    if not user:
        return jsonify({'success': False, 'message': 'User not found'}), 404
    
    # Get user's posts
    posts = Post.query.filter_by(user_id=user.id).all()
    
    return jsonify({
        'success': True,
        'user': user.to_dict(),
        'post_count': len(posts),
        'posts': [post.to_dict() for post in posts]
    }), 200

# Update user profile
@auth_bp.route('/profile', methods=['PUT'])
@jwt_required()
def update_profile():
    # Get user identity from JWT
    current_identity = get_jwt_identity()
    
    # Find user
    user = User.query.get(current_identity['user_id'])
    
    if not user:
        return jsonify({'success': False, 'message': 'User not found'}), 404
    
    data = request.get_json()
    
    # Update fields
    if 'username' in data and data['username'] != user.username:
        # Check if username is taken
        existing_user = User.query.filter_by(username=data['username']).first()
        if existing_user and existing_user.id != user.id:
            return jsonify({'success': False, 'message': 'Username already taken'}), 400
        user.username = data['username']
    
    # Save changes
    db.session.commit()
    
    return jsonify({
        'success': True,
        'message': 'Profile updated successfully',
        'user': user.to_dict()
    }), 200

# Change password
@auth_bp.route('/change-password', methods=['POST'])
@jwt_required()
def change_password():
    # Get user identity from JWT
    current_identity = get_jwt_identity()
    
    # Find user
    user = User.query.get(current_identity['user_id'])
    
    if not user:
        return jsonify({'success': False, 'message': 'User not found'}), 404
    
    data = request.get_json()
    
    # Validate input
    if not data or not data.get('old_password') or not data.get('new_password'):
        return jsonify({'success': False, 'message': 'Missing required fields'}), 400
    
    # Check old password
    if not user.check_password(data['old_password']):
        return jsonify({'success': False, 'message': 'Incorrect password'}), 401
    
    # Set new password
    user.set_password(data['new_password'])
    
    # Save changes
    db.session.commit()
    
    return jsonify({
        'success': True,
        'message': 'Password changed successfully'
    }), 200

# Request password reset
@auth_bp.route('/request-reset', methods=['POST'])
def request_password_reset():
    data = request.get_json()
    
    # Validate input
    if not data or not data.get('email'):
        return jsonify({'success': False, 'message': 'Missing email'}), 400
    
    # Find user by email
    user = User.query.filter_by(email=data['email']).first()
    
    if not user:
        # Don't reveal that the user doesn't exist
        return jsonify({'success': True, 'message': 'If the email exists, a reset link has been sent'}), 200
    
    # Generate reset token (simple UUID for now)
    import uuid
    reset_token = str(uuid.uuid4())
    
    # Store token with expiration time (24 hours)
    user.reset_token = reset_token
    user.reset_token_expires = datetime.utcnow() + timedelta(hours=24)
    db.session.commit()
    
    # Note: In a real app, you'd send an email with the reset link
    # For now, just return the token directly (for testing)
    return jsonify({
        'success': True, 
        'message': 'If the email exists, a reset link has been sent',
        'debug_token': reset_token  # Only for development
    }), 200

# Reset password with token
@auth_bp.route('/reset-password', methods=['POST'])
def reset_password():
    data = request.get_json()
    
    # Validate input
    if not data or not data.get('token') or not data.get('new_password'):
        return jsonify({'success': False, 'message': 'Missing required fields'}), 400
    
    # Find user by token
    user = User.query.filter_by(reset_token=data['token']).first()
    
    if not user or not user.reset_token_expires or user.reset_token_expires < datetime.utcnow():
        return jsonify({'success': False, 'message': 'Invalid or expired token'}), 400
    
    # Set new password
    user.set_password(data['new_password'])
    
    # Clear reset token
    user.reset_token = None
    user.reset_token_expires = None
    
    # Save changes
    db.session.commit()
    
    return jsonify({
        'success': True,
        'message': 'Password has been reset successfully'
    }), 200

# Upload user avatar
@auth_bp.route('/upload-avatar', methods=['POST'])
@jwt_required()
def upload_avatar():
    # Get user identity from JWT
    current_identity = get_jwt_identity()
    
    # Find user
    user = User.query.get(current_identity['user_id'])
    
    if not user:
        return jsonify({'success': False, 'message': 'User not found'}), 404
    
    data = request.get_json()
    
    # Validate input (base64 encoded image)
    if not data or not data.get('avatar'):
        return jsonify({'success': False, 'message': 'Missing avatar data'}), 400
    
    # Store avatar (base64 string)
    user.avatar = data['avatar']
    
    # Save changes
    db.session.commit()
    
    return jsonify({
        'success': True,
        'message': 'Avatar uploaded successfully',
        'user': user.to_dict()
    }), 200

# Verify email address
@auth_bp.route('/verify-email', methods=['POST'])
def verify_email():
    data = request.get_json()
    
    # Validate input
    if not data or not data.get('email') or not data.get('code'):
        return jsonify({'success': False, 'message': 'Missing required fields'}), 400
    
    # Find user by email
    user = User.query.filter_by(email=data['email']).first()
    
    if not user or not user.verification_code or user.verification_code != data['code']:
        return jsonify({'success': False, 'message': 'Invalid verification code'}), 400
    
    # Mark email as verified
    user.email_verified = True
    user.verification_code = None
    
    # Save changes
    db.session.commit()
    
    return jsonify({
        'success': True,
        'message': 'Email verified successfully'
    }), 200

# Initialize auth
def init_auth(app):
    login_manager.init_app(app)
    jwt.init_app(app)
    app.register_blueprint(auth_bp, url_prefix='/api/auth')
