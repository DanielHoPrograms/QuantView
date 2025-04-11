import streamlit as st
import requests
import time
import json
import base64
from io import BytesIO
from PIL import Image
import random
import string

def init_auth_session_state():
    """Initialize authentication session state variables"""
    if 'user' not in st.session_state:
        st.session_state.user = None
    if 'jwt_token' not in st.session_state:
        st.session_state.jwt_token = None
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    # Password reset state
    if 'reset_token' not in st.session_state:
        st.session_state.reset_token = None
    if 'reset_email' not in st.session_state:
        st.session_state.reset_email = None
    if 'show_reset_form' not in st.session_state:
        st.session_state.show_reset_form = False

def get_api_url():
    """Get API URL from session state or use default"""
    return st.session_state.get('api_server_url', 'http://0.0.0.0:5001')

def login_user_api(email, password):
    """Login user through API"""
    api_url = get_api_url()
    
    try:
        response = requests.post(
            f"{api_url}/api/auth/login",
            json={"email": email, "password": password}
        )
        
        if response.status_code == 200:
            data = response.json()
            st.session_state.user = data.get('user')
            st.session_state.jwt_token = data.get('access_token')
            st.session_state.logged_in = True
            return True, "Login successful"
        else:
            error_msg = response.json().get('message', 'Invalid email or password')
            return False, error_msg
    except Exception as e:
        return False, f"Error connecting to the server: {str(e)}"

def register_user_api(email, username, password):
    """Register new user through API"""
    api_url = get_api_url()
    
    try:
        response = requests.post(
            f"{api_url}/api/auth/register",
            json={"email": email, "username": username, "password": password}
        )
        
        if response.status_code == 201:
            data = response.json()
            st.session_state.user = data.get('user')
            st.session_state.jwt_token = data.get('access_token')
            st.session_state.logged_in = True
            return True, "Registration successful"
        else:
            error_msg = response.json().get('message', 'Registration failed')
            return False, error_msg
    except Exception as e:
        return False, f"Error connecting to the server: {str(e)}"

def logout_user():
    """Logout current user"""
    api_url = get_api_url()
    
    # Try to logout via API
    if st.session_state.jwt_token:
        try:
            headers = {"Authorization": f"Bearer {st.session_state.jwt_token}"}
            requests.post(f"{api_url}/api/auth/logout", headers=headers)
        except Exception as e:
            print(f"Error logging out: {str(e)}")
    
    # Clear session state regardless of API result
    st.session_state.user = None
    st.session_state.jwt_token = None
    st.session_state.logged_in = False
    
def get_user_posts_api(user_id=None):
    """Get posts for the current user"""
    api_url = get_api_url()
    
    if not user_id and st.session_state.user:
        user_id = st.session_state.user.get('id')
    
    if not user_id:
        return []
    
    try:
        headers = {}
        if st.session_state.jwt_token:
            headers["Authorization"] = f"Bearer {st.session_state.jwt_token}"
            
        response = requests.get(
            f"{api_url}/api/posts?user_id={user_id}",
            headers=headers
        )
        
        if response.status_code == 200:
            data = response.json()
            return data.get('posts', [])
        else:
            print(f"Error getting user posts: {response.status_code}")
            return []
    except Exception as e:
        print(f"Error connecting to the server: {str(e)}")
        return []

def update_profile_api(username=None):
    """Update user profile through API"""
    api_url = get_api_url()
    
    if not st.session_state.user or not st.session_state.jwt_token:
        return False, "You must be logged in to update your profile"
    
    try:
        headers = {"Authorization": f"Bearer {st.session_state.jwt_token}"}
        
        data = {}
        if username:
            data['username'] = username
            
        response = requests.put(
            f"{api_url}/api/auth/profile",
            headers=headers,
            json=data
        )
        
        if response.status_code == 200:
            data = response.json()
            st.session_state.user = data.get('user')
            return True, "Profile updated successfully"
        else:
            error_msg = response.json().get('message', 'Update failed')
            return False, error_msg
    except Exception as e:
        return False, f"Error connecting to the server: {str(e)}"

def change_password_api(old_password, new_password):
    """Change user password through API"""
    api_url = get_api_url()
    
    if not st.session_state.user or not st.session_state.jwt_token:
        return False, "You must be logged in to change your password"
    
    try:
        headers = {"Authorization": f"Bearer {st.session_state.jwt_token}"}
        
        response = requests.post(
            f"{api_url}/api/auth/change-password",
            headers=headers,
            json={"old_password": old_password, "new_password": new_password}
        )
        
        if response.status_code == 200:
            return True, "Password changed successfully"
        else:
            error_msg = response.json().get('message', 'Password change failed')
            return False, error_msg
    except Exception as e:
        return False, f"Error connecting to the server: {str(e)}"

def request_password_reset_api(email):
    """Request a password reset through API"""
    api_url = get_api_url()
    
    try:
        response = requests.post(
            f"{api_url}/api/auth/request-reset",
            json={"email": email}
        )
        
        if response.status_code == 200:
            data = response.json()
            
            # Store reset token in session state (for development only - in production this would be sent via email)
            if 'debug_token' in data:
                st.session_state.reset_token = data['debug_token']
                st.session_state.reset_email = email
                st.session_state.show_reset_form = True
            
            return True, data.get('message', 'Password reset requested')
        else:
            error_msg = response.json().get('message', 'Password reset request failed')
            return False, error_msg
    except Exception as e:
        return False, f"Error connecting to the server: {str(e)}"

def reset_password_api(token, new_password):
    """Reset password using a token through API"""
    api_url = get_api_url()
    
    try:
        response = requests.post(
            f"{api_url}/api/auth/reset-password",
            json={"token": token, "new_password": new_password}
        )
        
        if response.status_code == 200:
            # Clear reset token from session state
            st.session_state.reset_token = None
            st.session_state.reset_email = None
            st.session_state.show_reset_form = False
            
            return True, "Password has been reset successfully"
        else:
            error_msg = response.json().get('message', 'Password reset failed')
            return False, error_msg
    except Exception as e:
        return False, f"Error connecting to the server: {str(e)}"
        
def upload_avatar_api(avatar_data):
    """Upload user avatar through API"""
    api_url = get_api_url()
    
    if not st.session_state.user or not st.session_state.jwt_token:
        return False, "You must be logged in to upload an avatar"
    
    try:
        headers = {"Authorization": f"Bearer {st.session_state.jwt_token}"}
        
        response = requests.post(
            f"{api_url}/api/auth/upload-avatar",
            headers=headers,
            json={"avatar": avatar_data}
        )
        
        if response.status_code == 200:
            data = response.json()
            st.session_state.user = data.get('user')
            return True, "Avatar uploaded successfully"
        else:
            error_msg = response.json().get('message', 'Avatar upload failed')
            return False, error_msg
    except Exception as e:
        return False, f"Error connecting to the server: {str(e)}"

def display_login_ui():
    """Display login/registration UI"""
    init_auth_session_state()
    
    if st.session_state.logged_in:
        return display_profile_ui()
    
    st.subheader("📝 Login or Register")
    
    # Display password reset form if requested
    if st.session_state.show_reset_form and st.session_state.reset_token:
        return display_password_reset_form()
    
    # Create tabs for login, registration, and password reset
    login_tab, register_tab, reset_tab = st.tabs(["Login", "Register", "Forgot Password"])
    
    # Login tab
    with login_tab:
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_password")
        
        if st.button("Login", key="login_button"):
            if not email or not password:
                st.error("Please enter both email and password.")
            else:
                with st.spinner("Logging in..."):
                    success, message = login_user_api(email, password)
                    if success:
                        st.success(message)
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(message)
    
    # Registration tab
    with register_tab:
        reg_email = st.text_input("Email", key="reg_email")
        reg_username = st.text_input("Username", key="reg_username")
        reg_password = st.text_input("Password", type="password", key="reg_password")
        reg_password_confirm = st.text_input("Confirm Password", type="password", key="reg_password_confirm")
        
        if st.button("Register", key="reg_button"):
            if not reg_email or not reg_username or not reg_password:
                st.error("Please fill in all fields.")
            elif reg_password != reg_password_confirm:
                st.error("Passwords do not match.")
            else:
                with st.spinner("Registering..."):
                    success, message = register_user_api(reg_email, reg_username, reg_password)
                    if success:
                        st.success(message)
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(message)
    
    # Password reset request tab
    with reset_tab:
        st.write("Enter your email address to receive a password reset link.")
        reset_email = st.text_input("Email", key="reset_email")
        
        if st.button("Request Password Reset", key="reset_request_button"):
            if not reset_email:
                st.error("Please enter your email address.")
            else:
                with st.spinner("Requesting password reset..."):
                    success, message = request_password_reset_api(reset_email)
                    if success:
                        st.success(message)
                        # In dev mode, we immediately show the reset form
                        if st.session_state.show_reset_form:
                            time.sleep(1)
                            st.rerun()
                    else:
                        st.error(message)

def display_password_reset_form():
    """Display the password reset form"""
    st.subheader("Reset Your Password")
    
    st.write(f"Enter a new password for {st.session_state.reset_email}")
    
    new_password = st.text_input("New Password", type="password", key="new_reset_password")
    confirm_password = st.text_input("Confirm New Password", type="password", key="confirm_reset_password")
    
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("Back", key="back_to_login"):
            st.session_state.show_reset_form = False
            st.session_state.reset_token = None
            st.session_state.reset_email = None
            st.rerun()
    
    with col2:
        if st.button("Reset Password", key="reset_password_button"):
            if not new_password or not confirm_password:
                st.error("Please enter a new password and confirm it.")
            elif new_password != confirm_password:
                st.error("Passwords do not match.")
            else:
                with st.spinner("Resetting password..."):
                    success, message = reset_password_api(st.session_state.reset_token, new_password)
                    if success:
                        st.success(message)
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(message)

def display_profile_ui():
    """Display user profile if logged in"""
    if not st.session_state.logged_in or not st.session_state.user:
        return display_login_ui()
    
    user = st.session_state.user
    
    st.subheader(f"👤 Welcome, {user.get('username')}")
    
    # Create tabs for profile, posts, and settings
    profile_tab, posts_tab, settings_tab = st.tabs(["Profile", "My Posts", "Settings"])
    
    # Profile tab
    with profile_tab:
        col1, col2 = st.columns([1, 3])
        
        with col1:
            # Display user avatar if available
            if user.get('avatar'):
                try:
                    avatar_data = user.get('avatar')
                    st.image(avatar_data, width=150)
                except Exception as e:
                    st.error(f"Error displaying avatar: {str(e)}")
            else:
                # Display default avatar
                st.info("No avatar set")
                
            # Avatar upload option
            st.write("**Update Avatar:**")
            uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="avatar_uploader")
            
            if uploaded_file is not None:
                try:
                    # Read the file and encode as base64
                    img = Image.open(uploaded_file)
                    # Resize to a reasonable size
                    img = img.resize((150, 150))
                    buffered = BytesIO()
                    img.save(buffered, format="PNG")
                    img_str = base64.b64encode(buffered.getvalue()).decode()
                    
                    # Show preview
                    st.image(img, width=150, caption="Preview")
                    
                    # Upload button
                    if st.button("Upload Avatar", key="upload_avatar_button"):
                        with st.spinner("Uploading avatar..."):
                            success, message = upload_avatar_api(img_str)
                            if success:
                                st.success(message)
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error(message)
                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")
        
        with col2:
            # User information
            st.write(f"**Email:** {user.get('email')}")
            st.write(f"**Username:** {user.get('username')}")
            st.write(f"**Joined:** {user.get('created_at')}")
            
            # Progress section
            st.subheader("Progress & Statistics")
            
            # User level
            level = user.get('level', 1)
            st.write(f"**Level:** {level}")
            
            # Progress bar to next level
            st.progress(min(1.0, user.get('reputation', 0) / (level * 100)))
            
            # Stats in columns
            stats_col1, stats_col2 = st.columns(2)
            
            with stats_col1:
                st.write(f"**Reputation:** {user.get('reputation', 0)}")
                st.write(f"**Posts:** {user.get('posts_count', 0)}")
            
            with stats_col2:
                st.write(f"**Comments:** {user.get('comments_count', 0)}")
                st.write(f"**Likes received:** {user.get('likes_received', 0)}")

            # Email verification status
            is_verified = user.get('email_verified', False)
            if is_verified:
                st.success("Email Verified ✓")
            else:
                st.warning("Email Not Verified")
                if st.button("Verify Email", key="verify_email_button"):
                    st.info("A verification code has been sent to your email. Please check your inbox.")
        
        # Logout button at the bottom
        if st.button("Logout", key="logout_button"):
            logout_user()
            st.success("You have been logged out.")
            time.sleep(1)
            st.rerun()
    
    # User's posts tab
    with posts_tab:
        st.subheader("My Posts")
        posts = get_user_posts_api()
        
        if not posts:
            st.info("You haven't created any posts yet.")
        else:
            for post in posts:
                with st.expander(f"{post.get('title')} - {post.get('post_type')}"):
                    st.write(post.get('content'))
                    st.write(f"**Created:** {post.get('timestamp')}")
                    st.write(f"**Likes:** {post.get('likes', 0)}")
                    
                    # Show comments if any
                    comments = post.get('comments', [])
                    if comments:
                        st.write("**Comments:**")
                        for comment in comments:
                            st.text(f"{comment.get('author')}: {comment.get('text')}")
    
    # Settings tab
    with settings_tab:
        st.subheader("Update Profile")
        
        new_username = st.text_input("New Username", value=user.get('username'), key="settings_username")
        
        if st.button("Update Profile", key="update_profile_button"):
            if new_username != user.get('username'):
                success, message = update_profile_api(username=new_username)
                if success:
                    st.success(message)
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(message)
            else:
                st.info("No changes to save.")
        
        # Change password section
        st.subheader("Change Password")
        
        old_password = st.text_input("Current Password", type="password", key="old_password")
        new_password = st.text_input("New Password", type="password", key="new_password")
        confirm_password = st.text_input("Confirm New Password", type="password", key="confirm_password")
        
        if st.button("Change Password", key="change_password_button"):
            if not old_password or not new_password or not confirm_password:
                st.error("Please fill in all password fields.")
            elif new_password != confirm_password:
                st.error("New passwords do not match.")
            else:
                success, message = change_password_api(old_password, new_password)
                if success:
                    st.success(message)
                else:
                    st.error(message)

def display_auth_status():
    """Display current authentication status (for use in other pages)"""
    if st.session_state.get('logged_in', False) and st.session_state.get('user'):
        return st.session_state.user.get('username'), True
    return "Anonymous", False
