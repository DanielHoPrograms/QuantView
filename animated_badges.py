import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import time
import random
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import badge definitions from gamification_minimal.py
from gamification_minimal import BADGES

def display_animated_badge_progress():
    """
    Display the user's badge progress with animated interactions
    Enhances the original badge display with animations and interactivity
    """
    st.subheader("Your Achievement Badges")
    
    # Display total points with a counting animation
    if 'total_points' in st.session_state:
        points_container = st.empty()
        start_points = 0
        end_points = st.session_state.total_points
        steps = 20
        step_size = (end_points - start_points) / steps if steps > 0 else 0
        
        for i in range(steps + 1):
            current_points = int(start_points + (step_size * i))
            points_container.metric("Total Points", current_points)
            time.sleep(0.02)
            
        # Final update to ensure exact value
        points_container.metric("Total Points", end_points)
    else:
        st.metric("Total Points", 0)
    
    # Organize badges by category
    categories = {
        "learning": "Learning",
        "personalized": "Personalized Learning Path",
        "quiz": "Quiz Achievements",
        "usage": "Application Usage",
        "advanced": "Advanced"
    }
    
    # Create tabs for badge categories
    tabs = st.tabs(list(categories.values()))
    
    for i, (category_id, category_name) in enumerate(categories.items()):
        with tabs[i]:
            category_badges = {badge_id: badge for badge_id, badge in BADGES.items() if badge["category"] == category_id}
            
            if not category_badges:
                st.write("No badges in this category.")
                continue
            
            # Display badges in a grid
            cols = st.columns(min(3, len(category_badges)))
            
            for j, (badge_id, badge) in enumerate(category_badges.items()):
                with cols[j % len(cols)]:
                    badge_container = st.empty()
                    badge_earned = badge_id in st.session_state.get('user_badges', {})
                    
                    # Generate random animation parameters (for variety)
                    animation_delay = j * 0.15  # Stagger the animations
                    pulse_duration = random.uniform(2.0, 3.0)
                    rotation_duration = random.uniform(2.5, 4.0)
                    scale_duration = random.uniform(2.0, 3.0)
                    
                    if badge_earned:
                        # Earned badge - display with animations
                        badge_container.markdown(f"""
                        <div style="text-align: center; border: 2px solid #4CAF50; border-radius: 10px; padding: 10px; margin: 5px; 
                             transition: all 0.3s ease; 
                             animation: fadeIn 0.5s ease-in-out {animation_delay}s both, 
                                        badgePulse {pulse_duration}s ease-in-out infinite;">
                            <div style="font-size: 40px; 
                                      animation: badgeRotate {rotation_duration}s ease-in-out infinite, 
                                                 badgeScale {scale_duration}s ease-in-out infinite;">
                                {badge['icon']}
                            </div>
                            <div style="font-weight: bold; color: #4CAF50;">{badge['name']}</div>
                            <div style="font-size: 12px; color: #666;">{badge['description']}</div>
                            <div style="font-size: 14px; color: #4CAF50; margin-top: 5px;">+{badge['points']} points</div>
                            <div style="font-size: 11px; color: #999; margin-top: 5px;">
                                Earned: {st.session_state.user_badges.get(badge_id, {}).get('earned_date', 'N/A')}
                            </div>
                        </div>
                        
                        <style>
                        @keyframes fadeIn {{
                            0% {{ opacity: 0; transform: translateY(10px); }}
                            100% {{ opacity: 1; transform: translateY(0); }}
                        }}
                        
                        @keyframes badgePulse {{
                            0% {{ box-shadow: 0 0 0 0 rgba(76, 175, 80, 0.5); }}
                            70% {{ box-shadow: 0 0 0 10px rgba(76, 175, 80, 0); }}
                            100% {{ box-shadow: 0 0 0 0 rgba(76, 175, 80, 0); }}
                        }}
                        
                        @keyframes badgeRotate {{
                            0% {{ transform: rotate(-5deg); }}
                            50% {{ transform: rotate(5deg); }}
                            100% {{ transform: rotate(-5deg); }}
                        }}
                        
                        @keyframes badgeScale {{
                            0% {{ transform: scale(1); }}
                            50% {{ transform: scale(1.1); }}
                            100% {{ transform: scale(1); }}
                        }}
                        </style>
                        """, unsafe_allow_html=True)
                    else:
                        # Locked badge - display in gray with subtle animation
                        badge_container.markdown(f"""
                        <div style="text-align: center; border: 2px solid #ccc; border-radius: 10px; padding: 10px; margin: 5px; 
                              background-color: #f5f5f5; 
                              animation: fadeIn 0.5s ease-in-out {animation_delay}s both,
                                         lockedPulse 3s ease-in-out infinite;">
                            <div style="font-size: 30px; opacity: 0.5; filter: grayscale(100%);">{badge['icon']}</div>
                            <div style="font-weight: bold; color: #999;">{badge['name']}</div>
                            <div style="font-size: 12px; color: #999;">{badge['description']}</div>
                            <div style="font-size: 14px; color: #999; margin-top: 5px;">+{badge['points']} points</div>
                            <div style="font-size: 11px; color: #999; margin-top: 5px;">Locked</div>
                        </div>
                        
                        <style>
                        @keyframes fadeIn {{
                            0% {{ opacity: 0; transform: translateY(10px); }}
                            100% {{ opacity: 1; transform: translateY(0); }}
                        }}
                        
                        @keyframes lockedPulse {{
                            0% {{ opacity: 0.9; }}
                            50% {{ opacity: 1; }}
                            100% {{ opacity: 0.9; }}
                        }}
                        </style>
                        """, unsafe_allow_html=True)

def display_animated_badge_unlock(badge_id):
    """
    Display an animated badge unlock notification
    
    Args:
        badge_id: ID of the badge that was unlocked
    """
    if badge_id not in BADGES:
        return
    
    badge = BADGES[badge_id]
    
    # Create a container for the animation
    badge_modal = st.empty()
    
    # Display the badge unlock animation
    badge_modal.markdown(f"""
    <div style="position: fixed; top: 0; left: 0; width: 100%; height: 100%; 
               background-color: rgba(0,0,0,0.8); z-index: 9999; 
               display: flex; justify-content: center; align-items: center; 
               animation: modalFadeIn 0.5s ease-in-out;">
        <div style="background-color: #2E2E2E; border-radius: 15px; padding: 30px; text-align: center; 
                   max-width: 80%; animation: badgeAppear 0.5s ease-out 0.3s both;">
            <div style="font-size: 24px; color: #FFC107; margin-bottom: 15px;">🎉 Achievement Unlocked! 🎉</div>
            <div style="font-size: 80px; margin: 20px 0; animation: badgeBounce 1s ease-in-out 0.5s infinite alternate, 
                                                                    badgeGlow 2s ease-in-out 0.5s infinite;">
                {badge['icon']}
            </div>
            <div style="font-size: 26px; color: #4CAF50; font-weight: bold; margin: 15px 0;">
                {badge['name']}
            </div>
            <div style="font-size: 16px; color: #DDDDDD; margin-bottom: 15px;">
                {badge['description']}
            </div>
            <div style="font-size: 20px; color: #4CAF50; margin: 15px 0; animation: pointsAppear 0.5s ease-out 1s both;">
                +{badge['points']} points
            </div>
            <button onclick="this.parentElement.parentElement.style.display='none';" 
                    style="background-color: #0C7BDC; color: white; border: none; padding: 10px 20px; 
                           border-radius: 5px; cursor: pointer; font-size: 16px; margin-top: 15px;
                           animation: buttonAppear 0.5s ease-out 1.5s both;">
                Continue
            </button>
        </div>
    </div>
    
    <style>
    @keyframes modalFadeIn {{
        0% {{ opacity: 0; }}
        100% {{ opacity: 1; }}
    }}
    
    @keyframes badgeAppear {{
        0% {{ transform: scale(0.8); opacity: 0; }}
        100% {{ transform: scale(1); opacity: 1; }}
    }}
    
    @keyframes badgeBounce {{
        0% {{ transform: translateY(0); }}
        100% {{ transform: translateY(-10px); }}
    }}
    
    @keyframes badgeGlow {{
        0% {{ text-shadow: 0 0 10px rgba(255, 255, 0, 0.5); }}
        50% {{ text-shadow: 0 0 20px rgba(255, 215, 0, 0.8), 0 0 30px rgba(255, 215, 0, 0.5); }}
        100% {{ text-shadow: 0 0 10px rgba(255, 255, 0, 0.5); }}
    }}
    
    @keyframes pointsAppear {{
        0% {{ transform: scale(0); opacity: 0; }}
        50% {{ transform: scale(1.2); opacity: 1; }}
        100% {{ transform: scale(1); opacity: 1; }}
    }}
    
    @keyframes buttonAppear {{
        0% {{ transform: translateY(20px); opacity: 0; }}
        100% {{ transform: translateY(0); opacity: 1; }}
    }}
    </style>
    """, unsafe_allow_html=True)
    
    # Wait for user to click continue
    time.sleep(5)  # Auto-close after 5 seconds
    badge_modal.empty()

def display_achievement_statistics():
    """Display animated statistics about the user's achievements"""
    if 'user_badges' not in st.session_state:
        st.warning("You haven't earned any badges yet. Complete learning modules and activities to earn achievements!")
        return
    
    # Calculate achievement statistics
    total_badges = len(BADGES)
    earned_badges = len(st.session_state.user_badges)
    progress_percentage = (earned_badges / total_badges) * 100 if total_badges > 0 else 0
    
    # Group badges by category
    category_counts = {}
    category_earned = {}
    
    for category in set(badge["category"] for badge in BADGES.values()):
        category_badges = [badge_id for badge_id, badge in BADGES.items() if badge["category"] == category]
        category_counts[category] = len(category_badges)
        category_earned[category] = len([badge_id for badge_id in category_badges if badge_id in st.session_state.user_badges])
    
    # Create animated progress bars
    st.markdown("### Your Achievement Progress")
    
    # Overall progress
    overall_progress_container = st.empty()
    for i in range(101):
        target = min(i, int(progress_percentage))
        progress_color = "#0C7BDC" if target < 50 else "#FFC107" if target < 80 else "#4CAF50"
        
        overall_progress_container.markdown(f"""
        <div style="margin: 10px 0;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                <div style="font-weight: bold;">Overall Progress</div>
                <div>{target}% ({earned_badges}/{total_badges})</div>
            </div>
            <div style="background-color: #333; border-radius: 10px; height: 20px; position: relative;">
                <div style="position: absolute; left: 0; top: 0; height: 100%; width: {target}%; 
                           background-color: {progress_color}; border-radius: 10px;
                           transition: width 0.3s ease;">
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if i >= progress_percentage:
            break
            
        time.sleep(0.01)
    
    # Category progress bars
    for category, count in category_counts.items():
        earned = category_earned[category]
        category_percentage = (earned / count) * 100 if count > 0 else 0
        
        # Get human-readable category name
        category_name = {
            "learning": "Learning",
            "personalized": "Personalized Learning",
            "quiz": "Quiz Achievements",
            "usage": "Application Usage",
            "advanced": "Advanced"
        }.get(category, category.capitalize())
        
        category_container = st.empty()
        for i in range(101):
            target = min(i, int(category_percentage))
            progress_color = "#0C7BDC" if target < 50 else "#FFC107" if target < 80 else "#4CAF50"
            
            category_container.markdown(f"""
            <div style="margin: 10px 0;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                    <div>{category_name}</div>
                    <div>{target}% ({earned}/{count})</div>
                </div>
                <div style="background-color: #333; border-radius: 10px; height: 12px; position: relative;">
                    <div style="position: absolute; left: 0; top: 0; height: 100%; width: {target}%; 
                               background-color: {progress_color}; border-radius: 10px;
                               transition: width 0.3s ease;">
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if i >= category_percentage:
                break
                
            time.sleep(0.005)
    
    # Create radar chart for achievement categories
    categories = list(category_counts.keys())
    percentages = [
        (category_earned[cat] / category_counts[cat] * 100) 
        if category_counts[cat] > 0 else 0 
        for cat in categories
    ]
    
    # Get human-readable category names
    category_names = [
        {
            "learning": "Learning",
            "personalized": "Personalized Learning",
            "quiz": "Quiz Achievements",
            "usage": "Application Usage",
            "advanced": "Advanced"
        }.get(cat, cat.capitalize()) 
        for cat in categories
    ]
    
    # Create the radar chart with animation
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=[0] * len(categories),  # Start with all zeros for animation
        theta=category_names,
        fill='toself',
        name='Achievement Categories',
        line_color='#0C7BDC',
        fillcolor='rgba(12, 123, 220, 0.3)'
    ))
    
    # Update layout
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        showlegend=False,
        title="Achievement Categories",
        height=400,
        margin=dict(l=40, r=40, t=50, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    # Display the chart and animate it
    chart_container = st.empty()
    chart_container.plotly_chart(fig, use_container_width=True)
    
    # Animate the radar chart
    steps = 20
    for i in range(steps + 1):
        progress = i / steps
        current_values = [p * progress for p in percentages]
        
        fig.update_traces(r=current_values)
        chart_container.plotly_chart(fig, use_container_width=True)
        time.sleep(0.05)
    
    # Display some badges to unlock next
    st.markdown("### Badges to Unlock Next")
    
    # Find badges that haven't been earned yet
    locked_badges = {badge_id: badge for badge_id, badge in BADGES.items() 
                     if badge_id not in st.session_state.get('user_badges', {})}
    
    if not locked_badges:
        st.success("Congratulations! You've earned all available badges!")
    else:
        # Display a few recommended badges to unlock next
        recommended_badges = list(locked_badges.items())[:3]
        cols = st.columns(len(recommended_badges))
        
        for i, (badge_id, badge) in enumerate(recommended_badges):
            with cols[i]:
                st.markdown(f"""
                <div style="text-align: center; border: 2px solid #666; border-radius: 10px; 
                           padding: 15px; margin: 5px; background-color: #1E1E1E; 
                           animation: glowPulse 2s infinite alternate;">
                    <div style="font-size: 40px; opacity: 0.8;">{badge['icon']}</div>
                    <div style="font-weight: bold; color: #DDD; margin: 10px 0;">{badge['name']}</div>
                    <div style="font-size: 14px; color: #AAA;">{badge['description']}</div>
                    <div style="font-size: 16px; color: #0C7BDC; margin-top: 15px;">+{badge['points']} points</div>
                </div>
                
                <style>
                @keyframes glowPulse {{
                    0% {{ box-shadow: 0 0 5px rgba(12, 123, 220, 0.3); }}
                    100% {{ box-shadow: 0 0 15px rgba(12, 123, 220, 0.7); }}
                }}
                </style>
                """, unsafe_allow_html=True)

def award_badge_with_animation(badge_id):
    """
    Award a badge to the user with an animation if they don't already have it
    
    Args:
        badge_id: ID of the badge to award
    
    Returns:
        True if the badge was awarded, False if already owned
    """
    # Skip if badge doesn't exist
    if badge_id not in BADGES:
        return False
    
    # Skip if user already has the badge
    if 'user_badges' in st.session_state and badge_id in st.session_state.user_badges:
        return False
    
    # Initialize user_badges if it doesn't exist
    if 'user_badges' not in st.session_state:
        st.session_state.user_badges = {}
    
    # Initialize total_points if it doesn't exist
    if 'total_points' not in st.session_state:
        st.session_state.total_points = 0
    
    # Award the badge
    today = datetime.now().strftime("%Y-%m-%d")
    badge_points = BADGES[badge_id]['points']
    
    st.session_state.user_badges[badge_id] = {
        'earned_date': today
    }
    
    st.session_state.total_points += badge_points
    
    # Show the badge unlock animation
    display_animated_badge_unlock(badge_id)
    
    return True
