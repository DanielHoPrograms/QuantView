import streamlit as st
import time
import json
import pandas as pd
import numpy as np
import random
import plotly.graph_objects as go
from datetime import datetime

def add_page_transitions():
    """Add page transition animations using CSS and JS"""
    st.markdown("""
    <style>
    /* Page transition animations */
    .main {
        animation: fadeIn 0.5s ease-in-out;
    }
    
    @keyframes fadeIn {
        0% { opacity: 0; transform: translateY(10px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    
    /* Button hover effects */
    .stButton>button {
        transition: all 0.3s ease !important;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2) !important;
    }
    
    /* Input field focus animation */
    .stTextInput>div>div>input, .stNumberInput>div>div>input {
        transition: all 0.3s ease !important;
    }
    
    .stTextInput>div>div>input:focus, .stNumberInput>div>div>input:focus {
        border-color: #0C7BDC !important;
        box-shadow: 0 0 0 2px rgba(12, 123, 220, 0.3) !important;
    }
    
    /* Slider animation */
    .stSlider {
        transition: all 0.3s ease !important;
    }
    
    /* Expander animation */
    .streamlit-expanderHeader {
        transition: background-color 0.3s ease !important;
    }
    
    /* Card-like elements */
    .element-container div[data-testid="stVerticalBlock"] > div {
        transition: all 0.3s ease !important;
    }
    
    .element-container div[data-testid="stVerticalBlock"] > div:hover {
        transform: translateY(-2px) !important;
    }
    </style>
    """, unsafe_allow_html=True)

def loading_animation(message="Loading data...", container=None):
    """
    Display a loading animation with a custom message
    
    Args:
        message: Message to display with the animation
        container: Optional container to place the animation in (defaults to a new empty container)
    """
    # Create a container for the animation or use the provided one
    loading_container = container if container is not None else st.empty()
    
    # Create a pulsating dot animation
    dots = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    
    for i in range(10):  # Show animation for a brief period
        loading_container.markdown(f"""
        <div style="display: flex; align-items: center; margin: 10px 0; animation: fadeIn 0.5s ease-in-out;">
            <div style="font-size: 24px; margin-right: 10px; color: #0C7BDC; 
                       animation: pulse 1s infinite ease-in-out;">{dots[i % len(dots)]}</div>
            <div>{message}</div>
        </div>
        
        <style>
        @keyframes pulse {{
            from {{ opacity: 0.5; }}
            50% {{ opacity: 1; }}
            to {{ opacity: 0.5; }}
        }}
        
        @keyframes fadeIn {{
            from {{ opacity: 0; }}
            to {{ opacity: 1; }}
        }}
        </style>
        """, unsafe_allow_html=True)
        
        time.sleep(0.1)
    
    # Clear the animation
    loading_container.empty()
    
    return loading_container

def show_animated_notification(message, type="success", container=None):
    """
    Show an animated notification (success, info, warning, error)
    
    Args:
        message: Message to display
        type: Type of notification ('success', 'info', 'warning', 'error')
        container: Optional container to place the notification in
    """
    # Use provided container or create a new one
    notification_container = container if container else st.empty()
    
    # Set color based on notification type
    colors = {
        "success": "#4CAF50",
        "info": "#0C7BDC",
        "warning": "#FFA726",
        "error": "#F44336"
    }
    
    icons = {
        "success": "✓",
        "info": "ℹ",
        "warning": "⚠",
        "error": "✗"
    }
    
    color = colors.get(type, colors["info"])
    icon = icons.get(type, icons["info"])
    
    # Display the notification with animation
    notification_container.markdown(f"""
    <div style="border-left: 4px solid {color}; background-color: {color}22; 
               padding: 15px; border-radius: 4px; margin: 10px 0;
               animation: slideIn 0.5s ease-in-out, glow 2s infinite alternate;">
        <div style="display: flex; align-items: center;">
            <div style="font-size: 20px; margin-right: 10px; color: {color}; 
                       width: 24px; height: 24px; border-radius: 50%;
                       display: flex; justify-content: center; align-items: center;
                       animation: appear 0.5s ease-in-out 0.3s both;">
                {icon}
            </div>
            <div style="animation: fadeIn 0.5s ease-in-out 0.2s both;">
                {message}
            </div>
        </div>
    </div>
    
    <style>
    @keyframes slideIn {{
        0% {{ transform: translateX(-20px); opacity: 0; }}
        100% {{ transform: translateX(0); opacity: 1; }}
    }}
    
    @keyframes fadeIn {{
        0% {{ opacity: 0; }}
        100% {{ opacity: 1; }}
    }}
    
    @keyframes appear {{
        0% {{ transform: scale(0); }}
        50% {{ transform: scale(1.2); }}
        100% {{ transform: scale(1); }}
    }}
    
    @keyframes glow {{
        0% {{ box-shadow: 0 0 5px rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.1); }}
        100% {{ box-shadow: 0 0 15px rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.3); }}
    }}
    </style>
    """, unsafe_allow_html=True)
    
    # Return the container so it can be cleared later
    return notification_container

def create_animated_value_change(label, old_value, new_value, currency=False, percentage=False):
    """
    Create an animated transition for a changing value (for metrics like stock prices)
    
    Args:
        label: Label for the value
        old_value: Previous value
        new_value: New value
        currency: If True, format as currency
        percentage: If True, format as percentage
    """
    # Create a container for the animation
    container = st.empty()
    
    # Determine the formatting
    if currency:
        format_value = lambda x: f"${x:.2f}"
    elif percentage:
        format_value = lambda x: f"{x:.2f}%"
    else:
        format_value = lambda x: f"{x:.2f}"
    
    # Calculate value change
    change = new_value - old_value
    change_pct = (change / old_value) * 100 if old_value != 0 else 0
    
    # Determine color based on change
    color = "#4CAF50" if change >= 0 else "#F44336"
    arrow = "↑" if change >= 0 else "↓"
    
    # Step from old value to new value
    steps = 20
    step_size = (new_value - old_value) / steps if steps > 0 else 0
    
    for i in range(steps + 1):
        current_value = old_value + (step_size * i)
        current_change = (current_value - old_value) / old_value * 100 if old_value != 0 else 0
        
        # Update the display
        container.markdown(f"""
        <div style="text-align: center; padding: 10px; border-radius: 5px;">
            <div style="font-size: 14px; color: #999; margin-bottom: 5px;">{label}</div>
            <div style="font-size: 24px; font-weight: bold; margin-bottom: 5px;">
                {format_value(current_value)}
            </div>
            <div style="font-size: 14px; color: {color};">
                {arrow} {format_value(abs(current_change)) if percentage else format_value(abs(change))}
                {'' if percentage else ' (' + f"{abs(current_change):.2f}%" + ')'}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        time.sleep(0.02)
    
    # Return the container in case it needs to be cleared later
    return container

def create_typing_animation(text, speed=0.03, container=None):
    """
    Create a typing animation effect for text
    
    Args:
        text: Text to display with typing animation
        speed: Delay between characters in seconds
        container: Optional container to place the animation in
    """
    typing_container = container if container else st.empty()
    
    for i in range(len(text) + 1):
        typing_container.markdown(f"""
        <div style="font-family: monospace; font-size: 16px; white-space: pre-wrap;">
            {text[:i]}<span class="cursor">|</span>
        </div>
        
        <style>
        .cursor {{
            animation: blink 1s infinite;
        }}
        
        @keyframes blink {{
            0% {{ opacity: 0; }}
            40% {{ opacity: 0; }}
            50% {{ opacity: 1; }}
            90% {{ opacity: 1; }}
            100% {{ opacity: 0; }}
        }}
        </style>
        """, unsafe_allow_html=True)
        
        time.sleep(speed)
    
    # Return the container so it can be cleared later
    return typing_container

def create_chart_build_animation(fig, container=None, steps=5):
    """
    Gradually build a chart trace by trace for a dramatic reveal
    
    Args:
        fig: Plotly figure to animate
        container: Optional container to place the animation in
        steps: Number of animation steps
    """
    chart_container = container if container else st.empty()
    
    # Get the number of traces in the figure
    num_traces = len(fig.data)
    
    # Clone the figure to avoid modifying the original
    import copy
    animated_fig = copy.deepcopy(fig)
    
    # Start with all traces hidden
    for i in range(num_traces):
        animated_fig.data[i].visible = False
    
    # Gradually show traces
    for step in range(steps + 1):
        # Calculate how many traces to show in this step
        traces_to_show = int((step / steps) * num_traces)
        
        # Update trace visibility
        for i in range(num_traces):
            if i < traces_to_show:
                animated_fig.data[i].visible = True
            else:
                animated_fig.data[i].visible = False
        
        # Update opacity for all visible traces
        opacity = min(1.0, step / (steps * 0.7))
        for i in range(traces_to_show):
            if hasattr(animated_fig.data[i], 'opacity'):
                animated_fig.data[i].opacity = opacity
            if hasattr(animated_fig.data[i], 'marker'):
                if animated_fig.data[i].marker:
                    animated_fig.data[i].marker.opacity = opacity
            if hasattr(animated_fig.data[i], 'line'):
                if animated_fig.data[i].line:
                    animated_fig.data[i].line.width = 2 * opacity
        
        # Display the updated figure
        chart_container.plotly_chart(animated_fig, use_container_width=True)
        
        time.sleep(0.2)
    
    # Finally show the original figure with all traces
    chart_container.plotly_chart(fig, use_container_width=True)
    
    return chart_container
