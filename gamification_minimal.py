import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import time
from datetime import datetime

# Define achievement badges
BADGES = {
    # Learning achievements
    "first_steps": {
        "name": "First Steps", 
        "description": "Complete your first learning module",
        "icon": "🏁",
        "points": 10,
        "category": "learning"
    },
    "technical_novice": {
        "name": "Technical Novice", 
        "description": "Learn about your first technical indicator",
        "icon": "📊",
        "points": 20,
        "category": "learning"
    },
    "chart_reader": {
        "name": "Chart Reader", 
        "description": "Complete the candlestick chart module",
        "icon": "📈",
        "points": 30,
        "category": "learning"
    },
    "indicator_master": {
        "name": "Indicator Master", 
        "description": "Complete all technical indicator modules",
        "icon": "🧠",
        "points": 50,
        "category": "learning"
    },
    "risk_manager": {
        "name": "Risk Manager", 
        "description": "Complete the risk management module",
        "icon": "🛡️",
        "points": 40,
        "category": "learning"
    },
    "market_wisdom": {
        "name": "Market Wisdom", 
        "description": "Complete all learning modules",
        "icon": "🦉",
        "points": 100,
        "category": "learning"
    },
    
    # Personalized Learning Path achievements
    "path_starter": {
        "name": "Path Starter", 
        "description": "Complete your knowledge assessment",
        "icon": "🚶",
        "points": 10,
        "category": "personalized"
    },
    "path_explorer": {
        "name": "Path Explorer", 
        "description": "Complete 3 modules in your personalized learning path",
        "icon": "🧭",
        "points": 20,
        "category": "personalized"
    },
    "knowledge_climber": {
        "name": "Knowledge Climber", 
        "description": "Advance to the Intermediate knowledge level",
        "icon": "🧗",
        "points": 30,
        "category": "personalized"
    },
    "market_expert": {
        "name": "Market Expert", 
        "description": "Reach the Advanced knowledge level",
        "icon": "🏔️",
        "points": 50,
        "category": "personalized"
    },
    
    # Quiz achievements
    "quiz_taker": {
        "name": "Quiz Taker", 
        "description": "Complete your first quiz",
        "icon": "📝",
        "points": 10,
        "category": "quiz"
    },
    "knowledge_seeker": {
        "name": "Knowledge Seeker", 
        "description": "Complete 3 different quiz categories",
        "icon": "🔍",
        "points": 25,
        "category": "quiz"
    },
    "market_scholar": {
        "name": "Market Scholar", 
        "description": "Complete all quiz categories",
        "icon": "🎓",
        "points": 50,
        "category": "quiz"
    },
    "quiz_ace": {
        "name": "Quiz Ace", 
        "description": "Score 100% on a quiz with at least 5 questions",
        "icon": "🥇",
        "points": 30,
        "category": "quiz"
    },
    "chart_analyst": {
        "name": "Chart Analyst", 
        "description": "Correctly interpret a chart pattern",
        "icon": "📊",
        "points": 15,
        "category": "quiz"
    },
    
    # Watchlist achievements
    "diversified_investor": {
        "name": "Diversified Investor", 
        "description": "Add 5 different stocks to your watchlist",
        "icon": "🔎",
        "points": 15,
        "category": "watchlist"
    },
    "market_explorer": {
        "name": "Market Explorer", 
        "description": "Add 10 different stocks to your watchlist",
        "icon": "🌐",
        "points": 30,
        "category": "watchlist"
    },
    "sector_analyst": {
        "name": "Sector Analyst", 
        "description": "Add stocks from 5 different sectors to your watchlist",
        "icon": "📋",
        "points": 25,
        "category": "watchlist"
    },
    "global_investor": {
        "name": "Global Investor", 
        "description": "Add stocks from international markets to your watchlist",
        "icon": "🌎",
        "points": 35,
        "category": "watchlist"
    },
    
    # Backtesting achievements
    "backtest_explorer": {
        "name": "Backtest Explorer", 
        "description": "Run your first strategy backtest",
        "icon": "🧪",
        "points": 20,
        "category": "backtest"
    },
    "strategy_diversifier": {
        "name": "Strategy Diversifier", 
        "description": "Backtest 3 different trading strategies",
        "icon": "🔄",
        "points": 30,
        "category": "backtest"
    },
    "alpha_finder": {
        "name": "Alpha Finder", 
        "description": "Develop a strategy that outperforms the benchmark by 10%",
        "icon": "🏆",
        "points": 50,
        "category": "backtest"
    },
    "risk_optimizer": {
        "name": "Risk Optimizer", 
        "description": "Create a strategy with a Sharpe ratio above 1.5",
        "icon": "⚖️",
        "points": 40,
        "category": "backtest"
    },
    "backtest_master": {
        "name": "Backtest Master", 
        "description": "Optimize a strategy with at least 3 parameters",
        "icon": "🔧",
        "points": 45,
        "category": "backtest"
    }
}

# Define learning modules
LEARNING_MODULES = {
    "what_is_a_stock": {
        "name": "What is a Stock?",
        "description": "Learn the fundamentals of what stocks are and how they work",
        "badge": "first_steps",
        "estimated_time": "8 min",
        "difficulty": "Beginner"
    },
    "intro_to_markets": {
        "name": "Introduction to Stock Markets",
        "description": "Learn the basics of how stock markets work",
        "badge": "first_steps",
        "estimated_time": "10 min",
        "difficulty": "Beginner"
    },
    "candlestick_charts": {
        "name": "Understanding Candlestick Charts",
        "description": "Learn how to read and interpret candlestick charts",
        "badge": "chart_reader",
        "estimated_time": "15 min",
        "difficulty": "Beginner"
    },
    "intro_to_rsi": {
        "name": "Introduction to RSI",
        "description": "Learn about the Relative Strength Index and how to use it",
        "badge": "technical_novice",
        "estimated_time": "20 min",
        "difficulty": "Intermediate"
    },
    "intro_to_macd": {
        "name": "MACD Explained",
        "description": "Understanding the Moving Average Convergence Divergence indicator",
        "badge": "technical_novice",
        "estimated_time": "20 min", 
        "difficulty": "Intermediate"
    },
    "bollinger_bands": {
        "name": "Bollinger Bands Strategy",
        "description": "Learn how to use Bollinger Bands for trading decisions",
        "badge": "technical_novice",
        "estimated_time": "25 min",
        "difficulty": "Intermediate"
    },
    "advanced_indicators": {
        "name": "Advanced Technical Indicators",
        "description": "Master complex technical indicators and their combinations",
        "badge": "indicator_master",
        "estimated_time": "30 min",
        "difficulty": "Advanced"
    }
}

# Module content for different modules
MODULE_CONTENT = {
    "what_is_a_stock": """<div style="background-color: #000000 !important; color: #ffffff !important; padding: 15px; border-radius: 5px; margin-bottom: 20px; border: 1px solid #444444; font-size: 1rem;">
<h2 style="color: #ffffff !important; font-size: 1.5rem;">What is a Stock?</h2>

<p style="color: #ffffff !important; font-size: 1rem;">A stock (also known as equity or a share) represents ownership in a company. When you purchase a stock, you're buying a small piece of the company, which makes you a shareholder.</p>

<h3 style="color: #ffffff !important; font-size: 1.3rem;">Key Aspects of Stocks</h3>

<ul style="color: #ffffff !important; font-size: 1rem;">
    <li style="color: #ffffff !important;"><strong style="color: #ffffff !important;">Ownership Rights</strong>: Stocks entitle you to a portion of the company's assets and earnings</li>
    <li style="color: #ffffff !important;"><strong style="color: #ffffff !important;">Voting Rights</strong>: Common shareholders typically have voting rights in company decisions</li>
    <li style="color: #ffffff !important;"><strong style="color: #ffffff !important;">Dividends</strong>: Some companies distribute a portion of profits to shareholders</li>
    <li style="color: #ffffff !important;"><strong style="color: #ffffff !important;">Capital Appreciation</strong>: Stocks can increase in value over time</li>
</ul>

<h3 style="color: #ffffff !important; font-size: 1.3rem;">Types of Stocks</h3>

<ol style="color: #ffffff !important; font-size: 1rem;">
    <li style="color: #ffffff !important;"><strong style="color: #ffffff !important;">Common Stock</strong>: Standard shares with voting rights but lower priority in asset claims</li>
    <li style="color: #ffffff !important;"><strong style="color: #ffffff !important;">Preferred Stock</strong>: Higher claim on assets and earnings, typically with fixed dividends but limited voting rights</li>
    <li style="color: #ffffff !important;"><strong style="color: #ffffff !important;">Growth Stocks</strong>: Companies expected to grow at an above-average rate</li>
    <li style="color: #ffffff !important;"><strong style="color: #ffffff !important;">Value Stocks</strong>: Companies trading at a lower price relative to their fundamentals</li>
    <li style="color: #ffffff !important;"><strong style="color: #ffffff !important;">Dividend Stocks</strong>: Companies that pay regular dividends to shareholders</li>
</ol>

<h3 style="color: #ffffff !important; font-size: 1.3rem;">How Stocks Are Traded</h3>

<p style="color: #ffffff !important; font-size: 1rem;">Stocks are traded on exchanges like the New York Stock Exchange (NYSE) or NASDAQ. The price of a stock is determined by supply and demand – when more people want to buy than sell, prices rise, and vice versa.</p>
</div>""",
    
    "intro_to_markets": """<div style="background-color: #000000 !important; color: #ffffff !important; padding: 15px; border-radius: 5px; margin-bottom: 20px; border: 1px solid #444444; font-size: 1rem;">
<h2 style="color: #ffffff !important; font-size: 1.5rem;">Introduction to Stock Markets</h2>

<p style="color: #ffffff !important; font-size: 1rem;">The stock market is a marketplace where shares of publicly traded companies are bought and sold. It provides companies with capital to grow their business and offers investors an opportunity to share in the profits of successful companies.</p>

<h3 style="color: #ffffff !important; font-size: 1.3rem;">Key Stock Market Concepts</h3>

<ol style="color: #ffffff !important; font-size: 1rem;">
    <li style="color: #ffffff !important;"><strong style="color: #ffffff !important;">Stocks (Shares)</strong>: Represent ownership in a company</li>
    <li style="color: #ffffff !important;"><strong style="color: #ffffff !important;">Exchanges</strong>: Organized marketplaces where stocks are traded (e.g., NYSE, NASDAQ)</li>
    <li style="color: #ffffff !important;"><strong style="color: #ffffff !important;">Market Indices</strong>: Track performance of groups of stocks (e.g., S&P 500, Dow Jones)</li>
    <li style="color: #ffffff !important;"><strong style="color: #ffffff !important;">Bull vs Bear Markets</strong>: Rising markets vs falling markets</li>
    <li style="color: #ffffff !important;"><strong style="color: #ffffff !important;">Dividends</strong>: Portion of company profits paid to shareholders</li>
</ol>

<h3 style="color: #ffffff !important; font-size: 1.3rem;">Market Participants</h3>

<ul style="color: #ffffff !important; font-size: 1rem;">
    <li style="color: #ffffff !important;"><strong style="color: #ffffff !important;">Retail Investors</strong>: Individual investors like you</li>
    <li style="color: #ffffff !important;"><strong style="color: #ffffff !important;">Institutional Investors</strong>: Banks, pension funds, mutual funds</li>
    <li style="color: #ffffff !important;"><strong style="color: #ffffff !important;">Market Makers</strong>: Ensure liquidity by always being willing to buy or sell</li>
    <li style="color: #ffffff !important;"><strong style="color: #ffffff !important;">Brokers</strong>: Execute trades on behalf of investors</li>
</ul>

<h3 style="color: #ffffff !important; font-size: 1.3rem;">How Stock Prices Move</h3>

<p style="color: #ffffff !important; font-size: 1rem;">Stock prices move based on supply and demand, which are influenced by:</p>

<ul style="color: #ffffff !important; font-size: 1rem;">
    <li style="color: #ffffff !important;">Company performance and earnings</li>
    <li style="color: #ffffff !important;">Economic indicators and news</li>
    <li style="color: #ffffff !important;">Interest rates and inflation</li>
    <li style="color: #ffffff !important;">Market sentiment and psychology</li>
    <li style="color: #ffffff !important;">Industry trends and competitive landscape</li>
</ul>
</div>

<div style="background-color: #000000 !important; color: #ffffff !important; padding: 15px; border-radius: 5px; margin-bottom: 20px; border: 1px solid #444444; font-size: 1rem;">
<h3 style="color: #ffffff !important; font-size: 1.3rem;">Getting Started with Investing</h3>

<ol style="color: #ffffff !important; font-size: 1rem;">
    <li style="color: #ffffff !important;">Define your investment goals and time horizon</li>
    <li style="color: #ffffff !important;">Research companies and understand their business models</li>
    <li style="color: #ffffff !important;">Diversify your portfolio to manage risk</li>
    <li style="color: #ffffff !important;">Consider both fundamental and technical analysis</li>
    <li style="color: #ffffff !important;">Start with a long-term perspective rather than short-term trading</li>
</ol>
</div>""",
    
    "candlestick_charts": """<div style="background-color: #000000 !important; color: #ffffff !important; padding: 15px; border-radius: 5px; margin-bottom: 20px; border: 1px solid #444444; font-size: 1rem;">
<h2 style="color: #ffffff !important; font-size: 1.5rem;">Understanding Candlestick Charts</h2>

<p style="color: #ffffff !important; font-size: 1rem;">Candlestick charts originated in Japan in the 18th century and are now one of the most popular chart types for technical analysis.</p>

<h3 style="color: #ffffff !important; font-size: 1.3rem;">Anatomy of a Candlestick</h3>

<p style="color: #ffffff !important; font-size: 1rem;">Each candlestick represents a specific time period (e.g., 1 day, 1 hour) and shows four key price points:</p>

<ul style="color: #ffffff !important; font-size: 1rem;">
    <li style="color: #ffffff !important;"><strong style="color: #ffffff !important;">Open</strong>: Price at the beginning of the period</li>
    <li style="color: #ffffff !important;"><strong style="color: #ffffff !important;">Close</strong>: Price at the end of the period</li>
    <li style="color: #ffffff !important;"><strong style="color: #ffffff !important;">High</strong>: Highest price during the period</li>
    <li style="color: #ffffff !important;"><strong style="color: #ffffff !important;">Low</strong>: Lowest price during the period</li>
</ul>

<p style="color: #ffffff !important; font-size: 1rem;">The "body" of the candle represents the range between open and close, while the "wicks" or "shadows" show the high and low.</p>

<h3 style="color: #ffffff !important; font-size: 1.3rem;">Color Coding</h3>

<ul style="color: #ffffff !important; font-size: 1rem;">
    <li style="color: #ffffff !important;"><strong style="color: #ffffff !important;">Green/White Candle</strong>: Close is higher than open (bullish)</li>
    <li style="color: #ffffff !important;"><strong style="color: #ffffff !important;">Red/Black Candle</strong>: Close is lower than open (bearish)</li>
</ul>

<h3 style="color: #ffffff !important; font-size: 1.3rem;">Common Candlestick Patterns</h3>

<h4 style="color: #ffffff !important; font-size: 1.2rem;">Bullish Patterns</h4>

<ul style="color: #ffffff !important; font-size: 1rem;">
    <li style="color: #ffffff !important;"><strong style="color: #ffffff !important;">Hammer</strong>: Small body with long lower wick, appearing in downtrends</li>
    <li style="color: #ffffff !important;"><strong style="color: #ffffff !important;">Bullish Engulfing</strong>: Large green candle engulfs previous red candle</li>
    <li style="color: #ffffff !important;"><strong style="color: #ffffff !important;">Morning Star</strong>: Three-candle pattern showing potential reversal from downtrend</li>
</ul>

<h4 style="color: #ffffff !important; font-size: 1.2rem;">Bearish Patterns</h4>

<ul style="color: #ffffff !important; font-size: 1rem;">
    <li style="color: #ffffff !important;"><strong style="color: #ffffff !important;">Shooting Star</strong>: Small body with long upper wick, appearing in uptrends</li>
    <li style="color: #ffffff !important;"><strong style="color: #ffffff !important;">Bearish Engulfing</strong>: Large red candle engulfs previous green candle</li>
    <li style="color: #ffffff !important;"><strong style="color: #ffffff !important;">Evening Star</strong>: Three-candle pattern showing potential reversal from uptrend</li>
</ul>

<h4 style="color: #ffffff !important; font-size: 1.2rem;">Indecision Patterns</h4>

<ul style="color: #ffffff !important; font-size: 1rem;">
    <li style="color: #ffffff !important;"><strong style="color: #ffffff !important;">Doji</strong>: Open and close are virtually the same, showing indecision</li>
    <li style="color: #ffffff !important;"><strong style="color: #ffffff !important;">Spinning Top</strong>: Small body with upper and lower wicks of similar length</li>
</ul>

<h3 style="color: #ffffff !important; font-size: 1.3rem;">Using Candlestick Patterns</h3>

<ul style="color: #ffffff !important; font-size: 1rem;">
    <li style="color: #ffffff !important;">Look for patterns at key support and resistance levels</li>
    <li style="color: #ffffff !important;">Confirm patterns with other technical indicators or volume</li>
    <li style="color: #ffffff !important;">Consider the overall trend when interpreting patterns</li>
    <li style="color: #ffffff !important;">The longer the time frame, the more significant the pattern</li>
</ul>
</div>""",
    
    "intro_to_rsi": """<div style="background-color: #000000 !important; color: #ffffff !important; padding: 15px; border-radius: 5px; margin-bottom: 20px; border: 1px solid #444444; font-size: 1rem;">
<h2 style="color: #ffffff !important; font-size: 1.5rem;">Introduction to RSI (Relative Strength Index)</h2>

<p style="color: #ffffff !important; font-size: 1rem;">The Relative Strength Index (RSI) is a momentum oscillator that measures the speed and change of price movements on a scale from 0 to 100.</p>

<h3 style="color: #ffffff !important; font-size: 1.3rem;">Understanding RSI</h3>

<ul style="color: #ffffff !important; font-size: 1rem;">
    <li style="color: #ffffff !important;"><strong style="color: #ffffff !important;">Developed by</strong> J. Welles Wilder in 1978</li>
    <li style="color: #ffffff !important;"><strong style="color: #ffffff !important;">Default period</strong> is 14 days (but can be adjusted)</li>
    <li style="color: #ffffff !important;"><strong style="color: #ffffff !important;">Scale</strong> ranges from 0 to 100</li>
    <li style="color: #ffffff !important;"><strong style="color: #ffffff !important;">Overbought territory</strong> is typically above 70</li>
    <li style="color: #ffffff !important;"><strong style="color: #ffffff !important;">Oversold territory</strong> is typically below 30</li>
</ul>

<h3 style="color: #ffffff !important; font-size: 1.3rem;">How RSI is Calculated</h3>

<p style="color: #ffffff !important; font-size: 1rem;">RSI = 100 - (100 / (1 + RS))</p>

<p style="color: #ffffff !important; font-size: 1rem;">Where RS = Average Gain / Average Loss over the specified period</p>

<h3 style="color: #ffffff !important; font-size: 1.3rem;">Using RSI in Trading</h3>

<h4 style="color: #ffffff !important; font-size: 1.2rem;">Basic Signals</h4>

<ul style="color: #ffffff !important; font-size: 1rem;">
    <li style="color: #ffffff !important;"><strong style="color: #ffffff !important;">Oversold Conditions (RSI < 30)</strong>: Potential buy signal</li>
    <li style="color: #ffffff !important;"><strong style="color: #ffffff !important;">Overbought Conditions (RSI > 70)</strong>: Potential sell signal</li>
</ul>

<h4 style="color: #ffffff !important; font-size: 1.2rem;">Advanced Applications</h4>

<ul style="color: #ffffff !important; font-size: 1rem;">
    <li style="color: #ffffff !important;"><strong style="color: #ffffff !important;">Divergence</strong>: When price makes a new high/low but RSI doesn't, suggesting a potential reversal</li>
    <li style="color: #ffffff !important;"><strong style="color: #ffffff !important;">Failure Swings</strong>: When RSI crosses back above 30 (bullish) or below 70 (bearish) after a reversal</li>
    <li style="color: #ffffff !important;"><strong style="color: #ffffff !important;">Centerline Crossovers</strong>: RSI crossing above 50 is bullish, below 50 is bearish</li>
</ul>

<h3 style="color: #ffffff !important; font-size: 1.3rem;">RSI Settings and Adjustments</h3>

<ul style="color: #ffffff !important; font-size: 1rem;">
    <li style="color: #ffffff !important;"><strong style="color: #ffffff !important;">Time Period</strong>: Shorter periods (e.g., 7-10) increase sensitivity</li>
    <li style="color: #ffffff !important;"><strong style="color: #ffffff !important;">Threshold Levels</strong>: Can be adjusted based on market conditions (e.g., 80/20 in strong trends)</li>
</ul>

<h3 style="color: #ffffff !important; font-size: 1.3rem;">RSI Limitations</h3>

<ul style="color: #ffffff !important; font-size: 1rem;">
    <li style="color: #ffffff !important;">Can remain in overbought/oversold territory during strong trends</li>
    <li style="color: #ffffff !important;">May generate false signals in ranging markets</li>
    <li style="color: #ffffff !important;">Best used in conjunction with other indicators and analysis methods</li>
</ul>
</div>"""
}

# Quiz questions for different modules
QUIZ_QUESTIONS = {
    "intro_to_markets": [
        {
            "question": "What is a stock?",
            "options": [
                "A loan given to a company",
                "Ownership in a company",
                "A physical certificate",
                "A bond issued by a company"
            ],
            "correct": 1
        },
        {
            "question": "What does it mean when an investor 'buys the dip'?",
            "options": [
                "Purchasing stocks when prices are falling",
                "Buying only tech stocks",
                "Investing in cryptocurrency",
                "Purchasing dividend stocks"
            ],
            "correct": 0
        },
        {
            "question": "What is market capitalization?",
            "options": [
                "The total value of all stocks in the market",
                "The maximum price a stock has reached",
                "The total value of a company's outstanding shares",
                "The amount of money a company has in cash"
            ],
            "correct": 2
        }
    ],
    "intro_to_rsi": [
        {
            "question": "What does RSI stand for?",
            "options": [
                "Relative Strength Index",
                "Rapid Stock Indicator",
                "Return on Stock Investment",
                "Rate of Substantial Increase"
            ],
            "correct": 0
        },
        {
            "question": "What RSI value typically indicates an oversold condition?",
            "options": [
                "Over 70",
                "Between 40 and 60",
                "Under 30",
                "Exactly 50"
            ],
            "correct": 2
        },
        {
            "question": "What is the typical time period used for RSI calculation?",
            "options": [
                "7 days",
                "14 days",
                "30 days",
                "50 days"
            ],
            "correct": 1
        }
    ]
}

def init_gamification():
    """Initialize gamification state variables"""
    if 'user_badges' not in st.session_state:
        st.session_state.user_badges = {}
    
    if 'completed_modules' not in st.session_state:
        st.session_state.completed_modules = []
    
    if 'learning_progress' not in st.session_state:
        st.session_state.learning_progress = {}
    
    if 'backtests_run' not in st.session_state:
        st.session_state.backtests_run = []
        
    if 'total_points' not in st.session_state:
        st.session_state.total_points = 0
    
    if 'quiz_attempts' not in st.session_state:
        st.session_state.quiz_attempts = {}

def award_badge(badge_id):
    """Award a badge to the user if they don't already have it"""
    if badge_id in BADGES and badge_id not in st.session_state.user_badges:
        st.session_state.user_badges[badge_id] = {
            "earned_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "badge_info": BADGES[badge_id]
        }
        st.session_state.total_points += BADGES[badge_id]["points"]
        
        # Show animation with a flag to prevent duplicate showing
        if 'show_badge_unlock' not in st.session_state:
            st.session_state.show_badge_unlock = []
        
        # Add badge to the unlock queue
        st.session_state.show_badge_unlock.append(badge_id)
        
        return True
    return False

def display_badge_unlock(badge_id):
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
    
    # Wait briefly, then clear the modal
    time.sleep(0.1)  # Just wait enough for rendering, we'll handle closing later

def check_module_completion(module_id):
    """Check if a module has been completed and award the appropriate badge"""
    if module_id not in st.session_state.completed_modules:
        st.session_state.completed_modules.append(module_id)
        
        # Award badge based on module completion
        if module_id == "intro" and award_badge("first_steps"):
            st.success(f"🏆 You've earned the {BADGES['first_steps']['name']} badge!")
        elif module_id == "rsi" and award_badge("technical_novice"):
            st.success(f"🏆 You've earned the {BADGES['technical_novice']['name']} badge!")

def check_watchlist_achievements():
    """Check for watchlist-related achievements"""
    # Check for watchlist size achievements
    if 'selected_stocks' in st.session_state:
        num_stocks = len(st.session_state.selected_stocks)
        if num_stocks >= 5 and award_badge("diversified_investor"):
            st.success(f"🏆 You've earned the {BADGES.get('diversified_investor', {}).get('name', 'Diversified Investor')} badge!")
        if num_stocks >= 10 and award_badge("market_explorer"):
            st.success(f"🏆 You've earned the {BADGES.get('market_explorer', {}).get('name', 'Market Explorer')} badge!")

def check_personalized_learning_achievements():
    """Check for personalized learning path achievements"""
    # Simplified version - just check if learning_progress exists
    if 'learning_progress' in st.session_state and st.session_state.learning_progress:
        # Award initial assessment badge
        if award_badge("path_starter"):
            st.success(f"🏆 You've earned the {BADGES.get('path_starter', {}).get('name', 'Path Starter')} badge!")

def check_backtest_achievements(backtest_results, backtest_performance, benchmark_performance, strategy_type):
    """Check for backtest-related achievements"""
    # Track number of backtests run
    if 'backtests_run' not in st.session_state:
        st.session_state.backtests_run = []
    
    st.session_state.backtests_run.append({
        "strategy": strategy_type,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "performance": backtest_performance
    })
    
    # Award first backtest badge
    if len(st.session_state.backtests_run) == 1 and award_badge("backtest_explorer"):
        st.success(f"🏆 You've earned the {BADGES.get('backtest_explorer', {}).get('name', 'Backtest Explorer')} badge!")
    
    # Award strategy diversity badge
    unique_strategies = set([b["strategy"] for b in st.session_state.backtests_run])
    if len(unique_strategies) >= 3 and award_badge("strategy_diversifier"):
        st.success(f"🏆 You've earned the {BADGES.get('strategy_diversifier', {}).get('name', 'Strategy Diversifier')} badge!")

def display_badge_progress():
    """Display the user's badge progress"""
    st.header("🏆 Your Achievement Badges")
    
    # Initialize gamification state
    init_gamification()
    
    # Display overall progress
    total_badges = len(BADGES)
    earned_badges = len(st.session_state.user_badges)
    
    # Add some badges to showcase the functionality
    if st.session_state.user_badges == {}:
        # Award first_steps badge just for demonstration
        award_badge("first_steps")
    
    # Display total points with a counting animation
    points_container = st.empty()
    points_container.markdown(f"""
    <div style="margin: 15px 0; padding: 15px; background-color: #1E3A52; border-radius: 10px; text-align: center;">
        <div style="font-size: 1.2rem; margin-bottom: 5px;">Total Achievement Points</div>
        <div style="font-size: 2rem; font-weight: bold; color: #4CAF50;">
            {st.session_state.total_points}
        </div>
        <div style="font-size: 0.9rem; margin-top: 10px;">
            You've earned {earned_badges} out of {total_badges} badges 
            ({int(earned_badges/total_badges*100 if total_badges > 0 else 0)}% complete)
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Display badges by category
    categories = {
        "learning": "Learning Achievements",
        "personalized": "Personalized Learning Achievements",
        "quiz": "Quiz Achievements",
        "watchlist": "Watchlist Achievements",
        "backtest": "Backtesting Achievements"
    }
    
    # Create tabs for different badge categories
    tab_names = list(categories.values())
    badge_tabs = st.tabs(tab_names)
    
    # For each category, display earned and available badges
    for i, (category_key, category_name) in enumerate(categories.items()):
        with badge_tabs[i]:
            # Filter badges for this category
            category_badges = {badge_id: badge for badge_id, badge in BADGES.items() 
                             if badge.get("category") == category_key}
            
            if category_badges:
                # Create columns for displaying badges
                cols = st.columns(min(3, len(category_badges)))
                
                # Display badges in columns
                for j, (badge_id, badge) in enumerate(category_badges.items()):
                    col_idx = j % len(cols)
                    with cols[col_idx]:
                        # Check if badge is earned
                        is_earned = badge_id in st.session_state.user_badges
                        
                        # Generate random animation parameters (for variety)
                        animation_delay = j * 0.15  # Stagger the animations
                        pulse_duration = 3.0
                        rotation_duration = 4.0
                        scale_duration = 3.0
                        
                        if is_earned:
                            # Earned badge with animations
                            st.markdown(f"""
                            <div style="text-align: center; 
                                border: 2px solid #4CAF50; 
                                border-radius: 10px; 
                                padding: 15px; 
                                margin: 10px 0; 
                                background-color: #1E3A52;
                                animation: fadeIn 0.5s ease-in-out {animation_delay}s both, 
                                         badgePulse {pulse_duration}s ease-in-out infinite;">
                                <div style="font-size: 3rem; margin-bottom: 10px;
                                          animation: badgeRotate {rotation_duration}s ease-in-out infinite, 
                                                  badgeScale {scale_duration}s ease-in-out infinite;">
                                    {badge['icon']}
                                </div>
                                <div style="font-weight: bold; color: #4CAF50; font-size: 1.2rem;">{badge['name']}</div>
                                <div style="font-size: 0.8rem; margin: 8px 0;">{badge['description']}</div>
                                <div style="font-size: 1rem; color: #4CAF50; margin-top: 8px;">+{badge['points']} points</div>
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
                            # Locked badge with subtle animation
                            st.markdown(f"""
                            <div style="text-align: center; 
                                border: 2px solid #555; 
                                border-radius: 10px; 
                                padding: 15px; 
                                margin: 10px 0; 
                                background-color: #333333;
                                opacity: 0.7;
                                animation: fadeIn 0.5s ease-in-out {animation_delay}s both,
                                         lockedPulse 3s ease-in-out infinite;">
                                <div style="font-size: 3rem; margin-bottom: 10px; opacity: 0.6; filter: grayscale(100%);">
                                    {badge['icon']}
                                </div>
                                <div style="font-weight: bold; color: #999; font-size: 1.2rem;">{badge['name']}</div>
                                <div style="font-size: 0.8rem; margin: 8px 0; color: #888;">{badge['description']}</div>
                                <div style="font-size: 1rem; color: #777; margin-top: 8px;">{badge['points']} points</div>
                            </div>
                            
                            <style>
                            @keyframes fadeIn {{
                                0% {{ opacity: 0; transform: translateY(10px); }}
                                100% {{ opacity: 0.7; transform: translateY(0); }}
                            }}
                            
                            @keyframes lockedPulse {{
                                0% {{ opacity: 0.7; }}
                                50% {{ opacity: 0.8; }}
                                100% {{ opacity: 0.7; }}
                            }}
                            </style>
                            """, unsafe_allow_html=True)
            else:
                st.write("No badges in this category yet.")

def check_badge_unlocks():
    """Check if there are any badges to be unlocked and display animations for them"""
    if 'show_badge_unlock' not in st.session_state:
        return
    
    if st.session_state.show_badge_unlock:
        # Get the first badge in the queue
        badge_id = st.session_state.show_badge_unlock.pop(0)
        
        # Display the badge unlock animation
        display_badge_unlock(badge_id)

def learning_section():
    """Display the learning section with modules, quizzes, personalized learning and achievements"""
    st.title("📚 Learning Center")
    
    # Initialize gamification state
    init_gamification()
    
    # Check for badge unlocks at the start of the page
    check_badge_unlocks()
    
    # Create tabs for learning, interactive quiz, and achievements
    learning_tab, quiz_tab, personalized_tab, badges_tab = st.tabs([
        "Learning Modules", 
        "Interactive Quiz", 
        "Personalized Learning", 
        "Your Achievements"
    ])
    
    with learning_tab:
        display_learning_modules()
    
    with quiz_tab:
        try:
            # Try to import and use market_quiz module
            from market_quiz import quiz_section
            quiz_section()
        except Exception as e:
            st.error(f"Unable to load quiz module: {e}")
            st.write("The interactive quiz features are not available.")
    
    with personalized_tab:
        try:
            # Try to import and use learning_path module
            from learning_path import personalized_learning_path
            personalized_learning_path()
        except Exception as e:
            st.error(f"Unable to load personalized learning path: {e}")
            st.write("The personalized learning features are not available.")
            
            # Fallback content
            st.subheader("What is Personalized Learning?")
            st.write("""
            The personalized learning path adapts to your knowledge level and interests in trading and technical analysis.
            It starts with an assessment to determine your current knowledge level and then recommends specific modules
            and bite-sized tutorials tailored to your needs.
            """)
    
    with badges_tab:
        display_badge_progress()

def display_learning_modules():
    """Display the learning modules with progress tracking"""
    st.header("Learning Modules")
    st.write("Expand your knowledge of technical analysis and trading strategies through these interactive modules.")
    
    # Check if we're viewing a specific module
    if 'active_module' in st.session_state:
        view_learning_module(st.session_state.active_module)
        return
    
    # Initialize gamification state if not already done
    init_gamification()
    
    # Display modules from the LEARNING_MODULES dictionary
    for module_id, module in LEARNING_MODULES.items():
        completed = module_id in st.session_state.completed_modules
        
        st.markdown(f"""
        <div style="padding: 15px; border-radius: 5px; border: 1px solid #ddd; margin-bottom: 15px;">
            <div style="display: flex; justify-content: space-between;">
                <div>
                    <h3>{module['name']} {"✓" if completed else ""}</h3>
                    <p>{module['description']}</p>
                    <p><strong>Difficulty:</strong> {module['difficulty']} | <strong>Estimated Time:</strong> {module['estimated_time']}</p>
                </div>
                <div>
                    {'<div style="background-color: #4CAF50; color: white; padding: 5px 10px; border-radius: 5px; display: inline-block;">Completed</div>' if completed else ''}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Add buttons for module interaction
        col1, col2, col3 = st.columns([2, 1, 1])
        with col3:
            if completed:
                if st.button("Review", key=f"review_{module_id}", help="Review this module again"):
                    st.session_state.active_module = module_id
                    st.rerun()
            else:
                if st.button("Start", key=f"start_{module_id}", help="Start this module"):
                    st.session_state.active_module = module_id
                    st.rerun()

def view_learning_module(module_id):
    """View a specific learning module"""
    if module_id not in LEARNING_MODULES:
        st.error("Module not found.")
        if st.button("Back to Modules"):
            del st.session_state.active_module
            st.rerun()
        return
    
    module = LEARNING_MODULES[module_id]
    
    # Add back button
    col1, col2 = st.columns([6, 1])
    with col2:
        if st.button("← Back"):
            del st.session_state.active_module
            st.rerun()
    
    # Display module title and description
    st.title(module["name"])
    st.write(f"**Difficulty:** {module['difficulty']} | **Estimated Time:** {module['estimated_time']}")
    
    # Module content
    st.markdown("""
    ## Introduction
    Technical analysis is the study of historical price and volume patterns to predict future market movements. Unlike fundamental analysis, which focuses on a company's financial health, technical analysis concentrates on chart patterns, indicators, and statistical trends.
    
    ## Key Concepts
    
    ### 1. Chart Types
    - **Candlestick Charts**: Show open, high, low, and close prices in a single bar
    - **Line Charts**: Simple representation of closing prices over time
    - **Bar Charts**: Show open, high, low, and close in a more compact format
    
    ### 2. Technical Indicators
    - **Trend Indicators**: Moving Averages, MACD
    - **Momentum Indicators**: RSI, Stochastic Oscillator
    - **Volatility Indicators**: Bollinger Bands, ATR
    - **Volume Indicators**: OBV, Volume Profile
    
    ### 3. Chart Patterns
    - **Continuation Patterns**: Flags, Pennants, Triangles
    - **Reversal Patterns**: Head and Shoulders, Double Tops/Bottoms
    
    ## Why Technical Analysis Matters
    Technical analysis helps traders:
    1. Identify potential entry and exit points
    2. Manage risk with specific stop-loss levels
    3. Understand market psychology and sentiment
    4. Develop systematic trading strategies
    
    ## Common Myths about Technical Analysis
    - **Myth 1**: Technical analysis always works
    - **Reality**: No analysis method is perfect; it's about probability, not certainty
    - **Myth 2**: More indicators mean better analysis
    - **Reality**: Too many indicators often lead to confusion and analysis paralysis
    """)
    
    # Module quiz
    st.markdown("---")
    st.header("Knowledge Check")
    
    if module_id in QUIZ_QUESTIONS:
        # Display quiz
        st.write("Test your knowledge with these questions:")
        
        if 'quiz_answers' not in st.session_state:
            st.session_state.quiz_answers = {}
        
        questions = QUIZ_QUESTIONS[module_id]
        correct_count = 0
        total_questions = len(questions)
        
        for i, question in enumerate(questions):
            st.subheader(f"Question {i+1}: {question['question']}")
            
            # Generate a unique key for this question
            q_key = f"{module_id}_q{i}"
            
            # Display options as radio buttons
            selected_option = st.radio(
                "Select your answer:",
                question['options'],
                key=q_key
            )
            
            # Store the selected answer
            selected_index = question['options'].index(selected_option)
            st.session_state.quiz_answers[q_key] = selected_index
            
            # Check if correct
            if selected_index == question['correct']:
                correct_count += 1
                st.success("Correct!")
            else:
                st.error(f"Incorrect. The correct answer is: {question['options'][question['correct']]}")
            
            st.markdown("---")
        
        # Show results
        score_percentage = (correct_count / total_questions) * 100
        st.subheader(f"Quiz Results: {correct_count}/{total_questions} ({score_percentage:.0f}%)")
        
        if score_percentage >= 80:
            st.success("Congratulations! You've passed this module's quiz!")
            
            # Mark module as completed if not already
            if module_id not in st.session_state.completed_modules:
                st.session_state.completed_modules.append(module_id)
                
                # Award badge if applicable
                badge_id = module.get("badge")
                if badge_id and badge_id in BADGES and award_badge(badge_id):
                    st.balloons()
                    st.success(f"🏆 You've earned the {BADGES[badge_id]['name']} badge!")
        else:
            st.warning("You need to score at least 80% to complete this module. Try again!")
    else:
        st.info("No quiz questions available for this module yet.")
        
        # Option to mark as completed anyway
        if module_id not in st.session_state.completed_modules:
            if st.button("Mark as Completed"):
                st.session_state.completed_modules.append(module_id)
                
                # Award badge if applicable
                badge_id = module.get("badge")
                if badge_id and badge_id in BADGES and award_badge(badge_id):
                    st.balloons()
                    st.success(f"🏆 You've earned the {BADGES[badge_id]['name']} badge!")
                
                st.rerun()
