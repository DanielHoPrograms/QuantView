import streamlit as st
import pandas as pd
import numpy as np
import json
import os
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
    "challenge_taker": {
        "name": "Challenge Taker", 
        "description": "Complete your first learning challenge",
        "icon": "🎯",
        "points": 15,
        "category": "personalized"
    },
    "challenge_master": {
        "name": "Challenge Master", 
        "description": "Complete 5 learning challenges",
        "icon": "🏆",
        "points": 40,
        "category": "personalized"
    },
    
    # Quiz achievements
    "quiz_taker": {
        "name": "Quiz Taker", 
        "description": "Complete your first interactive quiz",
        "icon": "❓",
        "points": 15,
        "category": "quiz"
    },
    "quiz_ace": {
        "name": "Quiz Ace", 
        "description": "Score 100% on any interactive quiz",
        "icon": "🎯",
        "points": 25,
        "category": "quiz"
    },
    "chart_analyst": {
        "name": "Chart Analyst", 
        "description": "Successfully complete a chart reading quiz",
        "icon": "📊",
        "points": 30,
        "category": "quiz"
    },
    "knowledge_seeker": {
        "name": "Knowledge Seeker", 
        "description": "Complete quizzes in 3 different categories",
        "icon": "🔍",
        "points": 40,
        "category": "quiz"
    },
    "market_scholar": {
        "name": "Market Scholar", 
        "description": "Complete all categories of quizzes",
        "icon": "🎓",
        "points": 75,
        "category": "quiz"
    },
    
    # Application usage achievements
    "watchlist_builder": {
        "name": "Watchlist Builder", 
        "description": "Add 5 stocks to your watchlist",
        "icon": "👁️",
        "points": 15,
        "category": "usage"
    },
    "parameter_tweaker": {
        "name": "Parameter Tweaker", 
        "description": "Modify technical indicator parameters",
        "icon": "🔧",
        "points": 20,
        "category": "usage"
    },
    "first_signal": {
        "name": "First Signal", 
        "description": "Identify your first buy signal",
        "icon": "🎯",
        "points": 25,
        "category": "usage"
    },
    "backtest_rookie": {
        "name": "Backtest Rookie", 
        "description": "Complete your first strategy backtest",
        "icon": "🧪",
        "points": 30,
        "category": "usage"
    },
    "strategy_developer": {
        "name": "Strategy Developer", 
        "description": "Test 3 different trading strategies",
        "icon": "🧩",
        "points": 40,
        "category": "usage"
    },
    "performance_analyzer": {
        "name": "Performance Analyzer", 
        "description": "Compare 5 different backtests",
        "icon": "📋",
        "points": 50,
        "category": "usage"
    },
    
    # Advanced achievements
    "profitable_strategy": {
        "name": "Profitable Strategy", 
        "description": "Develop a strategy that outperforms buy & hold",
        "icon": "💰",
        "points": 75,
        "category": "advanced"
    },
    "sharpe_optimizer": {
        "name": "Sharpe Optimizer", 
        "description": "Create a strategy with Sharpe ratio > 1.0",
        "icon": "⚡",
        "points": 75,
        "category": "advanced"
    },
    "drawdown_defender": {
        "name": "Drawdown Defender", 
        "description": "Create a strategy with max drawdown < 15%",
        "icon": "🛑",
        "points": 75,
        "category": "advanced"
    },
    "market_master": {
        "name": "Market Master", 
        "description": "Earn all other badges",
        "icon": "👑",
        "points": 200,
        "category": "advanced"
    }
}

# Define learning modules
LEARNING_MODULES = {
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
        "estimated_time": "40 min",
        "difficulty": "Advanced"
    },
    "risk_management": {
        "name": "Risk Management Principles",
        "description": "Learn how to manage risk in your trading strategy",
        "badge": "risk_manager",
        "estimated_time": "30 min",
        "difficulty": "Advanced"
    }
}

# Quiz questions for each module
QUIZ_QUESTIONS = {
    "intro_to_markets": [
        {
            "question": "What is a stock?",
            "options": [
                "A loan given to a company",
                "A share of ownership in a company",
                "A bond issued by a government",
                "A physical product sold by a company"
            ],
            "correct": 1  # Index of correct answer
        },
        {
            "question": "What happens when a company goes public?",
            "options": [
                "It sells shares to private investors only",
                "It buys back all existing shares",
                "It sells shares on a stock exchange to the public",
                "It merges with another company"
            ],
            "correct": 2
        },
        {
            "question": "What is a market index?",
            "options": [
                "The price of a single stock",
                "A basket of stocks representing a market or sector",
                "The total number of shares traded in a day",
                "The interest rate set by the central bank"
            ],
            "correct": 1
        },
        {
            "question": "What is a bull market?",
            "options": [
                "A market where prices are falling",
                "A market where prices are rising",
                "A market with high volatility",
                "A market with low trading volume"
            ],
            "correct": 1
        },
        {
            "question": "What does 'market capitalization' refer to?",
            "options": [
                "The total value of a company's outstanding shares",
                "The total profit a company makes annually",
                "The number of employees in a company",
                "The age of a company since its founding"
            ],
            "correct": 0
        }
    ],
    "candlestick_charts": [
        {
            "question": "What does a green (or white) candlestick represent?",
            "options": [
                "The price closed lower than it opened",
                "The price closed higher than it opened",
                "The price didn't change",
                "Trading was halted"
            ],
            "correct": 1
        },
        {
            "question": "What is the 'body' of a candlestick?",
            "options": [
                "The entire range from high to low",
                "The difference between open and close prices",
                "The volume of trades",
                "The time period of the candle"
            ],
            "correct": 1
        },
        {
            "question": "What pattern is formed when a small-bodied candle follows a large candle in the opposite direction?",
            "options": [
                "Doji",
                "Hammer",
                "Engulfing pattern",
                "Harami pattern"
            ],
            "correct": 3
        },
        {
            "question": "What does a long lower shadow (wick) typically indicate?",
            "options": [
                "Buyers pushed the price up after it fell",
                "Sellers pushed the price down after it rose",
                "The price remained stable throughout the period",
                "The market was closed for part of the period"
            ],
            "correct": 0
        },
        {
            "question": "What is a 'doji' candlestick?",
            "options": [
                "A candle with a very long body",
                "A candle where open and close prices are nearly equal",
                "A candle with no shadows/wicks",
                "Two identical candles in a row"
            ],
            "correct": 1
        }
    ],
    # Add similar question sets for other modules
    "intro_to_rsi": [
        {
            "question": "What does RSI stand for?",
            "options": [
                "Relative Stock Index",
                "Relative Strength Index",
                "Rate of Stock Increase",
                "Rapid Selling Indicator"
            ],
            "correct": 1
        },
        {
            "question": "What is the standard range for RSI values?",
            "options": [
                "0 to 100",
                "-100 to +100",
                "0 to 1",
                "-1 to +1"
            ],
            "correct": 0
        },
        {
            "question": "What RSI value typically indicates an oversold condition?",
            "options": [
                "Above 70",
                "Below 30",
                "Exactly 50",
                "Above 90"
            ],
            "correct": 1
        },
        {
            "question": "What is typically considered the default period for RSI calculation?",
            "options": [
                "7 days",
                "14 days",
                "30 days",
                "50 days"
            ],
            "correct": 1
        },
        {
            "question": "What does RSI divergence indicate?",
            "options": [
                "Price and RSI are moving in the same direction",
                "Price and RSI are moving in opposite directions",
                "RSI is staying constant",
                "Trading volume is increasing"
            ],
            "correct": 1
        }
    ],
    # Additional module questions would be defined here
}

# Learning module content - HTML formatted for Streamlit
MODULE_CONTENT = {
    "intro_to_markets": """
    <h2>Introduction to Stock Markets</h2>
    
    <h3>What is a Stock Market?</h3>
    <p>A stock market is a public market where company shares are traded. It serves two primary functions:</p>
    <ul>
        <li>Allowing companies to raise capital by selling ownership shares to the public</li>
        <li>Providing investors with an opportunity to participate in the financial achievements of companies</li>
    </ul>
    
    <h3>Key Stock Market Concepts</h3>
    
    <h4>Stocks (Shares)</h4>
    <p>A stock represents a share of ownership in a company. When you own a stock, you own a piece of that company.</p>
    
    <h4>Exchanges</h4>
    <p>Stock exchanges are organized marketplaces where stocks are bought and sold. Examples include:</p>
    <ul>
        <li>New York Stock Exchange (NYSE)</li>
        <li>NASDAQ</li>
        <li>Tokyo Stock Exchange</li>
        <li>London Stock Exchange</li>
    </ul>
    
    <h4>Market Indices</h4>
    <p>Indices track the performance of a group of stocks, representing a specific market or sector:</p>
    <ul>
        <li>S&P 500 - 500 large U.S. companies</li>
        <li>Dow Jones Industrial Average - 30 major U.S. companies</li>
        <li>NASDAQ Composite - Companies listed on the NASDAQ exchange</li>
        <li>Nikkei 225 - Major Japanese companies</li>
    </ul>
    
    <h4>Bull vs. Bear Markets</h4>
    <ul>
        <li><strong>Bull Market:</strong> Extended period of rising stock prices (usually 20% or more)</li>
        <li><strong>Bear Market:</strong> Extended period of falling stock prices (usually 20% or more)</li>
    </ul>
    
    <h3>How Stock Prices are Determined</h3>
    <p>Stock prices are determined by supply and demand in the market. Factors that influence prices include:</p>
    <ul>
        <li>Company performance and earnings</li>
        <li>Economic conditions</li>
        <li>Industry trends</li>
        <li>Investor sentiment</li>
        <li>News and events</li>
    </ul>
    
    <h3>Types of Stock Analysis</h3>
    <ul>
        <li><strong>Fundamental Analysis:</strong> Evaluating a company's financial health, management, competitive advantages, and growth prospects</li>
        <li><strong>Technical Analysis:</strong> Studying price movements and trading volume to forecast future price movements</li>
    </ul>
    
    <p>This application focuses primarily on technical analysis tools, which we'll explore in subsequent modules.</p>
    """,
    
    "candlestick_charts": """
    <h2>Understanding Candlestick Charts</h2>
    
    <h3>What are Candlestick Charts?</h3>
    <p>Candlestick charts are a type of financial chart that shows price movements in a visually intuitive way. They originated in Japan in the 18th century for trading rice and were introduced to the Western world in the 1990s.</p>
    
    <h3>Anatomy of a Candlestick</h3>
    <p>Each candlestick represents a specific time period (day, hour, minute, etc.) and shows four key price points:</p>
    <ul>
        <li><strong>Open:</strong> The price at the beginning of the time period</li>
        <li><strong>Close:</strong> The price at the end of the time period</li>
        <li><strong>High:</strong> The highest price during the time period</li>
        <li><strong>Low:</strong> The lowest price during the time period</li>
    </ul>
    
    <h4>Components:</h4>
    <ul>
        <li><strong>Body:</strong> The rectangle between the open and close prices</li>
        <li><strong>Shadows/Wicks:</strong> The lines extending from the body to the high and low prices</li>
    </ul>
    
    <h3>Candlestick Colors</h3>
    <ul>
        <li><strong>Green/White Candle:</strong> Closing price is higher than opening price (bullish)</li>
        <li><strong>Red/Black Candle:</strong> Closing price is lower than opening price (bearish)</li>
    </ul>
    
    <h3>Common Candlestick Patterns</h3>
    
    <h4>Single Candlestick Patterns</h4>
    <ul>
        <li><strong>Doji:</strong> Open and close prices are nearly equal, indicating indecision</li>
        <li><strong>Hammer:</strong> Small body with a long lower shadow, potential bullish reversal signal</li>
        <li><strong>Shooting Star:</strong> Small body with a long upper shadow, potential bearish reversal signal</li>
    </ul>
    
    <h4>Multi-Candlestick Patterns</h4>
    <ul>
        <li><strong>Engulfing Pattern:</strong> A candle that completely "engulfs" the body of the previous candle</li>
        <li><strong>Harami:</strong> A small-bodied candle contained within the body of the previous candle</li>
        <li><strong>Morning Star:</strong> Three-candle pattern signaling a potential bullish reversal</li>
        <li><strong>Evening Star:</strong> Three-candle pattern signaling a potential bearish reversal</li>
    </ul>
    
    <h3>How to Read Candlestick Charts</h3>
    <p>When analyzing candlestick charts, look for:</p>
    <ul>
        <li>The overall trend direction</li>
        <li>Support and resistance levels</li>
        <li>Recognizable patterns that may signal reversals or continuations</li>
        <li>Volume confirmation of price movements</li>
    </ul>
    
    <p>Candlestick charts provide a wealth of information about market psychology and can help you make more informed trading decisions when combined with other technical indicators.</p>
    """,
    
    "intro_to_rsi": """
    <h2>Introduction to RSI (Relative Strength Index)</h2>
    
    <h3>What is RSI?</h3>
    <p>The Relative Strength Index (RSI) is a momentum oscillator that measures the speed and change of price movements. Developed by J. Welles Wilder Jr. in 1978, it oscillates between 0 and 100, helping traders identify overbought and oversold conditions.</p>
    
    <h3>How RSI is Calculated</h3>
    <p>The RSI calculation involves several steps:</p>
    <ol>
        <li>Calculate average gains and losses over a specified period (typically 14 days)</li>
        <li>Compute the relative strength (RS) as: RS = Average Gain / Average Loss</li>
        <li>Convert RS to RSI: RSI = 100 - (100 / (1 + RS))</li>
    </ol>
    
    <h3>Interpreting RSI Values</h3>
    <ul>
        <li><strong>RSI > 70:</strong> Generally considered overbought, may indicate a potential reversal or corrective pullback</li>
        <li><strong>RSI < 30:</strong> Generally considered oversold, may indicate a potential reversal or bullish movement</li>
        <li><strong>RSI = 50:</strong> Often used as a centerline, indicating neutral momentum</li>
    </ul>
    
    <h3>RSI Trading Strategies</h3>
    
    <h4>Overbought/Oversold Strategy</h4>
    <p>The most common RSI strategy:</p>
    <ul>
        <li>Buy when RSI falls below 30 and then rises back above it</li>
        <li>Sell when RSI rises above 70 and then falls back below it</li>
    </ul>
    
    <h4>RSI Divergence</h4>
    <p>A powerful signal that occurs when price and RSI move in opposite directions:</p>
    <ul>
        <li><strong>Bullish Divergence:</strong> Price makes lower lows, but RSI makes higher lows (potential upward reversal)</li>
        <li><strong>Bearish Divergence:</strong> Price makes higher highs, but RSI makes lower highs (potential downward reversal)</li>
    </ul>
    
    <h4>RSI Failure Swings</h4>
    <p>Another pattern to watch for:</p>
    <ul>
        <li><strong>Bullish Failure Swing:</strong> RSI falls below 30, rises, pulls back but stays above 30, then breaks above its prior high</li>
        <li><strong>Bearish Failure Swing:</strong> RSI rises above 70, falls, bounces but stays below 70, then breaks below its prior low</li>
    </ul>
    
    <h3>Adjusting RSI Settings</h3>
    <p>While 14 periods is the default setting, you can adjust it based on your trading style:</p>
    <ul>
        <li><strong>Shorter periods (e.g., 7-10):</strong> More sensitive, generates more signals, better for short-term trading</li>
        <li><strong>Longer periods (e.g., 20-30):</strong> Less sensitive, generates fewer but potentially more reliable signals</li>
    </ul>
    
    <h3>Limitations of RSI</h3>
    <ul>
        <li>Can remain in overbought/oversold territory for extended periods during strong trends</li>
        <li>May generate false signals in ranging markets</li>
        <li>Should be used in conjunction with other technical indicators and price action analysis</li>
    </ul>
    
    <p>The RSI is one of the most widely used technical indicators due to its simplicity and effectiveness in identifying potential entry and exit points.</p>
    """,
    
    # Additional module content would be defined here
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
        return True
    return False

def check_module_completion(module_id):
    """Check if a module has been completed and award the appropriate badge"""
    if module_id in LEARNING_MODULES and module_id not in st.session_state.completed_modules:
        badge_id = LEARNING_MODULES[module_id].get("badge")
        if badge_id:
            # Import here to avoid circular imports
            from animated_badges import award_badge_with_animation
            award_badge_with_animation(badge_id)
        
        st.session_state.completed_modules.append(module_id)
        
        # Check for completing all modules
        if set(LEARNING_MODULES.keys()).issubset(set(st.session_state.completed_modules)):
            from animated_badges import award_badge_with_animation
            award_badge_with_animation("market_wisdom")

def check_watchlist_achievements():
    """Check for watchlist-related achievements"""
    if len(st.session_state.selected_stocks) >= 5 and "watchlist_builder" not in st.session_state.user_badges:
        from animated_badges import award_badge_with_animation
        award_badge_with_animation("watchlist_builder")

def check_personalized_learning_achievements():
    """Check for personalized learning path achievements"""
    # Import here to avoid circular imports
    from animated_badges import award_badge_with_animation
    
    # Check if diagnostic test is completed
    if 'diagnostic_completed' in st.session_state and st.session_state.diagnostic_completed:
        # Award badge for completing assessment
        if "path_starter" not in st.session_state.user_badges:
            award_badge_with_animation("path_starter")
        
        # Check for completed modules in personalized path
        if 'learning_path_progress' in st.session_state:
            completed_modules = [m for m, p in st.session_state.learning_path_progress.items() 
                               if p.get("completed", False)]
            
            # Award badge for completing 3 modules
            if len(completed_modules) >= 3 and "path_explorer" not in st.session_state.user_badges:
                award_badge_with_animation("path_explorer")
            
        # Check knowledge level advancement
        if 'knowledge_level' in st.session_state:
            if st.session_state.knowledge_level == "intermediate" and "knowledge_climber" not in st.session_state.user_badges:
                award_badge_with_animation("knowledge_climber")
            
            if st.session_state.knowledge_level in ["advanced", "expert"] and "market_expert" not in st.session_state.user_badges:
                award_badge_with_animation("market_expert")
        
        # Check completed challenges
        if 'challenge_completed' in st.session_state:
            if len(st.session_state.challenge_completed) >= 1 and "challenge_taker" not in st.session_state.user_badges:
                award_badge_with_animation("challenge_taker")
                    
            if len(st.session_state.challenge_completed) >= 5 and "challenge_master" not in st.session_state.user_badges:
                award_badge_with_animation("challenge_master")

def check_backtest_achievements(backtest_results, backtest_performance, benchmark_performance, strategy_type):
    """Check for backtest-related achievements"""
    # Import here to avoid circular imports
    from animated_badges import award_badge_with_animation
    
    # First backtest
    if "backtest_rookie" not in st.session_state.user_badges:
        award_badge_with_animation("backtest_rookie")
    
    # Record strategy type
    st.session_state.backtests_run.append(strategy_type)
    
    # Check if user has tested 3 different strategies
    unique_strategies = set(st.session_state.backtests_run)
    if len(unique_strategies) >= 3 and "strategy_developer" not in st.session_state.user_badges:
        award_badge_with_animation("strategy_developer")
    
    # Check if strategy outperforms buy & hold
    if backtest_performance['total_return'] > benchmark_performance['total_return'] and "profitable_strategy" not in st.session_state.user_badges:
        award_badge_with_animation("profitable_strategy")
    
    # Check Sharpe ratio
    if backtest_performance['sharpe'] > 1.0 and "sharpe_optimizer" not in st.session_state.user_badges:
        award_badge_with_animation("sharpe_optimizer")
    
    # Check drawdown
    if backtest_performance['max_drawdown'] < 0.15 and "drawdown_defender" not in st.session_state.user_badges:
        award_badge_with_animation("drawdown_defender")
    
    # Check if all other badges are earned
    if len(st.session_state.user_badges) >= len(BADGES) - 1 and "market_master" not in st.session_state.user_badges:
        award_badge_with_animation("market_master")

def display_badge_progress():
    """Display the user's badge progress"""
    # Use the animated badge display instead of the static one
    from animated_badges import display_animated_badge_progress
    
    # Show animated badges
    display_animated_badge_progress()
    
    # Add a button to view detailed achievement statistics
    if st.button("View Achievement Statistics", key="view_achievement_stats"):
        st.session_state.show_achievement_stats = True
    
    # Show achievement statistics if requested
    if st.session_state.get('show_achievement_stats', False):
        from animated_badges import display_achievement_statistics
        display_achievement_statistics()
        
        # Add a button to close statistics
        if st.button("Close Statistics", key="close_achievement_stats"):
            st.session_state.show_achievement_stats = False

def display_learning_modules():
    """Display the learning modules with progress tracking and animations"""
    st.subheader("Learning Modules")
    st.markdown("Complete these modules to learn about stock market concepts and earn badges.")
    
    # Import animations
    from animations import create_typing_animation
    
    # Create dynamic introductory text
    intro_container = st.empty()
    create_typing_animation(
        "Select a module below to start your learning journey. Each module you complete will earn you achievement badges!",
        container=intro_container,
        speed=0.02
    )
    
    # Group modules by difficulty
    modules_by_difficulty = {}
    for module_id, module in LEARNING_MODULES.items():
        difficulty = module.get("difficulty", "Beginner")
        if difficulty not in modules_by_difficulty:
            modules_by_difficulty[difficulty] = []
        modules_by_difficulty[difficulty].append((module_id, module))
    
    # Create tabs for difficulty levels
    difficulties = ["Beginner", "Intermediate", "Advanced"]
    tabs = st.tabs(difficulties)
    
    for i, difficulty in enumerate(difficulties):
        with tabs[i]:
            if difficulty not in modules_by_difficulty:
                st.write(f"No {difficulty} modules available yet.")
                continue
            
            modules = modules_by_difficulty[difficulty]
            
            # Display module cards with animations
            for j, (module_id, module) in enumerate(modules):
                completed = module_id in st.session_state.completed_modules
                badge_id = module.get("badge")
                badge_earned = badge_id in st.session_state.user_badges if badge_id else False
                
                # Create animated module card with hover effects
                animation_delay = j * 0.2  # Stagger animations
                card_container = st.container()
                with card_container:
                    # Use HTML/CSS for animated card with hover effects
                    st.markdown(f"""
                    <div style="border: 1px solid #444; border-radius: 10px; padding: 15px; margin-bottom: 20px;
                              background-color: {'rgba(76, 175, 80, 0.1)' if completed else 'rgba(12, 123, 220, 0.05)'};
                              transition: all 0.3s ease; transform: translateY(0px);
                              animation: cardAppear 0.5s ease-out {animation_delay}s both;"
                         onmouseover="this.style.transform='translateY(-5px)'; this.style.boxShadow='0 10px 20px rgba(0,0,0,0.2)';"
                         onmouseout="this.style.transform='translateY(0px)'; this.style.boxShadow='none';">
                         
                        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                            <div style="flex: 3;">
                                <h3 style="color: {'#4CAF50' if completed else '#0C7BDC'};">
                                    {('✅ ' if completed else '') + module['name']}
                                </h3>
                                <p>{module['description']}</p>
                                <div style="display: flex; justify-content: flex-start; margin-bottom: 10px; gap: 20px;">
                                    <span>⏱️ {module.get('estimated_time', 'N/A')}</span>
                                    <span>📊 {difficulty}</span>
                                    {f'<span>{BADGES[badge_id]["icon"]} {BADGES[badge_id]["name"]} {"✓" if badge_earned else "🔸"}</span>' if badge_id and badge_id in BADGES else ''}
                                </div>
                            </div>
                            <div style="flex: 1; text-align: right;">
                                {'<div style="background-color: #4CAF50; color: white; padding: 5px 10px; border-radius: 5px; display: inline-block;">Completed</div>' if completed else ''}
                            </div>
                        </div>
                    </div>
                    
                    <style>
                    @keyframes cardAppear {
                        from { opacity: 0; transform: translateY(20px); }
                        to { opacity: 1; transform: translateY(0); }
                    }
                    </style>
                    """, unsafe_allow_html=True)
                    
                    # Add buttons for module interaction
                    col1, col2, col3 = st.columns([2, 1, 1])
                    with col3:
                        if completed:
                            if st.button("Review", key=f"review_{module_id}", help="Review this module again"):
                                st.session_state.active_module = module_id
                                st.rerun()
                        else:
                            if st.button("Start Learning", key=f"start_{module_id}", help="Begin this learning module"):
                                st.session_state.active_module = module_id
                                st.rerun()
                
                st.markdown("---")

def display_module_content(module_id):
    """Display the content for a specific learning module with animations"""
    if module_id not in LEARNING_MODULES:
        st.error("Module not found!")
        return
    
    # Import animations
    from animations import create_typing_animation, show_animated_notification, loading_animation
    
    module = LEARNING_MODULES[module_id]
    
    # Header with animation
    header_container = st.empty()
    loading_animation("Loading module content...", container=header_container)
    
    st.markdown(f"""
    <h1 style="animation: fadeIn 0.8s ease-out both;">{module['name']}</h1>
    <p style="font-style: italic; animation: fadeIn 1s ease-out 0.3s both;">{module['description']}</p>
    
    <style>
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Show badge info if available
    badge_id = module.get("badge")
    if badge_id and badge_id in BADGES:
        badge = BADGES[badge_id]
        badge_earned = badge_id in st.session_state.get('user_badges', {})
        
        st.markdown(f"""
        <div style="margin: 20px 0; padding: 15px; border-radius: 10px; 
                  background-color: {'rgba(76, 175, 80, 0.1)' if badge_earned else 'rgba(255, 193, 7, 0.1)'};
                  border: 1px solid {'#4CAF50' if badge_earned else '#FFC107'};
                  animation: badgeAppear 1s ease-out 0.5s both;">
            <div style="display: flex; align-items: center;">
                <div style="font-size: 40px; margin-right: 15px; animation: {'badgePulse 2s infinite' if badge_earned else 'none'}">
                    {badge['icon']}
                </div>
                <div>
                    <div style="font-weight: bold; font-size: 18px; margin-bottom: 5px;">
                        {badge['name']} {' ✓' if badge_earned else ''}
                    </div>
                    <div>{badge['description']}</div>
                    <div style="margin-top: 5px; color: {'#4CAF50' if badge_earned else '#999'};">
                        {f"Earned: {st.session_state.user_badges[badge_id].get('earned_date', 'N/A')}" if badge_earned else f"+{badge['points']} points when completed"}
                    </div>
                </div>
            </div>
        </div>
        
        <style>
        @keyframes badgeAppear {
            from { opacity: 0; transform: scale(0.9); }
            to { opacity: 1; transform: scale(1); }
        }
        
        @keyframes badgePulse {
            from { transform: scale(1); }
            50% { transform: scale(1.1); }
            to { transform: scale(1); }
        }
        </style>
        """, unsafe_allow_html=True)
    
    # Content tabs with animation
    content_tab, quiz_tab = st.tabs(["Learn", "Quiz"])
    
    with content_tab:
        learn_container = st.empty()
        loading_animation("Preparing learning content...", container=learn_container)
        
        if module_id in MODULE_CONTENT:
            st.markdown(f"""
            <div style="animation: contentFadeIn 1s ease-out 0.7s both;">
                {MODULE_CONTENT[module_id]}
            </div>
            
            <style>
            @keyframes contentFadeIn {
                from { opacity: 0; }
                to { opacity: 1; }
            }
            </style>
            """, unsafe_allow_html=True)
        else:
            st.info("Content for this module is under development.")
    
    with quiz_tab:
        if module_id in QUIZ_QUESTIONS:
            display_module_quiz(module_id)
        else:
            st.info("Quiz for this module is under development.")
    
    # Navigation buttons with animations
    st.markdown("""
    <div style="margin-top: 30px; animation: buttonsAppear 1s ease-out 1.2s both;">
    </div>
    
    <style>
    @keyframes buttonsAppear {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    </style>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.button("↩️ Back to Modules", help="Return to module selection"):
            show_animated_notification("Returning to module selection...", type="info")
            st.session_state.active_module = None
            st.rerun()
    
    with col3:
        if module_id in st.session_state.completed_modules:
            button_text = "✓ Already Completed"
            button_disabled = True
        else:
            button_text = "🏆 Mark as Completed"
            button_disabled = False
            
        if st.button(button_text, disabled=button_disabled):
            show_animated_notification("Marking module as completed...", type="success")
            check_module_completion(module_id)
            st.rerun()

def display_module_quiz(module_id):
    """Display a quiz for the given module"""
    questions = QUIZ_QUESTIONS[module_id]
    
    # Quiz state
    if module_id not in st.session_state.quiz_attempts:
        st.session_state.quiz_attempts[module_id] = {
            "current_question": 0,
            "correct_answers": 0,
            "completed": False,
            "answers": []
        }
    
    quiz_state = st.session_state.quiz_attempts[module_id]
    
    # If quiz is completed, show results
    if quiz_state["completed"]:
        score = quiz_state["correct_answers"]
        total = len(questions)
        percentage = (score / total) * 100
        
        st.markdown(f"### Quiz Results")
        st.markdown(f"You scored {score} out of {total} ({percentage:.1f}%)")
        
        if percentage >= 70:
            st.success("You passed! Great job understanding the concepts.")
            if module_id not in st.session_state.completed_modules:
                st.info("Don't forget to mark the module as completed to earn your badge!")
        else:
            st.warning("You didn't pass the quiz. Review the content and try again.")
        
        if st.button("Try Again"):
            st.session_state.quiz_attempts[module_id] = {
                "current_question": 0,
                "correct_answers": 0,
                "completed": False,
                "answers": []
            }
            st.rerun()
        
        # Show quiz review
        st.markdown("### Review Your Answers")
        for i, (question, answer) in enumerate(zip(questions, quiz_state["answers"])):
            correct = question["correct"] == answer
            status = "✅" if correct else "❌"
            st.markdown(f"{status} **Question {i+1}:** {question['question']}")
            st.markdown(f"Your answer: {question['options'][answer]}")
            if not correct:
                st.markdown(f"Correct answer: {question['options'][question['correct']]}")
            st.markdown("---")
        
        return
    
    # Display current question
    current_q_idx = quiz_state["current_question"]
    question = questions[current_q_idx]
    
    st.markdown(f"### Question {current_q_idx + 1} of {len(questions)}")
    st.markdown(f"**{question['question']}**")
    
    # Create radio buttons for options
    selected_option = st.radio(
        "Select your answer:",
        options=question['options'],
        key=f"quiz_{module_id}_{current_q_idx}"
    )
    
    selected_idx = question['options'].index(selected_option)
    
    # Next button
    if st.button("Submit Answer"):
        # Record answer
        quiz_state["answers"].append(selected_idx)
        
        # Check if correct
        if selected_idx == question["correct"]:
            quiz_state["correct_answers"] += 1
        
        # Move to next question or finish
        if current_q_idx + 1 < len(questions):
            quiz_state["current_question"] += 1
        else:
            quiz_state["completed"] = True
        
        st.rerun()

def learning_section():
    """Display the learning section with modules and badges"""
    st.title("📚 Learning Center")
    
    # Initialize gamification state
    init_gamification()
    
    # Create tabs for learning, interactive quiz, and achievements
    learning_tab, personalized_path_tab, quiz_tab, badges_tab = st.tabs([
        "Learning Modules", 
        "Personalized Learning Path", 
        "Interactive Quiz", 
        "Your Achievements"
    ])
    
    with learning_tab:
        if 'active_module' in st.session_state and st.session_state.active_module:
            display_module_content(st.session_state.active_module)
        else:
            display_learning_modules()
    
    with personalized_path_tab:
        # Import and display the personalized learning path
        from learning_path import personalized_learning_path
        personalized_learning_path()
    
    with quiz_tab:
        # Import and display the interactive quiz
        from market_quiz import quiz_section
        quiz_section()
    
    with badges_tab:
        display_badge_progress()
