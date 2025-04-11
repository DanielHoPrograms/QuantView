import streamlit as st
import random
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import yfinance as yf
from gamification_minimal import award_badge, LEARNING_MODULES, QUIZ_QUESTIONS, MODULE_CONTENT

# Define knowledge level thresholds
KNOWLEDGE_LEVELS = {
    "beginner": {
        "name": "Beginner",
        "description": "New to stock market investing and technical analysis",
        "threshold": 0,
        "color": "#00CC96"  # Green
    },
    "intermediate": {
        "name": "Intermediate",
        "description": "Familiar with basic concepts and some technical indicators",
        "threshold": 30,
        "color": "#FFA15A"  # Orange
    },
    "advanced": {
        "name": "Advanced",
        "description": "Experienced with multiple strategies and technical analysis",
        "threshold": 70,
        "color": "#636EFA"  # Blue
    },
    "expert": {
        "name": "Expert",
        "description": "Deep understanding of market mechanics and advanced strategies",
        "threshold": 90,
        "color": "#AB63FA"  # Purple
    }
}

# Bite-sized tutorials organized by concept
BITE_SIZED_TUTORIALS = {
    "market_basics": [
        {
            "id": "mb_101",
            "title": "What is a Stock?",
            "content": """
            <div style="background-color: #000000 !important; color: #ffffff !important; padding: 15px; border-radius: 5px; margin-bottom: 20px; border: 1px solid #444444; font-size: 1rem;">
                <h3 style="color: #ffffff !important; font-size: 1.2rem;">What is a Stock?</h3>
                <p style="color: #ffffff !important; font-size: 1rem;">A <strong style="color: #ffffff !important;">stock</strong> represents ownership in a company. When you buy a stock, you're purchasing a small piece of that company.</p>
                <p style="color: #ffffff !important; font-size: 1rem;">Key points:</p>
                <ul style="color: #ffffff !important; font-size: 1rem;">
                    <li style="color: #ffffff !important;">Stocks are also called <em style="color: #ffffff !important;">shares</em> or <em style="color: #ffffff !important;">equities</em></li>
                    <li style="color: #ffffff !important;">Owning stocks gives you voting rights in company decisions</li>
                    <li style="color: #ffffff !important;">Stocks may pay <em style="color: #ffffff !important;">dividends</em>, which are portions of company profits</li>
                    <li style="color: #ffffff !important;">Stock prices change based on supply and demand, and company performance</li>
                </ul>
                <p style="color: #ffffff !important; font-size: 1rem;">When you hear "the market is up," it usually refers to stock indexes like the S&P 500 rising in value.</p>
            </div>
            """,
            "estimated_time": 2,  # in minutes
            "related_concepts": ["market_basics"]
        },
        {
            "id": "mb_102",
            "title": "The Stock Market Explained",
            "content": """
            <div style="background-color: #000000 !important; color: #ffffff !important; padding: 15px; border-radius: 5px; margin-bottom: 20px; border: 1px solid #444444; font-size: 1rem;">
                <h3 style="color: #ffffff !important; font-size: 1.2rem;">The Stock Market Explained</h3>
                <p style="color: #ffffff !important; font-size: 1rem;">The <strong style="color: #ffffff !important;">stock market</strong> is where buyers and sellers trade stocks. Think of it as a marketplace for ownership in companies.</p>
                <p style="color: #ffffff !important; font-size: 1rem;">Key points:</p>
                <ul style="color: #ffffff !important; font-size: 1rem;">
                    <li style="color: #ffffff !important;">Major stock exchanges include NYSE and NASDAQ</li>
                    <li style="color: #ffffff !important;">Trading happens electronically through brokers</li>
                    <li style="color: #ffffff !important;">Market hours are typically 9:30 AM to 4:00 PM Eastern Time (US)</li>
                    <li style="color: #ffffff !important;">Stock prices are determined by supply and demand</li>
                </ul>
                <p style="color: #ffffff !important; font-size: 1rem;">The stock market provides companies a way to raise money and gives investors a way to build wealth.</p>
            </div>
            """,
            "estimated_time": 2,
            "related_concepts": ["market_basics"]
        },
        {
            "id": "mb_103",
            "title": "Bull vs. Bear Markets",
            "content": """
            <div style="background-color: #000000 !important; color: #ffffff !important; padding: 15px; border-radius: 5px; margin-bottom: 20px; border: 1px solid #444444; font-size: 1rem;">
                <h3 style="color: #ffffff !important; font-size: 1.2rem;">Bull vs. Bear Markets</h3>
                <p style="color: #ffffff !important; font-size: 1rem;">Market trends are often described as either <strong style="color: #ffffff !important;">bull</strong> or <strong style="color: #ffffff !important;">bear</strong> markets.</p>
                <p style="color: #ffffff !important; font-size: 1rem;">Key points:</p>
                <ul style="color: #ffffff !important; font-size: 1rem;">
                    <li style="color: #ffffff !important;"><strong style="color: #ffffff !important;">Bull Market</strong>: Prices are rising or expected to rise (optimistic market)</li>
                    <li style="color: #ffffff !important;"><strong style="color: #ffffff !important;">Bear Market</strong>: Prices are falling or expected to fall (pessimistic market)</li>
                    <li style="color: #ffffff !important;">Bull markets typically last longer than bear markets</li>
                    <li style="color: #ffffff !important;">A bear market is officially declared after a 20% drop from recent highs</li>
                </ul>
                <p style="color: #ffffff !important; font-size: 1rem;">Remember the mnemonics: a bull strikes upward with its horns, while a bear swipes downward with its paws.</p>
            </div>
            """,
            "estimated_time": 2,
            "related_concepts": ["market_basics"]
        }
    ],
    "chart_patterns": [
        {
            "id": "cp_101",
            "title": "Candlestick Basics",
            "content": """
            <div style="background-color: #000000 !important; color: #ffffff !important; padding: 15px; border-radius: 5px; margin-bottom: 20px; border: 1px solid #444444; font-size: 1rem;">
                <h3 style="color: #ffffff !important; font-size: 1.2rem;">Candlestick Basics</h3>
                <p style="color: #ffffff !important; font-size: 1rem;">A <strong style="color: #ffffff !important;">candlestick</strong> shows the price movement of an asset within a specific time period.</p>
                <p style="color: #ffffff !important; font-size: 1rem;">Key components:</p>
                <ul style="color: #ffffff !important; font-size: 1rem;">
                    <li style="color: #ffffff !important;"><strong style="color: #ffffff !important;">Body</strong>: The rectangle showing the opening and closing prices</li>
                    <li style="color: #ffffff !important;"><strong style="color: #ffffff !important;">Green/White Body</strong>: Closing price higher than opening price (bullish)</li>
                    <li style="color: #ffffff !important;"><strong style="color: #ffffff !important;">Red/Black Body</strong>: Closing price lower than opening price (bearish)</li>
                    <li style="color: #ffffff !important;"><strong style="color: #ffffff !important;">Wicks/Shadows</strong>: The lines extending from the body showing high and low prices</li>
                </ul>
                <p style="color: #ffffff !important; font-size: 1rem;">Candlesticks provide much more information than simple line charts, revealing the battle between buyers and sellers.</p>
            </div>
            """,
            "estimated_time": 3,
            "related_concepts": ["chart_patterns"]
        },
        {
            "id": "cp_102",
            "title": "Doji Patterns",
            "content": """
            <div style="background-color: #000000 !important; color: #ffffff !important; padding: 15px; border-radius: 5px; margin-bottom: 20px; border: 1px solid #444444; font-size: 1rem;">
            <h3 style="color: #ffffff !important; font-size: 1.2rem;">Doji Patterns</h3>
            <p style="color: #ffffff !important; font-size: 1rem;">A <strong style="color: #ffffff !important;">Doji</strong> candlestick forms when the opening and closing prices are virtually the same, creating a cross-like shape.</p>
            <p style="color: #ffffff !important; font-size: 1rem;">Key points:</p>
            <ul style="color: #ffffff !important; font-size: 1rem;">
                <li style="color: #ffffff !important;">Represents indecision in the market</li>
                <li style="color: #ffffff !important;">Often signals a potential reversal</li>
                <li style="color: #ffffff !important;"><strong style="color: #ffffff !important;">Long-legged Doji</strong>: Long wicks above and below, indicating high volatility</li>
                <li style="color: #ffffff !important;"><strong style="color: #ffffff !important;">Dragonfly Doji</strong>: Long lower wick, often bullish at bottoms</li>
                <li style="color: #ffffff !important;"><strong style="color: #ffffff !important;">Gravestone Doji</strong>: Long upper wick, often bearish at tops</li>
            </ul>
            <p style="color: #ffffff !important; font-size: 1rem;">Dojis are more significant when they appear after a strong trend, potentially signaling trader exhaustion.</p>
            </div>
            """,
            "estimated_time": 3,
            "related_concepts": ["chart_patterns"]
        }
    ],
    "technical_indicators": [
        {
            "id": "ti_101",
            "title": "RSI in 2 Minutes",
            "content": """
            <div style="background-color: #000000 !important; color: #ffffff !important; padding: 15px; border-radius: 5px; margin-bottom: 20px; border: 1px solid #444444; font-size: 1rem;">
            <h3 style="color: #ffffff !important; font-size: 1.2rem;">RSI in 2 Minutes</h3>
            <p style="color: #ffffff !important; font-size: 1rem;">The <strong style="color: #ffffff !important;">Relative Strength Index (RSI)</strong> measures the speed and change of price movements, indicating overbought or oversold conditions.</p>
            <p style="color: #ffffff !important; font-size: 1rem;">Key points:</p>
            <ul style="color: #ffffff !important; font-size: 1rem;">
                <li style="color: #ffffff !important;">RSI ranges from 0 to 100</li>
                <li style="color: #ffffff !important;">Above 70: Generally considered overbought (potential sell signal)</li>
                <li style="color: #ffffff !important;">Below 30: Generally considered oversold (potential buy signal)</li>
                <li style="color: #ffffff !important;">Default period is 14 days, but can be adjusted</li>
            </ul>
            <p style="color: #ffffff !important; font-size: 1rem;">RSI works best in ranging markets and can give false signals during strong trends.</p>
            </div>
            """,
            "estimated_time": 2,
            "related_concepts": ["technical_indicators"]
        },
        {
            "id": "ti_102",
            "title": "Moving Averages Simplified",
            "content": """
            <div style="background-color: #000000 !important; color: #ffffff !important; padding: 15px; border-radius: 5px; margin-bottom: 20px; border: 1px solid #444444; font-size: 1rem;">
            <h3 style="color: #ffffff !important; font-size: 1.2rem;">Moving Averages Simplified</h3>
            <p style="color: #ffffff !important; font-size: 1rem;"><strong style="color: #ffffff !important;">Moving averages</strong> smooth out price data to create a single flowing line, making it easier to identify trends.</p>
            <p style="color: #ffffff !important; font-size: 1rem;">Common types:</p>
            <ul style="color: #ffffff !important; font-size: 1rem;">
                <li style="color: #ffffff !important;"><strong style="color: #ffffff !important;">Simple Moving Average (SMA)</strong>: Average of closing prices over a period</li>
                <li style="color: #ffffff !important;"><strong style="color: #ffffff !important;">Exponential Moving Average (EMA)</strong>: Gives more weight to recent prices</li>
                <li style="color: #ffffff !important;">Common periods: 20-day (short-term), 50-day (medium-term), 200-day (long-term)</li>
            </ul>
            <p style="color: #ffffff !important; font-size: 1rem;">When a shorter MA crosses above a longer MA, it's often seen as a bullish signal (called a "golden cross" when 50-day crosses above 200-day).</p>
            </div>
            """,
            "estimated_time": 2,
            "related_concepts": ["technical_indicators"]
        },
        {
            "id": "ti_103",
            "title": "MACD Quick Explanation",
            "content": """
            <div style="background-color: #000000 !important; color: #ffffff !important; padding: 15px; border-radius: 5px; margin-bottom: 20px; border: 1px solid #444444; font-size: 1rem;">
            <h3 style="color: #ffffff !important; font-size: 1.2rem;">MACD Quick Explanation</h3>
            <p style="color: #ffffff !important; font-size: 1rem;">The <strong style="color: #ffffff !important;">Moving Average Convergence Divergence (MACD)</strong> is a trend-following momentum indicator that shows the relationship between two moving averages.</p>
            <p style="color: #ffffff !important; font-size: 1rem;">Key components:</p>
            <ul style="color: #ffffff !important; font-size: 1rem;">
                <li style="color: #ffffff !important;"><strong style="color: #ffffff !important;">MACD Line</strong>: The difference between 12-period and 26-period EMAs</li>
                <li style="color: #ffffff !important;"><strong style="color: #ffffff !important;">Signal Line</strong>: 9-period EMA of the MACD Line</li>
                <li style="color: #ffffff !important;"><strong style="color: #ffffff !important;">Histogram</strong>: The difference between MACD Line and Signal Line</li>
            </ul>
            <p style="color: #ffffff !important; font-size: 1rem;">Bullish signals occur when the MACD Line crosses above the Signal Line, bearish when it crosses below.</p>
            </div>
            """,
            "estimated_time": 3,
            "related_concepts": ["technical_indicators"]
        }
    ],
    "risk_management": [
        {
            "id": "rm_101",
            "title": "Position Sizing Basics",
            "content": """
            <div style="background-color: #000000 !important; color: #ffffff !important; padding: 15px; border-radius: 5px; margin-bottom: 20px; border: 1px solid #444444; font-size: 1rem;">
            <h3 style="color: #ffffff !important; font-size: 1.2rem;">Position Sizing Basics</h3>
            <p style="color: #ffffff !important; font-size: 1rem;"><strong style="color: #ffffff !important;">Position sizing</strong> refers to how much of your capital you allocate to a single trade. It's one of the most important aspects of risk management.</p>
            <p style="color: #ffffff !important; font-size: 1rem;">Key approaches:</p>
            <ul style="color: #ffffff !important; font-size: 1rem;">
                <li style="color: #ffffff !important;"><strong style="color: #ffffff !important;">Fixed percentage</strong>: Risk a fixed percentage (e.g., 1-2%) of your total capital per trade</li>
                <li style="color: #ffffff !important;"><strong style="color: #ffffff !important;">Fixed dollar amount</strong>: Risk the same dollar amount on each trade</li>
                <li style="color: #ffffff !important;"><strong style="color: #ffffff !important;">Volatility-based</strong>: Adjust position size based on an asset's volatility</li>
            </ul>
            <p style="color: #ffffff !important; font-size: 1rem;">Proper position sizing helps you survive drawdowns and maintain consistent risk exposure.</p>
            </div>
            """,
            "estimated_time": 2,
            "related_concepts": ["risk_management"]
        },
        {
            "id": "rm_102",
            "title": "Stop-Loss Orders",
            "content": """
            <div style="background-color: #000000 !important; color: #ffffff !important; padding: 15px; border-radius: 5px; margin-bottom: 20px; border: 1px solid #444444; font-size: 1rem;">
            <h3 style="color: #ffffff !important; font-size: 1.2rem;">Stop-Loss Orders</h3>
            <p style="color: #ffffff !important; font-size: 1rem;">A <strong style="color: #ffffff !important;">stop-loss order</strong> is an order placed with a broker to sell a security when it reaches a certain price, limiting your potential loss.</p>
            <p style="color: #ffffff !important; font-size: 1rem;">Key points:</p>
            <ul style="color: #ffffff !important; font-size: 1rem;">
                <li style="color: #ffffff !important;">Helps remove emotion from trading decisions</li>
                <li style="color: #ffffff !important;">Can be set at a specific price or percentage</li>
                <li style="color: #ffffff !important;"><strong style="color: #ffffff !important;">Hard stops</strong>: Fixed price levels</li>
                <li style="color: #ffffff !important;"><strong style="color: #ffffff !important;">Trailing stops</strong>: Move with the price in your favor</li>
            </ul>
            <p style="color: #ffffff !important; font-size: 1rem;">Place stop-losses at logical levels based on technical analysis, not arbitrary distances from your entry.</p>
            </div>
            """,
            "estimated_time": 2,
            "related_concepts": ["risk_management"]
        }
    ]
}

# Define prerequisite relationships between modules
MODULE_PREREQUISITES = {
    "intro_to_markets": [],
    "candlestick_charts": ["intro_to_markets"],
    "intro_to_rsi": ["candlestick_charts"],
    "intro_to_macd": ["intro_to_rsi"],
    "bollinger_bands": ["intro_to_rsi"],
    "advanced_indicators": ["intro_to_macd", "bollinger_bands"],
    "risk_management": ["intro_to_markets"]
}

# Diagnostic test questions
DIAGNOSTIC_QUESTIONS = [
    {
        "question": "What does a candlestick chart show?",
        "options": [
            "Only closing prices over time",
            "Open, high, low, and close prices for each period",
            "Only volume information",
            "Market sentiment scores"
        ],
        "correct_idx": 1,
        "level": "beginner"
    },
    {
        "question": "What typically happens when the RSI goes below 30?",
        "options": [
            "The stock is considered overbought",
            "The stock is considered oversold",
            "The stock has reached its peak",
            "Trading volume increases dramatically"
        ],
        "correct_idx": 1,
        "level": "beginner"
    },
    {
        "question": "What is the primary use of Bollinger Bands?",
        "options": [
            "To identify market trends",
            "To measure volume fluctuations",
            "To measure volatility and identify overbought/oversold conditions",
            "To predict specific price targets"
        ],
        "correct_idx": 2,
        "level": "intermediate"
    },
    {
        "question": "In MACD analysis, what is the signal line?",
        "options": [
            "The zero line in the middle of the indicator",
            "A moving average of the MACD line",
            "The histogram showing momentum",
            "A line showing trade volume"
        ],
        "correct_idx": 1,
        "level": "intermediate" 
    },
    {
        "question": "What does divergence between price and an oscillator (like RSI) typically indicate?",
        "options": [
            "A continuation of the current trend",
            "An increase in trading volume",
            "A potential reversal of the current trend",
            "Market manipulation"
        ],
        "correct_idx": 2,
        "level": "advanced"
    },
    {
        "question": "Which of these combinations would be most effective for confirming a trend reversal?",
        "options": [
            "MACD crossover with increasing volume",
            "A single candlestick pattern",
            "RSI above 50",
            "Price crossing the 200-day moving average once"
        ],
        "correct_idx": 0,
        "level": "advanced"
    },
    {
        "question": "In risk management, what is the primary purpose of position sizing?",
        "options": [
            "To maximize returns by increasing leverage",
            "To limit the amount of capital at risk on any single trade",
            "To impress other traders with large positions",
            "To simplify tax calculations"
        ],
        "correct_idx": 1,
        "level": "intermediate"
    },
    {
        "question": "What is a typical characteristic of price action during market consolidation?",
        "options": [
            "Strong directional movement with high volume",
            "Reduced volatility with price moving in a range",
            "Extremely high volatility with gaps between trading sessions",
            "Consistent new highs or lows"
        ],
        "correct_idx": 1,
        "level": "intermediate"
    },
    {
        "question": "Which of these would be classified as a momentum indicator?",
        "options": [
            "Moving Average",
            "Volume Profile",
            "RSI (Relative Strength Index)",
            "Support and Resistance Levels"
        ],
        "correct_idx": 2,
        "level": "beginner"
    },
    {
        "question": "When multiple timeframe analysis shows conflicting signals, which timeframe typically takes precedence?",
        "options": [
            "The timeframe that matches your trading horizon",
            "Always the longest timeframe",
            "Always the shortest timeframe",
            "Whichever timeframe shows the most favorable signal"
        ],
        "correct_idx": 0,
        "level": "advanced"
    }
]

# Additional practice questions by difficulty level
PRACTICE_QUESTIONS = {
    "beginner": [
        {
            "question": "What is a stock market index?",
            "options": [
                "A government agency that regulates trading",
                "A basket of stocks representing a segment of the market",
                "A type of retirement account",
                "A fee paid when trading stocks"
            ],
            "correct_idx": 1
        },
        {
            "question": "What is a bull market?",
            "options": [
                "A market where prices are falling",
                "A market where prices are rising",
                "A market with high volatility",
                "A market with low trading volume"
            ],
            "correct_idx": 1
        },
        {
            "question": "Which of these is a typical characteristic of a bearish candlestick?",
            "options": [
                "The close price is higher than the open price",
                "The close price is lower than the open price",
                "The high and low are identical",
                "There is no wick or shadow"
            ],
            "correct_idx": 1
        }
    ],
    "intermediate": [
        {
            "question": "What is the significance of the 200-day moving average?",
            "options": [
                "It represents the average price over 200 trading sessions",
                "It's commonly used to identify the long-term trend direction",
                "It's used to calculate RSI values",
                "Both A and B are correct"
            ],
            "correct_idx": 3
        },
        {
            "question": "In a MACD indicator, what does a bullish crossover refer to?",
            "options": [
                "When the MACD line crosses above the signal line",
                "When the MACD line crosses below the signal line",
                "When the MACD line crosses above the zero line",
                "When the histogram changes from negative to positive"
            ],
            "correct_idx": 0
        },
        {
            "question": "What does the stochastic oscillator measure?",
            "options": [
                "Price volatility over time",
                "Trading volume relative to price",
                "The relationship between closing price and price range over time",
                "The strength of a price trend"
            ],
            "correct_idx": 2
        }
    ],
    "advanced": [
        {
            "question": "What is the Elliott Wave Theory primarily used for?",
            "options": [
                "Measuring market volume patterns",
                "Identifying recurring wave patterns in price movements",
                "Calculating optimal position sizes",
                "Determining sector rotation in the market"
            ],
            "correct_idx": 1
        },
        {
            "question": "Which of these would NOT be considered a mean reversion strategy?",
            "options": [
                "Trading based on Bollinger Band extremes",
                "Trend following with moving averages",
                "RSI oversold/overbought signals",
                "Statistical arbitrage between correlated assets"
            ],
            "correct_idx": 1
        },
        {
            "question": "What is the primary focus of Ichimoku Cloud analysis?",
            "options": [
                "Identifying potential support and resistance areas",
                "Providing an all-in-one indicator showing momentum, trend, and support/resistance",
                "Determining exact entry and exit price points",
                "Analyzing market breadth and sentiment"
            ],
            "correct_idx": 1
        }
    ]
}

# Challenge scenarios by difficulty level
CHALLENGE_SCENARIOS = {
    "beginner": [
        {
            "title": "Market Volatility Analysis",
            "description": "Analyze a stock's price volatility during different market conditions",
            "task": "Identify periods of high volatility on the chart and explain potential causes",
            "points": 10
        },
        {
            "title": "Trend Identification Challenge",
            "description": "Practice identifying uptrends, downtrends, and consolidation periods",
            "task": "Mark the primary trend directions on the provided chart",
            "points": 15
        }
    ],
    "intermediate": [
        {
            "title": "Indicator Combination Strategy",
            "description": "Create a trading strategy using multiple indicators",
            "task": "Develop a strategy using RSI and moving averages, then backtest it on historical data",
            "points": 25
        },
        {
            "title": "Support/Resistance Mapping",
            "description": "Identify key support and resistance levels on multiple timeframes",
            "task": "Mark major support/resistance zones on the chart and explain their significance",
            "points": 20
        }
    ],
    "advanced": [
        {
            "title": "Market Regime Detection",
            "description": "Develop a system to identify different market regimes (trending, ranging, volatile)",
            "task": "Create an algorithm that can classify market conditions using multiple indicators",
            "points": 35
        },
        {
            "title": "Portfolio Risk Optimization",
            "description": "Optimize a portfolio for maximum risk-adjusted returns",
            "task": "Allocate assets in a portfolio to achieve the best Sharpe ratio",
            "points": 40
        }
    ]
}

def initialize_learning_path():
    """Initialize learning path state variables"""
    if 'knowledge_level' not in st.session_state:
        st.session_state.knowledge_level = "beginner"
    
    if 'knowledge_score' not in st.session_state:
        st.session_state.knowledge_score = 0
    
    if 'diagnostic_completed' not in st.session_state:
        st.session_state.diagnostic_completed = False
    
    if 'learning_path_modules' not in st.session_state:
        st.session_state.learning_path_modules = []
    
    if 'learning_path_progress' not in st.session_state:
        st.session_state.learning_path_progress = {}
    
    if 'module_difficulty_preference' not in st.session_state:
        st.session_state.module_difficulty_preference = "standard"  # standard, easier, harder
    
    if 'challenge_completed' not in st.session_state:
        st.session_state.challenge_completed = set()
        
    if 'practice_questions_answered' not in st.session_state:
        st.session_state.practice_questions_answered = {}
        
    # New adaptive learning variables
    if 'learning_style' not in st.session_state:
        st.session_state.learning_style = "visual"  # visual, textual, interactive
        
    if 'concept_mastery' not in st.session_state:
        # Track mastery level (0-100) for individual concepts
        st.session_state.concept_mastery = {
            "market_basics": 0,
            "chart_patterns": 0,
            "technical_indicators": 0,
            "risk_management": 0,
            "trading_psychology": 0,
            "fundamental_analysis": 0,
            "advanced_strategies": 0
        }
    
    if 'learning_pace' not in st.session_state:
        st.session_state.learning_pace = "medium"  # slow, medium, fast
        
    if 'spaced_repetition_queue' not in st.session_state:
        # Queue for concepts that need reinforcement
        st.session_state.spaced_repetition_queue = []
        
    if 'last_learning_session' not in st.session_state:
        st.session_state.last_learning_session = None
        
    if 'total_learning_time' not in st.session_state:
        st.session_state.total_learning_time = 0  # in minutes
        
    if 'learning_streak' not in st.session_state:
        st.session_state.learning_streak = 0  # consecutive days
        
    if 'personalized_feedback' not in st.session_state:
        st.session_state.personalized_feedback = {}
        
    # Bite-sized tutorials tracking
    if 'bite_sized_completed' not in st.session_state:
        st.session_state.bite_sized_completed = {}
        
    if 'current_bite_sized' not in st.session_state:
        st.session_state.current_bite_sized = None
        
    if 'daily_bite_sized_quota' not in st.session_state:
        st.session_state.daily_bite_sized_quota = 3  # recommended tutorials per day
        
    if 'last_bite_sized_date' not in st.session_state:
        st.session_state.last_bite_sized_date = None
        
    if 'recommended_bite_sized' not in st.session_state:
        st.session_state.recommended_bite_sized = []


def run_diagnostic_test():
    """Run initial diagnostic test to determine user knowledge level"""
    st.subheader("Knowledge Assessment")
    st.write("Let's determine your current knowledge level with a short quiz. This will help us customize your learning path.")
    
    if not st.session_state.diagnostic_completed:
        # Randomly select questions from each difficulty category to ensure balance
        diagnostic_questions = []
        for level in ["beginner", "intermediate", "advanced"]:
            level_questions = [q for q in DIAGNOSTIC_QUESTIONS if q["level"] == level]
            # Select up to 3 questions from each level, or all if fewer are available
            selected = random.sample(level_questions, min(3, len(level_questions)))
            diagnostic_questions.extend(selected)
        
        random.shuffle(diagnostic_questions)
        
        # Store selected questions in session state if not already there
        if 'diagnostic_questions' not in st.session_state:
            st.session_state.diagnostic_questions = diagnostic_questions
            st.session_state.diagnostic_current_q = 0
            st.session_state.diagnostic_score = 0
            st.session_state.diagnostic_answers = []
        
        # Get current question
        if st.session_state.diagnostic_current_q < len(st.session_state.diagnostic_questions):
            current_q = st.session_state.diagnostic_questions[st.session_state.diagnostic_current_q]
            
            # Display progress
            progress = st.session_state.diagnostic_current_q / len(st.session_state.diagnostic_questions)
            st.progress(progress)
            st.write(f"Question {st.session_state.diagnostic_current_q + 1} of {len(st.session_state.diagnostic_questions)}")
            
            # Display question
            st.write(f"**{current_q['question']}**")
            
            # Display options
            option_cols = st.columns(2)
            for i, option in enumerate(current_q["options"]):
                col_idx = i % 2
                with option_cols[col_idx]:
                    if st.button(option, key=f"diag_option_{i}", use_container_width=True):
                        # Record answer
                        is_correct = (i == current_q["correct_idx"])
                        if is_correct:
                            st.session_state.diagnostic_score += 1
                            
                            # Weight by difficulty level
                            if current_q["level"] == "advanced":
                                st.session_state.diagnostic_score += 2
                            elif current_q["level"] == "intermediate":
                                st.session_state.diagnostic_score += 1
                        
                        st.session_state.diagnostic_answers.append({
                            "question_idx": st.session_state.diagnostic_current_q,
                            "selected_option": i,
                            "correct_option": current_q["correct_idx"],
                            "is_correct": is_correct,
                            "level": current_q["level"]
                        })
                        
                        # Move to next question
                        st.session_state.diagnostic_current_q += 1
                        st.rerun()
        else:
            # Calculate final score as a percentage
            total_possible_score = len(st.session_state.diagnostic_questions)
            # Add extra points for intermediate and advanced questions
            for q in st.session_state.diagnostic_questions:
                if q["level"] == "advanced":
                    total_possible_score += 2
                elif q["level"] == "intermediate":
                    total_possible_score += 1
                    
            final_score = (st.session_state.diagnostic_score / total_possible_score) * 100
            st.session_state.knowledge_score = final_score
            
            # Determine knowledge level based on score
            for level_id, level_info in KNOWLEDGE_LEVELS.items():
                if final_score >= level_info["threshold"]:
                    st.session_state.knowledge_level = level_id
            
            # Mark diagnostic as completed
            st.session_state.diagnostic_completed = True
            
            # Generate learning path based on results
            generate_learning_path()
            
            st.success("Assessment completed! Your personalized learning path has been created.")
            st.rerun()
    else:
        # Show the results of the diagnostic test
        level = st.session_state.knowledge_level
        level_info = KNOWLEDGE_LEVELS[level]
        
        st.info(f"Your knowledge level: **{level_info['name']}** ({st.session_state.knowledge_score:.1f}%)")
        st.write(level_info["description"])
        
        # Option to retake the test
        if st.button("Retake Assessment", type="secondary"):
            # Reset diagnostic state
            if 'diagnostic_questions' in st.session_state:
                del st.session_state.diagnostic_questions
            if 'diagnostic_current_q' in st.session_state:
                del st.session_state.diagnostic_current_q
            if 'diagnostic_score' in st.session_state:
                del st.session_state.diagnostic_score
            if 'diagnostic_answers' in st.session_state:
                del st.session_state.diagnostic_answers
            
            st.session_state.diagnostic_completed = False
            st.rerun()


def generate_learning_path():
    """Generate personalized learning path based on knowledge level"""
    level = st.session_state.knowledge_level
    
    # For beginners and intermediates, include most modules
    if level == "beginner":
        # Start with introductory modules for beginners
        path = ["intro_to_markets", "candlestick_charts", "intro_to_rsi", "risk_management"]
    elif level == "intermediate":
        # Intermediates can skip the most basic modules
        path = ["candlestick_charts", "intro_to_rsi", "intro_to_macd", "bollinger_bands", "risk_management"]
    else:  # advanced and expert
        # Advanced users focus on more complex topics
        path = ["intro_to_macd", "bollinger_bands", "advanced_indicators", "risk_management"]
    
    # Ensure no duplicates and modules exist in LEARNING_MODULES
    path = [m for m in path if m in LEARNING_MODULES]
    
    # Store the generated path
    st.session_state.learning_path_modules = path
    
    # Initialize progress for each module
    for module_id in path:
        if module_id not in st.session_state.learning_path_progress:
            st.session_state.learning_path_progress[module_id] = {
                "started": False,
                "completed": False,
                "quiz_score": 0,
                "last_accessed": None
            }


def check_module_prerequisites(module_id):
    """Check if all prerequisites for a module have been completed"""
    if module_id not in MODULE_PREREQUISITES:
        return True
    
    for prereq in MODULE_PREREQUISITES[module_id]:
        if prereq not in st.session_state.learning_path_progress:
            return False
        if not st.session_state.learning_path_progress[prereq]["completed"]:
            return False
    
    return True


def display_learning_path():
    """Display the personalized learning path"""
    st.subheader("Your Learning Path")
    
    # Progress overview
    completed_modules = sum(1 for m in st.session_state.learning_path_progress.values() if m["completed"])
    total_modules = len(st.session_state.learning_path_modules)
    
    if total_modules > 0:
        st.progress(completed_modules / total_modules)
        st.write(f"Progress: {completed_modules}/{total_modules} modules completed")
    
    # Create path visualization
    path_data = []
    
    for i, module_id in enumerate(st.session_state.learning_path_modules):
        module = LEARNING_MODULES[module_id]
        progress = st.session_state.learning_path_progress.get(module_id, {})
        
        # Determine status
        if progress.get("completed", False):
            status = "Completed"
            status_color = "green"
        elif progress.get("started", False):
            status = "In Progress"
            status_color = "orange"
        else:
            status = "Not Started"
            status_color = "gray"
        
        # Check prerequisites
        prerequisites_met = check_module_prerequisites(module_id)
        
        path_data.append({
            "Module": module["name"],
            "Difficulty": module.get("difficulty", "Beginner"),
            "Status": status,
            "StatusColor": status_color,
            "Index": i,
            "PrerequisitesMet": prerequisites_met
        })
    
    # Display as a table
    if path_data:
        path_df = pd.DataFrame(path_data)
        
        # Display modules as clickable cards
        for i, row in path_df.iterrows():
            module_id = st.session_state.learning_path_modules[row["Index"]]
            
            # Determine card color based on status
            if row["Status"] == "Completed":
                card_bg = "#EAFAF1"  # Light green
                emoji = "✅"
            elif row["Status"] == "In Progress":
                card_bg = "#FEF9E7"  # Light yellow
                emoji = "🔄"
            else:
                card_bg = "#F8F9F9"  # Light gray
                emoji = "📘"
            
            # Apply different styling if prerequisites are not met
            if not row["PrerequisitesMet"]:
                card_bg = "#F2F3F4"  # Even lighter gray
                emoji = "🔒"
            
            # Create the module card
            st.markdown(f"""
            <div style="padding: 15px; border-radius: 5px; background-color: {card_bg}; margin-bottom: 10px;">
                <h3 style="margin-top: 0;">{emoji} {row['Module']}</h3>
                <p><strong>Difficulty:</strong> {row['Difficulty']}</p>
                <p><strong>Status:</strong> <span style="color: {row['StatusColor']};">{row['Status']}</span></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Button to start/continue module
            button_label = "Continue" if row["Status"] == "In Progress" else "Start"
            
            if row["Status"] != "Completed":
                button_disabled = not row["PrerequisitesMet"]
                
                if button_disabled:
                    st.warning("Complete the prerequisites first")
                
                if st.button(button_label, key=f"btn_module_{module_id}", disabled=button_disabled):
                    st.session_state.active_module = module_id
                    st.session_state.learning_path_progress[module_id]["started"] = True
                    st.session_state.learning_path_progress[module_id]["last_accessed"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.rerun()
            
            st.markdown("---")
    else:
        st.info("No modules in your learning path yet. Please complete the assessment.")


def adjust_module_difficulty():
    """Allow user to adjust module difficulty preference and learning style"""
    st.markdown("<h2 style='font-size: 32px; font-weight: bold;'>Customize Your Learning Experience</h2>", unsafe_allow_html=True)
    
    # Difficulty preference
    st.markdown("<div style='font-size: 24px; font-weight: bold;'>Content Difficulty:</div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Make It Easier", key="btn_easier", 
                     disabled=st.session_state.module_difficulty_preference == "easier",
                     use_container_width=True):
            st.session_state.module_difficulty_preference = "easier"
            st.rerun()
    
    with col2:
        if st.button("Standard Difficulty", key="btn_standard",
                     disabled=st.session_state.module_difficulty_preference == "standard",
                     use_container_width=True):
            st.session_state.module_difficulty_preference = "standard"
            st.rerun()
    
    with col3:
        if st.button("Challenge Me", key="btn_harder",
                     disabled=st.session_state.module_difficulty_preference == "harder",
                     use_container_width=True):
            st.session_state.module_difficulty_preference = "harder"
            st.rerun()
    
    # Show current difficulty preference
    pref_label = {
        "easier": "Easier - More explanations, simpler examples",
        "standard": "Standard - Balanced difficulty",
        "harder": "Challenging - Advanced concepts, fewer hints"
    }
    
    st.info(f"Current difficulty: **{pref_label[st.session_state.module_difficulty_preference]}**")
    
    # Learning style preference
    st.markdown("<div style='font-size: 24px; font-weight: bold;'>Learning Style:</div>", unsafe_allow_html=True)
    
    style_cols = st.columns(3)
    with style_cols[0]:
        if st.button("Visual Learning", key="btn_visual",
                   disabled=st.session_state.learning_style == "visual",
                   use_container_width=True):
            st.session_state.learning_style = "visual"
            st.rerun()
            
    with style_cols[1]:
        if st.button("Textual Learning", key="btn_textual",
                   disabled=st.session_state.learning_style == "textual",
                   use_container_width=True):
            st.session_state.learning_style = "textual"
            st.rerun()
            
    with style_cols[2]:
        if st.button("Interactive Learning", key="btn_interactive",
                   disabled=st.session_state.learning_style == "interactive",
                   use_container_width=True):
            st.session_state.learning_style = "interactive"
            st.rerun()
    
    # Show current learning style preference
    style_label = {
        "visual": "Visual - Charts, diagrams, and visual explanations",
        "textual": "Textual - Detailed written explanations and examples",
        "interactive": "Interactive - Learning through practice and hands-on activities"
    }
    
    st.info(f"Current learning style: **{style_label[st.session_state.learning_style]}**")
    
    # Learning pace preference
    st.markdown("<div style='font-size: 24px; font-weight: bold;'>Learning Pace:</div>", unsafe_allow_html=True)
    pace_cols = st.columns(3)
    
    with pace_cols[0]:
        if st.button("Slow Pace", key="btn_slow",
                   disabled=st.session_state.learning_pace == "slow",
                   use_container_width=True):
            st.session_state.learning_pace = "slow"
            st.rerun()
            
    with pace_cols[1]:
        if st.button("Medium Pace", key="btn_medium",
                   disabled=st.session_state.learning_pace == "medium",
                   use_container_width=True):
            st.session_state.learning_pace = "medium"
            st.rerun()
            
    with pace_cols[2]:
        if st.button("Fast Pace", key="btn_fast",
                   disabled=st.session_state.learning_pace == "fast",
                   use_container_width=True):
            st.session_state.learning_pace = "fast"
            st.rerun()
    
    # Show current learning pace preference
    pace_label = {
        "slow": "Slow - More frequent recaps and smaller content chunks",
        "medium": "Medium - Balanced pace with regular review points",
        "fast": "Fast - More content with fewer recaps for quick progression"
    }
    
    st.info(f"Current learning pace: **{pace_label[st.session_state.learning_pace]}**")


def get_recommended_modules():
    """Get module recommendations based on current progress and knowledge level"""
    # Get modules that are not completed or not in the current path
    incomplete_modules = [m for m in st.session_state.learning_path_modules 
                         if not st.session_state.learning_path_progress.get(m, {}).get("completed", False)]
    
    # Find modules not in the current path
    all_modules = list(LEARNING_MODULES.keys())
    unused_modules = [m for m in all_modules if m not in st.session_state.learning_path_modules]
    
    recommendations = []
    
    # First, recommend incomplete modules from the path where prerequisites are met
    for module_id in incomplete_modules:
        if check_module_prerequisites(module_id):
            recommendations.append({
                "module_id": module_id,
                "reason": "Next in your learning path",
                "priority": 1
            })
    
    # Then, look at unused modules that match the user's level
    user_level = st.session_state.knowledge_level
    for module_id in unused_modules:
        module = LEARNING_MODULES[module_id]
        module_difficulty = module.get("difficulty", "Beginner").lower()
        
        # Match module difficulty to user level
        if (user_level == "beginner" and module_difficulty == "beginner") or \
           (user_level == "intermediate" and module_difficulty in ["beginner", "intermediate"]) or \
           (user_level in ["advanced", "expert"] and module_difficulty in ["intermediate", "advanced"]):
            
            # Check prerequisites
            if check_module_prerequisites(module_id):
                recommendations.append({
                    "module_id": module_id,
                    "reason": f"Recommended for your {KNOWLEDGE_LEVELS[user_level]['name']} level",
                    "priority": 2
                })
    
    # Sort by priority
    recommendations.sort(key=lambda x: x["priority"])
    
    return recommendations[:3]  # Return top 3 recommendations


def display_practice_questions():
    """Display practice questions based on the user's knowledge level"""
    st.subheader("Practice Questions")
    
    # Get questions for user's level and below
    level = st.session_state.knowledge_level
    available_levels = []
    
    if level in ["beginner", "intermediate", "advanced", "expert"]:
        available_levels.append("beginner")
    if level in ["intermediate", "advanced", "expert"]:
        available_levels.append("intermediate")
    if level in ["advanced", "expert"]:
        available_levels.append("advanced")
    
    # Get questions from available levels
    available_questions = []
    for lvl in available_levels:
        if lvl in PRACTICE_QUESTIONS:
            available_questions.extend(PRACTICE_QUESTIONS[lvl])
    
    if not available_questions:
        st.info("No practice questions available for your level yet.")
        return
    
    # Randomly select a question if not already selected
    if 'current_practice_question' not in st.session_state:
        st.session_state.current_practice_question = random.choice(available_questions)
        
    # Display current question
    current_q = st.session_state.current_practice_question
    st.write(f"**{current_q['question']}**")
    
    # Display options
    option_cols = st.columns(2)
    for i, option in enumerate(current_q["options"]):
        col_idx = i % 2
        with option_cols[col_idx]:
            # Disable buttons if user has already answered
            disabled = 'practice_answered' in st.session_state and st.session_state.practice_answered
            
            # Determine button color based on correctness (after answering)
            button_type = "primary"
            if 'practice_answered' in st.session_state and st.session_state.practice_answered:
                if i == current_q["correct_idx"]:
                    button_type = "success"
                elif i == st.session_state.practice_selection:
                    button_type = "danger"
            
            if st.button(option, key=f"practice_option_{i}", disabled=disabled, type=button_type, use_container_width=True):
                st.session_state.practice_selection = i
                st.session_state.practice_answered = True
                
                # Record this question as answered
                question_id = hash(current_q["question"])
                if question_id not in st.session_state.practice_questions_answered:
                    st.session_state.practice_questions_answered[question_id] = {
                        "is_correct": (i == current_q["correct_idx"]),
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                
                # If correct, update knowledge score slightly
                if i == current_q["correct_idx"]:
                    level_boost = 0
                    for lvl_idx, lvl in enumerate(available_levels):
                        if current_q in PRACTICE_QUESTIONS.get(lvl, []):
                            level_boost = lvl_idx + 1
                    
                    # Small boost to knowledge score for correct answers
                    current_score = st.session_state.knowledge_score
                    st.session_state.knowledge_score = min(100, current_score + (level_boost * 0.5))
                    
                    # Check if this moved the user to a new level
                    update_knowledge_level()
                
                st.rerun()
    
    # Show explanation after answering
    if 'practice_answered' in st.session_state and st.session_state.practice_answered:
        correct_option = current_q["options"][current_q["correct_idx"]]
        
        if st.session_state.practice_selection == current_q["correct_idx"]:
            st.success(f"✅ Correct! The answer is: {correct_option}")
        else:
            st.error(f"❌ Incorrect. The correct answer is: {correct_option}")
        
        # Next question button
        if st.button("Next Question", type="primary"):
            if 'current_practice_question' in st.session_state:
                del st.session_state.current_practice_question
            if 'practice_answered' in st.session_state:
                del st.session_state.practice_answered
            if 'practice_selection' in st.session_state:
                del st.session_state.practice_selection
            st.rerun()


def update_knowledge_level():
    """Update user knowledge level based on current score"""
    current_score = st.session_state.knowledge_score
    current_level = st.session_state.knowledge_level
    
    # Find the highest level the user qualifies for
    new_level = current_level
    for level_id, level_info in KNOWLEDGE_LEVELS.items():
        if current_score >= level_info["threshold"]:
            new_level = level_id
    
    # Update if changed
    if new_level != current_level:
        st.session_state.knowledge_level = new_level
        
        # Regenerate learning path if level increased
        level_order = ["beginner", "intermediate", "advanced", "expert"]
        if level_order.index(new_level) > level_order.index(current_level):
            generate_learning_path()
            
            # Show a congratulations message
            new_level_name = KNOWLEDGE_LEVELS[new_level]["name"]
            return f"Congratulations! Your knowledge level has increased to {new_level_name}."
    
    return None


def update_spaced_repetition_queue():
    """
    Update the spaced repetition queue based on concept mastery levels
    and time since last review. Concepts with lower mastery and not
    recently reviewed will be prioritized.
    """
    # Get all concepts and their mastery levels
    concept_mastery = st.session_state.concept_mastery
    
    # Sort concepts by mastery level (ascending)
    sorted_concepts = sorted(concept_mastery.items(), key=lambda x: x[1])
    
    # Create a new queue with low-mastery concepts first
    new_queue = []
    
    # Add concepts with low mastery (below 70%)
    for concept, mastery in sorted_concepts:
        if mastery < 70:
            new_queue.append(concept)
    
    # Add other concepts for occasional review
    for concept, mastery in sorted_concepts:
        if mastery >= 70 and concept not in new_queue:
            # High mastery concepts are reviewed less frequently, they'll be
            # added at the end of the queue
            new_queue.append(concept)
    
    # Store the updated queue in session state
    st.session_state.spaced_repetition_queue = new_queue
    
    return new_queue


def track_learning_session():
    """
    Track the learning session time and update total time spent
    Also updates learning streak and triggers spaced repetition updates
    """
    # Calculate time spent if we have a session start time
    if 'session_start_time' in st.session_state:
        session_duration = (datetime.now() - st.session_state.session_start_time).total_seconds() / 60  # in minutes
        
        # Update total learning time
        st.session_state.total_learning_time += session_duration
        
        # Reset session start time
        del st.session_state.session_start_time
    
    # Update spaced repetition queue daily
    last_update = st.session_state.last_learning_session
    if last_update:
        last_date = datetime.strptime(last_update, "%Y-%m-%d %H:%M:%S").date()
        today = datetime.now().date()
        
        if last_date != today:
            # If it's a new day, update the spaced repetition queue
            update_spaced_repetition_queue()
            
            # Also update recommended bite-sized tutorials
            recommend_bite_sized_tutorials()


def recommend_bite_sized_tutorials():
    """
    Recommend bite-sized tutorials based on knowledge level and concept mastery.
    Focus on concepts with lower mastery levels.
    """
    # Get all concepts sorted by mastery level (ascending)
    concept_mastery = st.session_state.concept_mastery
    sorted_concepts = sorted(concept_mastery.items(), key=lambda x: x[1])
    
    # Reset recommendations
    recommendations = []
    
    # Start with the concept that has the lowest mastery
    for concept, mastery in sorted_concepts:
        # Only recommend tutorials for concepts below 80% mastery
        if mastery < 80 and concept in BITE_SIZED_TUTORIALS:
            # Get available tutorials for this concept
            concept_tutorials = BITE_SIZED_TUTORIALS[concept]
            
            # Filter out completed tutorials
            available_tutorials = [t for t in concept_tutorials 
                                  if t["id"] not in st.session_state.bite_sized_completed]
            
            # Add up to 2 tutorials from each concept
            if available_tutorials:
                # If user is a beginner, prioritize easier tutorials
                if st.session_state.knowledge_level == "beginner":
                    # Sort by estimated time (ascending)
                    available_tutorials.sort(key=lambda x: x["estimated_time"])
                
                # If user is advanced, prioritize more complex tutorials
                elif st.session_state.knowledge_level in ["advanced", "expert"]:
                    # Sort by estimated time (descending)
                    available_tutorials.sort(key=lambda x: x["estimated_time"], reverse=True)
                
                # Add tutorials to recommendations (at most 2 per concept)
                for tutorial in available_tutorials[:2]:
                    recommendations.append({
                        "concept": concept,
                        "tutorial": tutorial,
                        "mastery": mastery
                    })
    
    # Limit to daily quota
    daily_quota = st.session_state.daily_bite_sized_quota
    recommendations = recommendations[:daily_quota]
    
    # Save to session state
    st.session_state.recommended_bite_sized = recommendations
    
    # Update last recommendation date
    st.session_state.last_bite_sized_date = datetime.now().strftime("%Y-%m-%d")
    
    return recommendations


def display_bite_sized_tutorial(tutorial_id=None):
    """
    Display a bite-sized tutorial with the given ID.
    If no ID is provided, choose one from recommendations.
    
    Args:
        tutorial_id: ID of the tutorial to display (optional)
        
    Returns:
        True if a tutorial was displayed, False if not
    """
    # Add a global CSS style to ensure all text has proper color and background
    st.markdown("""
    <style>
    .tutorial-content {
        color: #ffffff !important;
        background-color: #000000 !important;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
        border: 1px solid #444444;
    }
    .tutorial-content h3 {
        color: #ffffff !important;
    }
    .tutorial-content p, .tutorial-content li {
        color: #ffffff !important;
    }
    </style>
    """, unsafe_allow_html=True)
    # Find the tutorial
    tutorial = None
    concept = None
    
    if tutorial_id:
        # Search for the specific tutorial
        for concept_name, tutorials in BITE_SIZED_TUTORIALS.items():
            for t in tutorials:
                if t["id"] == tutorial_id:
                    tutorial = t
                    concept = concept_name
                    break
            if tutorial:
                break
    else:
        # Get a tutorial from recommendations
        if st.session_state.recommended_bite_sized:
            recommended = st.session_state.recommended_bite_sized[0]
            tutorial = recommended["tutorial"]
            concept = recommended["concept"]
    
    if not tutorial:
        return False
    
    # Set as current tutorial
    st.session_state.current_bite_sized = {
        "tutorial": tutorial,
        "concept": concept,
        "start_time": datetime.now()
    }
    
    # Display the tutorial
    st.markdown(f"# {tutorial['title']}")
    # Add null check for concept before calling replace
    concept_display = concept.replace('_', ' ').title() if concept else "General"
    st.markdown(f"<small>Est. time: {tutorial['estimated_time']} min • Topic: {concept_display}</small>", unsafe_allow_html=True)
    
    # Display tutorial content with proper styling that works in dark mode
    # Process the content to add inline styling
    modified_content = tutorial["content"].strip()
    
    # For robustness, always reprocess the content with styling, even if it already has some styling
    # First, strip any existing div wrappers if they exist (to avoid nested styling conflicts)
    if modified_content.startswith('<div') and modified_content.endswith('</div>'):
        # Find the end of the opening div tag
        div_end = modified_content.find('>')
        if div_end > 0:
            # Extract content between opening and closing div tags
            inner_content = modified_content[div_end+1:-6]  # -6 to remove </div>
            modified_content = inner_content
    
    # Replace common HTML tags with styled versions for light text on dark background
    modified_content = modified_content.replace("<h3>", "<h3 style=\"color: #ffffff !important;\">")
    modified_content = modified_content.replace("<p>", "<p style=\"color: #ffffff !important;\">")
    modified_content = modified_content.replace("<ul>", "<ul style=\"color: #ffffff !important;\">")
    modified_content = modified_content.replace("<li>", "<li style=\"color: #ffffff !important;\">")
    modified_content = modified_content.replace("<strong>", "<strong style=\"color: #ffffff !important;\">")
    modified_content = modified_content.replace("<em>", "<em style=\"color: #ffffff !important;\">")
    
    # Wrap in a div with black background styling
    modified_content = f'''
    <div style="background-color: #000000 !important; color: #ffffff !important; 
                padding: 20px !important; border-radius: 8px !important; 
                margin-bottom: 20px !important; border: 2px solid #444444 !important;
                font-family: Arial, sans-serif !important; max-width: 100% !important;">
        {modified_content}
    </div>
    '''
    
    st.markdown(modified_content, unsafe_allow_html=True)
    
    # Buttons - display them vertically instead of horizontally
    # Back button
    if st.button("⬅️ Back to Learning Path", key="bite_sized_back", use_container_width=True):
        # Clear current tutorial
        st.session_state.current_bite_sized = None
        return False
    
    # Add some spacing
    st.write("")
    
    # Complete button
    if st.button("✅ Mark as Completed", key="bite_sized_complete", use_container_width=True):
        # Mark as completed
        if tutorial["id"] not in st.session_state.bite_sized_completed:
            st.session_state.bite_sized_completed[tutorial["id"]] = {
                "completed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "concept": concept
            }
            
            # If it was in recommendations, remove it
            st.session_state.recommended_bite_sized = [
                r for r in st.session_state.recommended_bite_sized
                if r["tutorial"]["id"] != tutorial["id"]
            ]
            
            # Increase concept mastery
            if concept in st.session_state.concept_mastery:
                current_mastery = st.session_state.concept_mastery[concept]
                st.session_state.concept_mastery[concept] = min(100, current_mastery + 5)
            
            # Show success message
            st.success("Tutorial completed! Your knowledge in this area has improved.")
            
            # Clear current tutorial
            st.session_state.current_bite_sized = None
            
            # If recommendations are now empty, create new ones
            if not st.session_state.recommended_bite_sized:
                recommend_bite_sized_tutorials()
            
            return False
    
    return True


def display_bite_sized_recommendations():
    """Display recommended bite-sized tutorials"""
    st.subheader("📚 Bite-Sized Learning")
    
    # Check if we need to update recommendations (once per day)
    today = datetime.now().strftime("%Y-%m-%d")
    if not st.session_state.last_bite_sized_date or st.session_state.last_bite_sized_date != today:
        recommend_bite_sized_tutorials()
    
    # If there's a current tutorial, display it
    if st.session_state.current_bite_sized:
        if display_bite_sized_tutorial(st.session_state.current_bite_sized["tutorial"]["id"]):
            return
    
    # Otherwise, show recommendations
    recommendations = st.session_state.recommended_bite_sized
    
    if not recommendations:
        st.info("No new bite-sized tutorials available today. Check back tomorrow!")
        return
    
    st.write("Quick 2-3 minute tutorials tailored to your learning needs:")
    
    # Display each recommended tutorial as a card
    for i, rec in enumerate(recommendations):
        tutorial = rec["tutorial"]
        concept = rec["concept"].replace('_', ' ').title()
        
        st.markdown(f"""
        <div style="padding: 15px; border-radius: 5px; background-color: #000000 !important; color: #ffffff !important; margin-bottom: 10px; border-left: 5px solid #3498DB;">
            <h4 style="margin-top: 0; color: #ffffff !important;">📖 {tutorial['title']}</h4>
            <p style="color: #ffffff !important;"><strong style="color: #ffffff !important;">Topic:</strong> {concept} • <strong style="color: #ffffff !important;">Time:</strong> {tutorial['estimated_time']} min</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Make tutorial start button full width
        if st.button(f"Start Tutorial", key=f"bite_sized_btn_{i}", use_container_width=True):
            # Set as current tutorial
            display_bite_sized_tutorial(tutorial["id"])
            st.rerun()
    
    # Show completed count
    completed_count = len(st.session_state.bite_sized_completed)
    if completed_count > 0:
        st.info(f"You've completed {completed_count} bite-sized tutorials! Great job keeping up with your learning.")
        
        # Option to see history - create a centered column for vertical button
        history_col1, history_col2, history_col3 = st.columns([1, 2, 1])
        with history_col2:
            if st.button("View Completed Tutorials History", use_container_width=True):
                st.session_state.show_bite_sized_history = True
                st.rerun()
    
    # Show history if requested
    if 'show_bite_sized_history' in st.session_state and st.session_state.show_bite_sized_history:
        display_bite_sized_history()


def display_bite_sized_history():
    """Display history of completed bite-sized tutorials"""
    st.subheader("Your Learning History")
    
    if not st.session_state.bite_sized_completed:
        st.info("You haven't completed any bite-sized tutorials yet.")
        return
    
    # Organize by concept
    tutorials_by_concept = {}
    for tutorial_id, data in st.session_state.bite_sized_completed.items():
        concept = data["concept"]
        completed_at = data["completed_at"]
        
        # Find the tutorial details
        tutorial = None
        for t in BITE_SIZED_TUTORIALS.get(concept, []):
            if t["id"] == tutorial_id:
                tutorial = t
                break
        
        if tutorial:
            if concept not in tutorials_by_concept:
                tutorials_by_concept[concept] = []
            
            tutorials_by_concept[concept].append({
                "id": tutorial_id,
                "title": tutorial["title"],
                "completed_at": completed_at
            })
    
    # Display by concept
    for concept, tutorials in tutorials_by_concept.items():
        concept_name = concept.replace('_', ' ').title() if concept else "General"
        with st.expander(f"{concept_name} ({len(tutorials)})"):
            for tutorial in sorted(tutorials, key=lambda x: x["completed_at"], reverse=True):
                completed_date = datetime.strptime(tutorial["completed_at"], "%Y-%m-%d %H:%M:%S").strftime("%b %d, %Y")
                st.markdown(f"✅ **{tutorial['title']}** - *{completed_date}*")
    
    # Create a centered column for the button to make it vertical
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("Hide History", use_container_width=True):
            st.session_state.show_bite_sized_history = False
            st.rerun()


def display_adaptive_content(module_id):
    """Display module content with adaptive content delivery based on user preferences"""
    if module_id not in LEARNING_MODULES:
        st.error("Module not found!")
        return
    
    module = LEARNING_MODULES[module_id]
    
    # Track learning session time
    if 'session_start_time' not in st.session_state:
        st.session_state.session_start_time = datetime.now()
        # Update last learning session timestamp
        st.session_state.last_learning_session = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Update learning streak
        if st.session_state.last_learning_session:
            last_date = datetime.strptime(st.session_state.last_learning_session, "%Y-%m-%d %H:%M:%S").date()
            today = datetime.now().date()
            yesterday = today - timedelta(days=1)
            
            if last_date == yesterday:
                st.session_state.learning_streak += 1
            elif last_date < yesterday:
                st.session_state.learning_streak = 1
    
    # Header with learning streak
    if st.session_state.learning_streak > 1:
        st.markdown(f"# {module['name']} 🔥 {st.session_state.learning_streak} day streak!")
    else:
        st.markdown(f"# {module['name']}")
    
    st.markdown(f"*{module['description']}*")
    
    # Display adaptive preferences
    difficulty_preference = st.session_state.module_difficulty_preference
    learning_style = st.session_state.learning_style
    learning_pace = st.session_state.learning_pace
    
    # Style preferences indicator
    preferences_col1, preferences_col2, preferences_col3 = st.columns(3)
    
    with preferences_col1:
        difficulty_icon = "📝" if difficulty_preference == "easier" else "📊" if difficulty_preference == "standard" else "🔥"
        difficulty_text = "Easier" if difficulty_preference == "easier" else "Standard" if difficulty_preference == "standard" else "Advanced"
        st.info(f"{difficulty_icon} Difficulty: {difficulty_text}")
    
    with preferences_col2:
        style_icon = "📊" if learning_style == "visual" else "📚" if learning_style == "textual" else "🔄"
        style_text = "Visual" if learning_style == "visual" else "Textual" if learning_style == "textual" else "Interactive"
        st.info(f"{style_icon} Style: {style_text}")
    
    with preferences_col3:
        pace_icon = "🐢" if learning_pace == "slow" else "🚶" if learning_pace == "medium" else "🏃"
        pace_text = "Slow" if learning_pace == "slow" else "Medium" if learning_pace == "medium" else "Fast"
        st.info(f"{pace_icon} Pace: {pace_text}")
    
    # Content tabs with adapted naming based on learning style
    if learning_style == "visual":
        tabs_names = ["Visual Guide", "Quiz", "Challenge"]
    elif learning_style == "textual":
        tabs_names = ["Detailed Explanation", "Quiz", "Challenge"]
    else:  # interactive
        tabs_names = ["Practice & Learn", "Quiz", "Challenge"]
    
    content_tab, quiz_tab, challenge_tab = st.tabs(tabs_names)
    
    with content_tab:
        if module_id in MODULE_CONTENT:
            # Get standard content - and remove leading whitespace that can cause formatting issues
            standard_content = MODULE_CONTENT[module_id].strip()
            
            # Real-time market example with actual current data
            try:
                # If module is about an indicator, show a real-time example
                indicator_examples = {
                    "intro_to_rsi": "RSI",
                    "intro_to_macd": "MACD",
                    "bollinger_bands": "Bollinger Bands"
                }
                
                if module_id in indicator_examples:
                    ticker = "SPY"  # Default example using S&P 500 ETF
                    end_date = datetime.now().strftime("%Y-%m-%d")
                    start_date = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")
                    
                    # Get stock data - wrap in try/except to handle potential API issues
                    try:
                        data = yf.download(ticker, start=start_date, end=end_date)
                        if not data.empty:
                            # Create real-time example visualization based on the indicator
                            fig = go.Figure()
                            fig.add_trace(go.Candlestick(
                                x=data.index,
                                open=data['Open'],
                                high=data['High'],
                                low=data['Low'],
                                close=data['Close'],
                                name=ticker
                            ))
                            
                            # Add indicator-specific visualization
                            indicator_name = indicator_examples[module_id]
                            
                            if indicator_name == "RSI":
                                # Calculate RSI
                                delta = data['Close'].diff()
                                gain = delta.where(delta > 0, 0)
                                loss = -delta.where(delta < 0, 0)
                                avg_gain = gain.rolling(window=14).mean()
                                avg_loss = loss.rolling(window=14).mean()
                                rs = avg_gain / avg_loss
                                rsi = 100 - (100 / (1 + rs))
                                
                                # Create RSI subplot
                                fig.add_trace(go.Scatter(x=data.index, y=rsi, name="RSI", line=dict(color='purple')))
                                fig.add_hline(y=70, line_dash="dash", line_color="red")
                                fig.add_hline(y=30, line_dash="dash", line_color="green")
                                fig.update_layout(title=f"Real-time {ticker} with RSI Indicator", height=600)
                            
                            elif indicator_name == "MACD":
                                # Calculate MACD
                                exp1 = data['Close'].ewm(span=12, adjust=False).mean()
                                exp2 = data['Close'].ewm(span=26, adjust=False).mean()
                                macd = exp1 - exp2
                                signal = macd.ewm(span=9, adjust=False).mean()
                                histogram = macd - signal
                                
                                # Create MACD subplot
                                fig.add_trace(go.Scatter(x=data.index, y=macd, name="MACD", line=dict(color='blue')))
                                fig.add_trace(go.Scatter(x=data.index, y=signal, name="Signal", line=dict(color='red')))
                                fig.add_trace(go.Bar(x=data.index, y=histogram, name="Histogram"))
                                fig.update_layout(title=f"Real-time {ticker} with MACD Indicator", height=600)
                            
                            elif indicator_name == "Bollinger Bands":
                                # Calculate Bollinger Bands
                                sma = data['Close'].rolling(window=20).mean()
                                std = data['Close'].rolling(window=20).std()
                                upper_band = sma + (std * 2)
                                lower_band = sma - (std * 2)
                                
                                # Add Bollinger Bands to chart
                                fig.add_trace(go.Scatter(x=data.index, y=sma, name="SMA", line=dict(color='blue')))
                                fig.add_trace(go.Scatter(x=data.index, y=upper_band, name="Upper Band", line=dict(color='green')))
                                fig.add_trace(go.Scatter(x=data.index, y=lower_band, name="Lower Band", line=dict(color='red')))
                                fig.update_layout(title=f"Real-time {ticker} with Bollinger Bands", height=600)
                            
                            # Generate a unique ID for this chart based on ticker and indicator
                            chart_id = f"{ticker}_{indicator_name}".replace(" ", "_").lower()
                            st.plotly_chart(fig, use_container_width=True, key=f"practice_chart_{chart_id}")
                            
                            # Show explanation relevant to the real-time example
                            st.subheader(f"Real-time {indicator_name} Analysis")
                            st.write(f"This is a current example of {indicator_name} applied to {ticker} (S&P 500 ETF). Use this chart to practice identifying signals in real market conditions.")
                    except Exception as e:
                        st.warning(f"Could not load real-time example: {str(e)}")
            except:
                pass  # If real-time example fails, continue with static content
                
            # Adapt content based on learning style
            if learning_style == "visual":
                # Emphasize charts and visual elements
                visual_header = """
                <div style="background-color: #000000 !important; color: #ffffff !important; padding: 15px; border-radius: 5px; margin-bottom: 20px; border: 1px solid #444444;">
                    <h3 style="color: #ffffff !important;">📊 Visual Learning Mode</h3>
                    <p style="color: #ffffff !important;">Content is presented with emphasis on charts, diagrams, and visual examples.</p>
                </div>
                """
                
                # Display header and content separately
                st.markdown(visual_header, unsafe_allow_html=True)
                st.markdown(standard_content, unsafe_allow_html=True)
                
                # Create empty string for backward compatibility
                visual_content = ""
                # Add interactive chart examples depending on the specific module
                st.markdown(visual_content, unsafe_allow_html=True)
                
            elif learning_style == "textual":
                # Emphasize detailed text explanations
                textual_header = """
                <div style="background-color: #000000 !important; color: #ffffff !important; padding: 15px; border-radius: 5px; margin-bottom: 20px; border: 1px solid #444444;">
                    <h3 style="color: #ffffff !important;">📚 Textual Learning Mode</h3>
                    <p style="color: #ffffff !important;">Content is presented with detailed explanations and examples.</p>
                </div>
                """
                
                # Display header and content separately
                st.markdown(textual_header, unsafe_allow_html=True)
                st.markdown(standard_content, unsafe_allow_html=True)
                
            elif learning_style == "interactive":
                # Focus on interactive elements and hands-on practice
                interactive_header = """
                <div style="background-color: #000000 !important; color: #ffffff !important; padding: 15px; border-radius: 5px; margin-bottom: 20px; border: 1px solid #444444;">
                    <h3 style="color: #ffffff !important;">🔄 Interactive Learning Mode</h3>
                    <p style="color: #ffffff !important;">Content is presented with focus on practice and hands-on activities.</p>
                </div>
                """
                
                # Display header and content separately
                st.markdown(interactive_header, unsafe_allow_html=True)
                st.markdown(standard_content, unsafe_allow_html=True)
                
                # Add interactive elements specific to this module
                st.subheader("Practice Section")
                st.write("Apply your knowledge with these interactive exercises:")
                
                # Simple interactive exercises based on module
                if module_id == "intro_to_markets":
                    # Market basics exercise
                    st.selectbox("What happens to stock prices when more people want to buy than sell?", 
                                ["Select an answer", "Prices tend to rise", "Prices tend to fall", "Prices remain the same"],
                                key="interactive_ex1")
                elif module_id == "candlestick_charts":
                    # Candlestick pattern recognition
                    st.selectbox("In a bullish engulfing pattern, the second candle:", 
                                ["Select an answer", "Completely engulfs the first bearish candle", "Is smaller than the first candle", "Has the same open and close as the first candle"],
                                key="interactive_ex2")
            
            # Additional content based on difficulty preference
            if difficulty_preference == "easier":
                # For easier mode, add more explanations and examples
                if learning_pace == "slow":
                    # For slow pace, break content into smaller chunks with more recaps
                    st.subheader("Key Concepts Recap")
                    st.write("Let's review the main points to make sure you understand them:")
                    
                    with st.expander("Click to review key terms"):
                        st.markdown("""
                        * **Market Trend** - The general direction in which a market is moving
                        * **Support Level** - Price level where buying interest is strong enough to overcome selling pressure
                        * **Resistance Level** - Price level where selling interest is strong enough to overcome buying pressure
                        * **Volume** - The number of shares traded during a given period
                        """)
                
                st.markdown(f"""
                <div style="background-color: #000000 !important; color: #ffffff !important; padding: 15px; border-radius: 5px; margin-top: 20px; border: 1px solid #444444;">
                    <h3 style="color: #ffffff !important;">📌 Summary</h3>
                    <p style="color: #ffffff !important;">The key points from this module are:</p>
                    <ul style="color: #ffffff !important;">
                        <li style="color: #ffffff !important;">Understanding {module['name']} helps you make better investment decisions</li>
                        <li style="color: #ffffff !important;">Practice identifying these patterns in real charts</li>
                        <li style="color: #ffffff !important;">Remember that no indicator works perfectly every time</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            elif difficulty_preference == "harder":
                # For harder mode, add more advanced concepts and challenges
                if learning_pace == "fast":
                    # For fast pace, add extra advanced content
                    st.subheader("Advanced Trading Concepts")
                    st.write("Since you prefer a faster pace, here are some additional advanced concepts to explore:")
                    
                    advanced_topics = {
                        "intro_to_rsi": "RSI divergence and failure swings",
                        "intro_to_macd": "MACD histogram analysis and zero-line crossovers",
                        "bollinger_bands": "Bollinger Band width as a volatility indicator",
                        "candlestick_charts": "Multi-candlestick patterns and their reliability statistics",
                        "risk_management": "Position sizing algorithms and risk-adjusted returns"
                    }
                    
                    if module_id in advanced_topics:
                        st.write(f"📊 **{advanced_topics[module_id]}** - Research this topic for deeper understanding")
                
                st.markdown(f"""
                <div style="background-color: #000000 !important; color: #ffffff !important; padding: 15px; border-radius: 5px; margin-top: 20px; border: 1px solid #444444;">
                    <h3 style="color: #ffffff !important;">🧠 Advanced Applications</h3>
                    <p style="color: #ffffff !important;">For those seeking deeper understanding:</p>
                    <ul style="color: #ffffff !important;">
                        <li style="color: #ffffff !important;">How would you combine {module['name']} with other indicators?</li>
                        <li style="color: #ffffff !important;">What are the limitations of this approach in different market conditions?</li>
                        <li style="color: #ffffff !important;">Consider how institutional investors might exploit retail traders using these indicators</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            # Track related concepts for this module
            related_concepts = {
                "intro_to_markets": ["market_basics"],
                "candlestick_charts": ["chart_patterns"],
                "intro_to_rsi": ["technical_indicators"],
                "intro_to_macd": ["technical_indicators"],
                "bollinger_bands": ["technical_indicators"],
                "advanced_indicators": ["technical_indicators", "advanced_strategies"],
                "risk_management": ["risk_management"]
            }
            
            # Update concept mastery
            if module_id in related_concepts:
                for concept in related_concepts[module_id]:
                    if concept in st.session_state.concept_mastery:
                        # Small increment for viewing the content
                        current_mastery = st.session_state.concept_mastery[concept]
                        st.session_state.concept_mastery[concept] = min(100, current_mastery + 2)
                        
            # For spaced repetition, show concepts that need review
            if st.session_state.spaced_repetition_queue:
                with st.expander("📝 Review Previous Concepts"):
                    st.write("Based on your learning history, these concepts could use review:")
                    for concept in st.session_state.spaced_repetition_queue[:3]:  # Show top 3
                        concept_display = concept.replace('_', ' ').title() if concept else "General"
                        st.markdown(f"* **{concept_display}**")
                        
                    if st.button("Mark reviewed", key="mark_reviewed"):
                        # Remove reviewed concepts from the queue
                        if st.session_state.spaced_repetition_queue:
                            st.session_state.spaced_repetition_queue = st.session_state.spaced_repetition_queue[3:]
                            st.rerun()
        else:
            st.info("Content for this module is under development.")
    
    with quiz_tab:
        if module_id in QUIZ_QUESTIONS:
            display_module_quiz(module_id)
        else:
            st.info("Quiz for this module is under development.")
    
    with challenge_tab:
        display_challenge(module_id)
    
    # Navigation buttons
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.button("Back to Learning Path"):
            st.session_state.active_module = None
            st.rerun()
    
    with col3:
        module_completed = module_id in st.session_state.learning_path_progress and \
                          st.session_state.learning_path_progress[module_id].get("completed", False)
        
        if module_completed:
            button_text = "Already Completed"
            button_disabled = True
        else:
            button_text = "Mark as Completed"
            button_disabled = False
            
        if st.button(button_text, disabled=button_disabled, type="primary"):
            # Mark as completed in learning path progress
            if module_id in st.session_state.learning_path_progress:
                st.session_state.learning_path_progress[module_id]["completed"] = True
            
            # Award badge if module has one
            badge_id = module.get("badge")
            if badge_id:
                award_badge(badge_id)
            
            # Update knowledge score
            difficulty_value = {"Beginner": 5, "Intermediate": 10, "Advanced": 15}
            module_difficulty = module.get("difficulty", "Beginner")
            st.session_state.knowledge_score = min(100, st.session_state.knowledge_score + difficulty_value.get(module_difficulty, 5))
            
            # Check if knowledge level increased
            level_up_message = update_knowledge_level()
            if level_up_message:
                st.success(level_up_message)
            
            st.success(f"Module '{module['name']}' marked as completed!")
            st.rerun()


def display_module_quiz(module_id):
    """Display quiz for a specific module with adaptive difficulty"""
    if module_id not in QUIZ_QUESTIONS:
        st.info("No quiz available for this module yet.")
        return
    
    # Get standard questions
    quiz_questions = QUIZ_QUESTIONS[module_id]
    
    # Adjust based on difficulty preference
    difficulty_preference = st.session_state.module_difficulty_preference
    
    if 'module_quiz_state' not in st.session_state:
        st.session_state.module_quiz_state = {}
    
    if module_id not in st.session_state.module_quiz_state:
        st.session_state.module_quiz_state[module_id] = {
            "current_question": 0,
            "correct_answers": 0,
            "total_questions": len(quiz_questions),
            "answers": [],
            "completed": False
        }
    
    quiz_state = st.session_state.module_quiz_state[module_id]
    
    # Display quiz progress
    if not quiz_state["completed"]:
        st.progress(quiz_state["current_question"] / quiz_state["total_questions"])
        st.write(f"Question {quiz_state['current_question'] + 1} of {quiz_state['total_questions']}")
        
        # Current question
        if quiz_state["current_question"] < quiz_state["total_questions"]:
            current_q = quiz_questions[quiz_state["current_question"]]
            
            st.write(f"**{current_q['question']}**")
            
            # Provide hints in easier mode
            if difficulty_preference == "easier" and "hint" in current_q:
                st.info(f"💡 Hint: {current_q['hint']}")
            
            # Display options
            option_cols = st.columns(2)
            for i, option in enumerate(current_q["options"]):
                col_idx = i % 2
                with option_cols[col_idx]:
                    if st.button(option, key=f"quiz_option_{i}_{module_id}", use_container_width=True):
                        is_correct = (i == current_q["correct_idx"])
                        
                        quiz_state["answers"].append({
                            "question_idx": quiz_state["current_question"],
                            "selected_option": i,
                            "is_correct": is_correct
                        })
                        
                        if is_correct:
                            quiz_state["correct_answers"] += 1
                            st.success("✅ Correct!")
                        else:
                            st.error(f"❌ Incorrect. The correct answer is: {current_q['options'][current_q['correct_idx']]}")
                            
                            # In easier mode, provide an explanation
                            if difficulty_preference == "easier" and "explanation" in current_q:
                                st.info(f"Explanation: {current_q['explanation']}")
                        
                        # Move to next question or complete quiz
                        quiz_state["current_question"] += 1
                        
                        if quiz_state["current_question"] >= quiz_state["total_questions"]:
                            quiz_state["completed"] = True
                            
                            # Update module progress
                            if module_id in st.session_state.learning_path_progress:
                                st.session_state.learning_path_progress[module_id]["quiz_score"] = \
                                    (quiz_state["correct_answers"] / quiz_state["total_questions"]) * 100
                            
                        st.rerun()
        
    else:
        # Quiz completed, show results
        score_percentage = (quiz_state["correct_answers"] / quiz_state["total_questions"]) * 100
        
        if score_percentage >= 80:
            st.success(f"Quiz completed! Score: {quiz_state['correct_answers']}/{quiz_state['total_questions']} ({score_percentage:.1f}%)")
            if module_id in st.session_state.learning_path_progress:
                st.session_state.learning_path_progress[module_id]["quiz_completed"] = True
        elif score_percentage >= 60:
            st.warning(f"Quiz completed! Score: {quiz_state['correct_answers']}/{quiz_state['total_questions']} ({score_percentage:.1f}%)")
        else:
            st.error(f"Quiz completed! Score: {quiz_state['correct_answers']}/{quiz_state['total_questions']} ({score_percentage:.1f}%)")
            st.write("You might want to review the material and try again.")
        
        # Option to reset and retry
        if st.button("Retry Quiz"):
            # Reset quiz state
            if module_id in st.session_state.module_quiz_state:
                del st.session_state.module_quiz_state[module_id]
            st.rerun()


def display_challenge(module_id):
    """Display challenge for a specific module based on knowledge level"""
    module = LEARNING_MODULES[module_id]
    module_difficulty = module.get("difficulty", "Beginner").lower()
    user_level = st.session_state.knowledge_level
    
    # Match user level with appropriate challenges
    challenge_level = "beginner"
    if user_level in ["intermediate", "advanced", "expert"]:
        challenge_level = "intermediate"
    if user_level in ["advanced", "expert"]:
        challenge_level = "advanced"
    
    # Additionally consider the difficulty preference
    if st.session_state.module_difficulty_preference == "easier":
        # Step down challenge level if possible
        if challenge_level == "advanced":
            challenge_level = "intermediate"
        elif challenge_level == "intermediate":
            challenge_level = "beginner"
    elif st.session_state.module_difficulty_preference == "harder":
        # Step up challenge level if possible
        if challenge_level == "beginner" and "intermediate" in CHALLENGE_SCENARIOS:
            challenge_level = "intermediate"
        elif challenge_level == "intermediate" and "advanced" in CHALLENGE_SCENARIOS:
            challenge_level = "advanced"
    
    # Get appropriate challenges
    if challenge_level in CHALLENGE_SCENARIOS and CHALLENGE_SCENARIOS[challenge_level]:
        challenges = CHALLENGE_SCENARIOS[challenge_level]
        selected_challenge = random.choice(challenges)
        
        # Create unique ID for this challenge
        challenge_id = f"{module_id}_{hash(selected_challenge['title'])}"
        challenge_completed = challenge_id in st.session_state.challenge_completed
        
        # Display challenge card
        st.markdown(f"""
        <div style="padding: 15px; border-radius: 5px; background-color: #000000 !important; color: #ffffff !important; margin-bottom: 10px; border: 1px solid #444444;">
            <h3 style="margin-top: 0; color: #ffffff !important;">🏆 {selected_challenge['title']}</h3>
            <p style="color: #ffffff !important;">{selected_challenge['description']}</p>
            <p style="color: #ffffff !important;"><strong style="color: #ffffff !important;">Task:</strong> {selected_challenge['task']}</p>
            <p style="color: #ffffff !important;"><strong style="color: #ffffff !important;">Points:</strong> {selected_challenge['points']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # For some challenges, provide a sample chart
        if "chart_required" in selected_challenge and selected_challenge["chart_required"]:
            # Generate a random stock chart
            try:
                tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "JPM"]
                ticker = random.choice(tickers)
                end_date = datetime.now() - timedelta(days=random.randint(7, 30))
                start_date = end_date - timedelta(days=90)
                
                df = yf.download(ticker, start=start_date, end=end_date, progress=False)
                
                if not df.empty:
                    fig = go.Figure()
                    
                    fig.add_trace(go.Candlestick(
                        x=df.index,
                        open=df['Open'],
                        high=df['High'],
                        low=df['Low'],
                        close=df['Close'],
                        name='Price'
                    ))
                    
                    fig.update_layout(
                        title=f"{ticker} Stock Price",
                        xaxis_title="Date",
                        yaxis_title="Price",
                        xaxis_rangeslider_visible=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True, key="challenge_chart")
            except Exception as e:
                st.error(f"Error generating chart: {e}")
        
        # Solution submission
        st.subheader("Your Solution")
        solution = st.text_area("Enter your solution or analysis here", 
                               placeholder="Explain your approach and findings...",
                               disabled=challenge_completed)
        
        # Submit solution
        if not challenge_completed:
            if st.button("Submit Solution", key=f"submit_{challenge_id}"):
                if solution.strip():
                    # Mark challenge as completed
                    st.session_state.challenge_completed.add(challenge_id)
                    
                    # Award points
                    st.session_state.knowledge_score = min(100, st.session_state.knowledge_score + (selected_challenge['points'] * 0.2))
                    
                    # Check if knowledge level increased
                    level_up_message = update_knowledge_level()
                    
                    st.success(f"Challenge completed! You earned {selected_challenge['points']} points.")
                    
                    if level_up_message:
                        st.success(level_up_message)
                    
                    st.rerun()
                else:
                    st.warning("Please enter your solution before submitting.")
        else:
            st.success("You've already completed this challenge!")
    else:
        st.info("No challenges available for your current knowledge level.")


def learning_progress_insights():
    """Display insights and analytics about learning progress"""
    st.subheader("Learning Progress Insights")
    
    # Prepare data
    if not st.session_state.learning_path_progress:
        st.info("Complete some modules to see learning insights.")
        return
    
    # Extract module completion data
    module_data = []
    completion_timestamps = []
    
    for module_id, progress in st.session_state.learning_path_progress.items():
        if module_id in LEARNING_MODULES:
            module = LEARNING_MODULES[module_id]
            
            # Get the last accessed timestamp
            last_accessed = progress.get("last_accessed")
            if last_accessed:
                try:
                    timestamp = datetime.strptime(last_accessed, "%Y-%m-%d %H:%M:%S")
                    completion_timestamps.append(timestamp)
                except:
                    pass
            
            module_data.append({
                "Module": module["name"],
                "Difficulty": module.get("difficulty", "Beginner"),
                "Completed": progress.get("completed", False),
                "Quiz Score": progress.get("quiz_score", 0),
                "Time Estimate": module.get("estimated_time", "Unknown")
            })
    
    # Display learning insights in tabs
    insight_tabs = st.tabs(["Overview", "Concept Mastery", "Learning Activity", "Recommendations"])
    
    with insight_tabs[0]:  # Overview tab
        st.subheader("Learning Overview")
        
        # Create summary metrics
        metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
        
        # Calculate key metrics
        total_modules = len(LEARNING_MODULES)
        completed_modules = sum(1 for m in module_data if m["Completed"])
        started_modules = len(module_data)
        avg_quiz_score = np.mean([m["Quiz Score"] for m in module_data if m["Quiz Score"] > 0]) if any(m["Quiz Score"] > 0 for m in module_data) else 0
        
        with metrics_col1:
            st.metric("Knowledge Level", KNOWLEDGE_LEVELS[st.session_state.knowledge_level]["name"])
        
        with metrics_col2:
            progress_pct = int((completed_modules / total_modules) * 100) if total_modules > 0 else 0
            st.metric("Overall Progress", f"{progress_pct}%")
        
        with metrics_col3:
            st.metric("Learning Streak", f"{st.session_state.learning_streak} days")
        
        with metrics_col4:
            st.metric("Avg Quiz Score", f"{avg_quiz_score:.1f}%")
        
        # Create progress chart
        if module_data:
            st.subheader("Module Progress")
            
            # Prepare data for chart
            chart_data = pd.DataFrame(module_data)
            
            if not chart_data.empty:
                # Create module completion bar chart
                chart_data["Completion Status"] = chart_data["Completed"].map({True: "Completed", False: "In Progress"})
                chart_data["Quiz Score"] = chart_data["Quiz Score"].fillna(0)
                
                # Sort by difficulty
                difficulty_order = {"Beginner": 0, "Intermediate": 1, "Advanced": 2}
                chart_data["Difficulty_Rank"] = chart_data["Difficulty"].map(lambda x: difficulty_order.get(x, 999))
                chart_data = chart_data.sort_values("Difficulty_Rank")
                
                # Create bar chart
                fig = px.bar(
                    chart_data, 
                    x="Module", 
                    y="Quiz Score",
                    color="Completion Status",
                    hover_data=["Difficulty"],
                    text="Quiz Score",
                    title="Module Completion & Quiz Scores",
                    color_discrete_map={"Completed": "#2ECC71", "In Progress": "#F39C12"},
                    height=400
                )
                
                fig.update_layout(
                    xaxis_title="Module Name",
                    yaxis_title="Quiz Score (%)",
                    yaxis=dict(range=[0, 100]),
                    xaxis=dict(tickangle=-45)
                )
                
                st.plotly_chart(fig, use_container_width=True, key="modules_completion_chart")
                
            # Create recent activity timeline
            if completion_timestamps:
                st.subheader("Learning Activity Timeline")
                
                # Prepare timeline data
                timeline_data = []
                for i, timestamp in enumerate(sorted(completion_timestamps)):
                    timeline_data.append({
                        "Date": timestamp,
                        "Activity": "Module Completion" if i < len(module_data) else "Practice Session",
                        "Value": 1
                    })
                
                if timeline_data:
                    timeline_df = pd.DataFrame(timeline_data)
                    timeline_df = timeline_df.set_index("Date")
                    
                    # Resample to daily activity count
                    activity_by_day = timeline_df.resample('D')["Value"].sum().reset_index()
                    activity_by_day = activity_by_day.rename(columns={"Value": "Activity Count"})
                    
                    # Create timeline chart
                    fig = px.bar(
                        activity_by_day,
                        x="Date",
                        y="Activity Count",
                        title="Daily Learning Activity",
                        height=300
                    )
                    
                    fig.update_layout(
                        xaxis_title="Date",
                        yaxis_title="Activities"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True, key="activity_timeline_chart")
    
    with insight_tabs[1]:  # Concept Mastery tab
        st.subheader("Concept Mastery Tracker")
        
        # Create concept mastery radar chart
        concept_mastery = st.session_state.concept_mastery
        concepts = list(concept_mastery.keys())
        mastery_values = [concept_mastery[c] for c in concepts]
        
        # Format concept names for display
        display_concepts = [c.replace('_', ' ').title() for c in concepts]
        
        # Create radar chart for concept mastery
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=mastery_values,
            theta=display_concepts,
            fill='toself',
            name='Current Mastery'
        ))
        
        # Add "expert" reference
        fig.add_trace(go.Scatterpolar(
            r=[85] * len(concepts),
            theta=display_concepts,
            fill='none',
            line=dict(dash='dot', color='gray'),
            name='Expert Level'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            title="Concept Mastery Levels",
            height=500,
        )
        
        st.plotly_chart(fig, use_container_width=True, key="concept_mastery_radar_chart")
        
        # Concept details
        concept_details = {
            "market_basics": "Understanding of market mechanics, terminology, and basic principles",
            "chart_patterns": "Ability to recognize and interpret price patterns in charts",
            "technical_indicators": "Knowledge of various technical indicators and their applications",
            "risk_management": "Understanding of risk principles and position sizing strategies",
            "trading_psychology": "Awareness of psychological factors that impact trading decisions",
            "fundamental_analysis": "Ability to evaluate companies based on financial data",
            "advanced_strategies": "Knowledge of complex trading strategies and their implementation"
        }
        
        # Display concept descriptions and progress
        st.subheader("Concept Details")
        for concept, mastery in concept_mastery.items():
            col1, col2 = st.columns([3, 1])
            
            with col1:
                concept_display = concept.replace('_', ' ').title() if concept else "General"
                st.markdown(f"**{concept_display}**")
                st.write(concept_details.get(concept, ""))
            
            with col2:
                # Create circular progress indicator
                if mastery < 30:
                    level_label = "Beginner"
                    color = "#F39C12"  # Orange
                elif mastery < 60:
                    level_label = "Intermediate"
                    color = "#3498DB"  # Blue
                elif mastery < 85:
                    level_label = "Advanced"
                    color = "#2ECC71"  # Green
                else:
                    level_label = "Expert"
                    color = "#9B59B6"  # Purple
                
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=mastery,
                    domain=dict(x=[0, 1], y=[0, 1]),
                    gauge=dict(
                        axis=dict(range=[0, 100]),
                        bar=dict(color=color),
                    ),
                    number=dict(suffix="%"),
                    title=dict(text=level_label)
                ))
                
                fig.update_layout(height=150, margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig, use_container_width=True, key=f"concept_gauge_{concept}")
            
            st.markdown("---")
    
    with insight_tabs[2]:  # Learning Activity tab
        st.subheader("Learning Activity Analysis")
        
        # Display total learning time
        total_hours = st.session_state.total_learning_time / 60 if st.session_state.total_learning_time else 0
        st.metric("Total Learning Time", f"{total_hours:.1f} hours")
        
        # Display learning streak data
        st.subheader("Learning Consistency")
        
        if st.session_state.learning_streak > 1:
            st.success(f"🔥 You're on a {st.session_state.learning_streak}-day learning streak! Keep it up!")
        elif st.session_state.learning_streak == 1:
            st.info("You've started your learning journey today. Come back tomorrow to build your streak!")
        else:
            st.warning("You haven't been active recently. Start learning today to build a streak!")
        
        # Display learning style preferences
        st.subheader("Your Learning Preferences")
        pref_col1, pref_col2, pref_col3 = st.columns(3)
        
        with pref_col1:
            style_icon = "📊" if st.session_state.learning_style == "visual" else "📚" if st.session_state.learning_style == "textual" else "🔄"
            style_name = "Visual" if st.session_state.learning_style == "visual" else "Textual" if st.session_state.learning_style == "textual" else "Interactive"
            st.info(f"{style_icon} **Learning Style:** {style_name}")
        
        with pref_col2:
            diff_icon = "📝" if st.session_state.module_difficulty_preference == "easier" else "📊" if st.session_state.module_difficulty_preference == "standard" else "🔥"
            diff_name = "Easier" if st.session_state.module_difficulty_preference == "easier" else "Standard" if st.session_state.module_difficulty_preference == "standard" else "Challenge"
            st.info(f"{diff_icon} **Difficulty:** {diff_name}")
            
        with pref_col3:
            pace_icon = "🐢" if st.session_state.learning_pace == "slow" else "🚶" if st.session_state.learning_pace == "medium" else "🏃"
            pace_name = "Slow" if st.session_state.learning_pace == "slow" else "Medium" if st.session_state.learning_pace == "medium" else "Fast"
            st.info(f"{pace_icon} **Learning Pace:** {pace_name}")
    
    with insight_tabs[3]:  # Recommendations tab
        st.subheader("Personalized Recommendations")
        
        # Create personalized feedback based on learning patterns
        if not st.session_state.personalized_feedback:
            # Generate initial feedback
            feedback = []
            
            # Check concept mastery for imbalances
            concept_values = list(st.session_state.concept_mastery.values())
            if any(concept_values):  # If we have some mastery data
                min_concept = min(st.session_state.concept_mastery.items(), key=lambda x: x[1])
                max_concept = max(st.session_state.concept_mastery.items(), key=lambda x: x[1])
                
                # If there's a big gap between min and max concept mastery
                if max_concept[1] - min_concept[1] > 30:
                    feedback.append({
                        "type": "improvement",
                        "title": "Knowledge Gap Detected",
                        "description": f"Your understanding of {min_concept[0].replace('_', ' ').title()} is significantly lower than other concepts. Focus on this area to become more well-rounded.",
                        "action": f"Complete modules related to {min_concept[0].replace('_', ' ')}."
                    })
            
            # Check learning style effectiveness
            if module_data and any(m["Quiz Score"] > 0 for m in module_data):
                avg_score = np.mean([m["Quiz Score"] for m in module_data if m["Quiz Score"] > 0])
                if avg_score < 70:
                    current_style = st.session_state.learning_style
                    suggested_style = "visual" if current_style != "visual" else "interactive"
                    
                    feedback.append({
                        "type": "suggestion",
                        "title": "Try a Different Learning Style",
                        "description": f"Your current {current_style} learning style might not be optimal based on your quiz scores.",
                        "action": f"Try switching to {suggested_style} learning mode in your settings."
                    })
            
            # Check for consistency
            if st.session_state.learning_streak < 2:
                feedback.append({
                    "type": "habit",
                    "title": "Build a Learning Habit",
                    "description": "Consistent daily learning leads to better retention and faster progress.",
                    "action": "Set a regular time each day for at least 15 minutes of learning."
                })
            
            # Save feedback
            st.session_state.personalized_feedback = feedback
        
        # Display personalized feedback
        if st.session_state.personalized_feedback:
            for i, item in enumerate(st.session_state.personalized_feedback):
                icon = "🔎" if item["type"] == "improvement" else "💡" if item["type"] == "suggestion" else "⏱️"
                
                st.markdown(f"""
                <div style="padding: 15px; border-left: 5px solid #3498DB; background-color: #EBF5FB; margin-bottom: 15px;">
                    <h4>{icon} {item['title']}</h4>
                    <p>{item['description']}</p>
                    <p><strong>Suggested Action:</strong> {item['action']}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Add a next steps recommendation based on progress
            next_steps = get_recommended_modules()
            if next_steps:
                st.subheader("Suggested Next Steps")
                for i, rec in enumerate(next_steps[:2]):  # Show top 2 recommendations
                    module_id = rec["module_id"]
                    module = LEARNING_MODULES[module_id]
                    
                    st.markdown(f"""
                    <div style="padding: 15px; border-left: 5px solid #2ECC71; background-color: #EAFAF1; margin-bottom: 15px;">
                        <h4>📚 {module['name']}</h4>
                        <p>{module['description']}</p>
                        <p><strong>Why:</strong> {rec['reason']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if st.button(f"Start {module['name']}", key=f"insight_btn_{module_id}"):
                        st.session_state.active_module = module_id
                        st.session_state.learning_path_progress[module_id] = st.session_state.learning_path_progress.get(module_id, {
                            "started": True,
                            "completed": False,
                            "quiz_score": 0,
                            "last_accessed": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        })
                        st.rerun()
        else:
            st.info("Complete more modules to receive personalized recommendations.")
    
    if module_data:
        module_df = pd.DataFrame(module_data)
        
        # Create completion rate chart by difficulty
        completion_by_difficulty = module_df.groupby("Difficulty").agg({
            "Completed": lambda x: (sum(x) / len(x)) * 100
        }).reset_index()
        
        fig = px.bar(
            completion_by_difficulty,
            x="Difficulty",
            y="Completed",
            color="Difficulty",
            title="Completion Rate by Difficulty Level",
            labels={"Completed": "Completion Rate (%)"}
        )
        
        fig.update_layout(xaxis_title="Difficulty Level", yaxis_title="Completion Rate (%)")
        st.plotly_chart(fig, use_container_width=True, key="completion_rate_chart")
        
        # Calculate overall stats
        num_completed = sum(module_df["Completed"])
        total_modules = len(module_df)
        completion_rate = (num_completed / total_modules) * 100 if total_modules > 0 else 0
        
        # Display summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Modules Completed", f"{num_completed}/{total_modules}")
        with col2:
            st.metric("Completion Rate", f"{completion_rate:.1f}%")
        with col3:
            st.metric("Knowledge Level", KNOWLEDGE_LEVELS[st.session_state.knowledge_level]["name"])
        
        # Learning activity over time
        if completion_timestamps:
            # Group completions by day
            completion_days = {}
            for timestamp in completion_timestamps:
                day = timestamp.strftime("%Y-%m-%d")
                if day in completion_days:
                    completion_days[day] += 1
                else:
                    completion_days[day] = 1
            
            # Create activity chart
            activity_data = [{"Date": day, "Modules": count} for day, count in completion_days.items()]
            if activity_data:
                activity_df = pd.DataFrame(activity_data)
                activity_df["Date"] = pd.to_datetime(activity_df["Date"])
                activity_df = activity_df.sort_values("Date")
                
                fig = px.line(
                    activity_df,
                    x="Date",
                    y="Modules",
                    markers=True,
                    title="Learning Activity Over Time"
                )
                
                fig.update_layout(xaxis_title="Date", yaxis_title="Modules Interacted With")
                st.plotly_chart(fig, use_container_width=True, key="activity_over_time_chart")
    else:
        st.info("No learning data available yet.")


def personalized_learning_path():
    """Main function for the personalized learning path section"""
    st.title("📚 Personalized Learning Path")
    
    # Initialize learning path state variables
    initialize_learning_path()
    
    # Check for achievements related to personalized learning path
    from gamification_minimal import check_personalized_learning_achievements
    check_personalized_learning_achievements()
    
    # Update session tracking
    if 'session_start_time' in st.session_state:
        track_learning_session()
    
    # If it's a new day, update the spaced repetition queue
    if st.session_state.last_learning_session:
        last_date = datetime.strptime(st.session_state.last_learning_session, "%Y-%m-%d %H:%M:%S").date()
        today = datetime.now().date()
        
        if last_date != today:
            update_spaced_repetition_queue()
    
    # Create tabs for different sections
    tabs = st.tabs(["Learning Path", "Bite-Sized Tutorials", "Assessment", "Practice", "Insights"])
    
    with tabs[0]:  # Learning Path tab
        # If the user was in the middle of a module, show that module
        if 'active_module' in st.session_state and st.session_state.active_module:
            # Before displaying content, make sure the session start time is set
            if 'session_start_time' not in st.session_state:
                st.session_state.session_start_time = datetime.now()
            
            display_adaptive_content(st.session_state.active_module)
            
            # Back button
            if st.button("◀️ Back to Learning Path", key="back_to_learning_path"):
                # Track time spent in this module
                track_learning_session()
                
                # Clear the active module
                st.session_state.active_module = None
                st.rerun()
        else:
            # Learning path overview
            # Show welcome message if first time or new day
            if (not st.session_state.last_learning_session or 
                datetime.strptime(st.session_state.last_learning_session, "%Y-%m-%d %H:%M:%S").date() != datetime.now().date()):
                
                welcome_col1, welcome_col2 = st.columns([3, 1])
                with welcome_col1:
                    st.subheader(f"Welcome to Your Learning Journey")
                    
                    if st.session_state.learning_streak > 1:
                        st.success(f"🔥 You're on a {st.session_state.learning_streak} day learning streak! Keep it up!")
                    elif st.session_state.learning_streak == 1:
                        st.info("Welcome back! Come back tomorrow to build your learning streak.")
                    else:
                        st.info("Welcome! Start your learning journey today.")
                    
                with welcome_col2:
                    # Show knowledge level
                    level = st.session_state.knowledge_level
                    level_info = KNOWLEDGE_LEVELS[level]
                    st.metric("Your Level", level_info["name"])
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                display_learning_path()
                
                # Show spaced repetition reminders
                if st.session_state.spaced_repetition_queue:
                    st.subheader("Review Reminders")
                    st.info("Review these concepts to reinforce your learning:")
                    
                    # Get top 3 concepts to review
                    for concept in st.session_state.spaced_repetition_queue[:3]:
                        concept_name = concept.replace('_', ' ').title() if concept else "General"
                        st.write(f"• {concept_name}")
                        
                        # Find modules related to this concept
                        related_modules = []
                        for module_id, module in LEARNING_MODULES.items():
                            module_concepts = []
                            if module_id == "intro_to_markets" and concept == "market_basics":
                                related_modules.append(module_id)
                            elif module_id in ["candlestick_charts"] and concept == "chart_patterns":
                                related_modules.append(module_id)
                            elif module_id in ["intro_to_rsi", "intro_to_macd", "bollinger_bands"] and concept == "technical_indicators":
                                related_modules.append(module_id)
                            elif module_id == "risk_management" and concept == "risk_management":
                                related_modules.append(module_id)
                        
                        # Show related module buttons
                        if related_modules:
                            review_cols = st.columns(min(3, len(related_modules)))
                            for i, module_id in enumerate(related_modules[:3]):
                                with review_cols[i]:
                                    module_name = LEARNING_MODULES[module_id]["name"]
                                    if st.button(f"Review {module_name}", key=f"review_{module_id}"):
                                        st.session_state.active_module = module_id
                                        st.session_state.learning_path_progress[module_id] = st.session_state.learning_path_progress.get(module_id, {
                                            "started": True,
                                            "completed": False,
                                            "quiz_score": 0,
                                            "last_accessed": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                        })
                                        st.rerun()
            
            with col2:
                st.subheader("Learning Recommendations")
                recommendations = get_recommended_modules()
                
                if recommendations:
                    for rec in recommendations:
                        module_id = rec["module_id"]
                        module = LEARNING_MODULES[module_id]
                        
                        st.markdown(f"""
                        <div style="padding: 10px; border-left: 3px solid #3366CC; background-color: #000000 !important; color: #ffffff !important; margin-bottom: 10px; border-radius: 5px; border: 1px solid #444444;">
                            <h4 style="margin-top: 0; color: #ffffff !important;">{module["name"]}</h4>
                            <p style="color: #ffffff !important;">{rec["reason"]} ✨</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if st.button(f"Start {module['name']}", key=f"rec_btn_{module_id}"):
                            st.session_state.active_module = module_id
                            st.session_state.learning_path_progress[module_id] = st.session_state.learning_path_progress.get(module_id, {
                                "started": True,
                                "completed": False,
                                "quiz_score": 0,
                                "last_accessed": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            })
                            st.rerun()
                else:
                    st.info("Complete your assessment to get personalized recommendations.")
                
                # Difficulty adjustment
                adjust_module_difficulty()
                
                # Highlight bite-sized tutorials
                st.markdown("""
                <div style="padding: 15px; border-radius: 5px; background-color: #000000 !important; color: #ffffff !important; margin-top: 20px; border: 1px solid #444444;">
                    <h4 style="margin-top: 0; color: #ffffff !important;">✨ Try Bite-Sized Tutorials</h4>
                    <p style="color: #ffffff !important;">Learn key concepts in just 2-3 minutes! Perfect for a quick knowledge boost.</p>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("Browse Bite-Sized Tutorials", key="browse_bite_sized", use_container_width=True):
                    # Switch to bite-sized tab
                    st.session_state.active_tab = "Bite-Sized Tutorials"
                    st.rerun()
    
    with tabs[1]:  # Bite-Sized Tutorials tab
        # Set the active tab in session state
        if st.session_state.get('active_tab') == "Bite-Sized Tutorials":
            st.session_state.active_tab = None  # Reset active tab after navigation
        
        # Check if we need to provide an initial recommendation
        if not st.session_state.recommended_bite_sized:
            recommend_bite_sized_tutorials()
        
        # Display bite-sized tutorials
        display_bite_sized_recommendations()
    
    with tabs[2]:  # Assessment tab
        run_diagnostic_test()
    
    with tabs[3]:  # Practice tab
        display_practice_questions()
        
    with tabs[4]:  # Insights tab
        learning_progress_insights()
