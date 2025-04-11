import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import random
from datetime import datetime, timedelta
import plotly.graph_objects as go
from gamification_minimal import award_badge

def generate_chart_question(chart_type="candlestick"):
    """Generate a question based on a real stock chart"""
    
    # List of popular stocks for quiz questions
    popular_stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "JPM", "V", "WMT"]
    
    # Randomly select a stock
    ticker = random.choice(popular_stocks)
    
    # Get random time period within the last 2 years
    end_date = datetime.now() - timedelta(days=random.randint(30, 90))  # End date between 1-3 months ago
    start_date = end_date - timedelta(days=30)  # 30 day period
    
    # Fetch data
    try:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if len(df) == 0:
            # Fallback to a different stock if data retrieval fails
            return generate_chart_question(chart_type)
        
        # Select a random 10-day segment from the data
        if len(df) > 15:
            start_idx = random.randint(0, len(df) - 15)
            chart_data = df.iloc[start_idx:start_idx+15]
        else:
            chart_data = df
        
        # Create chart
        fig = go.Figure()
        
        if chart_type == "candlestick":
            fig.add_trace(go.Candlestick(
                x=chart_data.index,
                open=chart_data['Open'],
                high=chart_data['High'],
                low=chart_data['Low'],
                close=chart_data['Close'],
                name='Price'
            ))
            
            # Generate a question based on the chart pattern
            # Look for specific patterns in the data
            price_changes = chart_data['Close'].pct_change().dropna()
            
            # Check if there's a significant uptrend
            uptrend = (price_changes > 0).sum() / len(price_changes) > 0.7
            
            # Check if there's a significant downtrend
            downtrend = (price_changes < 0).sum() / len(price_changes) > 0.7
            
            # Check for high volatility
            high_volatility = price_changes.std() > 0.02
            
            # Generate question and options based on the pattern
            if uptrend:
                question = "What is the overall trend shown in this chart?"
                options = [
                    "Strong uptrend",
                    "Strong downtrend",
                    "Sideways/consolidation",
                    "High volatility with no clear trend"
                ]
                correct_idx = 0
            elif downtrend:
                question = "What is the overall trend shown in this chart?"
                options = [
                    "Strong uptrend",
                    "Strong downtrend",
                    "Sideways/consolidation",
                    "High volatility with no clear trend"
                ]
                correct_idx = 1
            elif high_volatility:
                question = "What characteristic is most prominent in this chart?"
                options = [
                    "Low volatility",
                    "Strong directional trend",
                    "High volatility",
                    "Perfect price stability"
                ]
                correct_idx = 2
            else:
                question = "What pattern does this chart primarily show?"
                options = [
                    "Strong directional movement",
                    "Sideways/consolidation pattern",
                    "Head and shoulders pattern",
                    "Double top pattern"
                ]
                correct_idx = 1
                
            # Hide the ticker name to make it a pure pattern recognition question
            fig.update_layout(
                title=f"Stock Chart (Date and Ticker Hidden)",
                xaxis_title="",
                yaxis_title="Price",
                xaxis_rangeslider_visible=False,
                xaxis={"showticklabels": False}  # Hide dates
            )
            
            return {
                "question": question,
                "options": options,
                "correct_idx": correct_idx,
                "chart": fig,
                "ticker": ticker,
                "period": f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
            }
            
    except Exception as e:
        # If there's an error, try again with a different stock
        print(f"Error generating chart question: {e}")
        return generate_chart_question(chart_type)


def quiz_section():
    """Main function for the quiz section"""
    st.header("🧠 Interactive Market Quiz")
    
    # Initialize quiz state
    if 'quiz_current_question' not in st.session_state:
        st.session_state.quiz_current_question = None
    
    if 'quiz_score' not in st.session_state:
        st.session_state.quiz_score = 0
        
    if 'quiz_total_questions' not in st.session_state:
        st.session_state.quiz_total_questions = 0
        
    if 'quiz_completed_types' not in st.session_state:
        st.session_state.quiz_completed_types = set()
    
    # Quiz categories
    quiz_types = {
        "chart_reading": "📊 Chart Pattern Recognition",
        "technical_indicators": "📈 Technical Indicators Quiz",
        "market_basics": "🏛️ Market Basics Quiz"
    }
    
    # Create 2x2 grid of quiz type cards
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Choose a Quiz Category")
        
        for quiz_id, quiz_name in quiz_types.items():
            # Show a checkmark if this quiz type has been completed
            completed_indicator = "✅ " if quiz_id in st.session_state.quiz_completed_types else ""
            
            if st.button(f"{completed_indicator}{quiz_name}", key=f"btn_{quiz_id}", use_container_width=True):
                st.session_state.quiz_type = quiz_id
                st.session_state.quiz_in_progress = True
                st.session_state.quiz_current_question = None  # Reset current question
                st.session_state.quiz_score = 0
                st.session_state.quiz_total_questions = 0
                st.rerun()
                
        # Display overall stats
        st.subheader("Your Quiz Stats")
        st.metric("Quiz Categories Completed", len(st.session_state.quiz_completed_types))
        
        # Award badges based on quiz completion
        if len(st.session_state.quiz_completed_types) >= 1:
            award_badge("quiz_taker")
        
        if len(st.session_state.quiz_completed_types) >= 3:
            award_badge("knowledge_seeker")
            
        if len(st.session_state.quiz_completed_types) >= len(quiz_types):
            award_badge("market_scholar")
    
    with col2:
        if 'quiz_in_progress' in st.session_state and st.session_state.quiz_in_progress:
            quiz_type = st.session_state.quiz_type
            
            st.subheader(f"Taking: {quiz_types[quiz_type]}")
            
            # Display progress
            if st.session_state.quiz_total_questions > 0:
                st.progress(st.session_state.quiz_score / st.session_state.quiz_total_questions)
                st.caption(f"Score: {st.session_state.quiz_score}/{st.session_state.quiz_total_questions}")
            
            # Generate new question if needed
            if st.session_state.quiz_current_question is None:
                if quiz_type == "chart_reading":
                    st.session_state.quiz_current_question = generate_chart_question("candlestick")
                elif quiz_type == "technical_indicators":
                    # These would come from a predefined list of questions about indicators
                    st.session_state.quiz_current_question = {
                        "question": "What does a RSI value below 30 typically indicate?",
                        "options": [
                            "The stock is overbought",
                            "The stock is oversold",
                            "The stock is fairly valued",
                            "There is no volume in the stock"
                        ],
                        "correct_idx": 1
                    }
                elif quiz_type == "market_basics":
                    # These would come from a predefined list of questions about market basics
                    st.session_state.quiz_current_question = {
                        "question": "What is market capitalization?",
                        "options": [
                            "The total dollar value of a company's outstanding shares",
                            "The maximum price a stock has ever reached",
                            "The total number of shares available for trading",
                            "The company's annual revenue"
                        ],
                        "correct_idx": 0
                    }
            
            # Display current question
            current_q = st.session_state.quiz_current_question
            
            st.write(f"**Question:** {current_q['question']}")
            
            # Display chart if it exists in the question
            if "chart" in current_q:
                st.plotly_chart(current_q["chart"], use_container_width=True)
                
                # Show ticker and period after answering for educational purposes
                if "user_answered" in st.session_state and st.session_state.user_answered:
                    st.caption(f"Chart info: {current_q['ticker']} from {current_q['period']}")
            
            # Display answer options
            option_cols = st.columns(2)
            user_selected = None
            
            # Disable buttons if already answered
            disabled_state = False
            if "user_answered" in st.session_state and st.session_state.user_answered:
                disabled_state = True
            
            for i, option in enumerate(current_q["options"]):
                col_idx = i % 2
                button_key = f"option_{i}_{st.session_state.quiz_total_questions}"
                
                # Determine button color based on correctness (after answering)
                button_type = "primary"
                if "user_answered" in st.session_state and st.session_state.user_answered:
                    if i == current_q["correct_idx"]:
                        button_type = "success"
                    elif i == st.session_state.user_selection:
                        button_type = "danger"
                
                with option_cols[col_idx]:
                    if st.button(option, 
                                key=button_key, 
                                disabled=disabled_state,
                                type=button_type,
                                use_container_width=True):
                        user_selected = i
                        st.session_state.user_selection = i
                        st.session_state.user_answered = True
                        st.session_state.quiz_total_questions += 1
                        
                        if i == current_q["correct_idx"]:
                            st.session_state.quiz_score += 1
                            
                            # Check for a perfect score and award badge
                            if st.session_state.quiz_score == st.session_state.quiz_total_questions and st.session_state.quiz_total_questions >= 5:
                                award_badge("quiz_ace")
                                
                            # Award chart analyst badge for chart questions
                            if quiz_type == "chart_reading":
                                award_badge("chart_analyst")
                                
                        st.rerun()
            
            # Show explanation after answering
            if "user_answered" in st.session_state and st.session_state.user_answered:
                correct_option = current_q["options"][current_q["correct_idx"]]
                
                if st.session_state.user_selection == current_q["correct_idx"]:
                    st.success(f"✅ Correct! The answer is: {correct_option}")
                else:
                    st.error(f"❌ Incorrect. The correct answer is: {correct_option}")
                
                # Next question button
                if st.button("Next Question", type="primary", use_container_width=True):
                    # Mark quiz type as completed after 5 questions
                    if st.session_state.quiz_total_questions >= 5:
                        st.session_state.quiz_completed_types.add(quiz_type)
                    
                    # Reset for next question
                    st.session_state.quiz_current_question = None
                    st.session_state.user_answered = False
                    st.rerun()
                
                # End quiz button
                if st.button("End Quiz", use_container_width=True):
                    # Mark quiz type as completed after answering at least 5 questions
                    if st.session_state.quiz_total_questions >= 5:
                        st.session_state.quiz_completed_types.add(quiz_type)
                    
                    st.session_state.quiz_in_progress = False
                    st.rerun()
        else:
            st.info("Select a quiz category from the left to begin testing your market knowledge. Each quiz features real market data and interactive questions to help you learn.")
            
            # Show stock market tip of the day
            st.subheader("📝 Market Tip")
            tips = [
                "Always consider the overall market trend before making individual stock decisions.",
                "Technical indicators work best when used in combination rather than in isolation.",
                "Past performance is not always indicative of future results.",
                "High RSI values (>70) might indicate an overbought condition.",
                "A stock's volume often confirms the strength of a price trend.",
                "Diversification across sectors can help reduce portfolio risk.",
                "The best trading strategies balance potential returns with acceptable risk.",
                "Market patterns tend to repeat, but never in exactly the same way.",
                "Combining technical analysis with fundamental research often yields better results.",
                "Patience is key - sometimes the best trading decision is no trade at all."
            ]
            st.write(random.choice(tips))
