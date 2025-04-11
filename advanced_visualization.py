import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from advanced_charts import (
    create_renko_chart, 
    create_point_and_figure_chart, 
    create_comparison_chart, 
    create_interactive_chart,
    fetch_stock_events
)

def advanced_visualization_section():
    """Main function for the advanced visualization section"""
    st.header("📊 Advanced Visualization Tools")
    
    st.markdown("""
    Explore stocks using specialized chart types, compare multiple stocks, and mark important events.
    These advanced visualizations can help identify patterns that might not be visible in traditional charts.
    """)
    
    # Create tabs for different visualization types
    viz_tabs = st.tabs([
        "Chart Types", 
        "Multi-Stock Comparison", 
        "Interactive Annotations"
    ])
    
    # Tab 1: Advanced Chart Types
    with viz_tabs[0]:
        st.subheader("Advanced Chart Types")
        st.markdown("""
        Renko charts and Point & Figure charts filter out noise by focusing on price movements of a certain magnitude.
        These charts ignore time and only display significant price changes.
        """)
        
        # Select stock for advanced charts
        all_stocks = st.session_state.selected_stocks if 'selected_stocks' in st.session_state else ['AAPL']
        if not all_stocks:
            all_stocks = ['AAPL']
            
        selected_ticker = st.selectbox(
            "Select a stock",
            options=all_stocks,
            key="adv_chart_ticker"
        )
        
        # Select time period
        time_period_options = {
            "1 Month": 30,
            "3 Months": 90,
            "6 Months": 180,
            "1 Year": 365,
            "2 Years": 730,
            "5 Years": 1825
        }
        
        selected_period = st.selectbox(
            "Select time period",
            options=list(time_period_options.keys()),
            index=2,  # Default to 6 months
            key="adv_chart_period"
        )
        
        days = time_period_options[selected_period]
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Fetch the data
        try:
            stock = yf.Ticker(selected_ticker)
            df = stock.history(start=start_date, end=end_date)
            
            if len(df) > 0:
                # Select chart type
                chart_type = st.radio(
                    "Select chart type",
                    options=["Renko", "Point & Figure"],
                    horizontal=True,
                    key="adv_chart_type"
                )
                
                if chart_type == "Renko":
                    # Renko chart settings
                    auto_brick = st.checkbox("Auto-size bricks (using ATR)", value=True, key="renko_auto")
                    
                    brick_size = None
                    if not auto_brick:
                        current_price = df['Close'].iloc[-1]
                        suggested_brick = current_price * 0.02  # 2% default
                        brick_size = st.number_input(
                            "Brick size",
                            min_value=0.01,
                            max_value=float(current_price * 0.1),
                            value=float(suggested_brick),
                            step=0.01,
                            format="%.2f",
                            help="Size of each brick in dollars. Smaller values create more bricks.",
                            key="renko_brick_size"
                        )
                    
                    # Create and display the Renko chart
                    with st.spinner("Generating Renko chart..."):
                        renko_fig = create_renko_chart(df, brick_size=brick_size, auto_brick=auto_brick)
                        st.plotly_chart(renko_fig, use_container_width=True)
                        
                        st.markdown("""
                        **How to interpret Renko charts:**
                        - Each brick represents a fixed price movement
                        - Green bricks show upward movement
                        - Red bricks show downward movement
                        - Time is not evenly spaced, focusing purely on price action
                        - Helps filter market noise and identify trends more clearly
                        """)
                
                elif chart_type == "Point & Figure":
                    # Point & Figure chart settings
                    auto_box = st.checkbox("Auto-size boxes (based on price)", value=True, key="pnf_auto")
                    
                    box_size = None
                    if not auto_box:
                        current_price = df['Close'].iloc[-1]
                        suggested_box = current_price * 0.01  # 1% default
                        box_size = st.number_input(
                            "Box size",
                            min_value=0.01,
                            max_value=float(current_price * 0.05),
                            value=float(suggested_box),
                            step=0.01,
                            format="%.2f",
                            help="Size of each box in dollars. Smaller values create more detail.",
                            key="pnf_box_size"
                        )
                    
                    reversal = st.slider(
                        "Reversal amount (boxes)",
                        min_value=1,
                        max_value=5,
                        value=3,
                        help="Number of boxes needed to cause a reversal to a new column",
                        key="pnf_reversal"
                    )
                    
                    # Create and display the Point & Figure chart
                    with st.spinner("Generating Point & Figure chart..."):
                        pnf_fig = create_point_and_figure_chart(df, box_size=box_size, reversal=reversal, auto_box=auto_box)
                        st.plotly_chart(pnf_fig, use_container_width=True)
                        
                        st.markdown("""
                        **How to interpret Point & Figure charts:**
                        - X's represent upward price movements
                        - O's represent downward price movements
                        - Each column contains exclusively X's or O's
                        - A new column starts when price reverses by a set number of boxes
                        - Helps identify support/resistance levels and breakouts
                        """)
            
            else:
                st.warning(f"Could not retrieve data for {selected_ticker}")
                
        except Exception as e:
            st.error(f"Error generating chart: {str(e)}")
    
    # Tab 2: Multi-Stock Comparison
    with viz_tabs[1]:
        st.subheader("Compare Multiple Stocks")
        
        st.markdown("""
        Compare the performance of multiple stocks over time with normalized prices to see relative performance.
        This helps identify which stocks are outperforming or underperforming the others.
        """)
        
        # Get available stocks
        all_stocks = st.session_state.selected_stocks if 'selected_stocks' in st.session_state else ['AAPL']
        if len(all_stocks) < 2:
            all_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        
        # Allow user to select stocks for comparison
        selected_tickers = st.multiselect(
            "Select stocks to compare (2-10 recommended)",
            options=all_stocks + ['SPY', 'QQQ', 'DIA', 'IWM'],  # Add some common ETFs
            default=all_stocks[:min(3, len(all_stocks))],
            key="comparison_tickers"
        )
        
        # Check if any stocks are selected
        if not selected_tickers:
            st.warning("Please select at least one stock for comparison")
        else:
            # Select time period for comparison
            time_period_options = {
                "1 Month": 30,
                "3 Months": 90,
                "6 Months": 180,
                "1 Year": 365,
                "2 Years": 730,
                "5 Years": 1825,
                "Max": 0  # Special case for max available data
            }
            
            selected_period = st.selectbox(
                "Select time period",
                options=list(time_period_options.keys()),
                index=3,  # Default to 1 year
                key="comparison_period"
            )
            
            # Set start date based on selected period
            end_date = datetime.now()
            days = time_period_options[selected_period]
            
            if days > 0:
                start_date = end_date - timedelta(days=days)
            else:
                # For "Max", use a very old date that will get the maximum available data
                start_date = datetime(2000, 1, 1)
            
            # Option to normalize prices
            normalize = st.checkbox(
                "Normalize prices (base 100)",
                value=True,
                help="Scale all stocks to start at 100 for easier comparison of percentage changes",
                key="normalize_prices"
            )
            
            # Add benchmark option
            add_benchmark = st.checkbox(
                "Add S&P 500 benchmark (SPY)",
                value=True,
                help="Include S&P 500 ETF (SPY) as a benchmark for comparison",
                key="add_benchmark"
            )
            
            if add_benchmark and 'SPY' not in selected_tickers:
                selected_tickers.append('SPY')
            
            # Create and display the comparison chart
            if selected_tickers:
                with st.spinner("Generating comparison chart..."):
                    comp_fig = create_comparison_chart(
                        selected_tickers,
                        start_date=start_date,
                        end_date=end_date,
                        normalize=normalize
                    )
                    st.plotly_chart(comp_fig, use_container_width=True)
                    
                    # Add explanation
                    if normalize:
                        st.markdown("""
                        **How to interpret normalized comparison charts:**
                        - All stocks start at a base value of 100
                        - Values represent percentage changes from the starting point
                        - Example: A value of 110 means the stock increased 10% from the start date
                        - Allows direct comparison of performance regardless of share price
                        """)
                    else:
                        st.markdown("""
                        **Note on non-normalized comparison:**
                        - Stocks with higher share prices will appear at the top of the chart
                        - This view is better for comparing actual price levels rather than percentage changes
                        - Consider using normalized view for relative performance comparison
                        """)
    
    # Tab 3: Interactive Annotations
    with viz_tabs[2]:
        st.subheader("Interactive Charts with Event Annotations")
        
        st.markdown("""
        Charts with annotations for important events like earnings reports, dividends, stock splits, and news.
        These annotations provide context for price movements and help identify patterns around key events.
        """)
        
        # Select stock for annotated chart
        all_stocks = st.session_state.selected_stocks if 'selected_stocks' in st.session_state else ['AAPL']
        if not all_stocks:
            all_stocks = ['AAPL']
            
        selected_ticker = st.selectbox(
            "Select a stock",
            options=all_stocks,
            key="annotated_chart_ticker"
        )
        
        # Select time period
        time_period_options = {
            "3 Months": 90,
            "6 Months": 180,
            "1 Year": 365,
            "2 Years": 730,
            "5 Years": 1825
        }
        
        selected_period = st.selectbox(
            "Select time period",
            options=list(time_period_options.keys()),
            index=2,  # Default to 1 year
            key="annotated_chart_period"
        )
        
        days = time_period_options[selected_period]
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Select which events to show
        event_options = {
            "Earnings Reports": "earnings",
            "Dividends": "dividend",
            "Stock Splits": "split"
        }
        
        selected_events = st.multiselect(
            "Select events to display",
            options=list(event_options.keys()),
            default=list(event_options.keys()),
            key="selected_events"
        )
        
        selected_event_types = [event_options[event] for event in selected_events]
        
        # Option to add custom annotations
        add_custom = st.checkbox("Add custom annotations", key="add_custom")
        
        custom_events = []
        if add_custom:
            with st.expander("Custom Annotations", expanded=True):
                # Form for adding custom annotations
                with st.form("custom_annotation_form"):
                    annotation_date = st.date_input(
                        "Event date",
                        value=datetime.now() - timedelta(days=30),
                        min_value=start_date.date(),
                        max_value=end_date.date(),
                        key="annotation_date"
                    )
                    
                    annotation_desc = st.text_input(
                        "Event description",
                        value="",
                        placeholder="e.g., Product launch, CEO change, etc.",
                        key="annotation_desc"
                    )
                    
                    annotation_type = st.selectbox(
                        "Event type",
                        options=["info", "news", "earnings", "dividend", "split"],
                        index=0,
                        key="annotation_type"
                    )
                    
                    submit_button = st.form_submit_button("Add Annotation")
                    
                    if submit_button and annotation_desc:
                        # Convert date to datetime
                        event_datetime = datetime.combine(annotation_date, datetime.min.time())
                        
                        # Add to custom events
                        custom_events.append({
                            'date': event_datetime,
                            'description': annotation_desc,
                            'type': annotation_type
                        })
                        
                        st.success(f"Added custom annotation for {annotation_date}")
                
                # Display list of custom annotations
                if 'custom_annotations' not in st.session_state:
                    st.session_state.custom_annotations = {}
                
                ticker_key = f"{selected_ticker}_annotations"
                if ticker_key not in st.session_state.custom_annotations:
                    st.session_state.custom_annotations[ticker_key] = []
                
                # Add new custom events to session state
                if custom_events:
                    for event in custom_events:
                        st.session_state.custom_annotations[ticker_key].append(event)
                
                # Display and allow removal of existing annotations
                if st.session_state.custom_annotations[ticker_key]:
                    st.write("Existing custom annotations:")
                    for i, event in enumerate(st.session_state.custom_annotations[ticker_key]):
                        col1, col2, col3 = st.columns([2, 6, 1])
                        with col1:
                            st.write(f"{event['date'].date()}")
                        with col2:
                            st.write(f"{event['description']} ({event['type']})")
                        with col3:
                            if st.button("🗑️", key=f"delete_{i}"):
                                st.session_state.custom_annotations[ticker_key].pop(i)
                                st.rerun()
        
        # Fetch the data and create the chart
        try:
            with st.spinner("Fetching data and events..."):
                stock = yf.Ticker(selected_ticker)
                df = stock.history(start=start_date, end=end_date)
                
                if len(df) > 0:
                    # Fetch events for the selected stock
                    events = fetch_stock_events(selected_ticker, start_date, end_date)
                    
                    # Filter events based on selected types
                    filtered_events = [event for event in events if event['type'] in selected_event_types]
                    
                    # Add custom annotations from session state
                    ticker_key = f"{selected_ticker}_annotations"
                    if ticker_key in st.session_state.custom_annotations:
                        filtered_events.extend(st.session_state.custom_annotations[ticker_key])
                    
                    # Create interactive chart with annotations
                    fig = create_interactive_chart(df, selected_ticker, filtered_events)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add legend explanation
                    st.markdown("""
                    **Event Markers Legend:**
                    - 🔵 Circle: Earnings report
                    - 🟢 Triangle: Dividend
                    - 🟣 Square: Stock split
                    - 🟠 Star: News
                    - ⚪ Diamond: Info/Other
                    
                    Hover over markers to see event details.
                    """)
                else:
                    st.warning(f"Could not retrieve data for {selected_ticker}")
                    
        except Exception as e:
            st.error(f"Error generating annotated chart: {str(e)}")
