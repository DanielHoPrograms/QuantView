import streamlit as st
import pandas as pd
import yfinance as yf
import time
from datetime import datetime, timedelta
import json
import plotly.graph_objects as go
from utils import get_stock_data


def initialize_watchlists():
    """Initialize watchlist data structures in session state"""
    if 'watchlists' not in st.session_state:
        # Default watchlists with example data
        st.session_state.watchlists = {
            'Default': {
                'stocks': [],
                'created_date': datetime.now().strftime('%Y-%m-%d'),
                'description': 'Default watchlist',
                'color': '#1f77b4',  # Default blue color
                'view_mode': 'compact'
            },
            'Tech Stocks': {
                'stocks': ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META'],
                'created_date': datetime.now().strftime('%Y-%m-%d'),
                'description': 'Major tech companies',
                'color': '#ff7f0e',  # Orange
                'view_mode': 'detailed'
            }
        }
    
    if 'current_watchlist' not in st.session_state:
        # Set default watchlist as the current one
        st.session_state.current_watchlist = 'Default'
    
    if 'watchlist_display_mode' not in st.session_state:
        st.session_state.watchlist_display_mode = 'cards'  # Options: 'cards', 'table', 'charts'


def validate_ticker(ticker):
    """
    Verify if a ticker symbol is valid by attempting to fetch its data
    
    Args:
        ticker: Stock symbol to validate
    
    Returns:
        Boolean indicating if the ticker is valid
    """
    try:
        # Strip whitespace and convert to uppercase
        ticker = ticker.strip().upper()
        
        # Get ticker info
        stock = yf.Ticker(ticker)
        
        # Check if we can get recent price data
        hist = stock.history(period="1d")
        
        # Valid if we got data
        return len(hist) > 0
    except Exception:
        return False


def create_watchlist(name, description="", color="#1f77b4"):
    """
    Create a new watchlist
    
    Args:
        name: Name of the watchlist
        description: Optional description
        color: Color code for visual identification
    
    Returns:
        Boolean indicating success/failure
    """
    if not name or name.strip() == "":
        return False
    
    # Ensure name is unique
    if name in st.session_state.watchlists:
        return False
    
    # Create new watchlist
    st.session_state.watchlists[name] = {
        'stocks': [],
        'created_date': datetime.now().strftime('%Y-%m-%d'),
        'description': description,
        'color': color,
        'view_mode': 'compact'
    }
    
    # Set as current watchlist
    st.session_state.current_watchlist = name
    return True


def delete_watchlist(name):
    """
    Delete a watchlist
    
    Args:
        name: Name of the watchlist to delete
    
    Returns:
        Boolean indicating success/failure
    """
    # Prevent deleting if it's the only watchlist
    if len(st.session_state.watchlists) <= 1:
        return False
    
    # Delete the watchlist
    if name in st.session_state.watchlists:
        del st.session_state.watchlists[name]
        
        # If current watchlist was deleted, switch to the first available one
        if st.session_state.current_watchlist == name:
            st.session_state.current_watchlist = list(st.session_state.watchlists.keys())[0]
        
        return True
    
    return False


def add_to_watchlist(watchlist_name, ticker):
    """
    Add a stock to a watchlist
    
    Args:
        watchlist_name: Name of the watchlist
        ticker: Stock symbol to add
    
    Returns:
        Boolean indicating success/failure
    """
    # Check if watchlist exists
    if watchlist_name not in st.session_state.watchlists:
        return False
    
    # Validate ticker
    ticker = ticker.strip().upper()
    if not validate_ticker(ticker):
        return False
    
    # Check if ticker is already in the watchlist
    if ticker in st.session_state.watchlists[watchlist_name]['stocks']:
        return False
    
    # Add ticker to watchlist
    st.session_state.watchlists[watchlist_name]['stocks'].append(ticker)
    return True


def remove_from_watchlist(watchlist_name, ticker):
    """
    Remove a stock from a watchlist
    
    Args:
        watchlist_name: Name of the watchlist
        ticker: Stock symbol to remove
    
    Returns:
        Boolean indicating success/failure
    """
    # Check if watchlist exists
    if watchlist_name not in st.session_state.watchlists:
        return False
    
    # Check if ticker is in the watchlist
    ticker = ticker.strip().upper()
    if ticker not in st.session_state.watchlists[watchlist_name]['stocks']:
        return False
    
    # Remove ticker from watchlist
    st.session_state.watchlists[watchlist_name]['stocks'].remove(ticker)
    return True


def rename_watchlist(old_name, new_name):
    """
    Rename a watchlist
    
    Args:
        old_name: Current watchlist name
        new_name: New name for the watchlist
    
    Returns:
        Boolean indicating success/failure
    """
    if old_name not in st.session_state.watchlists or new_name in st.session_state.watchlists:
        return False
    
    # Create a new entry with the new name and copy the data
    st.session_state.watchlists[new_name] = st.session_state.watchlists[old_name].copy()
    
    # Delete the old entry
    del st.session_state.watchlists[old_name]
    
    # Update current watchlist if needed
    if st.session_state.current_watchlist == old_name:
        st.session_state.current_watchlist = new_name
    
    return True


def update_watchlist_settings(name, description=None, color=None, view_mode=None):
    """
    Update watchlist settings
    
    Args:
        name: Name of the watchlist
        description: New description (optional)
        color: New color (optional)
        view_mode: New view mode (optional)
    
    Returns:
        Boolean indicating success/failure
    """
    if name not in st.session_state.watchlists:
        return False
    
    # Update settings if provided
    if description is not None:
        st.session_state.watchlists[name]['description'] = description
    
    if color is not None:
        st.session_state.watchlists[name]['color'] = color
    
    if view_mode is not None:
        st.session_state.watchlists[name]['view_mode'] = view_mode
    
    return True


def get_watchlists_summary():
    """
    Get a summary of all watchlists
    
    Returns:
        DataFrame with watchlist summary
    """
    summaries = []
    
    for name, data in st.session_state.watchlists.items():
        # Get stock count and latest update time
        stock_count = len(data['stocks'])
        
        # Add to summary list
        summaries.append({
            'Name': name,
            'Stocks': stock_count,
            'Created': data['created_date'],
            'Description': data['description'],
            'Color': data['color']
        })
    
    return pd.DataFrame(summaries)


def get_watchlist_performance(watchlist_name, period="1mo"):
    """
    Calculate performance metrics for stocks in a watchlist
    
    Args:
        watchlist_name: Name of the watchlist
        period: Time period for performance calculation
    
    Returns:
        DataFrame with performance metrics
    """
    if watchlist_name not in st.session_state.watchlists:
        return pd.DataFrame()
    
    # Get stocks from watchlist
    stocks = st.session_state.watchlists[watchlist_name]['stocks']
    
    if not stocks:
        return pd.DataFrame()
    
    # Calculate performance for each stock
    performance_data = []
    
    for ticker in stocks:
        try:
            # Get historical data
            df = get_stock_data(ticker, period=period)
            
            if df is not None and len(df) > 0:
                # Calculate performance metrics
                start_price = df['Close'].iloc[0]
                end_price = df['Close'].iloc[-1]
                price_change = end_price - start_price
                percent_change = (price_change / start_price) * 100
                
                # Calculate volatility
                if len(df) > 1:
                    daily_returns = df['Close'].pct_change().dropna()
                    volatility = daily_returns.std() * (252 ** 0.5) * 100  # Annualized volatility in %
                else:
                    volatility = 0
                
                # Get current price and daily change
                current_price = end_price
                if len(df) > 1:
                    daily_change = ((df['Close'].iloc[-1] / df['Close'].iloc[-2]) - 1) * 100
                else:
                    daily_change = 0
                
                # Get company info
                try:
                    stock_info = yf.Ticker(ticker).info
                    company_name = stock_info.get('shortName', ticker)
                    sector = stock_info.get('sector', 'Unknown')
                    market_cap = stock_info.get('marketCap', 0)
                    market_cap_formatted = f"${market_cap / 1000000000:.2f}B" if market_cap else "N/A"
                except:
                    company_name = ticker
                    sector = "Unknown"
                    market_cap_formatted = "N/A"
                
                performance_data.append({
                    'Ticker': ticker,
                    'Company': company_name,
                    'Current Price': current_price,
                    'Daily Change %': daily_change,
                    'Period Change %': percent_change,
                    'Volatility %': volatility,
                    'Sector': sector,
                    'Market Cap': market_cap_formatted
                })
        except Exception as e:
            st.error(f"Error getting data for {ticker}: {e}")
    
    if performance_data:
        return pd.DataFrame(performance_data)
    else:
        return pd.DataFrame()


def display_watchlist_cards(watchlist_performance):
    """
    Display watchlist stocks as cards
    
    Args:
        watchlist_performance: DataFrame with watchlist performance data
    """
    if len(watchlist_performance) == 0:
        st.info("No stocks in this watchlist or performance data not available.")
        return
    
    # Display as cards
    columns = st.columns(3)
    for i, (_, stock) in enumerate(watchlist_performance.iterrows()):
        with columns[i % 3]:
            # Determine card color based on performance
            if stock['Period Change %'] > 0:
                card_bg = "linear-gradient(to right, #e6f7ff, #f0f9ff)"
                change_color = "green"
            elif stock['Period Change %'] < 0:
                card_bg = "linear-gradient(to right, #fff1f0, #fff7f6)"
                change_color = "red"
            else:
                card_bg = "linear-gradient(to right, #f9f9f9, #ffffff)"
                change_color = "grey"
            
            # Create card
            with st.container():
                st.markdown(f"""
                <div style="padding: 1rem; border-radius: 0.5rem; background: {card_bg}; 
                            margin-bottom: 1rem; border: 1px solid #ddd;">
                    <h3 style="margin: 0;">{stock['Ticker']}</h3>
                    <p style="color: #666; font-size: 0.9rem; margin: 0.2rem 0;">{stock['Company']}</p>
                    <h4 style="margin: 0.5rem 0;">${stock['Current Price']:.2f}</h4>
                    <p style="color: {change_color}; margin: 0;">
                        {'↑' if stock['Daily Change %'] >= 0 else '↓'}
                        {abs(stock['Daily Change %']):.2f}% Today
                    </p>
                    <p style="color: {change_color}; margin: 0;">
                        {'↑' if stock['Period Change %'] >= 0 else '↓'}
                        {abs(stock['Period Change %']):.2f}% Period
                    </p>
                    <p style="color: #666; font-size: 0.8rem; margin: 0.2rem 0;">
                        {stock['Sector']} | {stock['Market Cap']}
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Action buttons
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("📊 Analyze", key=f"analyze_{stock['Ticker']}"):
                        # Set as selected stock for analysis
                        if 'selected_stocks' in st.session_state and stock['Ticker'] not in st.session_state.selected_stocks:
                            st.session_state.selected_stocks.append(stock['Ticker'])
                            st.success(f"Added {stock['Ticker']} to analysis")
                            st.session_state.current_page = 'analysis'
                            st.rerun()
                
                with col2:
                    if st.button("🗑️ Remove", key=f"remove_{stock['Ticker']}"):
                        if remove_from_watchlist(st.session_state.current_watchlist, stock['Ticker']):
                            st.success(f"Removed {stock['Ticker']} from watchlist")
                            st.rerun()


def display_watchlist_table(watchlist_performance):
    """
    Display watchlist stocks as a table
    
    Args:
        watchlist_performance: DataFrame with watchlist performance data
    """
    if len(watchlist_performance) == 0:
        st.info("No stocks in this watchlist or performance data not available.")
        return
    
    # Visualize data as a stylized table
    # Create a helper function for styling
    def highlight_positive(val):
        if isinstance(val, (int, float)):
            if val > 0:
                return 'color: green'
            elif val < 0:
                return 'color: red'
        return ''
    
    # Apply styling to the dataframe
    styled_df = watchlist_performance.style.applymap(
        highlight_positive, 
        subset=['Daily Change %', 'Period Change %']
    )
    
    # Display table with sorting enabled
    st.dataframe(
        styled_df,
        column_order=['Ticker', 'Company', 'Current Price', 'Daily Change %', 
                     'Period Change %', 'Volatility %', 'Sector', 'Market Cap'],
        hide_index=True,
        use_container_width=True
    )


def display_watchlist_charts(watchlist_name, period="1mo"):
    """
    Display performance charts for stocks in a watchlist
    
    Args:
        watchlist_name: Name of the watchlist
        period: Time period for chart data
    """
    if watchlist_name not in st.session_state.watchlists:
        st.error("Watchlist not found.")
        return
    
    stocks = st.session_state.watchlists[watchlist_name]['stocks']
    
    if not stocks:
        st.info("No stocks in this watchlist.")
        return
    
    # Create a chart to compare performance
    fig = go.Figure()
    
    for ticker in stocks:
        try:
            # Get historical data
            df = get_stock_data(ticker, period=period)
            
            if df is not None and len(df) > 0:
                # Normalize prices to start at 100 for fair comparison
                normalized_prices = (df['Close'] / df['Close'].iloc[0]) * 100
                
                # Add line to chart
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=normalized_prices,
                    name=ticker,
                    mode='lines'
                ))
        except Exception as e:
            st.error(f"Error getting data for {ticker}: {e}")
    
    # Update layout
    fig.update_layout(
        title=f"{watchlist_name} Performance Comparison (Normalized to 100)",
        xaxis_title="Date",
        yaxis_title="Normalized Price",
        legend_title="Stocks",
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)


def export_watchlist(watchlist_name):
    """
    Export a watchlist as JSON
    
    Args:
        watchlist_name: Name of the watchlist
        
    Returns:
        JSON string representation of the watchlist
    """
    if watchlist_name not in st.session_state.watchlists:
        return None
    
    watchlist_data = st.session_state.watchlists[watchlist_name].copy()
    
    # Convert to serializable format
    watchlist_export = {
        'name': watchlist_name,
        'data': watchlist_data
    }
    
    return json.dumps(watchlist_export, indent=2)


def import_watchlist(watchlist_json):
    """
    Import a watchlist from JSON
    
    Args:
        watchlist_json: JSON string of the watchlist
        
    Returns:
        Boolean indicating success/failure
    """
    try:
        # Parse JSON
        watchlist_data = json.loads(watchlist_json)
        
        # Validate format
        if 'name' not in watchlist_data or 'data' not in watchlist_data:
            return False
        
        name = watchlist_data['name']
        data = watchlist_data['data']
        
        # Validate required fields
        required_fields = ['stocks', 'created_date', 'description', 'color', 'view_mode']
        if not all(field in data for field in required_fields):
            return False
        
        # Ensure name is unique or add suffix
        original_name = name
        counter = 1
        while name in st.session_state.watchlists:
            name = f"{original_name} ({counter})"
            counter += 1
        
        # Add to watchlists
        st.session_state.watchlists[name] = data
        
        return True
    except Exception:
        return False


def watchlist_page():
    """Main function for the watchlists page"""
    st.title("📋 Custom Watchlists")
    
    # Initialize watchlists if not already done
    initialize_watchlists()
    
    # Create 2 columns - sidebar for controls and main area for watchlist display
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Manage Watchlists")
        
        # Create new watchlist form
        with st.expander("Create New Watchlist", expanded=False):
            with st.form("new_watchlist_form"):
                new_name = st.text_input("Watchlist Name")
                new_description = st.text_area("Description")
                new_color = st.color_picker("Color", "#1f77b4")
                
                submit_button = st.form_submit_button("Create Watchlist")
                
                if submit_button and new_name:
                    if create_watchlist(new_name, new_description, new_color):
                        st.success(f"Created new watchlist: {new_name}")
                        st.rerun()
                    else:
                        st.error("Failed to create watchlist. Name may already exist.")
        
        # Display existing watchlists
        st.subheader("Your Watchlists")
        
        # Get watchlist summary
        watchlist_summary = get_watchlists_summary()
        
        # Create watchlist selector
        watchlist_options = list(st.session_state.watchlists.keys())
        selected_watchlist = st.selectbox(
            "Select Watchlist",
            options=watchlist_options,
            index=watchlist_options.index(st.session_state.current_watchlist)
        )
        
        # Set current watchlist based on selection
        if selected_watchlist != st.session_state.current_watchlist:
            st.session_state.current_watchlist = selected_watchlist
        
        # Display watchlist info
        if selected_watchlist in st.session_state.watchlists:
            watchlist_data = st.session_state.watchlists[selected_watchlist]
            
            st.info(f"**{len(watchlist_data['stocks'])}** stocks in this watchlist")
            st.caption(f"Created: {watchlist_data['description']}")
            
            # Watchlist settings
            with st.expander("Watchlist Settings", expanded=False):
                # Edit name
                new_watchlist_name = st.text_input("Rename Watchlist", value=selected_watchlist)
                if st.button("Update Name") and new_watchlist_name != selected_watchlist:
                    if rename_watchlist(selected_watchlist, new_watchlist_name):
                        st.success(f"Renamed to {new_watchlist_name}")
                        st.rerun()
                    else:
                        st.error("Failed to rename. Name may already exist.")
                
                # Edit description
                new_description = st.text_area("Update Description", value=watchlist_data['description'])
                if st.button("Update Description") and new_description != watchlist_data['description']:
                    update_watchlist_settings(selected_watchlist, description=new_description)
                    st.success("Description updated")
                    st.rerun()
                
                # Edit color
                new_color = st.color_picker("Update Color", value=watchlist_data['color'])
                if st.button("Update Color") and new_color != watchlist_data['color']:
                    update_watchlist_settings(selected_watchlist, color=new_color)
                    st.success("Color updated")
                    st.rerun()
                
                # View mode
                view_mode_options = ['compact', 'detailed']
                new_view_mode = st.selectbox(
                    "Display Density",
                    options=view_mode_options,
                    index=view_mode_options.index(watchlist_data['view_mode'])
                )
                if st.button("Update Display") and new_view_mode != watchlist_data['view_mode']:
                    update_watchlist_settings(selected_watchlist, view_mode=new_view_mode)
                    st.success("Display settings updated")
                    st.rerun()
                
                # Delete watchlist
                if st.button("🗑️ Delete Watchlist", use_container_width=True, type="primary"):
                    if delete_watchlist(selected_watchlist):
                        st.success(f"Deleted watchlist: {selected_watchlist}")
                        st.rerun()
                    else:
                        st.error("Cannot delete the only watchlist.")
            
            # Add stocks to watchlist
            with st.expander("Add Stocks", expanded=False):
                new_ticker = st.text_input("Enter Ticker Symbol").strip().upper()
                
                add_button = st.button("Add to Watchlist")
                
                if add_button and new_ticker:
                    if add_to_watchlist(selected_watchlist, new_ticker):
                        st.success(f"Added {new_ticker} to watchlist")
                        st.rerun()
                    else:
                        st.error(f"Failed to add {new_ticker}. Check if it's valid or already in the watchlist.")
            
            # Import/Export
            with st.expander("Import/Export", expanded=False):
                # Export
                if st.button("Export Watchlist") and selected_watchlist:
                    watchlist_json = export_watchlist(selected_watchlist)
                    if watchlist_json:
                        st.download_button(
                            label="Download JSON",
                            data=watchlist_json,
                            file_name=f"{selected_watchlist}_watchlist.json",
                            mime="application/json"
                        )
                
                # Import
                st.write("Import Watchlist")
                uploaded_file = st.file_uploader("Upload JSON file", type="json")
                
                if uploaded_file and st.button("Import"):
                    watchlist_json = uploaded_file.getvalue().decode("utf-8")
                    if import_watchlist(watchlist_json):
                        st.success("Watchlist imported successfully")
                        st.rerun()
                    else:
                        st.error("Failed to import watchlist. Invalid format.")
        
    with col2:
        st.subheader(f"{selected_watchlist} Watchlist")
        
        if 'description' in st.session_state.watchlists[selected_watchlist] and st.session_state.watchlists[selected_watchlist]['description']:
            st.caption(st.session_state.watchlists[selected_watchlist]['description'])
        
        # Display options
        display_options = ["Cards", "Table", "Performance Chart"]
        display_mode = st.radio("Display Mode", display_options, horizontal=True)
        
        # Time period selection for performance calculation
        period_options = {
            "1 Week": "1wk", 
            "1 Month": "1mo", 
            "3 Months": "3mo", 
            "6 Months": "6mo", 
            "1 Year": "1y",
            "Year to Date": "ytd"
        }
        selected_period = st.selectbox("Performance Period", list(period_options.keys()), index=1)
        period = period_options[selected_period]
        
        # Get watchlist performance data
        with st.spinner("Loading watchlist data..."):
            performance_data = get_watchlist_performance(selected_watchlist, period=period)
        
        # Display according to selected mode
        if display_mode == "Cards":
            display_watchlist_cards(performance_data)
        elif display_mode == "Table":
            display_watchlist_table(performance_data)
        else:  # Performance Chart
            display_watchlist_charts(selected_watchlist, period=period)
