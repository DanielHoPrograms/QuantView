import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

def get_company_info(ticker):
    """
    Get basic company information for a given ticker
    
    Args:
        ticker: Stock symbol
        
    Returns:
        Dictionary with company information
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Extract relevant company information
        company_info = {
            'name': info.get('longName', 'N/A'),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'website': info.get('website', 'N/A'),
            'country': info.get('country', 'N/A'),
            'employees': info.get('fullTimeEmployees', 'N/A'),
            'description': info.get('longBusinessSummary', 'N/A')
        }
        
        return company_info
    except Exception as e:
        st.error(f"Error fetching company information: {str(e)}")
        return None

def get_key_metrics(ticker):
    """
    Get key financial metrics for a given ticker
    
    Args:
        ticker: Stock symbol
        
    Returns:
        Dictionary with key metrics
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Extract relevant financial metrics
        metrics = {
            'Market Cap': info.get('marketCap', 'N/A'),
            'P/E Ratio': info.get('trailingPE', 'N/A'),
            'Forward P/E': info.get('forwardPE', 'N/A'),
            'P/S Ratio': info.get('priceToSalesTrailing12Months', 'N/A'),
            'P/B Ratio': info.get('priceToBook', 'N/A'),
            'EPS (TTM)': info.get('trailingEps', 'N/A'),
            'EPS Growth (YOY)': info.get('earningsGrowth', 'N/A'),
            'Revenue Growth (YOY)': info.get('revenueGrowth', 'N/A'),
            'Profit Margin': info.get('profitMargins', 'N/A'),
            'Operating Margin': info.get('operatingMargins', 'N/A'),
            'ROE': info.get('returnOnEquity', 'N/A'),
            'ROA': info.get('returnOnAssets', 'N/A'),
            'Beta': info.get('beta', 'N/A'),
            'Dividend Yield': info.get('dividendYield', 'N/A'),
            'Payout Ratio': info.get('payoutRatio', 'N/A'),
            'Debt to Equity': info.get('debtToEquity', 'N/A'),
            'Current Ratio': info.get('currentRatio', 'N/A'),
            'Quick Ratio': info.get('quickRatio', 'N/A'),
            '52-Week High': info.get('fiftyTwoWeekHigh', 'N/A'),
            '52-Week Low': info.get('fiftyTwoWeekLow', 'N/A'),
            'Average Volume': info.get('averageVolume', 'N/A')
        }
        
        # Format metrics
        for key, value in metrics.items():
            if value != 'N/A':
                if 'Ratio' in key or 'Margin' in key or 'ROE' in key or 'ROA' in key or 'Growth' in key or 'Yield' in key:
                    # Convert decimal to percentage
                    if isinstance(value, (int, float)) and abs(value) < 1:
                        metrics[key] = f"{value * 100:.2f}%"
                    else:
                        metrics[key] = f"{value:.2f}" if isinstance(value, (int, float)) else value
                elif 'Market Cap' in key or 'Volume' in key:
                    # Format large numbers
                    if isinstance(value, (int, float)):
                        if value >= 1e9:
                            metrics[key] = f"${value / 1e9:.2f}B"
                        elif value >= 1e6:
                            metrics[key] = f"${value / 1e6:.2f}M"
                        else:
                            metrics[key] = f"${value:,.0f}"
                else:
                    metrics[key] = f"{value:.2f}" if isinstance(value, (int, float)) else value
        
        return metrics
    except Exception as e:
        st.error(f"Error fetching key metrics: {str(e)}")
        return None

def get_financials(ticker):
    """
    Get financial statements for a given ticker
    
    Args:
        ticker: Stock symbol
        
    Returns:
        Dictionary with income statement, balance sheet, and cash flow data
    """
    try:
        stock = yf.Ticker(ticker)
        
        # Get financial statements
        income_statement = stock.income_stmt
        balance_sheet = stock.balance_sheet
        cash_flow = stock.cashflow
        
        return {
            'income_statement': income_statement,
            'balance_sheet': balance_sheet,
            'cash_flow': cash_flow
        }
    except Exception as e:
        st.error(f"Error fetching financial statements: {str(e)}")
        return None

def parse_screening_filters(filter_string):
    """
    Parse a user-entered filter string into a structured format
    
    Args:
        filter_string: String with filter criteria (e.g., "P/E < 15 AND MarketCap > 1B")
        
    Returns:
        List of dictionaries with filter criteria
    """
    try:
        if not filter_string:
            return []
            
        # Split filters by "AND" (case-insensitive)
        filters_raw = filter_string.split(' AND ')
        
        filters = []
        for filter_raw in filters_raw:
            # Identify operator
            if '<=' in filter_raw:
                metric, value = filter_raw.split('<=')
                operator = '<='
            elif '>=' in filter_raw:
                metric, value = filter_raw.split('>=')
                operator = '>='
            elif '<' in filter_raw:
                metric, value = filter_raw.split('<')
                operator = '<'
            elif '>' in filter_raw:
                metric, value = filter_raw.split('>')
                operator = '>'
            elif '=' in filter_raw:
                metric, value = filter_raw.split('=')
                operator = '='
            else:
                # Skip malformed filters
                continue
                
            # Clean up metric and value
            metric = metric.strip()
            value = value.strip()
            
            # Convert value to appropriate type
            try:
                # Handle billion and million suffixes
                if value.upper().endswith('B'):
                    value = float(value[:-1]) * 1e9
                elif value.upper().endswith('M'):
                    value = float(value[:-1]) * 1e6
                elif value.upper().endswith('K'):
                    value = float(value[:-1]) * 1e3
                elif value.endswith('%'):
                    value = float(value[:-1]) / 100
                else:
                    value = float(value)
            except ValueError:
                # If conversion fails, keep as string
                pass
                
            filters.append({
                'metric': metric,
                'operator': operator,
                'value': value
            })
            
        return filters
    except Exception as e:
        st.error(f"Error parsing filter string: {str(e)}")
        return []

def apply_screening_filters(tickers, filters):
    """
    Apply screening filters to a list of tickers
    
    Args:
        tickers: List of stock symbols
        filters: List of filter dictionaries
        
    Returns:
        Filtered DataFrame with stock data
    """
    try:
        # Mapping of user-friendly metric names to yfinance keys
        metric_mapping = {
            'P/E': 'trailingPE',
            'ForwardP/E': 'forwardPE',
            'P/S': 'priceToSalesTrailing12Months',
            'P/B': 'priceToBook',
            'EPS': 'trailingEps',
            'EPSGrowth': 'earningsGrowth',
            'RevenueGrowth': 'revenueGrowth',
            'ProfitMargin': 'profitMargins',
            'OperatingMargin': 'operatingMargins',
            'ROE': 'returnOnEquity',
            'ROA': 'returnOnAssets',
            'Beta': 'beta',
            'DividendYield': 'dividendYield',
            'PayoutRatio': 'payoutRatio',
            'DebtToEquity': 'debtToEquity',
            'CurrentRatio': 'currentRatio',
            'QuickRatio': 'quickRatio',
            'MarketCap': 'marketCap',
            'PriceToBookRatio': 'priceToBook',
            'PriceToSalesRatio': 'priceToSalesTrailing12Months',
            'TotalDebtToEquity': 'debtToEquity',
            '52WeekHigh': 'fiftyTwoWeekHigh',
            '52WeekLow': 'fiftyTwoWeekLow'
        }
        
        # Clean up metric names by removing spaces and making case-insensitive
        clean_mapping = {k.lower().replace(' ', ''): v for k, v in metric_mapping.items()}
        
        if not filters:
            return pd.DataFrame({'Ticker': tickers})
            
        # Collect data for all tickers
        data = []
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                
                # Skip tickers that don't have info data
                if not info:
                    continue
                    
                # Add ticker to info
                info['Ticker'] = ticker
                data.append(info)
            except Exception as e:
                st.warning(f"Error fetching data for {ticker}: {str(e)}")
                continue
                
        if not data:
            return pd.DataFrame({'Ticker': []})
            
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Apply filters
        for filter_item in filters:
            metric = filter_item['metric'].lower().replace(' ', '')
            operator = filter_item['operator']
            value = filter_item['value']
            
            # Look up yfinance key for the metric
            yf_key = clean_mapping.get(metric, metric)
            
            # Skip if key is not in DataFrame
            if yf_key not in df.columns:
                continue
                
            # Apply filter based on operator
            if operator == '<':
                df = df[df[yf_key] < value]
            elif operator == '<=':
                df = df[df[yf_key] <= value]
            elif operator == '>':
                df = df[df[yf_key] > value]
            elif operator == '>=':
                df = df[df[yf_key] >= value]
            elif operator == '=':
                df = df[df[yf_key] == value]
                
        # Select subset of columns for display
        display_columns = ['Ticker', 'longName', 'sector', 'industry', 'country']
        for filter_item in filters:
            metric = filter_item['metric'].lower().replace(' ', '')
            yf_key = clean_mapping.get(metric, metric)
            if yf_key in df.columns and yf_key not in display_columns:
                display_columns.append(yf_key)
                
        # Add important financial metrics if available
        for col in ['marketCap', 'trailingPE', 'forwardPE', 'dividendYield', 'beta']:
            if col in df.columns and col not in display_columns:
                display_columns.append(col)
                
        # Return relevant columns only
        result_df = df[[col for col in display_columns if col in df.columns]]
        
        return result_df
    except Exception as e:
        st.error(f"Error applying filters: {str(e)}")
        return pd.DataFrame({'Ticker': []})

def get_peer_comparison(ticker, peers=None):
    """
    Compare key metrics with industry peers
    
    Args:
        ticker: Main stock symbol
        peers: List of peer stock symbols (optional)
        
    Returns:
        Dictionary with comparison data
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Get sector and industry
        sector = info.get('sector', '')
        industry = info.get('industry', '')
        
        # If peers not provided, try to find them based on industry
        if not peers:
            # This is a simple approach to find stocks in the same industry
            # In a real application, you might want to use a more sophisticated approach
            st.info(f"Finding peers in the {industry} industry...")
            
            # For demonstration purposes, use a fixed set of peers by industry
            industry_peers = {
                'Technology': ['AAPL', 'MSFT', 'GOOGL', 'META', 'AMZN'],
                'Semiconductors': ['NVDA', 'AMD', 'INTC', 'TSM', 'QCOM'],
                'Financial Services': ['JPM', 'BAC', 'WFC', 'C', 'GS'],
                'Healthcare': ['JNJ', 'PFE', 'MRK', 'UNH', 'ABBV'],
                'Consumer Cyclical': ['AMZN', 'HD', 'NKE', 'SBUX', 'MCD'],
                'Communication Services': ['GOOGL', 'META', 'DIS', 'NFLX', 'VZ'],
                'Industrial': ['GE', 'HON', 'MMM', 'CAT', 'UPS'],
                'Consumer Defensive': ['PG', 'KO', 'PEP', 'WMT', 'COST'],
                'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG'],
                'Basic Materials': ['LIN', 'RIO', 'BHP', 'FCX', 'DOW'],
                'Real Estate': ['SPG', 'AMT', 'EQIX', 'PSA', 'WELL'],
                'Utilities': ['NEE', 'DUK', 'SO', 'D', 'AEP']
            }
            
            # Find the closest industry match
            for key, value in industry_peers.items():
                if key.lower() in industry.lower() or key.lower() in sector.lower():
                    peers = value
                    break
                    
            # If no match, use a default set
            if not peers:
                peers = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META']
                
        # Make sure the main ticker is in the peer list
        if ticker not in peers:
            peers.insert(0, ticker)
        else:
            # Move the main ticker to the front
            peers.remove(ticker)
            peers.insert(0, ticker)
            
        # Collect data for all peers
        peer_data = []
        for peer in peers:
            try:
                peer_stock = yf.Ticker(peer)
                peer_info = peer_stock.info
                
                if peer_info:
                    peer_data.append({
                        'Ticker': peer,
                        'Name': peer_info.get('shortName', peer),
                        'Market Cap': peer_info.get('marketCap', None),
                        'P/E Ratio': peer_info.get('trailingPE', None),
                        'Forward P/E': peer_info.get('forwardPE', None),
                        'P/S Ratio': peer_info.get('priceToSalesTrailing12Months', None),
                        'P/B Ratio': peer_info.get('priceToBook', None),
                        'EPS (TTM)': peer_info.get('trailingEps', None),
                        'Profit Margin': peer_info.get('profitMargins', None),
                        'Operating Margin': peer_info.get('operatingMargins', None),
                        'ROE': peer_info.get('returnOnEquity', None),
                        'ROA': peer_info.get('returnOnAssets', None),
                        'Beta': peer_info.get('beta', None),
                        'Dividend Yield': peer_info.get('dividendYield', None),
                        'Debt to Equity': peer_info.get('debtToEquity', None),
                        'Current Ratio': peer_info.get('currentRatio', None)
                    })
            except Exception as e:
                st.warning(f"Error fetching data for peer {peer}: {str(e)}")
                
        # Convert to DataFrame
        comparison_df = pd.DataFrame(peer_data)
        
        # Format percentages
        for col in ['Profit Margin', 'Operating Margin', 'ROE', 'ROA', 'Dividend Yield']:
            if col in comparison_df.columns:
                comparison_df[col] = comparison_df[col].apply(
                    lambda x: f"{x*100:.2f}%" if isinstance(x, (int, float)) else x
                )
                
        # Format large numbers
        if 'Market Cap' in comparison_df.columns:
            comparison_df['Market Cap'] = comparison_df['Market Cap'].apply(
                lambda x: f"${x/1e9:.2f}B" if isinstance(x, (int, float)) and x >= 1e9 else
                (f"${x/1e6:.2f}M" if isinstance(x, (int, float)) and x >= 1e6 else x)
            )
            
        return comparison_df
    except Exception as e:
        st.error(f"Error creating peer comparison: {str(e)}")
        return None

def plot_metric_comparison(comparison_df, metric):
    """
    Plot a comparison of a specific metric across peers
    
    Args:
        comparison_df: DataFrame with comparison data
        metric: Metric to compare
        
    Returns:
        Plotly figure
    """
    try:
        if metric not in comparison_df.columns:
            st.warning(f"Metric {metric} not available for comparison")
            return None
            
        # Convert percentage strings to floats for plotting
        plot_df = comparison_df.copy()
        if metric in ['Profit Margin', 'Operating Margin', 'ROE', 'ROA', 'Dividend Yield']:
            plot_df[metric] = plot_df[metric].apply(
                lambda x: float(x.strip('%')) / 100 if isinstance(x, str) and '%' in x else x
            )
            
        # Convert formatted market cap to numbers for plotting
        if metric == 'Market Cap':
            plot_df[metric] = plot_df[metric].apply(
                lambda x: float(x.strip('$B')) * 1e9 if isinstance(x, str) and 'B' in x else
                (float(x.strip('$M')) * 1e6 if isinstance(x, str) and 'M' in x else x)
            )
            
        # Filter out non-numeric values
        plot_df = plot_df[pd.to_numeric(plot_df[metric], errors='coerce').notna()]
        
        if len(plot_df) == 0:
            st.warning(f"No valid numeric data available for {metric}")
            return None
            
        # Create bar chart for comparison
        fig = px.bar(
            plot_df,
            x='Ticker',
            y=metric,
            title=f"{metric} Comparison",
            color='Ticker',
            text_auto='.2s',
            height=400
        )
        
        # Highlight the main ticker
        fig.update_traces(
            marker_color=['blue' if i == 0 else 'lightblue' for i in range(len(plot_df))],
            marker_line_color='rgb(8,48,107)',
            marker_line_width=1.5,
            opacity=0.8
        )
        
        # Add average line
        avg_value = plot_df[metric].mean()
        fig.add_shape(
            type='line',
            x0=-0.5,
            x1=len(plot_df) - 0.5,
            y0=avg_value,
            y1=avg_value,
            line=dict(color='red', width=2, dash='dash')
        )
        
        # Add annotation for average
        fig.add_annotation(
            x=len(plot_df) - 1,
            y=avg_value,
            text=f"Avg: {avg_value:.2f}",
            showarrow=False,
            yshift=10,
            font=dict(color='red')
        )
        
        # Format y-axis for percentages
        if metric in ['Profit Margin', 'Operating Margin', 'ROE', 'ROA', 'Dividend Yield']:
            fig.update_layout(yaxis_tickformat='.1%')
            
        # Format y-axis for market cap
        if metric == 'Market Cap':
            fig.update_layout(yaxis_tickformat='$~s')
            
        fig.update_layout(
            xaxis_title=None,
            yaxis_title=metric,
            plot_bgcolor='white',
            xaxis={'categoryorder': 'array', 'categoryarray': plot_df['Ticker'].tolist()}
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating comparison plot: {str(e)}")
        return None

def format_financial_statement(df):
    """
    Format a financial statement DataFrame for display
    
    Args:
        df: Financial statement DataFrame
        
    Returns:
        Formatted DataFrame
    """
    try:
        if df is None or df.empty:
            return pd.DataFrame()
            
        # Transpose the DataFrame to have dates as columns
        formatted_df = df.transpose()
        
        # Format large numbers for better readability
        formatted_df = formatted_df.applymap(lambda x: f"${x/1e9:.2f}B" if isinstance(x, (int, float)) and abs(x) >= 1e9 else
                                         (f"${x/1e6:.2f}M" if isinstance(x, (int, float)) and abs(x) >= 1e6 else
                                          (f"${x/1e3:.2f}K" if isinstance(x, (int, float)) and abs(x) >= 1e3 else
                                           (f"${x:.2f}" if isinstance(x, (int, float)) else x))))
        
        return formatted_df
    except Exception as e:
        st.error(f"Error formatting financial statement: {str(e)}")
        return pd.DataFrame()

def plot_financial_trend(ticker, metric_name, metric_key):
    """
    Plot historical trends for a financial metric
    
    Args:
        ticker: Stock symbol
        metric_name: Display name of the metric
        metric_key: Key of the metric in financial statements
        
    Returns:
        Plotly figure
    """
    try:
        stock = yf.Ticker(ticker)
        
        # Get annual financial data
        income_stmt = stock.income_stmt
        balance_sheet = stock.balance_sheet
        cashflow = stock.cashflow
        
        # Determine which financial statement contains the metric
        if metric_key in income_stmt.index:
            data = income_stmt.loc[metric_key]
        elif metric_key in balance_sheet.index:
            data = balance_sheet.loc[metric_key]
        elif metric_key in cashflow.index:
            data = cashflow.loc[metric_key]
        else:
            st.warning(f"Metric {metric_name} not found in financial statements")
            return None
            
        # Create a DataFrame for the metric
        df = pd.DataFrame({
            'Date': data.index,
            'Value': data.values
        })
        
        # Create line chart
        fig = px.line(
            df,
            x='Date',
            y='Value',
            title=f"{metric_name} Trend for {ticker}",
            markers=True,
            height=400
        )
        
        # Format y-axis for large numbers
        fig.update_layout(
            yaxis_tickformat='$~s',
            plot_bgcolor='white',
            xaxis_title=None,
            yaxis_title=metric_name
        )
        
        # Add data points values as text
        fig.update_traces(
            textposition='top center',
            texttemplate='$%{y:.1f}B' if df['Value'].max() >= 1e9 else ('$%{y:.1f}M' if df['Value'].max() >= 1e6 else '$%{y:.1f}K')
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating financial trend plot: {str(e)}")
        return None

def fundamental_analysis_section():
    """Main function for the fundamental analysis section"""
    
    st.header("📊 Fundamental Analysis")
    st.markdown("""
    Analyze company financials, key metrics, and compare with industry peers to make informed investment decisions.
    """)
    
    # Tab layout for the fundamental analysis section
    tabs = st.tabs(["Key Metrics", "Financial Statements", "Peer Comparison", "Fundamental Screener"])
    
    # Initialize state variables
    if 'fundamental_ticker' not in st.session_state:
        # Use the first selected stock or default to AAPL
        st.session_state.fundamental_ticker = st.session_state.selected_stocks[0] if st.session_state.selected_stocks else 'AAPL'
    
    with tabs[0]:  # Key Metrics Tab
        st.subheader("Key Financial Metrics")
        
        # Select ticker
        selected_ticker = st.selectbox(
            "Select a stock for fundamental analysis",
            options=st.session_state.selected_stocks if st.session_state.selected_stocks else ['AAPL'],
            index=st.session_state.selected_stocks.index(st.session_state.fundamental_ticker) if st.session_state.fundamental_ticker in st.session_state.selected_stocks else 0,
            key="key_metrics_ticker"
        )
        
        st.session_state.fundamental_ticker = selected_ticker
        
        # Show company info
        company_info = get_company_info(selected_ticker)
        if company_info:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown(f"### {company_info['name']} ({selected_ticker})")
                st.markdown(f"**Sector:** {company_info['sector']}")
                st.markdown(f"**Industry:** {company_info['industry']}")
                st.markdown(f"**Country:** {company_info['country']}")
                
                if company_info['website'] != 'N/A':
                    st.markdown(f"**Website:** [{company_info['website']}]({company_info['website']})")
                    
                if company_info['employees'] != 'N/A':
                    st.markdown(f"**Employees:** {company_info['employees']:,}")
            
            with col2:
                # Get current stock price
                stock = yf.Ticker(selected_ticker)
                hist = stock.history(period="1d")
                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
                    previous_close = stock.info.get('previousClose', current_price)
                    daily_change = ((current_price / previous_close) - 1) * 100
                    
                    price_col, change_col = st.columns([1, 1])
                    price_col.metric("Current Price", f"${current_price:.2f}")
                    change_col.metric("Daily Change", f"{daily_change:.2f}%", f"{daily_change:.2f}%")
                    
                    # 52-week range
                    week_52_high = stock.info.get('fiftyTwoWeekHigh', 'N/A')
                    week_52_low = stock.info.get('fiftyTwoWeekLow', 'N/A')
                    if week_52_high != 'N/A' and week_52_low != 'N/A':
                        st.markdown(f"**52-Week Range:** ${week_52_low:.2f} - ${week_52_high:.2f}")
                        
                        # Create 52-week range visual indicator
                        if current_price >= week_52_low and current_price <= week_52_high:
                            pct_in_range = (current_price - week_52_low) / (week_52_high - week_52_low) * 100
                            st.progress(int(pct_in_range))
                            st.caption(f"Current price is {pct_in_range:.1f}% of 52-week range")
                
            # Company description in expandable section
            with st.expander("Company Description"):
                st.write(company_info['description'])
                
            # Get key metrics
            metrics = get_key_metrics(selected_ticker)
            if metrics:
                st.subheader("Key Metrics")
                
                # Use columns layout for better space utilization
                metric_columns = st.columns(3)
                
                # Group metrics into categories
                valuation_metrics = ['P/E Ratio', 'Forward P/E', 'P/S Ratio', 'P/B Ratio', 'EPS (TTM)', 'Market Cap']
                profitability_metrics = ['EPS Growth (YOY)', 'Revenue Growth (YOY)', 'Profit Margin', 'Operating Margin', 'ROE', 'ROA']
                financial_health_metrics = ['Debt to Equity', 'Current Ratio', 'Quick Ratio', 'Dividend Yield', 'Payout Ratio', 'Beta']
                
                # Valuation metrics
                with metric_columns[0]:
                    st.markdown("#### Valuation")
                    for metric in valuation_metrics:
                        if metric in metrics:
                            st.metric(metric, metrics[metric])
                
                # Profitability metrics
                with metric_columns[1]:
                    st.markdown("#### Profitability")
                    for metric in profitability_metrics:
                        if metric in metrics:
                            st.metric(metric, metrics[metric])
                
                # Financial health metrics
                with metric_columns[2]:
                    st.markdown("#### Financial Health")
                    for metric in financial_health_metrics:
                        if metric in metrics:
                            st.metric(metric, metrics[metric])
                
    with tabs[1]:  # Financial Statements Tab
        st.subheader("Financial Statements")
        
        # Select ticker
        selected_ticker = st.selectbox(
            "Select a stock for financial statements",
            options=st.session_state.selected_stocks if st.session_state.selected_stocks else ['AAPL'],
            index=st.session_state.selected_stocks.index(st.session_state.fundamental_ticker) if st.session_state.fundamental_ticker in st.session_state.selected_stocks else 0,
            key="financial_statements_ticker"
        )
        
        st.session_state.fundamental_ticker = selected_ticker
        
        # Get financial statements
        financials = get_financials(selected_ticker)
        if financials:
            # Create tabs for different financial statements
            statement_tabs = st.tabs(["Income Statement", "Balance Sheet", "Cash Flow"])
            
            with statement_tabs[0]:  # Income Statement
                st.markdown("#### Income Statement")
                income_df = format_financial_statement(financials['income_statement'])
                if not income_df.empty:
                    st.dataframe(income_df, use_container_width=True)
                    
                    # Allow user to plot key income statement metrics
                    metrics = ['Total Revenue', 'Gross Profit', 'Operating Income', 'Net Income']
                    metric_keys = {
                        'Total Revenue': 'Total Revenue',
                        'Gross Profit': 'Gross Profit',
                        'Operating Income': 'Operating Income',
                        'Net Income': 'Net Income'
                    }
                    
                    selected_metric = st.selectbox(
                        "Plot Income Statement Metric",
                        options=metrics,
                        key="income_metric"
                    )
                    
                    fig = plot_financial_trend(selected_ticker, selected_metric, metric_keys[selected_metric])
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Income statement data not available")
            
            with statement_tabs[1]:  # Balance Sheet
                st.markdown("#### Balance Sheet")
                balance_df = format_financial_statement(financials['balance_sheet'])
                if not balance_df.empty:
                    st.dataframe(balance_df, use_container_width=True)
                    
                    # Allow user to plot key balance sheet metrics
                    metrics = ['Total Assets', 'Total Liabilities', 'Total Stockholder Equity', 'Cash and Cash Equivalents']
                    metric_keys = {
                        'Total Assets': 'Total Assets',
                        'Total Liabilities': 'Total Liabilities Net Minority Interest',
                        'Total Stockholder Equity': 'Stockholders Equity',
                        'Cash and Cash Equivalents': 'Cash And Cash Equivalents'
                    }
                    
                    selected_metric = st.selectbox(
                        "Plot Balance Sheet Metric",
                        options=metrics,
                        key="balance_metric"
                    )
                    
                    fig = plot_financial_trend(selected_ticker, selected_metric, metric_keys[selected_metric])
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Balance sheet data not available")
            
            with statement_tabs[2]:  # Cash Flow
                st.markdown("#### Cash Flow Statement")
                cashflow_df = format_financial_statement(financials['cash_flow'])
                if not cashflow_df.empty:
                    st.dataframe(cashflow_df, use_container_width=True)
                    
                    # Allow user to plot key cash flow metrics
                    metrics = ['Operating Cash Flow', 'Capital Expenditure', 'Free Cash Flow', 'Dividends Paid']
                    metric_keys = {
                        'Operating Cash Flow': 'Operating Cash Flow',
                        'Capital Expenditure': 'Capital Expenditure',
                        'Free Cash Flow': 'Free Cash Flow',
                        'Dividends Paid': 'Dividends Paid'
                    }
                    
                    selected_metric = st.selectbox(
                        "Plot Cash Flow Metric",
                        options=metrics,
                        key="cashflow_metric"
                    )
                    
                    fig = plot_financial_trend(selected_ticker, selected_metric, metric_keys[selected_metric])
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Cash flow data not available")
                
    with tabs[2]:  # Peer Comparison Tab
        st.subheader("Peer Comparison")
        
        # Select ticker
        selected_ticker = st.selectbox(
            "Select a stock for peer comparison",
            options=st.session_state.selected_stocks if st.session_state.selected_stocks else ['AAPL'],
            index=st.session_state.selected_stocks.index(st.session_state.fundamental_ticker) if st.session_state.fundamental_ticker in st.session_state.selected_stocks else 0,
            key="peer_comparison_ticker"
        )
        
        st.session_state.fundamental_ticker = selected_ticker
        
        # Option to manually specify peers
        use_custom_peers = st.checkbox("Specify custom peers", value=False)
        peers = None
        
        if use_custom_peers:
            peers_input = st.text_input(
                "Enter peer stock symbols (comma-separated)",
                placeholder="e.g., AAPL, MSFT, GOOGL"
            )
            
            if peers_input:
                peers = [ticker.strip().upper() for ticker in peers_input.split(',')]
        
        # Get peer comparison data
        with st.spinner("Comparing with industry peers..."):
            comparison_df = get_peer_comparison(selected_ticker, peers)
            
        if comparison_df is not None and not comparison_df.empty:
            # Show the comparison table
            st.dataframe(comparison_df, use_container_width=True)
            
            # Allow the user to select a metric for visual comparison
            available_metrics = [col for col in comparison_df.columns if col not in ['Ticker', 'Name']]
            
            selected_metric = st.selectbox(
                "Select metric for comparison",
                options=available_metrics,
                index=0 if available_metrics else 0,
                key="peer_metric"
            )
            
            # Create comparison plot
            if selected_metric:
                fig = plot_metric_comparison(comparison_df, selected_metric)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Provide analysis of the comparison
                    st.subheader("Comparison Analysis")
                    
                    # Try to convert values to numeric for comparison
                    numeric_values = pd.to_numeric(comparison_df[selected_metric].apply(
                        lambda x: float(x.strip('%')) / 100 if isinstance(x, str) and '%' in x else
                        (float(x.strip('$B')) * 1e9 if isinstance(x, str) and 'B' in x else
                         (float(x.strip('$M')) * 1e6 if isinstance(x, str) and 'M' in x else x))
                    ), errors='coerce')
                    
                    # Calculate statistics if we have numeric values
                    if not numeric_values.isna().all():
                        avg_value = numeric_values.mean()
                        median_value = numeric_values.median()
                        main_value = numeric_values.iloc[0]  # First value is the main ticker
                        
                        # Percentile of the main stock among peers
                        percentile = (numeric_values <= main_value).mean() * 100
                        
                        # Format values for display
                        if selected_metric in ['Profit Margin', 'Operating Margin', 'ROE', 'ROA', 'Dividend Yield']:
                            format_str = "{:.2%}"
                        elif selected_metric == 'Market Cap':
                            if avg_value >= 1e9:
                                avg_value /= 1e9
                                median_value /= 1e9
                                main_value /= 1e9
                                format_str = "${:.2f}B"
                            else:
                                avg_value /= 1e6
                                median_value /= 1e6
                                main_value /= 1e6
                                format_str = "${:.2f}M"
                        else:
                            format_str = "{:.2f}"
                            
                        # Create analysis text
                        analysis_text = f"""
                        **Analysis of {selected_metric}**:
                        
                        * {selected_ticker} value: {format_str.format(main_value)}
                        * Industry average: {format_str.format(avg_value)}
                        * Industry median: {format_str.format(median_value)}
                        * {selected_ticker} is at the {percentile:.0f}th percentile among peers
                        """
                        
                        # For valuation metrics, lower is generally better
                        valuation_metrics = ['P/E Ratio', 'Forward P/E', 'P/S Ratio', 'P/B Ratio']
                        
                        # For profitability metrics, higher is generally better
                        profitability_metrics = ['EPS (TTM)', 'Profit Margin', 'Operating Margin', 'ROE', 'ROA']
                        
                        if selected_metric in valuation_metrics:
                            if main_value < avg_value:
                                analysis_text += f"\n* {selected_ticker} is trading at a **discount** compared to industry average"
                            else:
                                analysis_text += f"\n* {selected_ticker} is trading at a **premium** compared to industry average"
                        elif selected_metric in profitability_metrics:
                            if main_value > avg_value:
                                analysis_text += f"\n* {selected_ticker} has **above average** {selected_metric} compared to peers"
                            else:
                                analysis_text += f"\n* {selected_ticker} has **below average** {selected_metric} compared to peers"
                        
                        st.markdown(analysis_text)
                        
                    else:
                        st.info(f"Unable to perform numeric comparison for {selected_metric}")
                else:
                    st.info(f"No valid data for plotting {selected_metric}")
        else:
            st.warning("Unable to retrieve peer comparison data")
            
    with tabs[3]:  # Fundamental Screener
        st.subheader("Fundamental Stock Screener")
        st.markdown("""
        Filter stocks based on fundamental criteria to find investment opportunities.
        """)
        
        # Use S&P 500 constituents as the default screening universe
        sp500_tickers = [
            # Technology
            'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'META', 'NVDA', 'AVGO', 'CSCO', 'ADBE', 'ORCL', 'CRM', 'ACN', 'IBM', 'INTC', 'AMD',
            # Healthcare
            'JNJ', 'UNH', 'PFE', 'MRK', 'ABT', 'TMO', 'ABBV', 'DHR', 'LLY',
            # Consumer
            'AMZN', 'TSLA', 'HD', 'PG', 'COST', 'KO', 'PEP', 'WMT', 'MCD', 'DIS', 'NFLX', 'NKE',
            # Financial
            'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'V', 'MA', 'AXP', 'BLK',
            # Energy
            'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'PSX', 'OXY',
            # Industrials
            'GE', 'HON', 'MMM', 'CAT', 'DE', 'BA', 'LMT', 'RTX', 'UPS', 'FDX'
        ]
        
        # Allow user to choose between S&P 500, watchlist, or custom list
        universe_options = ["S&P 500 Constituents", "Your Watchlist", "Custom Tickers"]
        selected_universe = st.radio("Select Screening Universe", universe_options)
        
        screening_tickers = []
        if selected_universe == "S&P 500 Constituents":
            screening_tickers = sp500_tickers
            st.info(f"Screening from a subset of {len(screening_tickers)} S&P 500 stocks")
        elif selected_universe == "Your Watchlist":
            screening_tickers = st.session_state.selected_stocks
            st.info(f"Screening from your watchlist with {len(screening_tickers)} stocks")
        else:  # Custom Tickers
            custom_tickers_input = st.text_input(
                "Enter ticker symbols to screen (comma-separated)",
                placeholder="e.g., AAPL, MSFT, GOOGL"
            )
            if custom_tickers_input:
                screening_tickers = [ticker.strip().upper() for ticker in custom_tickers_input.split(',')]
                st.info(f"Screening {len(screening_tickers)} custom tickers")
            else:
                st.warning("Please enter ticker symbols to screen")
        
        # Create filter input
        st.subheader("Define Screening Criteria")
        
        # Show common filter examples
        filter_examples = {
            "Low P/E Stocks": "P/E < 15",
            "High Dividend Yield": "DividendYield > 0.03",
            "Large Cap Technology": "MarketCap > 100B AND Sector = Technology",
            "Value Stocks": "P/E < 15 AND P/B < 2 AND DividendYield > 0.02",
            "Growth Stocks": "RevenueGrowth > 0.15 AND EPSGrowth > 0.10",
            "Profitable Companies": "ProfitMargin > 0.15 AND ROE > 0.15"
        }
        
        selected_example = st.selectbox(
            "Filter Examples (select to pre-fill)",
            options=[""] + list(filter_examples.keys())
        )
        
        filter_text = filter_examples.get(selected_example, "") if selected_example else ""
        
        filter_input = st.text_area(
            "Enter filter criteria",
            value=filter_text,
            placeholder="e.g., P/E < 15 AND MarketCap > 1B",
            help="Use AND to combine multiple filters. Available metrics: P/E, ForwardP/E, P/S, P/B, EPS, EPSGrowth, RevenueGrowth, ProfitMargin, ROE, ROA, DividendYield, DebtToEquity, Beta, MarketCap"
        )
        
        # Parse filters and apply screening
        if st.button("Run Screening", use_container_width=True) and screening_tickers:
            filters = parse_screening_filters(filter_input)
            
            if filters:
                st.write(f"Applying {len(filters)} filters to {len(screening_tickers)} stocks")
                
                with st.spinner("Screening stocks..."):
                    results_df = apply_screening_filters(screening_tickers, filters)
                
                if not results_df.empty:
                    st.success(f"Found {len(results_df)} stocks matching your criteria")
                    
                    # Format the results for display
                    display_df = results_df.copy()
                    
                    # Format market cap
                    if 'marketCap' in display_df.columns:
                        display_df['Market Cap'] = display_df['marketCap'].apply(
                            lambda x: f"${x/1e9:.2f}B" if isinstance(x, (int, float)) and x >= 1e9 else
                            (f"${x/1e6:.2f}M" if isinstance(x, (int, float)) and x >= 1e6 else x)
                        )
                        display_df = display_df.drop('marketCap', axis=1)
                    
                    # Format P/E ratio
                    if 'trailingPE' in display_df.columns:
                        display_df['P/E Ratio'] = display_df['trailingPE'].apply(
                            lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x
                        )
                        display_df = display_df.drop('trailingPE', axis=1)
                        
                    # Format Forward P/E
                    if 'forwardPE' in display_df.columns:
                        display_df['Forward P/E'] = display_df['forwardPE'].apply(
                            lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x
                        )
                        display_df = display_df.drop('forwardPE', axis=1)
                        
                    # Format dividend yield
                    if 'dividendYield' in display_df.columns:
                        display_df['Dividend Yield'] = display_df['dividendYield'].apply(
                            lambda x: f"{x*100:.2f}%" if isinstance(x, (int, float)) else x
                        )
                        display_df = display_df.drop('dividendYield', axis=1)
                        
                    # Format beta
                    if 'beta' in display_df.columns:
                        display_df['Beta'] = display_df['beta'].apply(
                            lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x
                        )
                        display_df = display_df.drop('beta', axis=1)
                        
                    # Rename columns for better display
                    if 'longName' in display_df.columns:
                        display_df = display_df.rename(columns={'longName': 'Company Name'})
                        
                    st.dataframe(display_df, use_container_width=True)
                    
                    # Allow adding stocks from results to watchlist
                    if display_df['Ticker'].tolist():
                        add_to_watchlist = st.multiselect(
                            "Add stocks to your watchlist",
                            options=display_df['Ticker'].tolist()
                        )
                        
                        if add_to_watchlist and st.button("Add Selected to Watchlist", use_container_width=True):
                            added_count = 0
                            for ticker in add_to_watchlist:
                                if ticker not in st.session_state.selected_stocks:
                                    st.session_state.selected_stocks.append(ticker)
                                    added_count += 1
                            
                            if added_count > 0:
                                st.success(f"Added {added_count} stocks to your watchlist")
                                st.rerun()
                else:
                    st.warning("No stocks match your filtering criteria")
            else:
                st.warning("Please enter valid filtering criteria")
