import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# Define sector mappings (will be populated from Yahoo Finance)
SECTOR_COLORS = {
    "Technology": "#1E88E5",
    "Healthcare": "#43A047",
    "Financial Services": "#FDD835",
    "Consumer Cyclical": "#FB8C00",
    "Communication Services": "#D81B60",
    "Industrials": "#8E24AA",
    "Consumer Defensive": "#00ACC1",
    "Energy": "#F4511E",
    "Basic Materials": "#5E35B1",
    "Real Estate": "#3949AB",
    "Utilities": "#00897B",
    "Unknown": "#757575"
}

def initialize_portfolio():
    """Initialize portfolio state variables"""
    if 'portfolio' not in st.session_state:
        st.session_state.portfolio = []
    
    if 'portfolio_history' not in st.session_state:
        st.session_state.portfolio_history = pd.DataFrame()
    
    if 'last_portfolio_update' not in st.session_state:
        st.session_state.last_portfolio_update = datetime.now() - timedelta(days=1)
    
    if 'risk_assessment' not in st.session_state:
        st.session_state.risk_assessment = {}

def get_stock_info(ticker):
    """Get detailed information about a stock"""
    try:
        # Try to get the latest price data first
        ticker_data = yf.download(ticker, period="1d")
        current_price = 0
        
        if not len(ticker_data) == 0:
            current_price = ticker_data['Close'].iloc[-1]
        
        # Get additional info using the Ticker method
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # If we couldn't get the price from download, try the info dict
        if current_price <= 0:
            current_price = info.get('currentPrice', 
                          info.get('regularMarketPrice',
                          info.get('previousClose', 
                          info.get('open', 0))))
        
        # Create a clean stock info dict with essential data
        stock_data = {
            'ticker': ticker,
            'name': info.get('shortName', info.get('longName', ticker)),
            'sector': info.get('sector', 'Unknown'),
            'industry': info.get('industry', 'Unknown'),
            'current_price': current_price,
            'market_cap': info.get('marketCap', 0),
            'pe_ratio': info.get('trailingPE', 0),
            'dividend_yield': info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0,
            'beta': info.get('beta', 0),
            'avg_volume': info.get('averageVolume', 0)
        }
        return stock_data
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return None

def add_to_portfolio(ticker, shares, purchase_price, purchase_date):
    """Add a stock to the portfolio"""
    stock_info = get_stock_info(ticker)
    
    if stock_info:
        # Create portfolio entry
        entry = {
            **stock_info,
            'shares': shares,
            'purchase_price': purchase_price,
            'purchase_date': purchase_date,
            'cost_basis': shares * purchase_price,
            'current_value': shares * stock_info['current_price'],
            'gain_loss_dollar': shares * (stock_info['current_price'] - purchase_price),
            'gain_loss_percent': ((stock_info['current_price'] - purchase_price) / purchase_price) * 100 if purchase_price > 0 else 0
        }
        
        # Check if stock already exists in portfolio
        exists = False
        for i, stock in enumerate(st.session_state.portfolio):
            if stock['ticker'] == ticker:
                # Update existing position (average down/up)
                total_shares = stock['shares'] + shares
                total_cost = stock['cost_basis'] + (shares * purchase_price)
                avg_price = total_cost / total_shares if total_shares > 0 else 0
                
                st.session_state.portfolio[i] = {
                    **stock_info,
                    'shares': total_shares,
                    'purchase_price': avg_price,
                    'purchase_date': f"Multiple ({stock['purchase_date']}, {purchase_date})",
                    'cost_basis': total_cost,
                    'current_value': total_shares * stock_info['current_price'],
                    'gain_loss_dollar': total_shares * (stock_info['current_price'] - avg_price),
                    'gain_loss_percent': ((stock_info['current_price'] - avg_price) / avg_price) * 100 if avg_price > 0 else 0
                }
                exists = True
                break
        
        # Add new position if it doesn't exist
        if not exists:
            st.session_state.portfolio.append(entry)
        
        update_portfolio_history()
        calculate_risk_metrics()
        return True
    return False

def remove_from_portfolio(ticker):
    """Remove a stock from the portfolio"""
    initial_length = len(st.session_state.portfolio)
    st.session_state.portfolio = [stock for stock in st.session_state.portfolio if stock['ticker'] != ticker]
    
    if len(st.session_state.portfolio) < initial_length:
        update_portfolio_history()
        calculate_risk_metrics()
        return True
    return False

def update_portfolio_prices():
    """Update current prices and values for all portfolio stocks"""
    if not st.session_state.portfolio:
        return
    
    for i, stock in enumerate(st.session_state.portfolio):
        try:
            ticker = stock['ticker']
            # Try to get the latest price data
            ticker_data = yf.download(ticker, period="1d")
            
            if not len(ticker_data) == 0:
                current_price = ticker_data['Close'].iloc[-1]
            else:
                # Fall back to the Ticker info method if download fails
                ticker_obj = yf.Ticker(ticker)
                ticker_info = ticker_obj.info
                current_price = ticker_info.get('currentPrice', 
                               ticker_info.get('regularMarketPrice',
                               ticker_info.get('previousClose', 
                               ticker_info.get('open', stock['current_price']))))
            
            # Make sure we have a valid price
            if current_price <= 0:
                current_price = stock['current_price']  # Use previous price if we can't get a new one
            
            # Update price and calculations
            st.session_state.portfolio[i]['current_price'] = current_price
            st.session_state.portfolio[i]['current_value'] = stock['shares'] * current_price
            st.session_state.portfolio[i]['gain_loss_dollar'] = stock['shares'] * (current_price - stock['purchase_price'])
            st.session_state.portfolio[i]['gain_loss_percent'] = ((current_price - stock['purchase_price']) / stock['purchase_price']) * 100 if stock['purchase_price'] > 0 else 0
        except Exception as e:
            st.warning(f"Could not update price for {stock['ticker']}: {str(e)}")
    
    st.session_state.last_portfolio_update = datetime.now()
    update_portfolio_history()
    calculate_risk_metrics()

def get_portfolio_summary():
    """Calculate portfolio summary metrics"""
    if not st.session_state.portfolio:
        return {
            'total_value': 0,
            'total_cost': 0,
            'total_gain_loss_dollar': 0,
            'total_gain_loss_percent': 0,
            'num_positions': 0,
            'num_winning': 0,
            'num_losing': 0,
            'best_performer': None,
            'worst_performer': None
        }
    
    total_value = sum(stock['current_value'] for stock in st.session_state.portfolio)
    total_cost = sum(stock['cost_basis'] for stock in st.session_state.portfolio)
    total_gain_loss = total_value - total_cost
    
    if total_cost > 0:
        total_gain_loss_percent = (total_gain_loss / total_cost) * 100
    else:
        total_gain_loss_percent = 0
    
    # Count winning and losing positions
    winning_positions = [s for s in st.session_state.portfolio if s['gain_loss_dollar'] > 0]
    losing_positions = [s for s in st.session_state.portfolio if s['gain_loss_dollar'] < 0]
    
    # Find best and worst performers
    if st.session_state.portfolio:
        sorted_by_percent = sorted(st.session_state.portfolio, key=lambda x: x['gain_loss_percent'], reverse=True)
        best_performer = sorted_by_percent[0] if sorted_by_percent else None
        worst_performer = sorted_by_percent[-1] if sorted_by_percent else None
    else:
        best_performer = None
        worst_performer = None
    
    return {
        'total_value': total_value,
        'total_cost': total_cost,
        'total_gain_loss_dollar': total_gain_loss,
        'total_gain_loss_percent': total_gain_loss_percent,
        'num_positions': len(st.session_state.portfolio),
        'num_winning': len(winning_positions),
        'num_losing': len(losing_positions),
        'best_performer': best_performer,
        'worst_performer': worst_performer
    }

def get_sector_allocation():
    """Calculate sector allocation percentages"""
    if not st.session_state.portfolio:
        return pd.DataFrame()
    
    # Create DataFrame with sector data
    sectors = {}
    total_value = sum(stock['current_value'] for stock in st.session_state.portfolio)
    
    for stock in st.session_state.portfolio:
        sector = stock['sector']
        if sector not in sectors:
            sectors[sector] = 0
        sectors[sector] += stock['current_value']
    
    # Convert to DataFrame with percentages
    sector_data = []
    for sector, value in sectors.items():
        percentage = (value / total_value) * 100 if total_value > 0 else 0
        sector_data.append({
            'sector': sector,
            'value': value,
            'percentage': percentage
        })
    
    return pd.DataFrame(sector_data)

def update_portfolio_history():
    """Update portfolio history with current value"""
    if not st.session_state.portfolio:
        return
    
    today = datetime.now().date()
    total_value = sum(stock['current_value'] for stock in st.session_state.portfolio)
    
    # If we already have an entry for today, update it
    if len(st.session_state.portfolio_history) == 0:
        history = pd.DataFrame([{'date': today, 'value': total_value}])
        st.session_state.portfolio_history = history
    else:
        # Convert to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(st.session_state.portfolio_history['date']):
            st.session_state.portfolio_history['date'] = pd.to_datetime(st.session_state.portfolio_history['date'])
        
        today_entry = st.session_state.portfolio_history[st.session_state.portfolio_history['date'].dt.date == today]
        
        if len(today_entry) > 0:
            st.session_state.portfolio_history.loc[st.session_state.portfolio_history['date'].dt.date == today, 'value'] = total_value
        else:
            new_row = pd.DataFrame([{'date': today, 'value': total_value}])
            st.session_state.portfolio_history = pd.concat([st.session_state.portfolio_history, new_row], ignore_index=True)
    
    # Sort by date
    st.session_state.portfolio_history = st.session_state.portfolio_history.sort_values('date')

def calculate_historical_returns():
    """Calculate historical returns for the portfolio"""
    if len(st.session_state.portfolio_history) == 0 or len(st.session_state.portfolio_history) < 2:
        return None
    
    df = st.session_state.portfolio_history.copy()
    df['daily_return'] = df['value'].pct_change()
    df['cumulative_return'] = (1 + df['daily_return']).cumprod() - 1
    
    return df

def calculate_risk_metrics():
    """Calculate key risk metrics for the portfolio"""
    if not st.session_state.portfolio:
        st.session_state.risk_assessment = {}
        return
    
    # Get historical data for beta calculation
    tickers = [stock['ticker'] for stock in st.session_state.portfolio]
    weights = [stock['current_value'] / sum(s['current_value'] for s in st.session_state.portfolio) 
              for stock in st.session_state.portfolio]
    
    # Get 1-year historical data for portfolio and S&P 500
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    try:
        # Get S&P 500 data (^GSPC is the ticker for S&P 500)
        sp500 = yf.download('^GSPC', start=start_date, end=end_date)
        sp500_returns = sp500['Adj Close'].pct_change().dropna()
        
        # Get portfolio stock data
        portfolio_data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
        
        # Handle single stock case
        if len(tickers) == 1:
            portfolio_data = pd.DataFrame({tickers[0]: portfolio_data})
        
        # Calculate portfolio returns
        portfolio_returns = portfolio_data.pct_change().dropna()
        
        # Calculate weighted portfolio returns
        weighted_returns = pd.DataFrame()
        
        for i, ticker in enumerate(tickers):
            if ticker in portfolio_returns.columns:
                weighted_returns[ticker] = portfolio_returns[ticker] * weights[i]
        
        portfolio_daily_returns = weighted_returns.sum(axis=1)
        
        # Align dates
        aligned_returns = pd.concat([portfolio_daily_returns, sp500_returns], axis=1).dropna()
        aligned_returns.columns = ['Portfolio', 'S&P500']
        
        # Calculate beta
        covariance = aligned_returns.cov().iloc[0, 1]
        market_variance = aligned_returns['S&P500'].var()
        beta = covariance / market_variance if market_variance > 0 else 1.0
        
        # Calculate volatility (annualized standard deviation)
        volatility = portfolio_daily_returns.std() * (252 ** 0.5)  # Annualized
        
        # Calculate Sharpe Ratio (assuming risk-free rate of 2%)
        risk_free_rate = 0.02
        mean_return = portfolio_daily_returns.mean() * 252  # Annualized
        sharpe_ratio = (mean_return - risk_free_rate) / volatility if volatility > 0 else 0
        
        # Calculate max drawdown
        cum_returns = (1 + portfolio_daily_returns).cumprod()
        max_drawdown = ((cum_returns.cummax() - cum_returns) / cum_returns.cummax()).max()
        
        # Calculate Value at Risk (VaR) at 95% confidence level
        var_95 = np.percentile(portfolio_daily_returns, 5)
        
        # Calculate correlation with S&P 500
        correlation = aligned_returns.corr().iloc[0, 1]
        
        # Calculate diversification score (based on sector allocation)
        sectors = {}
        total_value = sum(stock['current_value'] for stock in st.session_state.portfolio)
        
        for stock in st.session_state.portfolio:
            sector = stock['sector']
            if sector not in sectors:
                sectors[sector] = 0
            sectors[sector] += stock['current_value'] / total_value
        
        # Herfindahl-Hirschman Index (HHI) for concentration
        hhi = sum(weight ** 2 for weight in sectors.values())
        
        # Diversification score (inverse of HHI, normalized to 0-100)
        diversification_score = (1 - hhi) * 100
        
        # Overall risk score (0-100, higher is riskier)
        # Components: beta (30%), volatility (30%), diversification (20%), drawdown (20%)
        normalized_beta = min(100, max(0, (beta / 2) * 100))  # Normalize around 1.0
        normalized_volatility = min(100, max(0, (volatility / 0.3) * 100))  # Normalize around 30%
        normalized_drawdown = min(100, max(0, (max_drawdown / 0.4) * 100))  # Normalize around 40%
        
        risk_score = (
            0.3 * normalized_beta +
            0.3 * normalized_volatility +
            0.2 * (100 - diversification_score) +
            0.2 * normalized_drawdown
        )
        
        # Risk category
        if risk_score < 25:
            risk_category = "Low Risk"
        elif risk_score < 50:
            risk_category = "Moderate Risk"
        elif risk_score < 75:
            risk_category = "High Risk"
        else:
            risk_category = "Very High Risk"
        
        # Store all metrics
        st.session_state.risk_assessment = {
            'beta': beta,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'var_95': var_95,
            'correlation': correlation,
            'diversification_score': diversification_score,
            'risk_score': risk_score,
            'risk_category': risk_category,
            'calculated_date': datetime.now()
        }
    except Exception as e:
        st.error(f"Error calculating risk metrics: {str(e)}")
        st.session_state.risk_assessment = {
            'error': str(e),
            'calculated_date': datetime.now()
        }

def get_performance_metrics():
    """Calculate key performance metrics for the portfolio"""
    if not st.session_state.portfolio or len(st.session_state.portfolio_history) == 0:
        return {}
    
    try:
        # Get historical returns
        returns_data = calculate_historical_returns()
        
        if returns_data is None or len(returns_data) == 0:
            return {}
        
        # Calculate metrics
        latest_value = returns_data['value'].iloc[-1]
        initial_value = returns_data['value'].iloc[0]
        total_return = (latest_value / initial_value - 1) * 100
        
        # Calculate annualized return
        days = (returns_data['date'].iloc[-1] - returns_data['date'].iloc[0]).days
        if days > 0:
            annualized_return = ((1 + total_return/100) ** (365/days) - 1) * 100
        else:
            annualized_return = 0
        
        return {
            'current_value': latest_value,
            'initial_value': initial_value,
            'total_return': total_return,
            'annualized_return': annualized_return
        }
    except Exception as e:
        st.error(f"Error calculating performance metrics: {str(e)}")
        return {}

def plot_portfolio_value_history():
    """Create a line chart of portfolio value over time"""
    if len(st.session_state.portfolio_history) == 0:
        return None
    
    fig = px.line(
        st.session_state.portfolio_history, 
        x='date', 
        y='value',
        title='Portfolio Value History',
        labels={'date': 'Date', 'value': 'Portfolio Value ($)'}
    )
    
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Value ($)',
        hovermode='x unified'
    )
    
    return fig

def plot_sector_allocation():
    """Create a pie chart of sector allocation"""
    if not st.session_state.portfolio:
        return None
    
    sector_data = get_sector_allocation()
    
    if len(sector_data) == 0:
        return None
    
    # Add color mapping
    sector_data['color'] = sector_data['sector'].apply(lambda x: SECTOR_COLORS.get(x, SECTOR_COLORS['Unknown']))
    
    fig = px.pie(
        sector_data,
        values='percentage',
        names='sector',
        title='Sector Allocation',
        color='sector',
        color_discrete_map={sector: SECTOR_COLORS.get(sector, SECTOR_COLORS['Unknown']) 
                           for sector in sector_data['sector']}
    )
    
    fig.update_traces(
        textposition='inside', 
        textinfo='percent+label',
        hoverinfo='label+percent+value',
        marker=dict(line=dict(color='white', width=2))
    )
    
    return fig

def run_risk_assessment():
    """Run a one-click risk assessment for the portfolio"""
    if not st.session_state.portfolio:
        st.warning("Add stocks to your portfolio to run a risk assessment.")
        return
    
    # Calculate risk metrics
    calculate_risk_metrics()
    
    # Return True if assessment was successful
    return 'risk_score' in st.session_state.risk_assessment

def display_portfolio_summary():
    """Display portfolio summary cards"""
    summary = get_portfolio_summary()
    performance = get_performance_metrics()
    
    # Create three columns for summary cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Total Portfolio Value", 
            f"${summary['total_value']:,.2f}",
            f"{performance.get('total_return', 0):.2f}%" if performance else None
        )
        st.metric(
            "Total Cost Basis",
            f"${summary['total_cost']:,.2f}"
        )
    
    with col2:
        st.metric(
            "Total Gain/Loss",
            f"${summary['total_gain_loss_dollar']:,.2f}",
            f"{summary['total_gain_loss_percent']:.2f}%"
        )
        st.metric(
            "Number of Positions",
            f"{summary['num_positions']}",
            f"✅ {summary['num_winning']} | ❌ {summary['num_losing']}"
        )
    
    with col3:
        if 'risk_assessment' in st.session_state and 'risk_category' in st.session_state.risk_assessment:
            risk = st.session_state.risk_assessment
            st.metric(
                "Risk Level",
                risk['risk_category'],
                f"Score: {risk['risk_score']:.1f}/100"
            )
            st.metric(
                "Diversification",
                f"{risk['diversification_score']:.1f}%"
            )
        else:
            st.info("Run risk assessment for more metrics")
            st.button("Run Risk Assessment", on_click=run_risk_assessment, type="primary")

def display_portfolio_table():
    """Display the portfolio as a table"""
    if not st.session_state.portfolio:
        st.info("Your portfolio is empty. Add stocks using the form above.")
        return
    
    # Convert portfolio to DataFrame for display
    df = pd.DataFrame(st.session_state.portfolio)
    
    # Select and rename columns for display
    display_df = df[[
        'ticker', 'name', 'sector', 'shares', 'purchase_price', 
        'current_price', 'cost_basis', 'current_value', 
        'gain_loss_dollar', 'gain_loss_percent'
    ]].copy()
    
    display_df.columns = [
        'Ticker', 'Name', 'Sector', 'Shares', 'Purchase Price',
        'Current Price', 'Cost Basis', 'Current Value',
        'Gain/Loss ($)', 'Gain/Loss (%)'
    ]
    
    # Format numeric columns
    display_df['Purchase Price'] = display_df['Purchase Price'].map('${:,.2f}'.format)
    display_df['Current Price'] = display_df['Current Price'].map('${:,.2f}'.format)
    display_df['Cost Basis'] = display_df['Cost Basis'].map('${:,.2f}'.format)
    display_df['Current Value'] = display_df['Current Value'].map('${:,.2f}'.format)
    display_df['Gain/Loss ($)'] = display_df['Gain/Loss ($)'].map('${:,.2f}'.format)
    display_df['Gain/Loss (%)'] = display_df['Gain/Loss (%)'].map('{:,.2f}%'.format)
    
    # Display table
    st.dataframe(
        display_df,
        column_config={
            "Ticker": st.column_config.TextColumn("Ticker", width="small"),
            "Name": st.column_config.TextColumn("Name", width="medium"),
            "Sector": st.column_config.TextColumn("Sector", width="medium"),
            "Shares": st.column_config.NumberColumn("Shares", width="small"),
        },
        hide_index=True
    )

def display_risk_assessment():
    """Display detailed risk assessment metrics"""
    if 'risk_assessment' not in st.session_state or not st.session_state.risk_assessment:
        st.warning("Run a risk assessment first to see detailed metrics.")
        st.button("Run Risk Assessment", on_click=run_risk_assessment, type="primary")
        return
    
    risk = st.session_state.risk_assessment
    
    if 'error' in risk:
        st.error(f"Error in last risk assessment: {risk['error']}")
        st.button("Try Again", on_click=run_risk_assessment, type="primary")
        return
    
    # Create columns for risk metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Risk Metrics")
        metrics1 = [
            {"label": "Portfolio Beta", "value": f"{risk['beta']:.2f}", 
             "help": "Measure of portfolio volatility relative to the market. >1 means more volatile than the market."},
            {"label": "Volatility (Annual)", "value": f"{risk['volatility']*100:.2f}%", 
             "help": "Standard deviation of returns, annualized. Higher means more volatile."},
            {"label": "Maximum Drawdown", "value": f"{risk['max_drawdown']*100:.2f}%", 
             "help": "Largest peak-to-trough decline. Represents worst-case historical loss."},
            {"label": "Value at Risk (95%)", "value": f"{risk['var_95']*100:.2f}%", 
             "help": "Maximum daily loss expected with 95% confidence. Smaller negative number is better."},
        ]
        
        for metric in metrics1:
            st.metric(metric["label"], metric["value"], help=metric["help"])
    
    with col2:
        st.subheader("Performance Metrics")
        metrics2 = [
            {"label": "Sharpe Ratio", "value": f"{risk['sharpe_ratio']:.2f}", 
             "help": "Return per unit of risk. Higher is better, >1 is good."},
            {"label": "Market Correlation", "value": f"{risk['correlation']:.2f}", 
             "help": "Correlation with S&P 500. Closer to 1 means moves with market."},
            {"label": "Diversification Score", "value": f"{risk['diversification_score']:.1f}%", 
             "help": "Higher means better diversified across sectors."},
            {"label": "Overall Risk Score", "value": f"{risk['risk_score']:.1f}/100", 
             "help": "Proprietary risk score. Lower means less risky."},
        ]
        
        for metric in metrics2:
            st.metric(metric["label"], metric["value"], help=metric["help"])
    
    # Risk category explanation
    st.info(f"Risk Category: **{risk['risk_category']}** (Last calculated: {risk['calculated_date'].strftime('%Y-%m-%d %H:%M')})")
    
    # Risk category explanation
    risk_explanations = {
        "Low Risk": "Your portfolio shows low volatility and good diversification. It's likely to be resilient during market downturns but may underperform during bull markets.",
        "Moderate Risk": "Your portfolio has a balanced risk profile with moderate volatility. This is generally suitable for medium to long-term investors who can tolerate some market fluctuations.",
        "High Risk": "Your portfolio shows elevated volatility or concentration. Consider diversifying more across sectors or adding some defensive positions if you're concerned about short-term fluctuations.",
        "Very High Risk": "Your portfolio has significant risk exposure due to high volatility, concentration, or leverage. Consider rebalancing to reduce risk if this doesn't match your risk tolerance."
    }
    
    if risk['risk_category'] in risk_explanations:
        st.write(risk_explanations[risk['risk_category']])
    
    # Display risk recommendations
    st.subheader("Risk Management Recommendations")
    
    recommendations = []
    
    # Beta-based recommendations
    if risk['beta'] > 1.5:
        recommendations.append("• Your portfolio is significantly more volatile than the market. Consider adding some defensive stocks or bonds to reduce volatility.")
    elif risk['beta'] < 0.7:
        recommendations.append("• Your portfolio has lower volatility than the market. This is good for stability but might underperform during bull markets.")
    
    # Diversification recommendations
    if risk['diversification_score'] < 60:
        recommendations.append("• Your portfolio could benefit from better sector diversification. Consider adding positions in underrepresented sectors.")
    
    # Concentration recommendations
    if len(st.session_state.portfolio) < 5:
        recommendations.append("• Your portfolio has few positions, increasing specific company risk. Consider adding more stocks for better diversification.")
    
    # Display all recommendations
    if recommendations:
        for rec in recommendations:
            st.write(rec)
    else:
        st.write("• Your portfolio appears well-balanced from a risk perspective.")

def portfolio_section():
    """Main function for the portfolio page"""
    # Initialize portfolio state if needed
    initialize_portfolio()
    
    st.header("📊 Portfolio Tracker")
    st.markdown("""
    Track your stock portfolio performance, analyze risk metrics, and get insights on your investments.
    """)
    
    # Check if we need to update prices (every 15 minutes)
    time_since_update = datetime.now() - st.session_state.last_portfolio_update
    if time_since_update.total_seconds() > 900:  # 15 minutes in seconds
        update_portfolio_prices()
    
    # Portfolio management section
    with st.expander("Manage Portfolio", expanded=len(st.session_state.portfolio) == 0):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Add Stock to Portfolio")
            
            # Form for adding new stock
            with st.form("add_stock_form"):
                ticker = st.text_input("Ticker Symbol", max_chars=10).upper()
                col1a, col2a = st.columns(2)
                with col1a:
                    shares = st.number_input("Number of Shares", min_value=0.01, step=1.0, format="%.2f")
                with col2a:
                    purchase_price = st.number_input("Purchase Price per Share ($)", min_value=0.01, step=1.0, format="%.2f")
                purchase_date = st.date_input("Purchase Date", value=datetime.now().date())
                
                submitted = st.form_submit_button("Add to Portfolio")
                
                if submitted and ticker and shares > 0 and purchase_price > 0:
                    success = add_to_portfolio(ticker, shares, purchase_price, purchase_date.strftime('%Y-%m-%d'))
                    if success:
                        st.success(f"Added {ticker} to your portfolio!")
                    else:
                        st.error(f"Could not add {ticker}. Check the ticker symbol and try again.")
        
        with col2:
            st.subheader("Portfolio Actions")
            
            if st.button("Update Prices", type="primary", use_container_width=True):
                update_portfolio_prices()
                st.success("Prices updated successfully!")
            
            if st.button("Run Risk Assessment", use_container_width=True):
                if run_risk_assessment():
                    st.success("Risk assessment completed!")
                else:
                    st.error("Could not complete risk assessment. Add stocks to your portfolio first.")
            
            if st.session_state.portfolio:
                # Dropdown to remove stocks
                selected_ticker = st.selectbox(
                    "Select stock to remove:",
                    options=[stock['ticker'] for stock in st.session_state.portfolio],
                    index=None,
                    placeholder="Choose a stock..."
                )
                
                if selected_ticker and st.button("Remove from Portfolio", type="secondary", use_container_width=True):
                    if remove_from_portfolio(selected_ticker):
                        st.success(f"Removed {selected_ticker} from your portfolio!")
    
    # Only show portfolio data if we have stocks
    if st.session_state.portfolio:
        # Display portfolio summary
        st.subheader("Portfolio Summary")
        display_portfolio_summary()
        
        # Charts section
        col1, col2 = st.columns(2)
        
        with col1:
            # Portfolio value history chart
            value_chart = plot_portfolio_value_history()
            if value_chart:
                st.plotly_chart(value_chart, use_container_width=True)
            else:
                st.info("Not enough history data to show chart.")
        
        with col2:
            # Sector allocation chart
            sector_chart = plot_sector_allocation()
            if sector_chart:
                st.plotly_chart(sector_chart, use_container_width=True)
            else:
                st.info("No sector data available.")
        
        # Portfolio holdings table
        st.subheader("Portfolio Holdings")
        display_portfolio_table()
        
        # Risk assessment section
        st.subheader("Risk Analysis")
        display_risk_assessment()
    else:
        st.info("Add your first stock to get started!")
        
        # Example portfolio button
        if st.button("Add Example Portfolio (for demo)", use_container_width=True):
            # Add some example stocks
            example_stocks = [
                {"ticker": "AAPL", "shares": 10, "price": 175.50, "date": "2023-01-15"},
                {"ticker": "MSFT", "shares": 5, "price": 320.75, "date": "2023-02-10"},
                {"ticker": "GOOGL", "shares": 8, "price": 130.20, "date": "2023-03-05"},
                {"ticker": "AMZN", "shares": 12, "price": 145.30, "date": "2023-04-20"},
                {"ticker": "JNJ", "shares": 15, "price": 160.45, "date": "2023-05-12"}
            ]
            
            for stock in example_stocks:
                add_to_portfolio(stock["ticker"], stock["shares"], stock["price"], stock["date"])
            
            run_risk_assessment()
            st.success("Example portfolio added!")
            st.rerun()
