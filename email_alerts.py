import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

def is_email_configured():
    """Check if email credentials are configured"""
    email_address = os.environ.get("ALERT_EMAIL_ADDRESS")
    email_password = os.environ.get("ALERT_EMAIL_PASSWORD")
    
    return all([email_address, email_password])

def send_email_alert(to_email, subject, message_html):
    """
    Send an email alert using SMTP
    
    Args:
        to_email: Recipient's email address
        subject: Email subject
        message_html: HTML content of the email
        
    Returns:
        Success message if sent, error message if failed
    """
    if not is_email_configured():
        return "Email is not configured. Please set up the required environment variables."
    
    try:
        # Get email credentials from environment variables
        from_email = os.environ.get("ALERT_EMAIL_ADDRESS")
        email_password = os.environ.get("ALERT_EMAIL_PASSWORD")
        
        # Create message
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = from_email
        msg['To'] = to_email
        
        # Add HTML content
        html_part = MIMEText(message_html, 'html')
        msg.attach(html_part)
        
        # Connect to SMTP server and send email
        # Note: This assumes Gmail. For other providers, change the server details.
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(from_email, email_password)
        server.send_message(msg)
        server.quit()
        
        return f"Email alert sent to {to_email}"
    
    except Exception as e:
        return f"Error sending email: {str(e)}"

def create_buy_signal_email(ticker, price, signal_type, signal_strength, indicators, timestamp=None):
    """
    Create HTML content for a buy signal email
    
    Args:
        ticker: Stock symbol
        price: Current price
        signal_type: Type of signal (e.g., "BUY", "STRONG BUY", "SELL")
        signal_strength: Signal strength percentage
        indicators: Dictionary of indicators that triggered the signal
        timestamp: Optional timestamp for the signal
        
    Returns:
        HTML content for the email
    """
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
    # Set color based on signal type
    if 'BUY' in signal_type:
        signal_color = '#4CAF50'  # Green
        arrow = '▲'
    elif 'SELL' in signal_type:
        signal_color = '#F44336'  # Red
        arrow = '▼'
    else:
        signal_color = '#FFC107'  # Yellow
        arrow = '◆'
    
    # Create a nice HTML email with responsive design
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Stock Alert: {ticker}</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 600px;
                margin: 0 auto;
                padding: 20px;
            }}
            .header {{
                background-color: #f8f9fa;
                padding: 20px;
                text-align: center;
                border-radius: 5px 5px 0 0;
                border-bottom: 3px solid {signal_color};
            }}
            .content {{
                padding: 20px;
                background-color: white;
                border: 1px solid #ddd;
                border-top: none;
                border-radius: 0 0 5px 5px;
            }}
            .signal {{
                font-size: 24px;
                color: {signal_color};
                font-weight: bold;
                margin: 10px 0;
            }}
            .price {{
                font-size: 36px;
                font-weight: bold;
                color: #333;
            }}
            .indicators {{
                margin-top: 20px;
                border-top: 1px solid #eee;
                padding-top: 15px;
            }}
            .indicator {{
                margin-bottom: 10px;
            }}
            .footer {{
                margin-top: 20px;
                font-size: 12px;
                color: #777;
                text-align: center;
            }}
            .strength-meter {{
                height: 10px;
                background-color: #e0e0e0;
                border-radius: 5px;
                margin: 10px 0;
                overflow: hidden;
            }}
            .strength-value {{
                height: 100%;
                background-color: {signal_color};
                width: {signal_strength}%;
            }}
            .button {{
                display: inline-block;
                padding: 10px 20px;
                background-color: #3498db;
                color: white;
                text-decoration: none;
                border-radius: 5px;
                margin-top: 15px;
            }}
            @media screen and (max-width: 480px) {{
                body {{
                    padding: 10px;
                }}
                .header, .content {{
                    padding: 10px;
                }}
                .price {{
                    font-size: 28px;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>{ticker} Alert</h1>
        </div>
        <div class="content">
            <p>A trading signal has been detected:</p>
            <div class="signal">{arrow} {signal_type} SIGNAL {arrow}</div>
            <div class="price">${price:.2f}</div>
            
            <p>Signal Strength: {signal_strength:.0f}%</p>
            <div class="strength-meter">
                <div class="strength-value"></div>
            </div>
            
            <div class="indicators">
                <h3>Technical Indicators:</h3>
    """
    
    # Add each indicator
    for indicator, value in indicators.items():
        html += f'<div class="indicator"><strong>{indicator}:</strong> {value}</div>'
    
    html += f"""
            </div>
            
            <p><strong>Signal Time:</strong> {timestamp}</p>
            
            <a href="#" class="button">View Chart</a>
            
            <div class="footer">
                <p>This is an automated alert from your Stock Market Analysis Tool. 
                Please conduct your own research before making investment decisions.</p>
                <p>To unsubscribe from these alerts, click <a href="#">here</a>.</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    return html

def validate_email(email):
    """
    Basic validation for email addresses
    
    Args:
        email: Email address to validate
        
    Returns:
        True if valid format, False otherwise
    """
    # Very basic check for @ symbol and at least one dot after it
    if '@' not in email:
        return False
    
    # Split by @ and check for domain
    parts = email.split('@')
    if len(parts) != 2:
        return False
    
    domain = parts[1]
    if '.' not in domain:
        return False
    
    # Check for minimum lengths
    if len(parts[0]) < 1 or len(domain) < 3:
        return False
    
    return True
