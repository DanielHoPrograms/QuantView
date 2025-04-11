import os
from twilio.rest import Client

def is_twilio_configured():
    """Check if Twilio credentials are configured"""
    twilio_account_sid = os.environ.get("TWILIO_ACCOUNT_SID")
    twilio_auth_token = os.environ.get("TWILIO_AUTH_TOKEN")
    twilio_phone_number = os.environ.get("TWILIO_PHONE_NUMBER")
    
    return all([twilio_account_sid, twilio_auth_token, twilio_phone_number])

def send_price_alert(to_phone_number, ticker, price, signal_type, signal_strength=None):
    """
    Send a price alert SMS message using Twilio
    
    Args:
        to_phone_number: Recipient's phone number (format: +1XXXXXXXXXX)
        ticker: Stock symbol
        price: Current price
        signal_type: Type of signal (e.g., "BUY", "SELL")
        signal_strength: Optional signal strength percentage
    
    Returns:
        Success message if sent, error message if failed
    """
    if not is_twilio_configured():
        return "Twilio is not configured. Please set up the required environment variables."
    
    try:
        # Get Twilio credentials from environment variables
        account_sid = os.environ.get("TWILIO_ACCOUNT_SID")
        auth_token = os.environ.get("TWILIO_AUTH_TOKEN")
        from_number = os.environ.get("TWILIO_PHONE_NUMBER")
        
        # Create Twilio client
        client = Client(account_sid, auth_token)
        
        # Create message content
        if signal_strength:
            message_body = f"ALERT: {signal_type} signal for {ticker} at ${price:.2f} (Signal Strength: {signal_strength:.0f}%)"
        else:
            message_body = f"ALERT: {signal_type} signal for {ticker} at ${price:.2f}"
        
        # Send message
        message = client.messages.create(
            body=message_body,
            from_=from_number,
            to=to_phone_number
        )
        
        return f"Message sent with SID: {message.sid}"
    
    except Exception as e:
        return f"Error sending SMS: {str(e)}"

def validate_phone_number(phone_number):
    """
    Basic validation for phone numbers
    
    Args:
        phone_number: Phone number to validate
        
    Returns:
        Formatted phone number with + prefix if valid, None if invalid
    """
    # Remove any non-digit characters
    digits_only = ''.join(filter(str.isdigit, phone_number))
    
    # Check if we have a reasonable number of digits (min 10)
    if len(digits_only) < 10:
        return None
    
    # If the number doesn't start with +, add the + sign
    if not phone_number.startswith('+'):
        # If it's a US number (10 digits), add +1
        if len(digits_only) == 10:
            return f"+1{digits_only}"
        # Otherwise just add +
        return f"+{digits_only}"
    
    return phone_number
