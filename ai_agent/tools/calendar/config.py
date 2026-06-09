from pathlib import Path

_CONFIG_DIR = Path(__file__).parent  # resolves to wherever config.py lives


# all defaults are set here
DEFAULT_TIMEZONE = "Asia/Dubai"

# OAuth Scope : defines what permissions our application is requesting
# This permission scope allows the app to read and write Google Calendar data
# Read = check availability
# Write = create bookings
SCOPES = ["https://www.googleapis.com/auth/calendar"]

# downloaded from Google Cloud Console and it identifies our app to Google
DEFAULT_CREDENTIALS_PATH = str(_CONFIG_DIR / "credentials.json")

# generated automatically after the first successful login.
# contains: Access Token (short-lived) and Refresh Token (long-lived)
# this file allows the user to stay signed in between sessions
DEFAULT_TOKEN_PATH = str(_CONFIG_DIR / "token.json")
DEFAULT_LOG_DIR = "data/logs"
DEFAULT_MAX_LOG_SIZE_MB = 5