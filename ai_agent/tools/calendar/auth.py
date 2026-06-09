"""
This file handles Google Calendar API authentication.

Think of this file as the "login layer".

Before your app can read calendar events, check availability, or create bookings, 
it must prove to Google that the user has given permission.

This file:
1. Loads an existing Google access token if available.
2. Refreshes the token if it expired.
3. Opens a browser login only when needed.
4. Builds and returns an authenticated Google Calendar API service client.

Other files in the project can import get_calendar_service(), instead of repeating the authentication code.
"""

from __future__ import annotations # deals with the new/old python syntax compatibility
from pathlib import Path
from typing import Any, Dict, Tuple

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError  # type: ignore
from google.auth.exceptions import RefreshError

from config import SCOPES,DEFAULT_CREDENTIALS_PATH,DEFAULT_TOKEN_PATH

# Module-level service cache: avoids rebuilding the API client on every tool call
# Key: (credentials_path, token_path) tuple
_SERVICE_CACHE: Dict[Tuple[str, str], Any] = {}


# this is the main function other files should call
# authenticate with Google and return a Calendar API service
def get_calendar_service(

    credentials_path: str = DEFAULT_CREDENTIALS_PATH,
    token_path: str = DEFAULT_TOKEN_PATH,
    use_cache: bool = True,
    allow_interactive_auth: bool = False,
) -> Any:

# Flow 1) Return cached service if available (avoids repeated setup)
    # create a cache key based on the credential and token file paths
    cache_key = (credentials_path, token_path)
    # if we already created a service client earlier, return it immediately (saving time and avoiding repeated setups)
    if use_cache and cache_key in _SERVICE_CACHE:
        return _SERVICE_CACHE[cache_key]
    # Try to load saved user credentials from the strored token.json
    creds = _load_credentials(token_path)
    # if there are no credentials, or the credentials are invalid, refresh them or ask the user to log in
    if not creds or not creds.valid:
        creds = _refresh_or_login(
            creds=creds,
            credentials_path=credentials_path,
            token_path=token_path,
            allow_interactive_auth=allow_interactive_auth,
        )

    # building  the Google Calendar API client
    service = build("calendar",   # "calendar" means we are using the Calendar API
    "v3", # "v3" is the API version
    credentials=creds)

    # store the service in cache so future calls can reuse it
    if use_cache:
        _SERVICE_CACHE[cache_key] = service
    return service


# clear all cached Google Calendar service clients
# (useful in testing when you want each test to start fresh)
def clear_service_cache() -> None:
    _SERVICE_CACHE.clear()

# Flow 2) Load an existing token.json if present
def _load_credentials(token_path: str) -> Credentials | None:
    """
    Load saved credentials from token.json if it exists:

    If token.json exists, it means the user has already logged in before.
    If it does not exist, the app will need to authenticate again.

    """
    token_file = Path(token_path)

    if token_file.exists():
        return Credentials.from_authorized_user_file(
            filename=str(token_file),
            scopes=SCOPES,
        )
    return None


# Flow 3) Refresh an expired token automatically
def _refresh_or_login(
    creds: Credentials | None,
    credentials_path: str,
    token_path: str,
    allow_interactive_auth: bool,
) -> Credentials:
    """
    Authentication workflow for the Google Calendar API.

    This function acts as the gateway between our application and Google's servers to identity their identity and permissions.

    Authentication follows three possible paths:

    1. Refresh an expired access token automatically.
    2. Launch a browser login if no valid token exists.
    3. Raise an error if authentication is required but not allowed.

    """

    # this file stores the user's access token and refresh token locally
    token_file = Path(token_path)

    # CASE 1: existing token has expired
    # If we have a refresh token, Google can issue a new access token without asking the user to log in again
    if creds and creds.expired and creds.refresh_token:

        try:
            print("[auth] Existing token expired. Trying to refresh it...")
            # send a refresh request to Google servers
            creds.refresh(Request())
            # save the updated token information locally
            _save_token(creds, token_path)
            print("[auth] Token refreshed successfully.")
            return creds

        except RefreshError:
            # This happens if:
            # - the user revoked access from Google settings
            # - the refresh token expired
            # - their credentials were changed
    
            # instead of crashing the app, we delete the invalid token and continue to the browser-login step
            print("[auth] Saved token is expired or revoked.")
            print("[auth] Removing old token.json so you can log in again.")
            token_file.unlink(missing_ok=True)
            creds = None

    # CASE 2: interactive authentication not allowed
    
    # I want to authenticate manually through the terminal since we dont have a websie yet
    if not allow_interactive_auth:
        raise RuntimeError(
            "Google Calendar is not authenticated.\n"
            "Please run:\n"
            "python3 auth.py"
        )

    # CASE 3: credentials exists
    credentials_file = Path(credentials_path)

    if not credentials_file.exists():
        raise FileNotFoundError(
            f"Missing {credentials_path}.\n\n"
            "Download OAuth credentials from Google Cloud Console."
        )

    try:

        print("[auth] Opening browser for Google login...")
        # Flow 4) Launch a browser login when no valid token exists
        # this creates a local OAuth flow
        # the user:
        # logs into Google
        # grants Calendar permissions
        # google redirects back to our local application
        #
        flow = InstalledAppFlow.from_client_secrets_file(
            client_secrets_file=str(credentials_file),
            scopes=SCOPES,
        )

        # launch a temporary local web server and browser window
        # after the user approves access, Google sends an authorization code to this local server
        creds = flow.run_local_server(port=0)

        # save credentials so future runs skip login
        _save_token(creds, token_path)

        print("[auth] Google authentication successful.")

        return creds

    except Exception as e:

        # convert Google errors into user friendly messages
        raise RuntimeError(
            "Google login failed.\n"
            "Possible causes:\n"
            "- Browser closed before login completed\n"
            "- Calendar API not enabled\n"
            "- Incorrect credentials.json\n"
            "- OAuth consent screen not configured\n\n"
            f"Original error: {e}"
        ) from e



def _save_token(creds: Credentials, token_path: str) -> None:
    """
    Save credentials to token.json

    This allows the app to reuse the user's authorization in future runs
    """
    Path(token_path).write_text(creds.to_json(), encoding="utf-8")
    print(f"[auth] Token saved to {token_path}")


# CLI entry point — run once to authorise and generate token.json
if __name__ == "__main__":
    try:
        print("[auth] Starting Google Calendar authentication...")
        get_calendar_service(allow_interactive_auth=True, use_cache=False)
        print("[auth] Authentication complete. token.json is ready.")

    except Exception as e:
        print("\n[auth] Authentication failed.")
        print(e)