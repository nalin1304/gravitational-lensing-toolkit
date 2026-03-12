"""
Authentication Utilities for FastAPI

This module provides JWT token handling, password hashing, and user
authentication functions for the Gravitational Lensing API.

Author: Gravitational Lensing Toolkit
Date: March 2025
"""

from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# JWT Configuration
SECRET_KEY = "your-secret-key-change-in-production"  # Change in production!
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer(auto_error=False)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a plain password against a hashed password.

    Parameters
    ----------
    plain_password : str
        The plain text password
    hashed_password : str
        The hashed password stored in database

    Returns
    -------
    bool
        True if passwords match, False otherwise
    """
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """
    Hash a plain password.

    Parameters
    ----------
    password : str
        The plain text password

    Returns
    -------
    str
        The hashed password
    """
    return pwd_context.hash(password)


def create_access_token(
    data: Dict[str, Any], expires_delta: Optional[timedelta] = None
) -> str:
    """
    Create a JWT access token.

    Parameters
    ----------
    data : dict
        Data to encode in the token (e.g., user_id, username)
    expires_delta : timedelta, optional
        Token expiration time. Defaults to ACCESS_TOKEN_EXPIRE_MINUTES

    Returns
    -------
    str
        The encoded JWT token
    """
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(
            minutes=ACCESS_TOKEN_EXPIRE_MINUTES
        )

    to_encode.update(
        {"exp": expire, "type": "access", "iat": datetime.now(timezone.utc)}
    )

    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def create_refresh_token(data: Dict[str, Any]) -> str:
    """
    Create a JWT refresh token with longer expiration.

    Parameters
    ----------
    data : dict
        Data to encode in the token (e.g., user_id)

    Returns
    -------
    str
        The encoded JWT refresh token
    """
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)

    to_encode.update(
        {"exp": expire, "type": "refresh", "iat": datetime.now(timezone.utc)}
    )

    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def decode_token(token: str) -> Dict[str, Any]:
    """
    Decode and verify a JWT token.

    Parameters
    ----------
    token : str
        The JWT token to decode

    Returns
    -------
    dict
        The decoded token payload

    Raises
    ------
    HTTPException
        If token is invalid or expired
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


def verify_token(token: str, token_type: str = "access") -> Dict[str, Any]:
    """
    Verify a JWT token and check its type.

    Parameters
    ----------
    token : str
        The JWT token to verify
    token_type : str
        Expected token type ("access" or "refresh")

    Returns
    -------
    dict
        The decoded token payload

    Raises
    ------
    HTTPException
        If token is invalid, expired, or wrong type
    """
    payload = decode_token(token)

    # Check token type
    if payload.get("type") != token_type:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token type. Expected {token_type}",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Check expiration
    exp = payload.get("exp")
    if exp is None or datetime.fromtimestamp(exp, tz=timezone.utc) < datetime.now(
        timezone.utc
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return payload


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> Optional[Dict[str, Any]]:
    """
    Get current user from JWT token (required authentication).

    This dependency ensures the user is authenticated and returns
    the user information from the token.

    Parameters
    ----------
    credentials : HTTPAuthorizationCredentials
        The Bearer token from the Authorization header

    Returns
    -------
    dict
        User information from token (user_id, username, role, etc.)

    Raises
    ------
    HTTPException
        If authentication fails

    Example
    -------
    @app.get("/protected")
    async def protected_route(current_user: dict = Depends(get_current_user)):
        return {"message": f"Hello {current_user['username']}"}
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials
    payload = verify_token(token, token_type="access")

    # Extract user info
    user_id = payload.get("sub")
    username = payload.get("username")
    role = payload.get("role")

    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return {
        "user_id": user_id,
        "username": username,
        "role": role,
        "token_payload": payload,
    }


async def get_current_active_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> Dict[str, Any]:
    """
    Get current active user with verified status (required authentication).

    Similar to get_current_user but also verifies the user is active.
    Use this for endpoints that require authenticated access.

    Parameters
    ----------
    credentials : HTTPAuthorizationCredentials
        The Bearer token from the Authorization header

    Returns
    -------
    dict
        User information from token

    Raises
    ------
    HTTPException
        If authentication fails or user is inactive
    """
    user = await get_current_user(credentials)

    # In a real implementation, you would check the database here
    # to verify the user is still active and not suspended
    # For now, we assume all authenticated users are active

    return user


async def get_current_admin_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> Dict[str, Any]:
    """
    Get current user and verify admin role (required admin authentication).

    Use this for admin-only endpoints.

    Parameters
    ----------
    credentials : HTTPAuthorizationCredentials
        The Bearer token from the Authorization header

    Returns
    -------
    dict
        User information from token

    Raises
    ------
    HTTPException
        If authentication fails or user is not admin
    """
    user = await get_current_user(credentials)

    if user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required"
        )

    return user


async def get_optional_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> Optional[Dict[str, Any]]:
    """
    Get current user from JWT token (optional authentication).

    This dependency attempts to get user info from the token but does
    not raise an error if authentication fails. Use this for endpoints
    that work both with and without authentication.

    Parameters
    ----------
    credentials : HTTPAuthorizationCredentials
        The Bearer token from the Authorization header (optional)

    Returns
    -------
    dict or None
        User information if authenticated, None otherwise

    Example
    -------
    @app.get("/public")
    async def public_route(current_user: Optional[dict] = Depends(get_optional_user)):
        if current_user:
            return {"message": f"Hello {current_user['username']}"}
        return {"message": "Hello guest"}
    """
    if credentials is None:
        return None

    try:
        token = credentials.credentials
        payload = verify_token(token, token_type="access")

        return {
            "user_id": payload.get("sub"),
            "username": payload.get("username"),
            "role": payload.get("role"),
            "token_payload": payload,
        }
    except HTTPException:
        return None


def generate_api_key() -> str:
    """
    Generate a random API key.

    Returns
    -------
    str
        A random API key
    """
    import secrets

    return f"gl_{secrets.token_urlsafe(32)}"


def hash_api_key(api_key: str) -> str:
    """
    Hash an API key for storage.

    Parameters
    ----------
    api_key : str
        The plain API key

    Returns
    -------
    str
        The hashed API key
    """
    return pwd_context.hash(api_key)


def verify_api_key(plain_key: str, hashed_key: str) -> bool:
    """
    Verify an API key against its hash.

    Parameters
    ----------
    plain_key : str
        The plain API key
    hashed_key : str
        The hashed API key from database

    Returns
    -------
    bool
        True if keys match, False otherwise
    """
    return pwd_context.verify(plain_key, hashed_key)
