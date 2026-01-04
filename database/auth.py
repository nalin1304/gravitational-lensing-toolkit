"""
Authentication and authorization utilities

Provides JWT token handling, password hashing, and user authentication

Author: Phase 12 Implementation
Date: October 2025
"""

from datetime import datetime, timedelta
from typing import Optional, Dict
from passlib.context import CryptContext
from jose import JWTError, jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, APIKeyHeader
from sqlalchemy.orm import Session

import os
import secrets

from .database import get_db
from .models import User, ApiKey, UserRole


# ============================================================================
# Configuration
# ============================================================================

# JWT Configuration
SECRET_KEY = os.getenv("SECRET_KEY", secrets.token_urlsafe(32))
ALGORITHM = "HS256"
# P0 SECURITY FIX: Reduced from 30 days (43200 minutes) to 15 minutes
# This limits the window of opportunity for token theft/misuse
ACCESS_TOKEN_EXPIRE_MINUTES = 15
REFRESH_TOKEN_EXPIRE_DAYS = 7

# Password hashing
pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto",
    bcrypt__rounds=12  # Reduce rounds for testing
)

# OAuth2 scheme
# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/v1/auth/login")
oauth2_scheme_optional = OAuth2PasswordBearer(tokenUrl="api/v1/auth/login", auto_error=False)

# API Key header
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


# ============================================================================
# Password Utilities
# ============================================================================

def hash_password(password: str) -> str:
    """
    Hash a password using bcrypt
    
    Args:
        password: Plain text password
        
    Returns:
        str: Hashed password
    """
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a password against a hash
    
    Args:
        plain_password: Plain text password
        hashed_password: Hashed password to compare against
        
    Returns:
        bool: True if password matches
    """
    return pwd_context.verify(plain_password, hashed_password)


# ============================================================================
# JWT Token Utilities
# ============================================================================

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a JWT access token
    
    Args:
        data: Payload data to encode
        expires_delta: Token expiration time
        
    Returns:
        str: Encoded JWT token
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire, "type": "access"})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def create_refresh_token(data: dict) -> str:
    """
    Create a JWT refresh token
    
    Args:
        data: Payload data to encode
        
    Returns:
        str: Encoded JWT refresh token
    """
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire, "type": "refresh"})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def decode_token(token: str) -> Dict:
    """
    Decode and validate a JWT token
    
    Args:
        token: JWT token to decode
        
    Returns:
        dict: Decoded token payload
        
    Raises:
        HTTPException: If token is invalid or expired
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Could not validate credentials: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ============================================================================
# User Authentication
# ============================================================================

def authenticate_user(db: Session, username: str, password: str) -> Optional[User]:
    """
    Authenticate a user by username and password
    
    Args:
        db: Database session
        username: Username or email
        password: Plain text password
        
    Returns:
        User: Authenticated user object or None
    """
    # Try username first
    user = db.query(User).filter(User.username == username).first()
    
    # Try email if username not found
    if not user:
        user = db.query(User).filter(User.email == username).first()
    
    # Verify password
    if not user or not verify_password(password, user.hashed_password):
        return None
    
    # Check if user is active
    if not user.is_active:
        return None
    
    return user


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    api_key: Optional[str] = Depends(api_key_header),
    db: Session = Depends(get_db)
) -> User:
    """
    Get current authenticated user from JWT token or API key
    
    Args:
        token: JWT access token
        api_key: API key header
        db: Database session
        
    Returns:
        User: Current authenticated user
        
    Raises:
        HTTPException: If authentication fails
    """
    # Try API key first
    if api_key:
        return await get_user_from_api_key(api_key, db)
    
    # Try JWT token
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Decode token
    payload = decode_token(token)
    user_id: int = payload.get("sub")
    
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Get user from database
    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Check if user is active
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive"
        )
    
    # Update last login
    user.last_login = datetime.utcnow()
    db.commit()
    
    return user


async def get_user_from_api_key(api_key: str, db: Session) -> User:
    """
    Get user from API key
    
    Args:
        api_key: API key
        db: Database session
        
    Returns:
        User: User associated with API key
        
    Raises:
        HTTPException: If API key is invalid
        
    Note:
        API keys are stored as bcrypt hashes. Since bcrypt includes a random salt,
        we cannot simply hash the input and compare. Instead, we iterate over
        active API keys and use verify_password() for secure comparison.
    """
    # Get all active API keys (we need to verify against each one)
    # In production with many API keys, consider a different hashing scheme (e.g., SHA-256)
    # that allows direct lookup, or store a prefix/identifier alongside the hash.
    active_api_keys = db.query(ApiKey).filter(
        ApiKey.is_active == True
    ).all()
    
    # Find matching API key using secure verification
    api_key_obj = None
    for key in active_api_keys:
        if verify_password(api_key, key.key_hash):
            api_key_obj = key
            break
    
    if not api_key_obj:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    
    # Check expiration
    if api_key_obj.expires_at and api_key_obj.expires_at < datetime.utcnow():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key has expired"
        )
    
    # Get user
    user = db.query(User).filter(User.id == api_key_obj.user_id).first()
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key or user inactive"
        )
    
    # Update last used
    api_key_obj.last_used = datetime.utcnow()
    db.commit()
    
    return user


async def get_optional_user(
    token: Optional[str] = Depends(oauth2_scheme_optional),
    api_key: Optional[str] = Depends(api_key_header),
    db: Session = Depends(get_db)
) -> Optional[User]:
    """
    Get current authenticated user (optional)
    
    Returns:
        User: Authenticated user or None
    """
    try:
        if api_key:
            return await get_user_from_api_key(api_key, db)
        
        if not token:
            return None
        
        payload = decode_token(token)
        user_id = payload.get("sub")
        
        if user_id is None:
            return None
        
        user = db.query(User).filter(User.id == user_id).first()
        if user and user.is_active:
            return user
            
        return None
    except Exception:
        return None


async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """
    Get current active user (convenience function)
    
    Args:
        current_user: Current user from get_current_user
        
    Returns:
        User: Current active user
        
    Raises:
        HTTPException: If user is inactive
    """
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user"
        )
    return current_user


async def get_current_admin_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """
    Get current user and verify admin role
    
    Args:
        current_user: Current user from get_current_user
        
    Returns:
        User: Current admin user
        
    Raises:
        HTTPException: If user is not admin
    """
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    return current_user


async def get_current_researcher_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """
    Get current user and verify researcher or admin role
    
    Args:
        current_user: Current user from get_current_user
        
    Returns:
        User: Current researcher or admin user
        
    Raises:
        HTTPException: If user is not researcher or admin
    """
    if current_user.role not in [UserRole.RESEARCHER, UserRole.ADMIN]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Researcher or admin role required"
        )
    return current_user


# ============================================================================
# API Key Management
# ============================================================================

def generate_api_key() -> str:
    """
    Generate a new API key
    
    Returns:
        str: New API key (32 bytes, URL-safe)
    """
    return secrets.token_urlsafe(32)


def create_api_key(
    db: Session,
    user_id: int,
    name: str,
    scopes: list = None,
    expires_days: Optional[int] = None
) -> tuple[ApiKey, str]:
    """
    Create a new API key for a user
    
    Args:
        db: Database session
        user_id: User ID
        name: API key name
        scopes: List of permission scopes
        expires_days: Number of days until expiration
        
    Returns:
        tuple: (ApiKey object, plain text API key)
    """
    # Generate API key
    plain_key = generate_api_key()
    key_hash = hash_password(plain_key)
    key_prefix = plain_key[:8]
    
    # Set expiration
    expires_at = None
    if expires_days:
        expires_at = datetime.utcnow() + timedelta(days=expires_days)
    
    # Create API key object
    api_key = ApiKey(
        user_id=user_id,
        key_hash=key_hash,
        key_prefix=key_prefix,
        name=name,
        scopes=scopes or ["read"],
        expires_at=expires_at
    )
    
    db.add(api_key)
    db.commit()
    db.refresh(api_key)
    
    return api_key, plain_key


def revoke_api_key(db: Session, api_key_id: int, user_id: int) -> bool:
    """
    Revoke (deactivate) an API key
    
    Args:
        db: Database session
        api_key_id: API key ID
        user_id: User ID (for authorization)
        
    Returns:
        bool: True if revoked successfully
    """
    api_key = db.query(ApiKey).filter(
        ApiKey.id == api_key_id,
        ApiKey.user_id == user_id
    ).first()
    
    if not api_key:
        return False
    
    api_key.is_active = False
    db.commit()
    return True


# ============================================================================
# OAuth2 Utilities (Placeholder for future implementation)
# ============================================================================

def verify_oauth_token(provider: str, token: str) -> Optional[Dict]:
    """
    Verify OAuth token from provider (Google, GitHub, etc.)
    
    Args:
        provider: OAuth provider name ('google', 'github')
        token: OAuth access token or ID token
        
    Returns:
        dict: User info from provider or None
        Contains: email, name, provider, provider_id
        
    Examples:
        >>> user_info = verify_oauth_token('google', google_token)
        >>> if user_info:
        ...     print(f"Authenticated: {user_info['email']}")
    
    Notes:
        Requires environment variables:
        - GOOGLE_CLIENT_ID: For Google OAuth
        - GITHUB_OAUTH_TOKEN: For GitHub OAuth (optional)
    """
    provider = provider.lower()
    
    try:
        if provider == 'google':
            return _verify_google_token(token)
        elif provider == 'github':
            return _verify_github_token(token)
        else:
            raise ValueError(f"Unsupported OAuth provider: {provider}")
    except Exception as e:
        print(f"OAuth verification failed: {e}")
        return None


def _verify_google_token(token: str) -> Optional[Dict]:
    """Verify Google OAuth2 token."""
    try:
        from google.oauth2 import id_token
        from google.auth.transport import requests
        import os
        
        client_id = os.getenv('GOOGLE_CLIENT_ID')
        if not client_id:
            raise ValueError("GOOGLE_CLIENT_ID environment variable not set")
        
        idinfo = id_token.verify_oauth2_token(
            token, requests.Request(), client_id
        )
        
        return {
            'email': idinfo.get('email'),
            'name': idinfo.get('name'),
            'provider': 'google',
            'provider_id': idinfo.get('sub'),
            'email_verified': idinfo.get('email_verified', False),
            'picture': idinfo.get('picture')
        }
    except ValueError as e:
        print(f"Google token verification failed: {e}")
        return None
    except ImportError:
        raise ImportError("google-auth required: pip install google-auth")


def _verify_github_token(token: str) -> Optional[Dict]:
    """Verify GitHub OAuth token."""
    try:
        from github import Github, GithubException
        
        g = Github(token)
        user = g.get_user()
        
        try:
            return {
                'email': user.email or f"{user.login}@github.com",
                'name': user.name or user.login,
                'provider': 'github',
                'provider_id': str(user.id),
                'username': user.login,
                'avatar_url': user.avatar_url,
                'email_verified': True
            }
        except GithubException as e:
            print(f"GitHub API error: {e}")
            return None
    except ImportError:
        raise ImportError("PyGithub required: pip install PyGithub")
    except Exception as e:
        print(f"GitHub token verification failed: {e}")
        return None
