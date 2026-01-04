"""
Authentication routes for FastAPI

Provides login, registration, token refresh, and user management endpoints

Author: Phase 12 Implementation
Date: October 2025
SECURITY: P1 fixes applied November 2025 (rate limiting)
"""

from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.orm import Session
from typing import Optional, List
from datetime import timedelta
from slowapi import Limiter
from slowapi.util import get_remote_address

from pathlib import Path

from database import (
    get_db,
    authenticate_user,
    create_access_token,
    create_refresh_token,
    decode_token,
    get_current_user,
    get_current_active_user,
    get_current_admin_user,
    create_user,
    get_user,
    get_user_by_email,
    get_user_by_username,
    get_users,
    update_user,
    delete_user,
    create_api_key,
    revoke_api_key,
    User,
    UserRole,
    create_audit_log,
    ACCESS_TOKEN_EXPIRE_MINUTES
)


# ============================================================================
# Router Setup
# ============================================================================

router = APIRouter(prefix="/api/v1/auth", tags=["Authentication"])

# P1 SECURITY FIX: Rate limiter to prevent brute-force attacks
limiter = Limiter(key_func=get_remote_address)


# ============================================================================
# Request/Response Models
# ============================================================================

class UserRegister(BaseModel):
    """User registration request"""
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8)
    full_name: Optional[str] = None


class UserLogin(BaseModel):
    """User login request"""
    username: str
    password: str


class TokenResponse(BaseModel):
    """Token response"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    """User information response"""
    id: int
    email: str
    username: str
    full_name: Optional[str]
    role: str
    is_active: bool
    is_verified: bool
    created_at: str
    
    class Config:
        from_attributes = True


class UserUpdate(BaseModel):
    """User update request"""
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    password: Optional[str] = Field(None, min_length=8)


class ApiKeyCreate(BaseModel):
    """API key creation request"""
    name: str = Field(..., min_length=1, max_length=100)
    scopes: List[str] = Field(default=["read"], description="Permission scopes")
    expires_days: Optional[int] = Field(None, ge=1, le=365)


class ApiKeyResponse(BaseModel):
    """API key response"""
    id: int
    key_prefix: str
    name: str
    api_key: Optional[str] = None  # Only returned on creation
    scopes: List[str]
    created_at: str
    expires_at: Optional[str]
    
    class Config:
        from_attributes = True


# ============================================================================
# Authentication Endpoints
# ============================================================================

@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(user_data: UserRegister, db: Session = Depends(get_db)):
    """
    Register a new user
    
    Creates a new user account with the provided credentials.
    """
    # Check if email already exists
    existing_user = get_user_by_email(db, user_data.email)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Check if username already exists
    existing_user = get_user_by_username(db, user_data.username)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already taken"
        )
    
    # Create user
    user = create_user(
        db=db,
        email=user_data.email,
        username=user_data.username,
        password=user_data.password,
        full_name=user_data.full_name,
        role=UserRole.USER
    )
    
    # Create audit log
    create_audit_log(
        db=db,
        user_id=user.id,
        action="register",
        resource_type="user",
        resource_id=user.id
    )
    
    return user


@router.post("/login", response_model=TokenResponse)
@limiter.limit("5/minute")  # P1 SECURITY FIX: Limit login attempts to prevent brute-force
async def login(
    request: Request,  # Required for rate limiting
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    """
    Login with username/email and password
    
    Returns access and refresh tokens for authentication.

    SECURITY: Rate limited to 5 attempts per minute per IP address to prevent brute-force attacks.
    """
    # Authenticate user
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create tokens
    access_token = create_access_token(
        data={"sub": user.id, "username": user.username, "role": user.role}
    )
    refresh_token = create_refresh_token(
        data={"sub": user.id}
    )
    
    # Update last login
    update_user(db, user.id, last_login=user.last_login)
    
    # Create audit log
    create_audit_log(
        db=db,
        user_id=user.id,
        action="login",
        resource_type="user",
        resource_id=user.id
    )
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer"
    }


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(refresh_token: str, db: Session = Depends(get_db)):
    """
    Refresh access token using refresh token
    
    Generates a new access token when the current one expires.
    """
    # Decode refresh token
    payload = decode_token(refresh_token)
    
    # Check token type
    if payload.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type"
        )
    
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )
    
    # Get user
    user = get_user(db, user_id)
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive"
        )
    
    # Create new tokens
    new_access_token = create_access_token(
        data={"sub": user.id, "username": user.username, "role": user.role}
    )
    new_refresh_token = create_refresh_token(
        data={"sub": user.id}
    )
    
    return {
        "access_token": new_access_token,
        "refresh_token": new_refresh_token,
        "token_type": "bearer"
    }


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: User = Depends(get_current_active_user)):
    """
    Get current user information
    
    Returns details about the authenticated user.
    """
    return current_user


@router.put("/me", response_model=UserResponse)
async def update_current_user(
    user_data: UserUpdate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Update current user information
    
    Allows users to update their profile details.
    """
    update_data = {}
    
    if user_data.email:
        # Check if email is already used
        existing = get_user_by_email(db, user_data.email)
        if existing and existing.id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already in use"
            )
        update_data["email"] = user_data.email
    
    if user_data.full_name is not None:
        update_data["full_name"] = user_data.full_name
    
    if user_data.password:
        from database import hash_password
        update_data["hashed_password"] = hash_password(user_data.password)
    
    # Update user
    updated_user = update_user(db, current_user.id, **update_data)
    
    # Create audit log
    create_audit_log(
        db=db,
        user_id=current_user.id,
        action="update_profile",
        resource_type="user",
        resource_id=current_user.id,
        changes=update_data
    )
    
    return updated_user


# ============================================================================
# API Key Management
# ============================================================================

@router.post("/api-keys", response_model=ApiKeyResponse, status_code=status.HTTP_201_CREATED)
async def create_user_api_key(
    key_data: ApiKeyCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Create a new API key
    
    Generates an API key for programmatic access to the API.
    """
    # Create API key
    api_key_obj, plain_key = create_api_key(
        db=db,
        user_id=current_user.id,
        name=key_data.name,
        scopes=key_data.scopes,
        expires_days=key_data.expires_days
    )
    
    # Create audit log
    create_audit_log(
        db=db,
        user_id=current_user.id,
        action="create_api_key",
        resource_type="api_key",
        resource_id=api_key_obj.id
    )
    
    # Return with plain API key (only time it's visible)
    response = ApiKeyResponse.from_orm(api_key_obj)
    response.api_key = plain_key
    
    return response


@router.delete("/api-keys/{api_key_id}", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_user_api_key(
    api_key_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Revoke an API key
    
    Deactivates the specified API key.
    """
    success = revoke_api_key(db, api_key_id, current_user.id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found"
        )
    
    # Create audit log
    create_audit_log(
        db=db,
        user_id=current_user.id,
        action="revoke_api_key",
        resource_type="api_key",
        resource_id=api_key_id
    )
    
    return None


# ============================================================================
# Admin Endpoints
# ============================================================================

@router.get("/users", response_model=List[UserResponse])
async def list_users(
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """
    List all users (Admin only)
    
    Returns a paginated list of all users in the system.
    """
    users = get_users(db, skip=skip, limit=limit)
    return users


@router.get("/users/{user_id}", response_model=UserResponse)
async def get_user_by_id(
    user_id: int,
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """
    Get user by ID (Admin only)
    
    Returns detailed information about a specific user.
    """
    user = get_user(db, user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    return user


@router.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user_by_id(
    user_id: int,
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """
    Delete user (Admin only)
    
    Soft-deletes a user account (deactivates it).
    """
    if user_id == current_user.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete your own account"
        )
    
    success = delete_user(db, user_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Create audit log
    create_audit_log(
        db=db,
        user_id=current_user.id,
        action="delete_user",
        resource_type="user",
        resource_id=user_id
    )
    
    return None
