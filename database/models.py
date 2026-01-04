"""
Database models for Gravitational Lensing Analysis Platform

Models:
- User: User accounts and authentication
- Analysis: Analysis sessions and results
- Job: Processing jobs and status
- Result: Stored analysis results
- ApiKey: API key management

Author: Phase 12 Implementation
Date: October 2025
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, ForeignKey, JSON, Text, Enum as SQLEnum
from sqlalchemy.orm import DeclarativeBase, relationship
from sqlalchemy.sql import func
from datetime import datetime
import enum


# SQLAlchemy 2.0 style declarative base
class Base(DeclarativeBase):
    """Base class for all database models (SQLAlchemy 2.0 pattern)."""
    pass


# ============================================================================
# Enums
# ============================================================================

class UserRole(str, enum.Enum):
    """User role enumeration"""
    ADMIN = "admin"
    RESEARCHER = "researcher"
    USER = "user"
    GUEST = "guest"


class JobStatus(str, enum.Enum):
    """Job status enumeration"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AnalysisType(str, enum.Enum):
    """Analysis type enumeration"""
    SYNTHETIC = "synthetic"
    REAL_DATA = "real_data"
    INFERENCE = "inference"
    BATCH = "batch"
    CUSTOM = "custom"


# ============================================================================
# Models
# ============================================================================

class User(Base):
    """
    User account model
    
    Stores user information, authentication, and preferences
    """
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255))
    
    # Role and permissions
    role = Column(SQLEnum(UserRole), default=UserRole.USER, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    
    # OAuth
    oauth_provider = Column(String(50))  # google, github, etc.
    oauth_id = Column(String(255))
    
    # Preferences
    preferences = Column(JSON, default={})
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    last_login = Column(DateTime(timezone=True))
    
    # Relationships
    analyses = relationship("Analysis", back_populates="user", cascade="all, delete-orphan")
    jobs = relationship("Job", back_populates="user", cascade="all, delete-orphan")
    api_keys = relationship("ApiKey", back_populates="user", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<User(id={self.id}, email={self.email}, role={self.role})>"


class ApiKey(Base):
    """
    API key management
    
    Stores API keys for programmatic access
    """
    __tablename__ = "api_keys"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    key_hash = Column(String(255), unique=True, index=True, nullable=False)
    key_prefix = Column(String(10), nullable=False)  # First 8 chars for display
    name = Column(String(100))
    
    # Permissions
    scopes = Column(JSON, default=["read"])  # read, write, admin
    
    # Status
    is_active = Column(Boolean, default=True, nullable=False)
    last_used = Column(DateTime(timezone=True))
    
    # Rate limiting
    rate_limit = Column(Integer, default=1000)  # requests per hour
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    expires_at = Column(DateTime(timezone=True))
    
    # Relationships
    user = relationship("User", back_populates="api_keys")
    
    def __repr__(self):
        return f"<ApiKey(id={self.id}, prefix={self.key_prefix}, user_id={self.user_id})>"


class Analysis(Base):
    """
    Analysis session model
    
    Stores analysis configurations and metadata
    """
    __tablename__ = "analyses"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Analysis details
    name = Column(String(255), nullable=False)
    description = Column(Text)
    type = Column(SQLEnum(AnalysisType), nullable=False)
    
    # Configuration
    config = Column(JSON, nullable=False)  # Analysis parameters
    
    # Status
    status = Column(SQLEnum(JobStatus), default=JobStatus.PENDING, nullable=False)
    progress = Column(Float, default=0.0)
    
    # Sharing
    is_public = Column(Boolean, default=False, nullable=False)
    shared_with = Column(JSON, default=[])  # List of user IDs
    
    # Tags for organization
    tags = Column(JSON, default=[])
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    completed_at = Column(DateTime(timezone=True))
    
    # Relationships
    user = relationship("User", back_populates="analyses")
    results = relationship("Result", back_populates="analysis", cascade="all, delete-orphan")
    jobs = relationship("Job", back_populates="analysis", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Analysis(id={self.id}, name={self.name}, type={self.type})>"


class Job(Base):
    """
    Processing job model
    
    Tracks individual computation jobs
    """
    __tablename__ = "jobs"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    analysis_id = Column(Integer, ForeignKey("analyses.id"))
    
    # Job details
    job_type = Column(String(50), nullable=False)  # synthetic, inference, etc.
    job_id = Column(String(100), unique=True, index=True, nullable=False)  # UUID
    
    # Configuration
    parameters = Column(JSON, nullable=False)
    
    # Status
    status = Column(SQLEnum(JobStatus), default=JobStatus.PENDING, nullable=False)
    progress = Column(Float, default=0.0)
    
    # Error handling
    error_message = Column(Text)
    retry_count = Column(Integer, default=0)
    
    # Performance metrics
    duration_seconds = Column(Float)
    memory_mb = Column(Float)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    
    # Relationships
    user = relationship("User", back_populates="jobs")
    analysis = relationship("Analysis", back_populates="jobs")
    results = relationship("Result", back_populates="job", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Job(id={self.id}, job_id={self.job_id}, status={self.status})>"


class Result(Base):
    """
    Analysis result model
    
    Stores computation results and outputs
    """
    __tablename__ = "results"
    
    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(Integer, ForeignKey("jobs.id"), nullable=False)
    analysis_id = Column(Integer, ForeignKey("analyses.id"))
    
    # Result data
    result_type = Column(String(50), nullable=False)  # convergence_map, inference, etc.
    data = Column(JSON, nullable=False)  # Main result data
    result_metadata = Column(JSON, default={})  # Renamed from metadata
    
    # File storage
    file_path = Column(String(500))  # Path to stored file (e.g., .npy, .fits)
    file_size_mb = Column(Float)
    
    # Quality metrics
    confidence_score = Column(Float)
    uncertainty = Column(JSON)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Relationships
    job = relationship("Job", back_populates="results")
    analysis = relationship("Analysis", back_populates="results")
    
    def __repr__(self):
        return f"<Result(id={self.id}, type={self.result_type}, job_id={self.job_id})>"


class Notification(Base):
    """
    User notification model
    
    Stores notifications for users (job completion, errors, etc.)
    """
    __tablename__ = "notifications"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Notification details
    title = Column(String(255), nullable=False)
    message = Column(Text, nullable=False)
    type = Column(String(50))  # success, error, warning, info
    
    # Related entities
    job_id = Column(Integer, ForeignKey("jobs.id"))
    analysis_id = Column(Integer, ForeignKey("analyses.id"))
    
    # Status
    is_read = Column(Boolean, default=False, nullable=False)
    read_at = Column(DateTime(timezone=True))
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    def __repr__(self):
        return f"<Notification(id={self.id}, title={self.title}, read={self.is_read})>"


class AuditLog(Base):
    """
    Audit log model
    
    Tracks all important actions for security and debugging
    """
    __tablename__ = "audit_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    
    # Action details
    action = Column(String(100), nullable=False)  # create, update, delete, login, etc.
    resource_type = Column(String(50))  # user, analysis, job, etc.
    resource_id = Column(Integer)
    
    # Request details
    ip_address = Column(String(50))
    user_agent = Column(String(500))
    
    # Change tracking
    changes = Column(JSON)  # Before/after values
    
    # Timestamp
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    def __repr__(self):
        return f"<AuditLog(id={self.id}, action={self.action}, user_id={self.user_id})>"


class SharedLink(Base):
    """
    Shared link model
    
    Allows sharing analyses via public links
    """
    __tablename__ = "shared_links"
    
    id = Column(Integer, primary_key=True, index=True)
    analysis_id = Column(Integer, ForeignKey("analyses.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Link details
    token = Column(String(100), unique=True, index=True, nullable=False)
    
    # Permissions
    can_view = Column(Boolean, default=True, nullable=False)
    can_edit = Column(Boolean, default=False, nullable=False)
    can_download = Column(Boolean, default=True, nullable=False)
    
    # Access control
    max_uses = Column(Integer)  # null = unlimited
    use_count = Column(Integer, default=0)
    
    # Status
    is_active = Column(Boolean, default=True, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    expires_at = Column(DateTime(timezone=True))
    last_accessed = Column(DateTime(timezone=True))
    
    def __repr__(self):
        return f"<SharedLink(id={self.id}, token={self.token[:8]}..., analysis_id={self.analysis_id})>"


# ============================================================================
# Helper Functions
# ============================================================================

def create_tables(engine):
    """Create all database tables"""
    Base.metadata.create_all(bind=engine)


def drop_tables(engine):
    """Drop all database tables (use with caution!)"""
    Base.metadata.drop_all(bind=engine)
