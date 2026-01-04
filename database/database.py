"""
Database connection and session management

Provides database engine, session factory, and dependency injection for FastAPI

Author: Phase 12 Implementation
Date: October 2025
"""

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from typing import Generator
import os
from contextlib import contextmanager

from .models import Base


# ============================================================================
# Database Configuration
# ============================================================================

# Get database URL from environment or use SQLite for development
# SECURITY: Never hardcode credentials! Use environment variables.
# For local development, create a .env file with:
#   DATABASE_URL=postgresql://user:password@localhost:5432/lensing_db
# For production, set this in your deployment environment
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///./lensing_dev.db"  # Safe default for development
)

# For testing, allow SQLite
if "sqlite" in DATABASE_URL.lower():
    # SQLite-specific configuration
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=False  # Set to True for SQL debugging
    )
else:
    # PostgreSQL configuration
    engine = create_engine(
        DATABASE_URL,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,  # Verify connections before using
        pool_recycle=3600,  # Recycle connections after 1 hour
        echo=False  # Set to True for SQL debugging
    )


# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# ============================================================================
# Database Event Listeners
# ============================================================================

@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_conn, connection_record):
    """Enable foreign keys for SQLite"""
    if "sqlite" in str(engine.url):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


# ============================================================================
# Database Functions
# ============================================================================

def init_db():
    """Initialize database (create tables)"""
    Base.metadata.create_all(bind=engine)


def drop_db():
    """Drop all tables (use with caution!)"""
    Base.metadata.drop_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    """
    Dependency for FastAPI routes
    
    Usage:
        @app.get("/users")
        def get_users(db: Session = Depends(get_db)):
            return db.query(User).all()
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def get_db_context():
    """
    Context manager for database sessions
    
    Usage:
        with get_db_context() as db:
            user = db.query(User).first()
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def check_db_connection() -> bool:
    """
    Check if database connection is working
    
    Returns:
        bool: True if connection successful, False otherwise
    """
    try:
        with engine.connect() as conn:
            conn.execute("SELECT 1")
        return True
    except Exception as e:
        print(f"Database connection failed: {e}")
        return False


def get_db_info() -> dict:
    """
    Get database information
    
    Returns:
        dict: Database type, URL, and connection status
    """
    return {
        "type": engine.url.drivername,
        "database": engine.url.database,
        "host": engine.url.host,
        "port": engine.url.port,
        "connected": check_db_connection(),
        "pool_size": engine.pool.size() if hasattr(engine.pool, 'size') else None
    }


# ============================================================================
# Testing Utilities
# ============================================================================

def create_test_engine():
    """Create in-memory SQLite engine for testing"""
    test_engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool
    )
    Base.metadata.create_all(bind=test_engine)
    return test_engine


def create_test_session():
    """Create test database session"""
    test_engine = create_test_engine()
    TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    return TestSessionLocal()
