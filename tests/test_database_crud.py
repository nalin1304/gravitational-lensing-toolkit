"""
Phase 12 Tests - Database, Authentication, and User Management

Tests for:
- Database models and CRUD operations
- Authentication (JWT, API keys)
- User management
- Analysis and job tracking

Author: Phase 12 Implementation
Date: October 2025
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from database.models import Base, UserRole, JobStatus, AnalysisType
from database.database import get_db
from database import (
    create_user,
    get_user,
    authenticate_user,
    create_access_token,
    create_api_key,
    create_analysis,
    create_job,
    create_result
)
from api.main import app


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def test_db():
    """Create in-memory SQLite database for testing"""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False}
    )
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture
def client(test_db):
    """Create test client with database override"""
    def override_get_db():
        try:
            yield test_db
        finally:
            pass
    
    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()


@pytest.fixture
def test_user(test_db):
    """Create a test user"""
    user = create_user(
        db=test_db,
        email="test@example.com",
        username="testuser",
        password="testpass123",
        full_name="Test User",
        role=UserRole.USER
    )
    return user


@pytest.fixture
def test_admin(test_db):
    """Create a test admin user"""
    admin = create_user(
        db=test_db,
        email="admin@example.com",
        username="admin",
        password="adminpass123",
        full_name="Admin User",
        role=UserRole.ADMIN
    )
    return admin


@pytest.fixture
def auth_headers(test_user):
    """Create authorization headers for test user"""
    token = create_access_token(
        data={"sub": test_user.id, "username": test_user.username, "role": test_user.role}
    )
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def admin_headers(test_admin):
    """Create authorization headers for admin user"""
    token = create_access_token(
        data={"sub": test_admin.id, "username": test_admin.username, "role": test_admin.role}
    )
    return {"Authorization": f"Bearer {token}"}


# ============================================================================
# Database Model Tests
# ============================================================================

class TestDatabaseModels:
    """Test database models"""
    
    def test_create_user(self, test_db):
        """Test creating a user"""
        user = create_user(
            db=test_db,
            email="newuser@example.com",
            username="newuser",
            password="password123",
            full_name="New User"
        )
        
        assert user.id is not None
        assert user.email == "newuser@example.com"
        assert user.username == "newuser"
        assert user.role == UserRole.USER
        assert user.is_active is True
        assert user.hashed_password != "password123"
    
    def test_user_unique_email(self, test_db, test_user):
        """Test that email must be unique"""
        with pytest.raises(Exception):
            create_user(
                db=test_db,
                email=test_user.email,
                username="different",
                password="pass123"
            )
    
    def test_user_unique_username(self, test_db, test_user):
        """Test that username must be unique"""
        with pytest.raises(Exception):
            create_user(
                db=test_db,
                email="different@example.com",
                username=test_user.username,
                password="pass123"
            )
    
    def test_create_analysis(self, test_db, test_user):
        """Test creating an analysis"""
        analysis = create_analysis(
            db=test_db,
            user_id=test_user.id,
            name="Test Analysis",
            type=AnalysisType.SYNTHETIC,
            config={"mass": 1e12, "grid_size": 64},
            description="Test description"
        )
        
        assert analysis.id is not None
        assert analysis.user_id == test_user.id
        assert analysis.name == "Test Analysis"
        assert analysis.type == AnalysisType.SYNTHETIC
        assert analysis.status == JobStatus.PENDING
    
    def test_create_job(self, test_db, test_user):
        """Test creating a job"""
        job = create_job(
            db=test_db,
            user_id=test_user.id,
            job_type="synthetic",
            job_id="test-job-123",
            parameters={"mass": 1e12}
        )
        
        assert job.id is not None
        assert job.user_id == test_user.id
        assert job.job_id == "test-job-123"
        assert job.status == JobStatus.PENDING


# ============================================================================
# Authentication Tests
# ============================================================================

class TestAuthentication:
    """Test authentication functionality"""
    
    def test_authenticate_valid_user(self, test_db, test_user):
        """Test authentication with valid credentials"""
        user = authenticate_user(test_db, "testuser", "testpass123")
        assert user is not None
        assert user.id == test_user.id
    
    def test_authenticate_with_email(self, test_db, test_user):
        """Test authentication with email"""
        user = authenticate_user(test_db, "test@example.com", "testpass123")
        assert user is not None
        assert user.id == test_user.id
    
    def test_authenticate_invalid_password(self, test_db, test_user):
        """Test authentication with invalid password"""
        user = authenticate_user(test_db, "testuser", "wrongpassword")
        assert user is None
    
    def test_authenticate_nonexistent_user(self, test_db):
        """Test authentication with nonexistent user"""
        user = authenticate_user(test_db, "nonexistent", "password")
        assert user is None
    
    def test_create_access_token(self, test_user):
        """Test access token creation"""
        token = create_access_token(
            data={"sub": test_user.id, "username": test_user.username}
        )
        assert token is not None
        assert isinstance(token, str)
        assert len(token) > 0
    
    def test_api_key_creation(self, test_db, test_user):
        """Test API key creation"""
        api_key_obj, plain_key = create_api_key(
            db=test_db,
            user_id=test_user.id,
            name="Test API Key",
            scopes=["read", "write"]
        )
        
        assert api_key_obj.id is not None
        assert api_key_obj.user_id == test_user.id
        assert api_key_obj.name == "Test API Key"
        assert api_key_obj.scopes == ["read", "write"]
        assert plain_key is not None
        assert len(plain_key) > 0


# ============================================================================
# API Endpoint Tests
# ============================================================================

class TestAuthEndpoints:
    """Test authentication API endpoints"""
    
    def test_register_user(self, client):
        """Test user registration"""
        response = client.post(
            "/api/v1/auth/register",
            json={
                "email": "newapi@example.com",
                "username": "newapi",
                "password": "securepass123",
                "full_name": "New API User"
            }
        )
        
        assert response.status_code == 201
        data = response.json()
        assert data["email"] == "newapi@example.com"
        assert data["username"] == "newapi"
        assert data["role"] == "user"
    
    def test_register_duplicate_email(self, client, test_user):
        """Test registration with duplicate email"""
        response = client.post(
            "/api/v1/auth/register",
            json={
                "email": test_user.email,
                "username": "different",
                "password": "password123"
            }
        )
        
        assert response.status_code == 400
        assert "already registered" in response.json()["detail"].lower()
    
    def test_login_success(self, client, test_user):
        """Test successful login"""
        response = client.post(
            "/api/v1/auth/login",
            data={
                "username": "testuser",
                "password": "testpass123"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"
    
    def test_login_invalid_credentials(self, client, test_user):
        """Test login with invalid credentials"""
        response = client.post(
            "/api/v1/auth/login",
            data={
                "username": "testuser",
                "password": "wrongpassword"
            }
        )
        
        assert response.status_code == 401
    
    def test_get_current_user(self, client, auth_headers):
        """Test getting current user info"""
        response = client.get(
            "/api/v1/auth/me",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["username"] == "testuser"
        assert data["email"] == "test@example.com"
    
    def test_get_current_user_unauthorized(self, client):
        """Test getting current user without auth"""
        response = client.get("/api/v1/auth/me")
        assert response.status_code == 401


class TestAnalysisEndpoints:
    """Test analysis API endpoints"""
    
    def test_create_analysis(self, client, auth_headers):
        """Test creating an analysis"""
        response = client.post(
            "/api/v1/analyses",
            headers=auth_headers,
            json={
                "name": "API Test Analysis",
                "description": "Test analysis from API",
                "type": "synthetic",
                "config": {"mass": 1e12, "grid_size": 64},
                "tags": ["test", "synthetic"]
            }
        )
        
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "API Test Analysis"
        assert data["type"] == "synthetic"
        assert data["status"] == "pending"
    
    def test_create_analysis_unauthorized(self, client):
        """Test creating analysis without auth"""
        response = client.post(
            "/api/v1/analyses",
            json={
                "name": "Test",
                "type": "synthetic",
                "config": {}
            }
        )
        assert response.status_code == 401
    
    def test_list_analyses(self, client, auth_headers, test_db, test_user):
        """Test listing analyses"""
        # Create test analysis
        create_analysis(
            db=test_db,
            user_id=test_user.id,
            name="Test Analysis",
            type=AnalysisType.SYNTHETIC,
            config={"test": "data"}
        )
        
        response = client.get(
            "/api/v1/analyses",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0
    
    def test_get_analysis(self, client, auth_headers, test_db, test_user):
        """Test getting specific analysis"""
        # Create test analysis
        analysis = create_analysis(
            db=test_db,
            user_id=test_user.id,
            name="Get Test",
            type=AnalysisType.SYNTHETIC,
            config={"test": "data"}
        )
        
        response = client.get(
            f"/api/v1/analyses/{analysis.id}",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == analysis.id
        assert data["name"] == "Get Test"
    
    def test_get_analysis_not_found(self, client, auth_headers):
        """Test getting nonexistent analysis"""
        response = client.get(
            "/api/v1/analyses/9999",
            headers=auth_headers
        )
        assert response.status_code == 404


class TestAdminEndpoints:
    """Test admin API endpoints"""
    
    def test_list_users_as_admin(self, client, admin_headers):
        """Test listing users as admin"""
        response = client.get(
            "/api/v1/auth/users",
            headers=admin_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    def test_list_users_as_regular_user(self, client, auth_headers):
        """Test listing users as regular user (should fail)"""
        response = client.get(
            "/api/v1/auth/users",
            headers=auth_headers
        )
        assert response.status_code == 403


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegrationWorkflows:
    """Test complete workflows"""
    
    def test_full_analysis_workflow(self, client, auth_headers, test_db, test_user):
        """Test complete analysis workflow"""
        # 1. Create analysis
        response = client.post(
            "/api/v1/analyses",
            headers=auth_headers,
            json={
                "name": "Workflow Test",
                "type": "synthetic",
                "config": {"mass": 1e12},
                "description": "Full workflow test"
            }
        )
        assert response.status_code == 201
        analysis_id = response.json()["id"]
        
        # 2. Get analysis
        response = client.get(
            f"/api/v1/analyses/{analysis_id}",
            headers=auth_headers
        )
        assert response.status_code == 200
        assert response.json()["name"] == "Workflow Test"
        
        # 3. Update analysis
        response = client.put(
            f"/api/v1/analyses/{analysis_id}",
            headers=auth_headers,
            json={"name": "Updated Workflow Test"}
        )
        assert response.status_code == 200
        assert response.json()["name"] == "Updated Workflow Test"
        
        # 4. List analyses
        response = client.get(
            "/api/v1/analyses",
            headers=auth_headers
        )
        assert response.status_code == 200
        assert len(response.json()) > 0
        
        # 5. Delete analysis
        response = client.delete(
            f"/api/v1/analyses/{analysis_id}",
            headers=auth_headers
        )
        assert response.status_code == 204


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
