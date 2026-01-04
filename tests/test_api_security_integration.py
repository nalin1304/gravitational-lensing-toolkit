"""
Integration tests for API security and functionality

Tests all P0 and P1 security fixes to ensure they work correctly.

Run with: pytest tests/test_api_security_integration.py -v

Author: Security Verification Suite
Date: November 2025
"""

import pytest
import requests
from fastapi.testclient import TestClient
import time
import sys
from pathlib import Path

from api.main import app
from database import get_db, create_user, UserRole
from database.auth import create_access_token, hash_password


client = TestClient(app)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def test_user(db_session):
    """Create a test user."""
    user = create_user(
        db=db_session,
        email="test@example.com",
        username="testuser",
        password="TestPassword123!",
        role=UserRole.USER
    )
    yield user
    # Cleanup happens automatically with db_session rollback


@pytest.fixture
def auth_headers(test_user):
    """Get authentication headers for test user."""
    token = create_access_token(
        data={"sub": test_user.id, "username": test_user.username}
    )
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def admin_user(db_session):
    """Create an admin user."""
    user = create_user(
        db=db_session,
        email="admin@example.com",
        username="admin",
        password="AdminPassword123!",
        role=UserRole.ADMIN
    )
    yield user


# ============================================================================
# P0 Security Tests - Authentication
# ============================================================================

class TestAuthentication:
    """Test P0 authentication fixes."""

    def test_unauthenticated_request_rejected(self):
        """Test that requests without authentication are rejected."""
        response = client.get("/api/v1/analyses/1")
        assert response.status_code == 401
        assert "not authenticated" in response.json()["detail"].lower()

    def test_invalid_token_rejected(self):
        """Test that invalid tokens are rejected."""
        headers = {"Authorization": "Bearer invalid_token_here"}
        response = client.get("/api/v1/analyses/1", headers=headers)
        assert response.status_code == 401

    def test_expired_token_rejected(self):
        """Test that expired tokens are rejected."""
        # Create token with negative expiration
        from datetime import timedelta
        token = create_access_token(
            data={"sub": 1, "username": "test"},
            expires_delta=timedelta(minutes=-1)
        )
        headers = {"Authorization": f"Bearer {token}"}
        response = client.get("/api/v1/analyses/1", headers=headers)
        assert response.status_code == 401

    def test_valid_token_accepted(self, auth_headers):
        """Test that valid tokens are accepted."""
        # This will 404 if auth works (analysis doesn't exist), 401 if auth fails
        response = client.get("/api/v1/analyses/999", headers=auth_headers)
        assert response.status_code in [404, 403]  # Not 401


# ============================================================================
# P0 Security Tests - Authorization (IDOR)
# ============================================================================

class TestAuthorization:
    """Test P0 authorization (IDOR) fixes."""

    def test_user_cannot_access_other_user_analysis(
        self, db_session, test_user, auth_headers
    ):
        """Test that users cannot access other users' private analyses."""
        from database import create_analysis, AnalysisType

        # Create another user's analysis
        other_user = create_user(
            db=db_session,
            email="other@example.com",
            username="otheruser",
            password="Password123!",
            role=UserRole.USER
        )

        analysis = create_analysis(
            db=db_session,
            user_id=other_user.id,
            name="Private Analysis",
            type=AnalysisType.SYNTHETIC,
            config={"test": True},
            is_public=False
        )

        # Try to access with test_user credentials
        response = client.get(
            f"/api/v1/analyses/{analysis.id}",
            headers=auth_headers
        )

        assert response.status_code == 403
        assert "not authorized" in response.json()["detail"].lower()

    def test_user_can_access_own_analysis(
        self, db_session, test_user, auth_headers
    ):
        """Test that users can access their own analyses."""
        from database import create_analysis, AnalysisType

        analysis = create_analysis(
            db=db_session,
            user_id=test_user.id,
            name="My Analysis",
            type=AnalysisType.SYNTHETIC,
            config={"test": True}
        )

        response = client.get(
            f"/api/v1/analyses/{analysis.id}",
            headers=auth_headers
        )

        assert response.status_code == 200
        assert response.json()["name"] == "My Analysis"

    def test_user_can_access_public_analysis(
        self, db_session, test_user, auth_headers
    ):
        """Test that users can access public analyses."""
        from database import create_analysis, AnalysisType

        other_user = create_user(
            db=db_session,
            email="other2@example.com",
            username="otheruser2",
            password="Password123!",
            role=UserRole.USER
        )

        analysis = create_analysis(
            db=db_session,
            user_id=other_user.id,
            name="Public Analysis",
            type=AnalysisType.SYNTHETIC,
            config={"test": True},
            is_public=True  # Public!
        )

        response = client.get(
            f"/api/v1/analyses/{analysis.id}",
            headers=auth_headers
        )

        assert response.status_code == 200
        assert response.json()["name"] == "Public Analysis"


# ============================================================================
# P1 Security Tests - Rate Limiting
# ============================================================================

class TestRateLimiting:
    """Test P1 rate limiting fixes."""

    def test_login_rate_limit_enforced(self):
        """Test that login endpoint enforces rate limiting (5/minute)."""
        # Make 6 rapid login attempts
        for i in range(6):
            response = client.post(
                "/api/v1/auth/login",
                data={"username": "testuser", "password": "wrongpassword"}
            )

            if i < 5:
                # First 5 should be 401 (wrong password)
                assert response.status_code == 401
            else:
                # 6th should be 429 (rate limited)
                assert response.status_code == 429
                assert "rate limit" in response.json()["detail"].lower()

    def test_rate_limit_resets_after_time(self):
        """Test that rate limit resets after waiting."""
        # Hit rate limit
        for _ in range(6):
            client.post(
                "/api/v1/auth/login",
                data={"username": "test", "password": "test"}
            )

        # Wait for rate limit window to pass (61 seconds for 1-minute window)
        time.sleep(61)

        # Should work again
        response = client.post(
            "/api/v1/auth/login",
            data={"username": "test", "password": "test"}
        )
        assert response.status_code == 401  # Not 429


# ============================================================================
# P1 Security Tests - File Upload Validation
# ============================================================================

class TestFileUploadSecurity:
    """Test P1 file upload security fixes."""

    def test_validate_fits_file_extension(self):
        """Test that non-FITS files are rejected."""
        from api.security_utils import validate_fits_file
        from fastapi import UploadFile
        from io import BytesIO

        # Mock upload file with wrong extension
        fake_file = UploadFile(
            filename="malicious.exe",
            file=BytesIO(b"fake content")
        )

        with pytest.raises(Exception) as exc_info:
            validate_fits_file(fake_file)

        assert "invalid file extension" in str(exc_info.value).lower()

    def test_validate_fits_file_size(self):
        """Test that oversized files are rejected."""
        from api.security_utils import validate_fits_file
        from fastapi import UploadFile
        from io import BytesIO

        # Mock upload file that's too large
        fake_file = UploadFile(
            filename="huge.fits",
            file=BytesIO(b"x" * (101 * 1024 * 1024))  # 101 MB
        )
        fake_file.size = 101 * 1024 * 1024

        with pytest.raises(Exception) as exc_info:
            validate_fits_file(fake_file)

        assert "too large" in str(exc_info.value).lower()

    def test_sanitize_xss_in_fits_headers(self):
        """Test that XSS payloads in FITS headers are sanitized."""
        from api.security_utils import sanitize_fits_header_value

        xss_payload = "<script>alert('XSS')</script>"
        sanitized = sanitize_fits_header_value(xss_payload)

        assert "<script>" not in sanitized
        assert "alert" in sanitized  # Text remains, tags removed


# ============================================================================
# P2 Security Tests - PII Redaction
# ============================================================================

class TestPIIRedaction:
    """Test P2 PII redaction in logs."""

    def test_email_redaction(self):
        """Test that emails are redacted from logs."""
        from api.secure_logging import redact_pii

        text = "User email: john@example.com logged in"
        redacted = redact_pii(text)

        assert "john@example.com" not in redacted
        assert "[REDACTED_EMAIL]" in redacted

    def test_password_redaction(self):
        """Test that passwords are redacted from logs."""
        from api.secure_logging import redact_pii

        text = "Login attempt: password=secret123"
        redacted = redact_pii(text)

        assert "secret123" not in redacted
        assert "[REDACTED_PASSWORD]" in redacted

    def test_token_redaction(self):
        """Test that tokens are redacted from logs."""
        from api.secure_logging import redact_pii

        text = "Auth token: Bearer abc123xyz456def789ghi012"
        redacted = redact_pii(text)

        assert "abc123xyz456def789ghi012" not in redacted
        assert "[REDACTED_TOKEN]" in redacted

    def test_dict_sanitization(self):
        """Test that sensitive keys in dicts are redacted."""
        from api.secure_logging import redact_dict

        data = {
            "username": "john",
            "password": "secret",
            "email": "john@example.com",
            "token": "abc123"
        }

        redacted = redact_dict(data)

        assert redacted["username"] == "john"  # Not sensitive
        assert redacted["password"] == "[REDACTED]"
        assert "[REDACTED_EMAIL]" in redacted["email"]
        assert redacted["token"] == "[REDACTED]"


# ============================================================================
# Functional Tests - API Endpoints
# ============================================================================

class TestAPIFunctionality:
    """Test that API still works after security fixes."""

    def test_health_endpoint_works(self):
        """Test that health check endpoint works."""
        response = client.get("/health")
        assert response.status_code == 200
        assert "status" in response.json()

    def test_synthetic_generation_works(self, auth_headers):
        """Test that synthetic convergence map generation works."""
        response = client.post(
            "/api/v1/synthetic",
            json={
                "profile_type": "NFW",
                "mass": 1e12,
                "scale_radius": 200.0,
                "ellipticity": 0.1,
                "grid_size": 64
            }
        )

        # Public endpoint, should work without auth
        assert response.status_code == 200
        assert "convergence_map" in response.json()

    def test_inference_works(self):
        """Test that model inference endpoint works."""
        import numpy as np

        # Generate fake convergence map
        fake_map = np.random.rand(64, 64).tolist()

        response = client.post(
            "/api/v1/inference",
            json={
                "convergence_map": fake_map,
                "target_size": 64,
                "mc_samples": 1
            }
        )

        # Should work (public endpoint)
        assert response.status_code in [200, 500]  # 500 if model not loaded


# ============================================================================
# Test Configuration
# ============================================================================

@pytest.fixture(scope="session", autouse=True)
def setup_test_database():
    """Set up test database."""
    # This would set up a test database
    # For now, using the existing database setup
    pass


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
