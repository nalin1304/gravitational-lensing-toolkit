"""
API Integration Tests

Tests the complete API security and functionality after remediation.

Author: P2 Quality Assurance
Date: November 2025
"""

import pytest
import requests
from typing import Dict, Optional
import time
import os


class TestAPIBase:
    """Base class for API tests with common utilities."""

    BASE_URL = os.getenv("API_URL", "http://localhost:8000")

    def get_auth_headers(self, token: Optional[str] = None) -> Dict[str, str]:
        """Get authorization headers."""
        if token:
            return {"Authorization": f"Bearer {token}"}
        return {}

    def register_user(self, username: str, email: str, password: str) -> Dict:
        """Helper to register a test user."""
        response = requests.post(
            f"{self.BASE_URL}/api/v1/auth/register",
            json={
                "username": username,
                "email": email,
                "password": password
            }
        )
        return response

    def login_user(self, username: str, password: str) -> Optional[str]:
        """Helper to login and get token."""
        response = requests.post(
            f"{self.BASE_URL}/api/v1/auth/login",
            data={
                "username": username,
                "password": password
            }
        )
        if response.status_code == 200:
            return response.json()["access_token"]
        return None


class TestAuthentication(TestAPIBase):
    """Test authentication security fixes."""

    def test_unauthenticated_access_denied(self):
        """Test that protected endpoints require authentication."""
        response = requests.get(f"{self.BASE_URL}/api/v1/analyses/1")
        assert response.status_code == 401, "Should return 401 Unauthorized"

    def test_invalid_token_rejected(self):
        """Test that invalid tokens are rejected."""
        headers = self.get_auth_headers("invalid_token_xyz")
        response = requests.get(
            f"{self.BASE_URL}/api/v1/analyses/1",
            headers=headers
        )
        assert response.status_code == 401, "Should reject invalid token"

    def test_valid_authentication_works(self):
        """Test that valid authentication works."""
        # Register and login
        username = f"test_user_{int(time.time())}"
        email = f"{username}@example.com"
        password = "SecurePass123!"

        # Register
        reg_response = self.register_user(username, email, password)
        assert reg_response.status_code == 201, "Registration should succeed"

        # Login
        token = self.login_user(username, password)
        assert token is not None, "Login should return token"

        # Access protected endpoint
        headers = self.get_auth_headers(token)
        response = requests.get(
            f"{self.BASE_URL}/api/v1/auth/me",
            headers=headers
        )
        assert response.status_code == 200, "Should access with valid token"


class TestAuthorization(TestAPIBase):
    """Test authorization (IDOR) security fixes."""

    def test_cannot_access_other_users_data(self):
        """Test that users cannot access other users' analyses."""
        # Create two users
        user1 = f"user1_{int(time.time())}"
        user2 = f"user2_{int(time.time())}"

        self.register_user(user1, f"{user1}@example.com", "Pass123!")
        self.register_user(user2, f"{user2}@example.com", "Pass123!")

        token1 = self.login_user(user1, "Pass123!")
        token2 = self.login_user(user2, "Pass123!")

        # User 1 creates an analysis
        headers1 = self.get_auth_headers(token1)
        create_response = requests.post(
            f"{self.BASE_URL}/api/v1/analyses",
            headers=headers1,
            json={
                "name": "Test Analysis",
                "type": "synthetic",
                "config": {"mass": 1e12}
            }
        )
        analysis_id = create_response.json()["id"]

        # User 2 tries to access User 1's analysis
        headers2 = self.get_auth_headers(token2)
        access_response = requests.get(
            f"{self.BASE_URL}/api/v1/analyses/{analysis_id}",
            headers=headers2
        )

        # Should be denied (403) or not found (404)
        assert access_response.status_code in [403, 404], \
            "Should not allow access to other users' data"


class TestRateLimiting(TestAPIBase):
    """Test rate limiting security fixes."""

    def test_login_rate_limiting(self):
        """Test that login endpoint is rate limited."""
        # Attempt 6 logins in quick succession
        responses = []
        for i in range(6):
            response = requests.post(
                f"{self.BASE_URL}/api/v1/auth/login",
                data={
                    "username": f"nonexistent_user_{i}",
                    "password": "wrong_password"
                }
            )
            responses.append(response.status_code)

        # Last request should be rate limited (429)
        assert 429 in responses, "Should return 429 Too Many Requests"


class TestInputValidation(TestAPIBase):
    """Test file upload and input validation."""

    def test_invalid_file_extension_rejected(self):
        """Test that non-FITS files are rejected."""
        # This test assumes an upload endpoint exists
        # If not implemented yet, this will be a placeholder
        pass

    def test_synthetic_map_validation(self):
        """Test parameter validation for synthetic map generation."""
        # Test invalid profile type
        response = requests.post(
            f"{self.BASE_URL}/api/v1/synthetic",
            json={
                "profile_type": "InvalidProfile",
                "mass": 1e12,
                "scale_radius": 200.0,
                "ellipticity": 0.2,
                "grid_size": 64
            }
        )
        assert response.status_code == 422, "Should reject invalid profile type"

        # Test invalid mass range
        response = requests.post(
            f"{self.BASE_URL}/api/v1/synthetic",
            json={
                "profile_type": "NFW",
                "mass": 1e20,  # Too large
                "scale_radius": 200.0,
                "ellipticity": 0.2,
                "grid_size": 64
            }
        )
        assert response.status_code == 422, "Should reject out-of-range mass"


class TestEncryption(TestAPIBase):
    """Test encryption and SSL configuration."""

    def test_database_ssl_enabled(self):
        """Test that PostgreSQL SSL is enabled."""
        # This would require database access
        # Placeholder for actual implementation
        pass


class TestPIIProtection(TestAPIBase):
    """Test PII protection in logs and responses."""

    def test_error_messages_no_pii(self):
        """Test that error messages don't expose PII."""
        # Attempt invalid login
        response = requests.post(
            f"{self.BASE_URL}/api/v1/auth/login",
            data={
                "username": "test@example.com",
                "password": "wrong"
            }
        )

        # Check that response doesn't contain email
        response_text = response.text.lower()
        assert "test@example.com" not in response_text, \
            "Error message should not expose email"


class TestHealthAndMetrics(TestAPIBase):
    """Test health checks and monitoring."""

    def test_health_endpoint(self):
        """Test that health endpoint works."""
        response = requests.get(f"{self.BASE_URL}/health")
        assert response.status_code == 200, "Health check should succeed"

        data = response.json()
        assert "status" in data, "Should include status"
        assert "gpu_available" in data, "Should include GPU status"

    def test_metrics_endpoint(self):
        """Test that Prometheus metrics endpoint works."""
        response = requests.get(f"{self.BASE_URL}/metrics")
        # Metrics endpoint might not exist yet
        # This is a placeholder for when it's implemented
        pass


class TestAPIFunctionality(TestAPIBase):
    """Test core API functionality."""

    def test_synthetic_map_generation(self):
        """Test synthetic convergence map generation."""
        response = requests.post(
            f"{self.BASE_URL}/api/v1/synthetic",
            json={
                "profile_type": "NFW",
                "mass": 1e12,
                "scale_radius": 200.0,
                "ellipticity": 0.2,
                "grid_size": 64
            }
        )

        assert response.status_code == 200, "Synthetic generation should succeed"

        data = response.json()
        assert "convergence_map" in data, "Should return convergence map"
        assert "metadata" in data, "Should return metadata"

    def test_model_inference(self):
        """Test PINN model inference."""
        # First generate a map
        map_response = requests.post(
            f"{self.BASE_URL}/api/v1/synthetic",
            json={
                "profile_type": "NFW",
                "mass": 1e12,
                "scale_radius": 200.0,
                "ellipticity": 0.0,
                "grid_size": 64
            }
        )

        convergence_map = map_response.json()["convergence_map"]

        # Run inference
        inference_response = requests.post(
            f"{self.BASE_URL}/api/v1/inference",
            json={
                "convergence_map": convergence_map,
                "target_size": 64,
                "mc_samples": 1
            }
        )

        assert inference_response.status_code == 200, "Inference should succeed"

        data = inference_response.json()
        assert "predictions" in data, "Should return predictions"
        assert "M_vir" in data["predictions"], "Should predict M_vir"


# ============================================================================
# Test Runners
# ============================================================================

def run_security_tests():
    """Run all security-related tests."""
    print("=" * 60)
    print("Running Security Test Suite")
    print("=" * 60)

    pytest.main([
        __file__,
        "-v",
        "-k", "test_unauthenticated or test_invalid_token or test_cannot_access or test_login_rate",
        "--tb=short"
    ])


def run_functionality_tests():
    """Run all functionality tests."""
    print("=" * 60)
    print("Running Functionality Test Suite")
    print("=" * 60)

    pytest.main([
        __file__,
        "-v",
        "-k", "test_synthetic or test_model_inference or test_health",
        "--tb=short"
    ])


def run_all_tests():
    """Run complete test suite."""
    print("=" * 60)
    print("Running Complete Test Suite")
    print("=" * 60)

    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--html=test_report.html",
        "--self-contained-html"
    ])


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "security":
            run_security_tests()
        elif sys.argv[1] == "functionality":
            run_functionality_tests()
        else:
            run_all_tests()
    else:
        run_all_tests()
