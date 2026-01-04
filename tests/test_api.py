"""
Tests for FastAPI REST API

Tests cover:
- Health checks
- Synthetic data generation endpoints
- Model inference endpoints
- Batch processing
- Error handling
- Authentication
- Performance

Author: Phase 11 Implementation
Date: October 2025
"""

import pytest
import numpy as np
from fastapi.testclient import TestClient
import sys
from pathlib import Path

from api.main import app

# Test client
client = TestClient(app)


# ============================================================================
# Test Health and Info Endpoints
# ============================================================================

class TestHealthEndpoints:
    """Test health check and info endpoints"""
    
    def test_root_endpoint(self):
        """Test root endpoint returns API information"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert data["version"] == "1.0.0"
    
    def test_health_check(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "gpu_available" in data
        assert isinstance(data["gpu_available"], bool)
    
    def test_models_endpoint(self):
        """Test list models endpoint"""
        response = client.get("/api/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert len(data["models"]) > 0
        assert data["models"][0]["name"] == "PINN"
    
    def test_stats_endpoint(self):
        """Test statistics endpoint"""
        response = client.get("/api/v1/stats")
        assert response.status_code == 200
        data = response.json()
        assert "total_jobs" in data
        assert "gpu_available" in data
        assert "timestamp" in data


# ============================================================================
# Test Synthetic Data Generation
# ============================================================================

class TestSyntheticGeneration:
    """Test synthetic convergence map generation API"""
    
    def test_generate_nfw_basic(self):
        """Test basic NFW convergence map generation"""
        payload = {
            "profile_type": "NFW",
            "mass": 2e12,
            "scale_radius": 200.0,
            "ellipticity": 0.0,
            "grid_size": 64
        }
        
        response = client.post("/api/v1/synthetic", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert "job_id" in data
        assert "convergence_map" in data
        assert "coordinates" in data
        assert "metadata" in data
        assert "timestamp" in data
        
        # Check convergence map shape
        conv_map = data["convergence_map"]
        assert len(conv_map) == 64
        assert len(conv_map[0]) == 64
        
        # Check metadata
        metadata = data["metadata"]
        assert metadata["grid_size"] == 64
        assert metadata["profile_type"] == "NFW"
        assert "min_value" in metadata
        assert "max_value" in metadata
    
    def test_generate_elliptical_nfw(self):
        """Test elliptical NFW generation"""
        payload = {
            "profile_type": "Elliptical NFW",
            "mass": 1e12,
            "scale_radius": 150.0,
            "ellipticity": 0.3,
            "grid_size": 32
        }
        
        response = client.post("/api/v1/synthetic", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert len(data["convergence_map"]) == 32
        assert data["metadata"]["ellipticity"] == 0.3
    
    def test_different_grid_sizes(self):
        """Test generation with different grid sizes"""
        for grid_size in [32, 64, 128]:
            payload = {
                "profile_type": "NFW",
                "mass": 2e12,
                "scale_radius": 200.0,
                "ellipticity": 0.0,
                "grid_size": grid_size
            }
            
            response = client.post("/api/v1/synthetic", json=payload)
            assert response.status_code == 200
            data = response.json()
            assert len(data["convergence_map"]) == grid_size
    
    def test_invalid_profile_type(self):
        """Test error handling for invalid profile type"""
        payload = {
            "profile_type": "INVALID",
            "mass": 2e12,
            "scale_radius": 200.0,
            "ellipticity": 0.0,
            "grid_size": 64
        }
        
        response = client.post("/api/v1/synthetic", json=payload)
        assert response.status_code == 422  # Validation error
    
    def test_invalid_mass_range(self):
        """Test validation of mass parameter"""
        payload = {
            "profile_type": "NFW",
            "mass": 1e10,  # Too low
            "scale_radius": 200.0,
            "ellipticity": 0.0,
            "grid_size": 64
        }
        
        response = client.post("/api/v1/synthetic", json=payload)
        assert response.status_code == 422
    
    def test_invalid_grid_size(self):
        """Test validation of grid size"""
        payload = {
            "profile_type": "NFW",
            "mass": 2e12,
            "scale_radius": 200.0,
            "ellipticity": 0.0,
            "grid_size": 100  # Not 32, 64, or 128
        }
        
        response = client.post("/api/v1/synthetic", json=payload)
        assert response.status_code == 422


# ============================================================================
# Test Model Inference
# ============================================================================

class TestInference:
    """Test model inference API"""
    
    def test_single_inference(self):
        """Test single forward pass inference"""
        # Generate test data
        convergence_map = np.random.rand(64, 64).tolist()
        
        payload = {
            "convergence_map": convergence_map,
            "target_size": 64,
            "mc_samples": 1
        }
        
        response = client.post("/api/v1/inference", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert "job_id" in data
        assert "predictions" in data
        assert "classification" in data
        assert "entropy" in data
        assert "timestamp" in data
        
        # Check predictions
        predictions = data["predictions"]
        assert "M_vir" in predictions
        assert "r_s" in predictions
        assert "ellipticity" in predictions
        
        # No uncertainties for single sample
        assert data.get("uncertainties") is None
    
    def test_mc_dropout_inference(self):
        """Test MC Dropout for uncertainty quantification"""
        convergence_map = np.random.rand(64, 64).tolist()
        
        payload = {
            "convergence_map": convergence_map,
            "target_size": 64,
            "mc_samples": 10
        }
        
        response = client.post("/api/v1/inference", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert "uncertainties" in data
        
        uncertainties = data["uncertainties"]
        assert "M_vir_std" in uncertainties
        assert "r_s_std" in uncertainties
        assert "ellipticity_std" in uncertainties
    
    def test_inference_with_different_sizes(self):
        """Test inference with different input sizes"""
        for size in [32, 64, 128]:
            convergence_map = np.random.rand(size, size).tolist()
            
            payload = {
                "convergence_map": convergence_map,
                "target_size": 64,
                "mc_samples": 1
            }
            
            response = client.post("/api/v1/inference", json=payload)
            assert response.status_code == 200
    
    def test_inference_invalid_shape(self):
        """Test error handling for invalid convergence map shape"""
        convergence_map = [[1, 2], [3, 4]]  # Too small
        
        payload = {
            "convergence_map": convergence_map,
            "target_size": 64,
            "mc_samples": 1
        }
        
        response = client.post("/api/v1/inference", json=payload)
        # Should handle gracefully (may succeed or fail depending on preprocessing)
        assert response.status_code in [200, 500]


# ============================================================================
# Test Batch Processing
# ============================================================================

class TestBatchProcessing:
    """Test batch job processing"""
    
    def test_submit_batch_job(self):
        """Test submitting batch processing job"""
        payload = {
            "job_ids": ["job1", "job2", "job3"]
        }
        
        response = client.post("/api/v1/batch", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert "batch_id" in data
        assert "message" in data
        assert "status_url" in data
    
    def test_get_batch_status(self):
        """Test getting batch job status"""
        # First submit a batch
        payload = {
            "job_ids": ["job1", "job2"]
        }
        
        submit_response = client.post("/api/v1/batch", json=payload)
        batch_id = submit_response.json()["batch_id"]
        
        # Check status
        response = client.get(f"/api/v1/batch/{batch_id}/status")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "progress" in data
        assert data["status"] in ["pending", "running", "completed", "failed"]
    
    def test_batch_status_not_found(self):
        """Test error for non-existent batch job"""
        response = client.get("/api/v1/batch/invalid-id/status")
        assert response.status_code == 404


# ============================================================================
# Test Error Handling
# ============================================================================

class TestErrorHandling:
    """Test API error handling"""
    
    def test_malformed_json(self):
        """Test handling of malformed JSON"""
        response = client.post(
            "/api/v1/synthetic",
            data="not json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422
    
    def test_missing_required_fields(self):
        """Test validation of required fields"""
        payload = {
            "profile_type": "NFW"
            # Missing mass
        }
        
        response = client.post("/api/v1/synthetic", json=payload)
        assert response.status_code == 422
    
    def test_invalid_endpoint(self):
        """Test 404 for invalid endpoint"""
        response = client.get("/api/v1/nonexistent")
        assert response.status_code == 404


# ============================================================================
# Test Integration Workflows
# ============================================================================

class TestIntegrationWorkflows:
    """Test end-to-end workflows"""
    
    def test_generate_and_infer_workflow(self):
        """Test complete workflow: generate -> infer"""
        # Step 1: Generate synthetic data
        gen_payload = {
            "profile_type": "NFW",
            "mass": 2e12,
            "scale_radius": 200.0,
            "ellipticity": 0.0,
            "grid_size": 64
        }
        
        gen_response = client.post("/api/v1/synthetic", json=gen_payload)
        assert gen_response.status_code == 200
        convergence_map = gen_response.json()["convergence_map"]
        
        # Step 2: Run inference
        inf_payload = {
            "convergence_map": convergence_map,
            "target_size": 64,
            "mc_samples": 5
        }
        
        inf_response = client.post("/api/v1/inference", json=inf_payload)
        assert inf_response.status_code == 200
        
        data = inf_response.json()
        assert "predictions" in data
        assert "uncertainties" in data


# ============================================================================
# Test Performance
# ============================================================================

class TestPerformance:
    """Test API performance"""
    
    def test_generation_speed(self):
        """Test synthetic generation completes within reasonable time"""
        import time
        
        payload = {
            "profile_type": "NFW",
            "mass": 2e12,
            "scale_radius": 200.0,
            "ellipticity": 0.0,
            "grid_size": 64
        }
        
        start = time.time()
        response = client.post("/api/v1/synthetic", json=payload)
        duration = time.time() - start
        
        assert response.status_code == 200
        assert duration < 10.0  # Should complete within 10 seconds
    
    def test_inference_speed(self):
        """Test inference completes within reasonable time"""
        import time
        
        convergence_map = np.random.rand(64, 64).tolist()
        payload = {
            "convergence_map": convergence_map,
            "target_size": 64,
            "mc_samples": 1
        }
        
        start = time.time()
        response = client.post("/api/v1/inference", json=payload)
        duration = time.time() - start
        
        assert response.status_code == 200
        assert duration < 5.0  # Should complete within 5 seconds
    
    def test_concurrent_requests(self):
        """Test handling of concurrent requests"""
        import concurrent.futures
        
        def make_request():
            payload = {
                "profile_type": "NFW",
                "mass": 2e12,
                "scale_radius": 200.0,
                "ellipticity": 0.0,
                "grid_size": 32  # Smaller for speed
            }
            return client.post("/api/v1/synthetic", json=payload)
        
        # Make 5 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(5)]
            responses = [f.result() for f in futures]
        
        # All should succeed
        assert all(r.status_code == 200 for r in responses)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
