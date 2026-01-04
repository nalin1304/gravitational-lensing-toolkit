"""
API Monitoring and Metrics Collection

Provides Prometheus-compatible metrics for monitoring API performance,
security events, and system health.

Author: P2 Infrastructure Improvements
Date: November 2025
"""

from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Request, Response
from typing import Callable
import time
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# Prometheus Metrics
# ============================================================================

# Request metrics
REQUEST_COUNT = Counter(
    'api_requests_total',
    'Total number of API requests',
    ['method', 'endpoint', 'status']
)

REQUEST_DURATION = Histogram(
    'api_request_duration_seconds',
    'API request duration in seconds',
    ['method', 'endpoint']
)

# Security metrics
AUTH_FAILURES = Counter(
    'api_auth_failures_total',
    'Total number of authentication failures',
    ['endpoint', 'reason']
)

RATE_LIMIT_HITS = Counter(
    'api_rate_limit_hits_total',
    'Total number of rate limit violations',
    ['endpoint']
)

# System metrics
MODEL_INFERENCE_COUNT = Counter(
    'model_inference_total',
    'Total number of model inference requests',
    ['model_type']
)

MODEL_INFERENCE_DURATION = Histogram(
    'model_inference_duration_seconds',
    'Model inference duration in seconds',
    ['model_type']
)

ACTIVE_CONNECTIONS = Gauge(
    'api_active_connections',
    'Number of active API connections'
)

# Database metrics
DB_QUERY_COUNT = Counter(
    'db_queries_total',
    'Total number of database queries',
    ['query_type']
)

DB_QUERY_DURATION = Histogram(
    'db_query_duration_seconds',
    'Database query duration in seconds',
    ['query_type']
)

# Error metrics
ERROR_COUNT = Counter(
    'api_errors_total',
    'Total number of API errors',
    ['error_type', 'endpoint']
)


# ============================================================================
# Middleware for Automatic Metrics Collection
# ============================================================================

async def metrics_middleware(request: Request, call_next: Callable) -> Response:
    """
    Middleware to automatically collect request metrics.

    Tracks:
    - Request count by method/endpoint/status
    - Request duration
    - Active connections
    - Errors

    Usage:
        app.middleware("http")(metrics_middleware)
    """
    # Increment active connections
    ACTIVE_CONNECTIONS.inc()

    # Extract request info
    method = request.method
    path = request.url.path

    # Start timer
    start_time = time.time()

    try:
        # Process request
        response = await call_next(request)

        # Record metrics
        duration = time.time() - start_time
        status = response.status_code

        REQUEST_COUNT.labels(method=method, endpoint=path, status=status).inc()
        REQUEST_DURATION.labels(method=method, endpoint=path).observe(duration)

        return response

    except Exception as e:
        # Record error
        duration = time.time() - start_time
        error_type = type(e).__name__

        ERROR_COUNT.labels(error_type=error_type, endpoint=path).inc()
        REQUEST_COUNT.labels(method=method, endpoint=path, status=500).inc()
        REQUEST_DURATION.labels(method=method, endpoint=path).observe(duration)

        logger.error(f"Request failed: {method} {path} - {error_type}: {str(e)}")
        raise

    finally:
        # Decrement active connections
        ACTIVE_CONNECTIONS.dec()


# ============================================================================
# Metrics Endpoint
# ============================================================================

async def metrics_endpoint() -> Response:
    """
    Prometheus metrics endpoint.

    Returns metrics in Prometheus exposition format.

    Usage:
        @app.get("/metrics")
        async def metrics():
            return await metrics_endpoint()
    """
    metrics_output = generate_latest()
    return Response(content=metrics_output, media_type=CONTENT_TYPE_LATEST)


# ============================================================================
# Helper Functions for Manual Metric Recording
# ============================================================================

def record_auth_failure(endpoint: str, reason: str):
    """
    Record an authentication failure.

    Args:
        endpoint: API endpoint where auth failed
        reason: Reason for failure (e.g., "invalid_token", "expired_token")

    Example:
        record_auth_failure("/api/v1/analyses", "invalid_token")
    """
    AUTH_FAILURES.labels(endpoint=endpoint, reason=reason).inc()
    logger.warning(f"Auth failure at {endpoint}: {reason}")


def record_rate_limit_hit(endpoint: str):
    """
    Record a rate limit violation.

    Args:
        endpoint: API endpoint where rate limit was hit

    Example:
        record_rate_limit_hit("/api/v1/auth/login")
    """
    RATE_LIMIT_HITS.labels(endpoint=endpoint).inc()
    logger.warning(f"Rate limit hit at {endpoint}")


def record_model_inference(model_type: str, duration: float):
    """
    Record a model inference request.

    Args:
        model_type: Type of model used (e.g., "PINN", "CNN")
        duration: Inference duration in seconds

    Example:
        start = time.time()
        result = model.predict(data)
        record_model_inference("PINN", time.time() - start)
    """
    MODEL_INFERENCE_COUNT.labels(model_type=model_type).inc()
    MODEL_INFERENCE_DURATION.labels(model_type=model_type).observe(duration)


def record_db_query(query_type: str, duration: float):
    """
    Record a database query.

    Args:
        query_type: Type of query (e.g., "select", "insert", "update")
        duration: Query duration in seconds

    Example:
        start = time.time()
        result = db.query(User).filter(...).all()
        record_db_query("select", time.time() - start)
    """
    DB_QUERY_COUNT.labels(query_type=query_type).inc()
    DB_QUERY_DURATION.labels(query_type=query_type).observe(duration)


# ============================================================================
# Health Check Helpers
# ============================================================================

class HealthStatus:
    """Track application health status."""

    def __init__(self):
        self.healthy = True
        self.issues = []

    def add_issue(self, component: str, message: str):
        """Add a health issue."""
        self.healthy = False
        self.issues.append({"component": component, "message": message})
        logger.error(f"Health check failed for {component}: {message}")

    def clear_issues(self):
        """Clear all health issues."""
        self.healthy = True
        self.issues = []

    def to_dict(self) -> dict:
        """Convert to dictionary for API response."""
        return {
            "healthy": self.healthy,
            "status": "healthy" if self.healthy else "unhealthy",
            "issues": self.issues
        }


# Global health status
_health_status = HealthStatus()


def get_health_status() -> HealthStatus:
    """Get current health status."""
    return _health_status


def check_component_health(component: str, check_func: Callable) -> bool:
    """
    Check health of a component.

    Args:
        component: Component name (e.g., "database", "model", "cache")
        check_func: Function that returns True if healthy, False otherwise

    Returns:
        True if healthy, False otherwise

    Example:
        def check_db():
            try:
                db.execute("SELECT 1")
                return True
            except:
                return False

        check_component_health("database", check_db)
    """
    try:
        if check_func():
            return True
        else:
            _health_status.add_issue(component, "Health check returned False")
            return False
    except Exception as e:
        _health_status.add_issue(component, str(e))
        return False
