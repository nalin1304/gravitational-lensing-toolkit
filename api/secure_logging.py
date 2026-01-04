"""
Secure logging utilities with PII redaction

Provides logging functions that automatically redact sensitive information.

Author: P2 Security Remediation
Date: November 2025
"""

import logging
import re
from typing import Any, Dict
from functools import wraps


# Patterns to detect and redact PII
PII_PATTERNS = {
    'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
    'password': re.compile(r'(password|passwd|pwd)["\']?\s*[:=]\s*["\']?([^"\'\s,}]+)', re.IGNORECASE),
    'token': re.compile(r'(token|jwt|bearer)["\']?\s*[:=]\s*["\']?([A-Za-z0-9._-]{20,})', re.IGNORECASE),
    'api_key': re.compile(r'(api[_-]?key|apikey)["\']?\s*[:=]\s*["\']?([A-Za-z0-9._-]{20,})', re.IGNORECASE),
    'credit_card': re.compile(r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b'),
    'ssn': re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
}


def redact_pii(text: str) -> str:
    """
    Redact PII from text string.

    Automatically detects and redacts:
    - Email addresses
    - Passwords
    - Tokens and API keys
    - Credit card numbers
    - Social Security Numbers

    Args:
        text: Text that may contain PII

    Returns:
        Text with PII redacted

    Example:
        >>> redact_pii("User email: john@example.com, token: abc123xyz...")
        "User email: [REDACTED_EMAIL], token: [REDACTED_TOKEN]"
    """
    if not text:
        return text

    redacted = text

    # Redact emails
    redacted = PII_PATTERNS['email'].sub('[REDACTED_EMAIL]', redacted)

    # Redact passwords (keep the key name, redact the value)
    redacted = PII_PATTERNS['password'].sub(r'\1: [REDACTED_PASSWORD]', redacted)

    # Redact tokens
    redacted = PII_PATTERNS['token'].sub(r'\1: [REDACTED_TOKEN]', redacted)

    # Redact API keys
    redacted = PII_PATTERNS['api_key'].sub(r'\1: [REDACTED_API_KEY]', redacted)

    # Redact credit cards
    redacted = PII_PATTERNS['credit_card'].sub('[REDACTED_CC]', redacted)

    # Redact SSNs
    redacted = PII_PATTERNS['ssn'].sub('[REDACTED_SSN]', redacted)

    return redacted


def redact_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Redact PII from dictionary values.

    Recursively processes nested dictionaries and redacts sensitive keys.

    Args:
        data: Dictionary that may contain PII

    Returns:
        Dictionary with PII redacted

    Example:
        >>> redact_dict({"email": "john@example.com", "name": "John"})
        {"email": "[REDACTED_EMAIL]", "name": "John"}
    """
    if not isinstance(data, dict):
        return data

    sensitive_keys = {
        'password', 'passwd', 'pwd', 'token', 'jwt', 'bearer',
        'api_key', 'apikey', 'secret', 'private_key', 'access_token',
        'refresh_token', 'session_id', 'cookie', 'authorization'
    }

    redacted = {}
    for key, value in data.items():
        key_lower = key.lower().replace('_', '').replace('-', '')

        # Redact entire value if key is sensitive
        if any(sensitive in key_lower for sensitive in sensitive_keys):
            redacted[key] = '[REDACTED]'
        # Recursively process nested dicts
        elif isinstance(value, dict):
            redacted[key] = redact_dict(value)
        # Redact string values
        elif isinstance(value, str):
            redacted[key] = redact_pii(value)
        else:
            redacted[key] = value

    return redacted


class SecureLogger:
    """
    Logger wrapper that automatically redacts PII.

    Drop-in replacement for standard Python logger.

    Usage:
        logger = SecureLogger(__name__)
        logger.info("User registered: john@example.com")
        # Output: "User registered: [REDACTED_EMAIL]"
    """

    def __init__(self, name: str):
        """
        Initialize secure logger.

        Args:
            name: Logger name (typically __name__)
        """
        self.logger = logging.getLogger(name)

    def _log(self, level: int, msg: str, *args, **kwargs):
        """Internal logging method with PII redaction."""
        # Redact message
        safe_msg = redact_pii(str(msg))

        # Redact extra data if present
        if 'extra' in kwargs and isinstance(kwargs['extra'], dict):
            kwargs['extra'] = redact_dict(kwargs['extra'])

        self.logger.log(level, safe_msg, *args, **kwargs)

    def debug(self, msg: str, *args, **kwargs):
        """Log debug message with PII redaction."""
        self._log(logging.DEBUG, msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs):
        """Log info message with PII redaction."""
        self._log(logging.INFO, msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs):
        """Log warning message with PII redaction."""
        self._log(logging.WARNING, msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs):
        """Log error message with PII redaction."""
        self._log(logging.ERROR, msg, *args, **kwargs)

    def critical(self, msg: str, *args, **kwargs):
        """Log critical message with PII redaction."""
        self._log(logging.CRITICAL, msg, *args, **kwargs)

    def exception(self, msg: str, *args, **kwargs):
        """Log exception with PII redaction."""
        safe_msg = redact_pii(str(msg))
        self.logger.exception(safe_msg, *args, **kwargs)


def secure_log_decorator(func):
    """
    Decorator to automatically log function calls with PII redaction.

    Example:
        @secure_log_decorator
        def create_user(email: str, password: str):
            # Implementation...
            pass

        # Logs: "Calling create_user with args: ([REDACTED_EMAIL],)"
    """
    logger = SecureLogger(func.__module__)

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Redact args and kwargs for logging
        safe_args = tuple(redact_pii(str(arg)) if isinstance(arg, str) else arg for arg in args)
        safe_kwargs = redact_dict(kwargs) if kwargs else {}

        logger.info(f"Calling {func.__name__} with args: {safe_args}, kwargs: {safe_kwargs}")

        try:
            result = func(*args, **kwargs)
            logger.info(f"{func.__name__} completed successfully")
            return result
        except Exception as e:
            logger.error(f"{func.__name__} failed: {redact_pii(str(e))}")
            raise

    return wrapper


# Convenience function to get a secure logger
def get_secure_logger(name: str) -> SecureLogger:
    """
    Get a secure logger instance.

    Args:
        name: Logger name (typically __name__)

    Returns:
        SecureLogger instance

    Example:
        from api.secure_logging import get_secure_logger
        logger = get_secure_logger(__name__)
        logger.info("User data: email@example.com")
    """
    return SecureLogger(name)
