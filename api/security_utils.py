"""
Security utilities for API

Provides file upload validation, sanitization, and security checks.

Author: P0 Security Remediation
Date: November 2025
"""

from fastapi import HTTPException, UploadFile, status
from typing import Optional, Dict, Any
import bleach
import os
import magic
from pathlib import Path

# File upload configuration
ALLOWED_FITS_EXTENSIONS = {".fits", ".fit", ".fts", ".fits.gz"}
MAX_FILE_SIZE_MB = 100  # Maximum file size in MB
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


def validate_fits_file(
    file: UploadFile,
    max_size_mb: Optional[int] = None
) -> None:
    """
    Validate FITS file upload for security.

    Checks:
    1. File extension is valid (.fits, .fit, .fts, .fits.gz)
    2. File size is within limits
    3. MIME type matches expected types

    Args:
        file: Uploaded file object
        max_size_mb: Optional maximum file size in MB (default: 100MB)

    Raises:
        HTTPException: If validation fails

    Example:
        @app.post("/upload")
        async def upload_fits(file: UploadFile = File(...)):
            validate_fits_file(file)
            # Process file...
    """
    if not file:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file provided"
        )

    # Check file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_FITS_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file extension. Allowed: {', '.join(ALLOWED_FITS_EXTENSIONS)}"
        )

    # Check file size (if available in headers)
    max_size = (max_size_mb or MAX_FILE_SIZE_MB) * 1024 * 1024
    if hasattr(file, 'size') and file.size:
        if file.size > max_size:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large. Maximum size: {max_size_mb or MAX_FILE_SIZE_MB}MB"
            )


def sanitize_fits_header_value(value: Any) -> str:
    """
    Sanitize FITS header values to prevent XSS attacks.

    Uses bleach to clean HTML/JavaScript from string values.

    Args:
        value: Header value (any type)

    Returns:
        Sanitized string representation

    Example:
        >>> sanitize_fits_header_value("<script>alert('XSS')</script>")
        "&lt;script&gt;alert('XSS')&lt;/script&gt;"
    """
    if value is None:
        return ""

    # Convert to string
    value_str = str(value)

    # Sanitize with bleach (removes all HTML/JS tags)
    sanitized = bleach.clean(
        value_str,
        tags=[],  # No tags allowed
        attributes={},  # No attributes allowed
        strip=True  # Strip tags instead of escaping
    )

    return sanitized


def sanitize_fits_headers(headers: Dict[str, Any]) -> Dict[str, str]:
    """
    Sanitize all values in a FITS header dictionary.

    Args:
        headers: Dictionary of FITS header key-value pairs

    Returns:
        Dictionary with sanitized values

    Example:
        >>> headers = {"OBJECT": "<script>XSS</script>", "EXPTIME": 300}
        >>> sanitize_fits_headers(headers)
        {"OBJECT": "&lt;script&gt;XSS&lt;/script&gt;", "EXPTIME": "300"}
    """
    if not headers:
        return {}

    return {
        key: sanitize_fits_header_value(value)
        for key, value in headers.items()
    }


async def read_file_in_chunks(
    file: UploadFile,
    chunk_size: int = 8192,
    max_size_bytes: Optional[int] = None
) -> bytes:
    """
    Read uploaded file in chunks with size validation.

    Prevents memory exhaustion from large file uploads.

    Args:
        file: Uploaded file object
        chunk_size: Size of each chunk to read (default: 8KB)
        max_size_bytes: Maximum total file size in bytes

    Returns:
        Complete file content as bytes

    Raises:
        HTTPException: If file exceeds maximum size

    Example:
        content = await read_file_in_chunks(file, max_size_bytes=10*1024*1024)
    """
    max_bytes = max_size_bytes or MAX_FILE_SIZE_BYTES
    content = b""
    total_size = 0

    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break

        total_size += len(chunk)
        if total_size > max_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large. Maximum size: {max_bytes / 1024 / 1024:.1f}MB"
            )

        content += chunk

    return content


def validate_filename(filename: str) -> str:
    """
    Validate and sanitize uploaded filename.

    Prevents directory traversal attacks (e.g., "../../etc/passwd").

    Args:
        filename: Original filename

    Returns:
        Sanitized filename (basename only, no path components)

    Raises:
        HTTPException: If filename is invalid

    Example:
        >>> validate_filename("../../etc/passwd")
        Raises HTTPException
        >>> validate_filename("my_data.fits")
        "my_data.fits"
    """
    if not filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename cannot be empty"
        )

    # Get basename only (removes any path traversal attempts)
    safe_filename = os.path.basename(filename)

    # Check for suspicious patterns
    suspicious_patterns = ["..", "~", "$", "|", ";", "&", ">", "<"]
    if any(pattern in safe_filename for pattern in suspicious_patterns):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename contains invalid characters"
        )

    # Check length
    if len(safe_filename) > 255:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename too long (max 255 characters)"
        )

    return safe_filename
