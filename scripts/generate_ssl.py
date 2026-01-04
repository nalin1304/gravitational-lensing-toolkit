"""
Generate Self-Signed SSL Certificates (`database/ssl/`)
For development usage only.

Usage:
    python scripts/generate_ssl.py
"""

import os
import subprocess
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_ssl_certs():
    project_root = Path(__file__).parent.parent
    ssl_dir = project_root / "database" / "ssl"
    
    # Create directory if not exists
    ssl_dir.mkdir(parents=True, exist_ok=True)
    
    key_path = ssl_dir / "server.key"
    crt_path = ssl_dir / "server.crt"
    
    if key_path.exists() and crt_path.exists():
        logger.info("✅ SSL certificates already exist.")
        return

    logger.info("🔐 Generating self-signed SSL certificates...")
    
    # OpenSSL command
    cmd = [
        "openssl", "req", "-x509", "-newkey", "rsa:4096",
        "-keyout", str(key_path),
        "-out", str(crt_path),
        "-days", "365",
        "-nodes",
        "-subj", "/CN=localhost"
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        # Set permissions (read-only for owner)
        key_path.chmod(0o600)
        crt_path.chmod(0o644)
        
        logger.info(f"✅ Generated: {crt_path}")
        logger.info(f"✅ Generated: {key_path}")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to generate certificates: {e}")
        logger.error(e.stderr.decode())
    except FileNotFoundError:
        logger.error("❌ 'openssl' command not found. Please install OpenSSL.")

if __name__ == "__main__":
    generate_ssl_certs()
