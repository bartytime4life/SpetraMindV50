"""
Hash and git utilities for reproducibility tracking.

Implements config hashing and git SHA tracking as required by the architecture.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Optional


def git_sha() -> str:
    """
    Get current git SHA.
    
    Returns:
        Git SHA string or "NA" if not available
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "NA"


def hash_configs(config_dir: str = "configs") -> str:
    """
    Compute hash of all config files for reproducibility.
    
    Args:
        config_dir: Directory containing config files
        
    Returns:
        SHA256 hash (first 12 characters)
    """
    hasher = hashlib.sha256()
    
    config_path = Path(config_dir)
    if not config_path.exists():
        return "no-configs"
        
    # Sort files for deterministic hashing
    config_files = sorted(config_path.rglob("*.yaml")) + sorted(config_path.rglob("*.yml"))
    
    for config_file in config_files:
        try:
            content = config_file.read_bytes()
            hasher.update(config_file.name.encode('utf-8'))  # Include filename
            hasher.update(content)
        except (OSError, PermissionError):
            # Skip files that can't be read
            continue
            
    return hasher.hexdigest()[:12]


def compute_file_hash(file_path: str) -> str:
    """
    Compute SHA256 hash of a file.
    
    Args:
        file_path: Path to file
        
    Returns:
        SHA256 hash string
    """
    hasher = hashlib.sha256()
    
    try:
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except (OSError, PermissionError):
        return "error"


def compute_data_hash(data: bytes) -> str:
    """
    Compute SHA256 hash of data.
    
    Args:
        data: Bytes to hash
        
    Returns:
        SHA256 hash string
    """
    return hashlib.sha256(data).hexdigest()


def create_run_manifest(
    config_hash: Optional[str] = None,
    git_sha_val: Optional[str] = None,
    additional_info: Optional[dict] = None
) -> dict:
    """
    Create run manifest for reproducibility.
    
    Args:
        config_hash: Configuration hash (computed if None)
        git_sha_val: Git SHA (computed if None)
        additional_info: Additional information to include
        
    Returns:
        Run manifest dictionary
    """
    import platform
    import sys
    from datetime import datetime, timezone
    
    if config_hash is None:
        config_hash = hash_configs()
    if git_sha_val is None:
        git_sha_val = git_sha()
        
    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "git_sha": git_sha_val,
        "config_hash": config_hash,
        "environment": {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "hostname": platform.node(),
            "user": platform.node()  # Simplified
        }
    }
    
    if additional_info:
        manifest.update(additional_info)
        
    return manifest


def save_run_manifest(
    output_path: str = "run_hash_summary_v50.json",
    **kwargs
) -> None:
    """
    Save run manifest to file.
    
    Args:
        output_path: Path to save manifest
        **kwargs: Additional arguments for create_run_manifest
    """
    import json
    
    manifest = create_run_manifest(**kwargs)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(manifest, f, indent=2)


def verify_reproducibility(
    manifest_path: str,
    check_git: bool = True,
    check_configs: bool = True
) -> dict:
    """
    Verify reproducibility by comparing current state with manifest.
    
    Args:
        manifest_path: Path to run manifest
        check_git: Whether to check git SHA
        check_configs: Whether to check config hash
        
    Returns:
        Verification result
    """
    import json
    
    result = {
        "manifest_exists": False,
        "git_match": None,
        "config_match": None,
        "issues": []
    }
    
    manifest_file = Path(manifest_path)
    if not manifest_file.exists():
        result["issues"].append(f"Manifest file not found: {manifest_path}")
        return result
        
    result["manifest_exists"] = True
    
    try:
        with open(manifest_file) as f:
            manifest = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        result["issues"].append(f"Cannot read manifest: {e}")
        return result
        
    # Check git SHA
    if check_git and "git_sha" in manifest:
        current_sha = git_sha()
        expected_sha = manifest["git_sha"]
        result["git_match"] = (current_sha == expected_sha)
        if not result["git_match"]:
            result["issues"].append(f"Git SHA mismatch: expected {expected_sha}, got {current_sha}")
            
    # Check config hash
    if check_configs and "config_hash" in manifest:
        current_hash = hash_configs()
        expected_hash = manifest["config_hash"]
        result["config_match"] = (current_hash == expected_hash)
        if not result["config_match"]:
            result["issues"].append(f"Config hash mismatch: expected {expected_hash}, got {current_hash}")
            
    return result