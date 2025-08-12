"""
Data contracts and schema validation.

Implements the data contracts from the architecture:
- Directory structure validation
- File format validation  
- Schema compliance checking
- Submission format validation
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import numpy as np


class DataContract:
    """
    Data contract specification and validation.
    
    Defines the expected structure and format of data files
    according to the architecture specification.
    """
    
    # Expected directory structure
    EXPECTED_DIRS = [
        "data/raw/fgs1",
        "data/raw/airs_ch0", 
        "data/calibrated/fgs1",
        "data/calibrated/airs",
        "data/features/fgs1_white",
        "data/features/airs_bins",
        "data/splits"
    ]
    
    # Expected file patterns
    FGS1_CALIBRATED_PATTERN = "fgs1_{planet}.npz"
    AIRS_CALIBRATED_PATTERN = "airs_{planet}.npz"
    FGS1_FEATURES_PATTERN = "fgs1_white_{planet}.npz"
    AIRS_FEATURES_PATTERN = "airs_bins_{planet}.npz"
    
    # Expected tensor shapes and dtypes
    CALIBRATED_FGS1_SCHEMA = {
        "frames": {"dtype": "float32", "ndim": 3},  # [T, H, W]
        "variance": {"dtype": "float32", "ndim": 3},  # [T, H, W]
        "mask": {"dtype": "bool", "ndim": 3}  # [T, H, W]
    }
    
    CALIBRATED_AIRS_SCHEMA = {
        "frames": {"dtype": "float32", "ndim": 3},  # [T, H, W]
        "variance": {"dtype": "float32", "ndim": 3},  # [T, H, W]
        "mask": {"dtype": "bool", "ndim": 3},  # [T, H, W]
        "trace_meta": {"dtype": "object", "ndim": 0}  # JSON metadata
    }
    
    FEATURES_FGS1_SCHEMA = {
        "flux": {"dtype": "float32", "ndim": 1},  # [T]
        "time": {"dtype": "float64", "ndim": 1},  # [T]
        "centroid": {"dtype": "float32", "ndim": 2},  # [T, 2]
        "jitter": {"dtype": "float32", "ndim": 2}  # [T, 2]
    }
    
    FEATURES_AIRS_SCHEMA = {
        "flux": {"dtype": "float32", "ndim": 2},  # [T, 283]
        "time": {"dtype": "float64", "ndim": 1}  # [T]
    }
    
    SUBMISSION_SCHEMA = {
        "required_columns": [f"mu_{i}" for i in range(283)] + [f"sigma_{i}" for i in range(283)],
        "dtypes": {"mu_*": "float32", "sigma_*": "float32"},
        "constraints": {
            "sigma_positive": "All sigma values must be positive",
            "finite_values": "All values must be finite (no NaN/inf)"
        }
    }


def validate_directory_structure(base_path: str) -> Dict[str, bool]:
    """
    Validate expected directory structure.
    
    Args:
        base_path: Base path to check
        
    Returns:
        Dict mapping directory paths to existence status
    """
    base = Path(base_path)
    results = {}
    
    for expected_dir in DataContract.EXPECTED_DIRS:
        dir_path = base / expected_dir
        results[expected_dir] = dir_path.exists() and dir_path.is_dir()
        
    return results


def validate_npz_schema(
    file_path: str, 
    expected_schema: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Validate NPZ file against expected schema.
    
    Args:
        file_path: Path to NPZ file
        expected_schema: Expected schema specification
        
    Returns:
        Validation result with errors and warnings
    """
    result = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "file_exists": False,
        "keys_found": [],
        "keys_missing": [],
        "shape_mismatches": [],
        "dtype_mismatches": []
    }
    
    file_path = Path(file_path)
    if not file_path.exists():
        result["valid"] = False
        result["errors"].append(f"File does not exist: {file_path}")
        return result
        
    result["file_exists"] = True
    
    try:
        with np.load(file_path) as data:
            result["keys_found"] = list(data.keys())
            
            # Check for missing keys
            expected_keys = set(expected_schema.keys())
            found_keys = set(data.keys())
            result["keys_missing"] = list(expected_keys - found_keys)
            
            if result["keys_missing"]:
                result["valid"] = False
                result["errors"].append(f"Missing required keys: {result['keys_missing']}")
            
            # Check each key's schema
            for key, schema in expected_schema.items():
                if key not in data:
                    continue
                    
                array = data[key]
                
                # Check dtype
                expected_dtype = schema.get("dtype")
                if expected_dtype and str(array.dtype) != expected_dtype:
                    result["dtype_mismatches"].append({
                        "key": key,
                        "expected": expected_dtype,
                        "found": str(array.dtype)
                    })
                    result["warnings"].append(f"Dtype mismatch for {key}: expected {expected_dtype}, got {array.dtype}")
                
                # Check ndim
                expected_ndim = schema.get("ndim")
                if expected_ndim is not None and array.ndim != expected_ndim:
                    result["shape_mismatches"].append({
                        "key": key,
                        "expected_ndim": expected_ndim,
                        "found_ndim": array.ndim,
                        "found_shape": array.shape
                    })
                    result["valid"] = False
                    result["errors"].append(f"Shape mismatch for {key}: expected {expected_ndim}D, got {array.ndim}D (shape: {array.shape})")
                    
    except Exception as e:
        result["valid"] = False
        result["errors"].append(f"Error loading NPZ file: {str(e)}")
        
    return result


def validate_submission_csv(file_path: str) -> Dict[str, Any]:
    """
    Validate submission CSV file against schema.
    
    Args:
        file_path: Path to submission CSV
        
    Returns:
        Validation result
    """
    result = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "file_exists": False,
        "shape": None,
        "columns_found": [],
        "columns_missing": [],
        "value_issues": []
    }
    
    file_path = Path(file_path)
    if not file_path.exists():
        result["valid"] = False
        result["errors"].append(f"File does not exist: {file_path}")
        return result
        
    result["file_exists"] = True
    
    try:
        import pandas as pd
        df = pd.read_csv(file_path)
        
        result["shape"] = df.shape
        result["columns_found"] = list(df.columns)
        
        # Check required columns
        expected_cols = DataContract.SUBMISSION_SCHEMA["required_columns"]
        result["columns_missing"] = [col for col in expected_cols if col not in df.columns]
        
        if result["columns_missing"]:
            result["valid"] = False
            result["errors"].append(f"Missing required columns: {result['columns_missing'][:10]}...")  # Show first 10
        
        # Check sigma positivity
        sigma_cols = [col for col in df.columns if col.startswith("sigma_")]
        for col in sigma_cols:
            if col in df.columns:
                negative_count = (df[col] <= 0).sum()
                if negative_count > 0:
                    result["valid"] = False
                    result["errors"].append(f"Found {negative_count} non-positive values in {col}")
        
        # Check for NaN/inf values
        nan_cols = df.columns[df.isna().any()].tolist()
        if nan_cols:
            result["valid"] = False
            result["errors"].append(f"Found NaN values in columns: {nan_cols[:5]}...")
            
        inf_cols = df.columns[np.isinf(df.select_dtypes(include=[np.number])).any()].tolist()
        if inf_cols:
            result["valid"] = False  
            result["errors"].append(f"Found infinite values in columns: {inf_cols[:5]}...")
            
    except ImportError:
        result["warnings"].append("pandas not available for CSV validation")
    except Exception as e:
        result["valid"] = False
        result["errors"].append(f"Error loading CSV file: {str(e)}")
        
    return result


def validate_planet_data(planet_id: str, base_path: str) -> Dict[str, Any]:
    """
    Validate all data files for a specific planet.
    
    Args:
        planet_id: Planet identifier
        base_path: Base data directory path
        
    Returns:
        Comprehensive validation result
    """
    base = Path(base_path)
    result = {
        "planet_id": planet_id,
        "overall_valid": True,
        "files_checked": [],
        "fgs1_calibrated": None,
        "airs_calibrated": None,
        "fgs1_features": None,
        "airs_features": None
    }
    
    # Check FGS1 calibrated
    fgs1_cal_path = base / "data/calibrated/fgs1" / f"fgs1_{planet_id}.npz"
    if fgs1_cal_path.exists():
        result["fgs1_calibrated"] = validate_npz_schema(str(fgs1_cal_path), DataContract.CALIBRATED_FGS1_SCHEMA)
        result["files_checked"].append(str(fgs1_cal_path))
        if not result["fgs1_calibrated"]["valid"]:
            result["overall_valid"] = False
    
    # Check AIRS calibrated  
    airs_cal_path = base / "data/calibrated/airs" / f"airs_{planet_id}.npz"
    if airs_cal_path.exists():
        result["airs_calibrated"] = validate_npz_schema(str(airs_cal_path), DataContract.CALIBRATED_AIRS_SCHEMA)
        result["files_checked"].append(str(airs_cal_path))
        if not result["airs_calibrated"]["valid"]:
            result["overall_valid"] = False
    
    # Check FGS1 features
    fgs1_feat_path = base / "data/features/fgs1_white" / f"fgs1_white_{planet_id}.npz"
    if fgs1_feat_path.exists():
        result["fgs1_features"] = validate_npz_schema(str(fgs1_feat_path), DataContract.FEATURES_FGS1_SCHEMA)
        result["files_checked"].append(str(fgs1_feat_path))
        if not result["fgs1_features"]["valid"]:
            result["overall_valid"] = False
    
    # Check AIRS features
    airs_feat_path = base / "data/features/airs_bins" / f"airs_bins_{planet_id}.npz"
    if airs_feat_path.exists():
        result["airs_features"] = validate_npz_schema(str(airs_feat_path), DataContract.FEATURES_AIRS_SCHEMA)
        result["files_checked"].append(str(airs_feat_path))
        if not result["airs_features"]["valid"]:
            result["overall_valid"] = False
    
    return result


def validate_schema(file_path: str, schema_type: str) -> Dict[str, Any]:
    """
    Main schema validation function.
    
    Args:
        file_path: Path to file to validate
        schema_type: Type of schema ('fgs1_calibrated', 'airs_calibrated', 
                    'fgs1_features', 'airs_features', 'submission')
        
    Returns:
        Validation result
    """
    if schema_type == "fgs1_calibrated":
        return validate_npz_schema(file_path, DataContract.CALIBRATED_FGS1_SCHEMA)
    elif schema_type == "airs_calibrated":
        return validate_npz_schema(file_path, DataContract.CALIBRATED_AIRS_SCHEMA)
    elif schema_type == "fgs1_features":
        return validate_npz_schema(file_path, DataContract.FEATURES_FGS1_SCHEMA)
    elif schema_type == "airs_features":
        return validate_npz_schema(file_path, DataContract.FEATURES_AIRS_SCHEMA)
    elif schema_type == "submission":
        return validate_submission_csv(file_path)
    else:
        return {
            "valid": False,
            "errors": [f"Unknown schema type: {schema_type}"],
            "warnings": []
        }


def create_validation_report(base_path: str, planet_ids: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Create comprehensive validation report.
    
    Args:
        base_path: Base data directory
        planet_ids: List of planet IDs to check (optional)
        
    Returns:
        Full validation report
    """
    report = {
        "base_path": base_path,
        "timestamp": None,  # Would add timestamp in real implementation
        "directory_structure": validate_directory_structure(base_path),
        "planets": {},
        "summary": {
            "total_planets": 0,
            "valid_planets": 0,
            "invalid_planets": 0,
            "missing_files": 0
        }
    }
    
    # If no planet IDs provided, try to discover them
    if planet_ids is None:
        # Try to find planet IDs from existing files
        planet_ids = []
        
        # Look for FGS1 calibrated files
        fgs1_cal_dir = Path(base_path) / "data/calibrated/fgs1"
        if fgs1_cal_dir.exists():
            for file_path in fgs1_cal_dir.glob("fgs1_*.npz"):
                planet_id = file_path.stem.replace("fgs1_", "")
                if planet_id not in planet_ids:
                    planet_ids.append(planet_id)
    
    # Validate each planet
    for planet_id in planet_ids:
        planet_result = validate_planet_data(planet_id, base_path)
        report["planets"][planet_id] = planet_result
        
        report["summary"]["total_planets"] += 1
        if planet_result["overall_valid"]:
            report["summary"]["valid_planets"] += 1
        else:
            report["summary"]["invalid_planets"] += 1
            
    return report