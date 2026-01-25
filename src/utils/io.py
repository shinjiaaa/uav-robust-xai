"""I/O utilities for file operations."""

import json
import yaml
from pathlib import Path
from typing import Any, Dict, Optional


def load_yaml(path: Path) -> Dict[str, Any]:
    """Load YAML configuration file.
    
    Args:
        path: Path to YAML file
        
    Returns:
        Dictionary with configuration
    """
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_yaml(data: Dict[str, Any], path: Path):
    """Save dictionary to YAML file.
    
    Args:
        data: Dictionary to save
        path: Output path
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)


def load_json(path: Path) -> Dict[str, Any]:
    """Load JSON file.
    
    Args:
        path: Path to JSON file
        
    Returns:
        Dictionary with data
    """
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Any, path: Path, indent: int = 2):
    """Save data to JSON file.
    
    Args:
        data: Data to save (must be JSON serializable)
        path: Output path
        indent: JSON indentation
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def ensure_dir(path: Path):
    """Ensure directory exists, create if not.
    
    Args:
        path: Directory path
    """
    path.mkdir(parents=True, exist_ok=True)
