
"""
Central configuration loader.
EVERY module imports this to get config values.
"""
import os
import yaml
from typing import Dict, Any
from pathlib import Path

# Global config object (loaded once, used everywhere)
_config: Dict[str, Any] = None

def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load YAML configuration file.
    
    Args:
        config_path: Path to config.yaml. If None, uses default.
    
    Returns:
        Dictionary with all configuration parameters.
    """
    global _config
    
    if _config is not None:
        return _config  # Return cached config (singleton pattern)
    
    if config_path is None:
        # Find config relative to this file
        root_dir = Path(__file__).parent.parent.parent
        config_path = root_dir / "config" / "config.yaml"
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        _config = yaml.safe_load(f)
    
    return _config

def get_config(key: str, default=None) -> Any:
    """
    Get a specific config value using dot notation.
    
    Examples:
        get_config("training.lgbm.num_leaves") -> 31
        get_config("api.port") -> 8000
    """
    config = load_config()
    keys = key.split('.')
    
    value = config
    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        else:
            return default
    
    return value

# Auto-load when imported
_config = load_config()

# ============================================================
# PRO TIP: Type-safe config access with dataclasses (optional)
# ============================================================
# If you want IDE autocomplete, define dataclasses like this:
# 
# from dataclasses import dataclass
# 
# @dataclass
# class TrainingConfig:
#     n_splits: int
#     test_size: int
# 
# @dataclass
# class Config:
#     training: TrainingConfig
# 
# config = Config(
#     training=TrainingConfig(
#         n_splits=load_config()['training']['validation']['n_splits'],
#         test_size=load_config()['training']['validation']['test_size']
#     )
# )
# 
# Then use: config.training.n_splits (with autocomplete!)

