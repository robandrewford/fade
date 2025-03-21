"""
OCR configuration management.

This module provides utilities for loading and managing OCR configuration.
"""

import json
import os
import logging
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger(__name__)

# Default config path
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "ocr_config.json")


def load_ocr_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load OCR configuration from a JSON file.
    
    Args:
        config_path: Path to the configuration file. If None, use the default.
        
    Returns:
        Dictionary containing OCR configuration.
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
    
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        logger.info(f"Loaded OCR configuration from {config_path}")
        return config
    except Exception as e:
        logger.warning(f"Failed to load OCR configuration from {config_path}: {e}")
        logger.info("Using default OCR configuration")
        # Return a minimal default configuration
        return {
            "detection": {
                "db_thresh": 0.3,
                "db_box_thresh": 0.3,
                "limit_side_len": 960,
            },
            "recognition": {
                "algorithm": "SVTR_LCNet",
            },
            "performance": {
                "use_mp": True,
                "total_process_num": 4,
            }
        }


def determine_gpu_usage(config: Dict[str, Any]) -> bool:
    """
    Determine whether to use GPU based on configuration and availability.
    
    Args:
        config: OCR configuration dictionary.
        
    Returns:
        Boolean indicating whether to use GPU.
    """
    # Check config setting first
    use_gpu = config.get("performance", {}).get("use_gpu")
    
    # If explicitly set to True or False, respect that
    if isinstance(use_gpu, bool):
        if use_gpu and not torch.cuda.is_available():
            logger.warning("GPU requested but not available. Falling back to CPU.")
            return False
        return use_gpu
    
    # Auto-detect GPU availability
    has_gpu = torch.cuda.is_available()
    if has_gpu:
        logger.info("GPU detected and will be used for OCR")
    else:
        logger.info("No GPU detected, using CPU for OCR")
    
    return has_gpu


def determine_batch_size(config: Dict[str, Any]) -> int:
    """
    Determine optimal batch size based on configuration and available memory.
    
    Args:
        config: OCR configuration dictionary.
        
    Returns:
        Optimal batch size as an integer.
    """
    base_batch_size = config.get("recognition", {}).get("batch_num", 10)
    
    # If auto-tuning is disabled, just return the configured value
    auto_tune = config.get("performance", {}).get("auto_tune_batch_size", False)
    if not auto_tune:
        return base_batch_size
    
    # Check if GPU is being used
    if not determine_gpu_usage(config):
        # For CPU, we'll just use the configured value
        return base_batch_size
    
    try:
        # Get available GPU memory
        gpu_mem_limit = config.get("performance", {}).get("gpu_mem_limit", 0.8)
        total_memory = torch.cuda.get_device_properties(0).total_memory
        reserved_memory = torch.cuda.memory_reserved(0)
        allocated_memory = torch.cuda.memory_allocated(0)
        free_memory = total_memory - reserved_memory - allocated_memory
        
        # Use a simple heuristic: 
        # - Each image takes ~100MB at max resolution
        # - Leave 20% memory as buffer
        available_memory = free_memory * gpu_mem_limit
        estimated_max_batch = int(available_memory / (100 * 1024 * 1024))
        
        # Ensure batch size is at least 1 and not too large
        optimal_batch = max(1, min(estimated_max_batch, 32))
        
        logger.info(f"Auto-tuned batch size from {base_batch_size} to {optimal_batch}")
        return optimal_batch
    
    except Exception as e:
        logger.warning(f"Failed to auto-tune batch size: {e}")
        return base_batch_size


def create_paddleocr_kwargs(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create keyword arguments for PaddleOCR initialization from configuration.
    
    Args:
        config: OCR configuration dictionary.
        
    Returns:
        Dictionary of keyword arguments for PaddleOCR initialization.
    """
    # Determine GPU usage
    use_gpu = determine_gpu_usage(config)
    
    # Determine batch size
    batch_size = determine_batch_size(config)
    
    # Extract parameters from config
    detection_config = config.get("detection", {})
    recognition_config = config.get("recognition", {})
    classification_config = config.get("classification", {})
    performance_config = config.get("performance", {})
    language = config.get("languages", {}).get("default", "en")
    
    # Create PaddleOCR kwargs
    kwargs = {
        # Language setting
        "lang": language,
        
        # Detection parameters
        "det_algorithm": detection_config.get("algorithm", "DB"),
        "det_db_thresh": detection_config.get("db_thresh", 0.3),
        "det_db_box_thresh": detection_config.get("db_box_thresh", 0.3),
        "det_db_unclip_ratio": detection_config.get("db_unclip_ratio", 1.5),
        "det_db_score_mode": detection_config.get("db_score_mode", "fast"),
        "use_dilation": detection_config.get("use_dilation", False),
        "det_limit_side_len": detection_config.get("limit_side_len", 960),
        "det_limit_type": detection_config.get("limit_type", "max"),
        
        # Recognition parameters
        "rec_algorithm": recognition_config.get("algorithm", "SVTR_LCNet"),
        "rec_batch_num": batch_size,
        
        # Classification parameters
        "use_angle_cls": classification_config.get("use_angle_cls", False),
        "cls_batch_num": classification_config.get("cls_batch_num", 6),
        "cls_thresh": classification_config.get("cls_thresh", 0.9),
        
        # Performance parameters
        "use_gpu": use_gpu,
        "use_mp": performance_config.get("use_mp", True),
        "total_process_num": performance_config.get("total_process_num", 4),
        "show_log": performance_config.get("show_log", False),
    }
    
    # Add model paths if specified
    det_model_path = config.get("models", {}).get("text_detection", {}).get("path")
    if det_model_path:
        kwargs["det_model_dir"] = det_model_path
        
    rec_model_path = config.get("models", {}).get("text_recognition", {}).get("path")
    if rec_model_path:
        kwargs["rec_model_dir"] = rec_model_path
    
    return kwargs


def initialize_ocr(config_path: Optional[str] = None) -> "PaddleOCR":
    """
    Initialize OCR with optimal configuration.
    
    Args:
        config_path: Path to the configuration file. If None, use the default.
        
    Returns:
        Initialized PaddleOCR instance.
    """
    # Import here to avoid circular import
    from paddleocr import PaddleOCR
    
    # Load configuration
    config = load_ocr_config(config_path)
    
    # Create PaddleOCR kwargs
    kwargs = create_paddleocr_kwargs(config)
    
    logger.info(f"Initializing PaddleOCR with parameters: {kwargs}")
    
    # Initialize OCR
    return PaddleOCR(**kwargs) 