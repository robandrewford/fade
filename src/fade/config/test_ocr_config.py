"""
Test script for OCR configuration.

This script tests the OCR configuration and initialization.
"""

import os
import logging
import argparse

# Import directly from the local module
from fade.config.ocr import initialize_ocr, load_ocr_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_ocr_config(config_path=None):
    """Test OCR configuration and initialization."""
    logger.info("Testing OCR configuration and initialization")
    
    # Load configuration
    config = load_ocr_config(config_path)
    logger.info(f"Loaded configuration: {config}")
    
    # Initialize OCR
    ocr = initialize_ocr(config_path)
    logger.info("OCR initialized successfully")
    
    # Check OCR parameters (get attributes that exist)
    logger.info(f"OCR model parameters loaded successfully")
    logger.info(f"OCR use_mp: {getattr(ocr, 'use_mp', 'Not available')}")
    logger.info(f"OCR rec_algorithm: {getattr(ocr, 'rec_algorithm', 'Not available')}")
    logger.info(f"OCR lang: {getattr(ocr, 'lang', 'Not available')}")
    
    return ocr


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test OCR configuration")
    parser.add_argument("--config", help="Path to OCR configuration file")
    args = parser.parse_args()
    
    ocr = test_ocr_config(args.config)
    print("\nOCR configuration test completed successfully")
    print("Use the initialized OCR instance with: ocr.ocr(image_path)") 