"""
Script to update OCR configuration.

Allows users to update their OCR configuration with command-line arguments.
"""

import os
import json
import argparse
import logging

# Import directly from the local module
from fade.config.ocr import DEFAULT_CONFIG_PATH, load_ocr_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_parser():
    """Create argument parser for OCR configuration updates."""
    parser = argparse.ArgumentParser(description="Update OCR configuration")
    
    # General options
    parser.add_argument("--config", help="Path to OCR configuration file", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--save", help="Save updated configuration", action="store_true")
    
    # Detection options
    parser.add_argument("--det-thresh", type=float, help="Detection threshold")
    parser.add_argument("--det-box-thresh", type=float, help="Detection box threshold")
    parser.add_argument("--det-unclip-ratio", type=float, help="Detection unclip ratio")
    parser.add_argument("--det-limit-side", type=int, help="Detection limit side length")
    
    # Recognition options
    parser.add_argument("--rec-algorithm", help="Recognition algorithm, e.g. SVTR_LCNet, PP-OCRv4")
    parser.add_argument("--rec-batch-num", type=int, help="Recognition batch size")
    
    # Classification options
    parser.add_argument("--use-angle-cls", action="store_true", help="Use angle classification")
    parser.add_argument("--no-angle-cls", action="store_true", help="Don't use angle classification")
    
    # Performance options
    parser.add_argument("--use-gpu", action="store_true", help="Use GPU")
    parser.add_argument("--cpu-only", action="store_true", help="Use CPU only")
    parser.add_argument("--use-mp", action="store_true", help="Use multiprocessing")
    parser.add_argument("--no-mp", action="store_true", help="Don't use multiprocessing")
    parser.add_argument("--processes", type=int, help="Number of processes for multiprocessing")
    
    # Language options
    parser.add_argument("--lang", help="Default language for OCR")
    
    # Other options
    parser.add_argument("--reset", action="store_true", help="Reset configuration to defaults")
    
    return parser


def update_config(config, args):
    """Update OCR configuration with command-line arguments."""
    # Detection options
    if args.det_thresh is not None:
        config["detection"]["db_thresh"] = args.det_thresh
    if args.det_box_thresh is not None:
        config["detection"]["db_box_thresh"] = args.det_box_thresh
    if args.det_unclip_ratio is not None:
        config["detection"]["db_unclip_ratio"] = args.det_unclip_ratio
    if args.det_limit_side is not None:
        config["detection"]["limit_side_len"] = args.det_limit_side
    
    # Recognition options
    if args.rec_algorithm is not None:
        config["recognition"]["algorithm"] = args.rec_algorithm
    if args.rec_batch_num is not None:
        config["recognition"]["batch_num"] = args.rec_batch_num
    
    # Classification options
    if args.use_angle_cls:
        config["classification"]["use_angle_cls"] = True
    if args.no_angle_cls:
        config["classification"]["use_angle_cls"] = False
    
    # Performance options
    if args.use_gpu:
        config["performance"]["use_gpu"] = True
    if args.cpu_only:
        config["performance"]["use_gpu"] = False
    if args.use_mp:
        config["performance"]["use_mp"] = True
    if args.no_mp:
        config["performance"]["use_mp"] = False
    if args.processes is not None:
        config["performance"]["total_process_num"] = args.processes
    
    # Language options
    if args.lang is not None:
        config["languages"]["default"] = args.lang
    
    return config


def save_config(config, path):
    """Save OCR configuration to file."""
    try:
        with open(path, "w") as f:
            json.dump(config, f, indent=4)
        logger.info(f"Saved configuration to {path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save configuration: {e}")
        return False


def main():
    """Main function for updating OCR configuration."""
    parser = get_parser()
    args = parser.parse_args()
    
    # Load existing configuration
    config = load_ocr_config(args.config)
    
    # Reset configuration if requested
    if args.reset:
        logger.info("Resetting configuration to defaults")
        # Load default configuration
        default_config = {
            "detection": {
                "algorithm": "DB",
                "db_thresh": 0.3,
                "db_box_thresh": 0.3,
                "db_unclip_ratio": 1.5,
                "db_score_mode": "fast",
                "use_dilation": False,
                "limit_side_len": 960,
                "limit_type": "max"
            },
            "recognition": {
                "algorithm": "SVTR_LCNet",
                "batch_num": 10
            },
            "classification": {
                "use_angle_cls": False,
                "cls_batch_num": 6,
                "cls_thresh": 0.9
            },
            "languages": {
                "default": "en",
                "available": ["en", "fr", "es", "de", "zh", "ja", "ko"]
            },
            "performance": {
                "use_mp": True,
                "total_process_num": 4,
                "show_log": False,
                "use_gpu": None,
                "gpu_mem_limit": 0.8,
                "auto_tune_batch_size": True
            },
            "models": {
                "text_detection": {
                    "name": "DB++",
                    "path": None
                },
                "text_recognition": {
                    "name": "SVTR_LCNet",
                    "path": None,
                    "alternatives": ["PP-OCRv4", "PP-OCRv3", "CRNN"]
                }
            }
        }
        config = default_config
    else:
        # Update configuration with command-line arguments
        config = update_config(config, args)
    
    # Print updated configuration
    logger.info("Updated configuration:")
    print(json.dumps(config, indent=4))
    
    # Save configuration if requested
    if args.save:
        save_config(config, args.config)
    else:
        logger.info("Configuration not saved. Use --save to save changes.")


if __name__ == "__main__":
    main() 