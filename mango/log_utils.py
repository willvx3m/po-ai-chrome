#!/usr/bin/env python3

import logging
import os
from typing import Optional


def configure_logging(log_dir: Optional[str] = None, log_filename: str = "log.txt") -> None:
	logger = logging.getLogger()
	logger.setLevel(logging.INFO)

	# Avoid duplicate handlers if reconfigured
	if logger.handlers:
		for h in list(logger.handlers):
			logger.removeHandler(h)

	formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

	# Console
	ch = logging.StreamHandler()
	ch.setLevel(logging.INFO)
	ch.setFormatter(formatter)
	logger.addHandler(ch)

	# File in specified dir or current file's directory
	if log_dir is None:
		log_dir = os.path.abspath(os.path.dirname(__file__))
	os.makedirs(log_dir, exist_ok=True)
	log_path = os.path.join(log_dir, log_filename)
	fh = logging.FileHandler(log_path)
	fh.setLevel(logging.INFO)
	fh.setFormatter(formatter)
	logger.addHandler(fh)

	logging.info(f"Logging to: {log_path}") 