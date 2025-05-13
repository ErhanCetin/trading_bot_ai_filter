"""
runner package

Bu paket, sinyal üretimi, veri zamanlaması ve canlı işlem süreçlerini yönetir.
"""

__version__ = "0.1.0"

import logging

logger = logging.getLogger("runner")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())
logger.info("🚀 Runner paketi yüklendi.")
