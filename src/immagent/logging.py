"""Logging configuration for immagent.

Usage:
    import logging
    import immagent

    # Enable debug logging
    logging.basicConfig(level=logging.DEBUG)

    # Or configure the immagent logger specifically
    immagent_logger = logging.getLogger("immagent")
    immagent_logger.setLevel(logging.DEBUG)
    immagent_logger.addHandler(logging.StreamHandler())
"""

import logging

# Create the library logger
logger = logging.getLogger("immagent")

# Add a NullHandler to avoid "No handler found" warnings
# Users can add their own handlers to configure logging
logger.addHandler(logging.NullHandler())
