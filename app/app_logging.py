"""
App logging helpers.
"""

import os


DEBUG = os.getenv("CVLA_DEBUG") == "1"


def dlog(msg: str):
    if DEBUG:
        print(msg)
