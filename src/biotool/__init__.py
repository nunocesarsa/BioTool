"""biotool: local, CPU-based multi-model bioacoustics processing."""

import os as _os

_os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # keep TensorFlow's start-up chatter out of the log

__version__ = "0.3.0"
