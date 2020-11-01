"""Paths.

Exports a number of paths which are used by the other modules.
"""

from os.path import abspath, dirname, join

current_dir = abspath(dirname(__file__))

SRC_DIR = abspath(join(current_dir, "../.."))
PROJECT_ROOT = abspath(join(SRC_DIR, ".."))
DATA_DIR = abspath(join(PROJECT_ROOT, "data"))
MODELS_DIR = abspath(join(PROJECT_ROOT, "models"))
MODEL_CONFIG_DIR = abspath(join(PROJECT_ROOT, "model-config"))
