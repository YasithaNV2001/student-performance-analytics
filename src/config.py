"""Shared settings: file locations, the random seed and the pass mark."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / 'data' / 'raw'
PROCESSED_DIR = ROOT / 'data' / 'processed'
FIGURES_DIR = ROOT / 'figures'
REPORTS_DIR = ROOT / 'reports'
MODELS_DIR = ROOT / 'models'

RANDOM_STATE = 42   # used by every split, model and clustering, so results are reproducible
PASS_MARK = 10      # UCI grades are out of 20; a final grade of 10 or more is a pass
