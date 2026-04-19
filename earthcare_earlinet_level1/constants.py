# File for defining constants

from loguru import logger
import os

logger.remove()
logger.add("[<level>{level:<8}]</level> | <cyan>{name}</cyan> | <green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green>: <level>{message}</level>")

GB_LAT = 44.344
GB_LON = 26.012


S_MOL = 8.503438157

EARTH_RADIUS_KM = 6371.009

FOLDER = None
FOLDERS = None
ROOT = None
GLOB_PATTERN = None
FILES = None
RADII = [50, 100, 200]
GB = "both"
OUTPUT_DIRECTORY = os.path.join(os.getcwd(), "data_dir", "output")


SMOOTH_WIN_M_DEFAULT = 500.0
SEARCH_BOUNDS_M_A = (15500.0, 18000.0)
SEARCH_BOUNDS_M_G = (8000.0, 11000.0)
FALLBACK_BAND_M_A = (15850.0, 17350.0)
FALLBACK_BAND_M_G = (9000.0, 10500.0)

FIGSIZE = (12, 24)
TITLE_SIZE = 24
LABEL_SIZE = 22
TICK_SIZE = 20
LEGEND_SIZE = 20
SAVE_PLOTS = False