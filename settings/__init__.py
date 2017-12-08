import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

DB_CONNECTION_STRING = os.environ.get("DB_CONNECTION_STRING")
NUM_WORKERS = int(os.environ.get("NUM_WORKERS"))

SAMP_RATE = int(os.environ.get("SAMP_RATE", 20))
N_SMOOTH = int(os.environ.get("N_SMOOTH", 20))
Z_THRESHOLD = int(os.environ.get("Z_THRESHOLD", 6))
