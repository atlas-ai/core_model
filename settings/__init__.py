import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

DB_CONNECTION_STRING = os.environ.get("DB_CONNECTION_STRING")
NUM_WORKERS = int(os.environ.get("NUM_WORKERS"))
