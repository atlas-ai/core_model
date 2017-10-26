from os.path import join, dirname
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

DB_CONNECTION_STRING = os.environ.get("DB_CONNECTION_STRING")
NUM_WORKERS = os.environ.get("NUM_WORKERS")
