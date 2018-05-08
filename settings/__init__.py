import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

DB_CONNECTION_STRING = os.environ.get("DB_CONNECTION_STRING")
NUM_WORKERS = int(os.environ.get("NUM_WORKERS"))

SAMP_RATE = int(os.environ.get("SAMP_RATE", 20))
N_SMOOTH = int(os.environ.get("N_SMOOTH", 20))
Z_THRESHOLD = int(os.environ.get("Z_THRESHOLD", 6))
TURN_THRESHOLD = float(os.environ.get("TURN_THRESHOLD", 0.8))
LANE_CHANGE_THRESHOLD = float(os.environ.get("LANE_CHANGE_THRESHOLD", 0.6))


print (DB_CONNECTION_STRING)
print (NUM_WORKERS)

# VARIABLES FOR DB POSTGRAPHILE SETUP
SMS_API_ACCOUNT=os.environ.get("SMS_API_ACCOUNT")
SMS_API_PASSWORD=os.environ.get("SMS_API_PASSWORD")

ATLAS_DBNAME=os.environ.get("ATLAS_DBNAME")
DB_CONNECTION_STRING=os.environ.get("DB_CONNECTION_STRING")
POSTGRAPHILE_PORT=os.environ.get("POSTGRAPHILE_PORT")
POSTGRAPHILE_SECRET=os.environ.get("POSTGRAPHILE_SECRET")
POSGRAPHILE_LOGIN_PW = os.environ.get("POSGRAPHILE_LOGIN_PW")
