import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())


SMS_API_ACCOUNT=os.environ.get("SMS_API_ACCOUNT")
SMS_API_PASSWORD=os.environ.get("SMS_API_PASSWORD")


ATLAS_DBNAME=os.environ.get("ATLAS_DBNAME")
DB_CONNECTION_STRING=os.environ.get("DB_CONNECTION_STRING")


POSTGRAPHILE_PORT=os.environ.get("POSTGRAPHILE_PORT")
POSTGRAPHILE_SECRET=os.environ.get("POSTGRAPHILE_SECRET")
