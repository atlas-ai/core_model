import psycopg2
import settings

from urllib.parse import urlparse


def connect_db():
    result = urlparse(settings.DB_CONNECTION_STRING)
    username = result.username
    password = result.password
    database = result.path[1:]
    hostname = result.hostname
    port = result.port

    connection = psycopg2.connect(
        database=database,
        user=username,
        password=password,
        host=hostname,
        port=port)
    return connection
