import psycopg2
import settings
import pandas as pd

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


def get_measurements(timestamp_from, timestamp_to, track_uuid, engine):
    query = """
                SELECT *
                FROM measurement
                WHERE (data->>'t')::numeric >= ('{timestamp_from}'::numeric - 15)
                    AND (data->>'t')::numeric <= '{timestamp_to}'
                    AND (data->>'track_uuid')::uuid = '{track_uuid}'::uuid
            """.format(timestamp_from=timestamp_from,
                       timestamp_to=timestamp_to,
                       track_uuid=track_uuid)
    return pd.read_sql_query(query, con=engine)


def get_detected_events_for_track(track_uuid, engine):
    query = """
                SELECT *
                FROM detected_events
                WHERE id = '{track_uuid}'
                ORDER BY s_utc
            """.format(track_uuid=track_uuid)
    return pd.read_sql_query(query, con=engine)