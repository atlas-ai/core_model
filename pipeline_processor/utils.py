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
                WHERE (data->>'t')::numeric >= '{timestamp_from}'
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


def get_measurements_for_replay(track_uuid, engine):
    query = """
        SELECT data->>'track_uuid' AS track_uuid,
               (data->>'t')::numeric AS t,
               (data->>'alt')::numeric AS alt,
               (data->>'g_x')::numeric AS g_x,
               (data->>'g_y')::numeric AS g_y,
               (data->>'g_z')::numeric AS g_z,
               (data->>'lat')::numeric AS lat,
               (data->>'m_x')::numeric AS m_x,
               (data->>'m_y')::numeric AS m_y,
               (data->>'m_z')::numeric AS m_z,
               (data->>'long')::numeric AS long,
               (data->>'speed')::numeric AS speed,
               (data->>'course')::numeric AS course,
               (data->>'att_yaw')::numeric AS att_yaw,
               data->>'heading' AS heading,
               (data->>'att_roll')::numeric AS att_roll,
               (data->>'user_a_x')::numeric AS user_a_x,
               (data->>'user_a_y')::numeric AS user_a_y,
               (data->>'user_a_z')::numeric AS user_a_z,
               (data->>'att_pitch')::numeric AS att_pitch,
               (data->>'rot_rate_x')::numeric AS rot_rate_x,
               (data->>'rot_rate_y')::numeric AS rot_rate_y,
               (data->>'rot_rate_z')::numeric AS rot_rate_z,
               data->>'name' AS name
        FROM measurement
        WHERE data->>'track_uuid' = '{track_uuid}'
            AND ((data->'name') IS NULL OR data->>'name' NOT LIKE 'replay')
        ORDER BY data->>'t'
                   """.format(track_uuid=track_uuid)
    return pd.read_sql_query(query, con=engine)
