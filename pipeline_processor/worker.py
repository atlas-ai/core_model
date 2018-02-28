import json
import logging
import settings
import pandas as pd
import cleaning as fin

from enum import Enum
from multiprocessing import Process
from sqlalchemy import create_engine
from pipeline_processor import replay
from work_flow import execute_algorithm, clean_results
from pipeline_processor.utils import connect_db, get_detected_events_for_track, get_measurements


class Status(Enum):
    UNPROCESSED = 'unprocessed'
    PROCESSING = 'processing'
    PROCESSED = 'processed'


class Events(Enum):
    TRACK_FINISHED = 'track_finished'
    REPLAY = 'replay'


class Worker(Process):
    def __init__(self, queue):
        super(Worker, self).__init__()
        self.queue = queue
        self.engine = create_engine(settings.DB_CONNECTION_STRING)

    def terminate(self):
        print('\nWORKER TERMINATED\n')

    def run(self):
        print('WORKER STARTED')

        for data in iter(self.queue.get, None):
            data = json.loads(data)
            payload_data = data['payload']['data']

            if 'name' in payload_data and payload_data['name'] == Events.REPLAY.value:
                replay.run(track_uuid=payload_data['original_track_uuid'],
                           new_track_uuid=payload_data['new_track_uuid'],
                           engine=self.engine)
                return

            print('NEW WORKER LAUNCHED, TRACK_UUID', payload_data['track_uuid'])
            print('TIMESTAMP FROM:', data['oldest_unprocessed_timestamp'], 'TO:', payload_data['t'], 'DIFF:',
                  (payload_data['t'] - data['oldest_unprocessed_timestamp']))

            # Query needed measurements data
            df = get_measurements(timestamp_to=payload_data['t'], track_uuid=payload_data['track_uuid'],
                                  engine=self.engine)

            # Emulating Main.write_acc()
            df_data = df['data'].apply(lambda x: pd.Series(x))

            gps_data = df_data[['t', 'lat', 'long', 'alt', 'course', 'speed']]
            imu_data = df_data[['t', 'att_pitch', 'att_roll', 'att_yaw', 'rot_rate_x', 'rot_rate_y', 'rot_rate_z',
                                'g_x', 'g_y', 'g_z', 'user_a_x', 'user_a_y', 'user_a_z', 'm_x', 'm_y', 'm_z']]

            gps = fin.gps_data(gps_data)
            imu = fin.imu_data(imu_data)

            try:
                df_sum = execute_algorithm(imu, gps, payload_data['track_uuid'], samp_rate=settings.SAMP_RATE,
                                           n_smooth=settings.N_SMOOTH, z_threshold=settings.Z_THRESHOLD,
                                           turn_threshold=settings.TURN_THRESHOLD,
                                           lane_change_threshold=settings.LANE_CHANGE_THRESHOLD)

                if not df_sum.empty:
                    print('UNIQUE ALGORITHM RESULTS:', df_sum['type'].unique())
                    # Check if data has been processed already
                    query = """
                            SELECT status
                            FROM measurement
                            WHERE (data->>'t')::numeric = '{timestamp_from}'::numeric
                                AND (data->>'track_uuid')::uuid = '{track_uuid}'::uuid
                            LIMIT 1
                        """.format(timestamp_from=data['oldest_unprocessed_timestamp'],
                                   track_uuid=payload_data['track_uuid'])
                    df_processed = pd.read_sql_query(query, con=self.engine)

                    # If data hasn't been processed yet then store the results
                    if df_processed.empty or df_processed.ix[0]['status'] != Status.PROCESSED.value:
                        df_sum.to_sql(name='detected_events', con=self.engine, if_exists='append', index=False)
                        print('RESULTS SAVED')
                else:
                    print('ALGORITHM DIDN\'T RETURN ANYTHING')
            except BaseException as e:
                logging.exception("AN EXCEPTION OCURRED")

            # No matter if the algorithm returned any results or not, update measurements data and set it as processed
            con = connect_db()
            cursor = con.cursor()

            query = """
                    INSERT INTO measurement_processed (data)
                    SELECT data FROM measurement_incoming
                    WHERE (data->>'t')::numeric <= '{timestamp_to}'
                        AND (data->>'track_uuid')::uuid = '{track_uuid}'::uuid;
            """.format(track_uuid=payload_data['track_uuid'], timestamp_to=float(payload_data['t'])-15)

            cursor.execute(query)

#            print("SET measurement processed TRACK_UUID {track_uuid} "
#                  "FROM {timestamp_from} TO {timestamp_to}".format(track_uuid=payload_data['track_uuid'],
#                                                                   timestamp_from=data['oldest_unprocessed_timestamp'],
#                                                                   timestamp_to=payload_data['t']))

            query = """
                    DELETE FROM measurement_incoming
                    WHERE (data->>'t')::numeric <= '{timestamp_to}'
                          AND (data->>'track_uuid')::uuid = '{track_uuid}'::uuid;
                    """.format(track_uuid=payload_data['track_uuid'], timestamp_to=float(payload_data['t'])-15)

            cursor.execute(query)
            con.commit()

            # Check if the data coming is a "track_finished" event and if so, call the track cleanup function
            if 'name' in payload_data and payload_data['name'] == Events.TRACK_FINISHED.value:
                print('track_finished EVENT RECEIVED')
                df_detected_events = get_detected_events_for_track(track_uuid=payload_data['track_uuid'],
                                                                   engine=self.engine)
                df_cleaned_detected_events = clean_results(track_uuid=payload_data['track_uuid'],
                                                           df_detected_events=df_detected_events)

                df_cleaned_detected_events.to_sql(name='cleaned_events', con=self.engine, if_exists='append', index=False)
                print('CLEANED EVENTS SAVED, UUID: {0}'.format(payload_data['track_uuid']))
