import json
import settings
import pandas as pd
import cleaning as fin

from enum import Enum
from work_flow import convert_frame
from multiprocessing import Process
from sqlalchemy import create_engine
from pipeline_processor.utils import connect_db
from work_flow import apply_filter, event_detection_model, acc_detection_model, evt_evaluation_model, \
    acc_evaluation_model, evaluation_summary


class Status(Enum):
    UNPROCESSED = 'unprocessed'
    PROCESSING = 'processing'
    PROCESSED = 'processed'


class Worker(Process):
    def __init__(self, queue):
        super(Worker, self).__init__()
        self.queue = queue
        self.engine = create_engine(settings.DB_CONNECTION_STRING)

    def run(self):
        print('Worker started')

        for data in iter(self.queue.get, None):
            data = json.loads(data)

            print('\nNEW WORKER LAUNCHED, TRACK_UUID', data['payload']['data']['track_uuid'])
            print('TIMESTAMP FROM:', data['oldest_unprocessed_timestamp'], 'TO:', data['payload']['data']['t'], 'DIFF:',
                  (data['payload']['data']['t'] - data['oldest_unprocessed_timestamp']))

            # Query needed measurements data (add 15 seconds (15000 millis) more for overlap data)
            query = """
                        SELECT *
                        FROM measurement
                        WHERE (data->>'t')::numeric >= ('{timestamp_from}'::numeric - 15)
                            AND (data->>'t')::numeric <= '{timestamp_to}'
                            AND (data->>'track_uuid')::uuid = '{track_uuid}'::uuid
                    """.format(timestamp_from=data['oldest_unprocessed_timestamp'],
                               timestamp_to=data['payload']['data']['t'],
                               track_uuid=data['payload']['data']['track_uuid'])

            df = pd.read_sql_query(query, con=self.engine)

            # Emulating Main.write_acc()
            df_data = df['data'].apply(lambda x: pd.Series(x))

            gps_data = df_data[['t', 'lat', 'long', 'alt', 'course', 'speed']]
            imu_data = df_data[['t', 'att_pitch', 'att_roll', 'att_yaw', 'rot_rate_x', 'rot_rate_y', 'rot_rate_z',
                                'g_x', 'g_y', 'g_z', 'user_a_x', 'user_a_y', 'user_a_z', 'm_x', 'm_y', 'm_z']]

            gps = fin.gps_data(gps_data)
            imu = fin.imu_data(imu_data)

            df_fc = convert_frame(imu, gps)
            acc_x, acc_y, rot_z, crs, spd = apply_filter(df_fc, n_smooth=20)
            df_evt = event_detection_model(rot_z, crs, spd)
            df_acc = acc_detection_model(acc_x, crs, spd, z_threshold=6)
            df_evt_eva = evt_evaluation_model(acc_x, acc_y, spd, df_evt)
            df_acc_eva = acc_evaluation_model(df_acc, z_threshold=6)
            df_sum = evaluation_summary(data['payload']['data']['track_uuid'], df_evt_eva, df_acc_eva)

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
                               track_uuid=data['payload']['data']['track_uuid'])
                df_processed = pd.read_sql_query(query, con=self.engine)

                # If data hasn't been processed yet then store the results
                if df_processed.empty or df_processed.ix[0]['status'] != Status.PROCESSED.value:
                    df_sum.to_sql(name='detected_events', con=self.engine, if_exists='append', index=False)
                    print('RESULTS SAVED')
            else:
                print('ALGORITHM DIDN\'T RETURN ANYTHING')

            # No matter if the algorithm returned any results or not, update measurements data and set it as processed
            query = """
                    UPDATE measurement
                    SET status = '{new_status}'
                    WHERE (data->>'t')::numeric >= '{timestamp_from}'::numeric
                        AND (data->>'t')::numeric <= '{timestamp_to}'
                        AND (data->>'track_uuid')::uuid = '{track_uuid}'::uuid
                    """.format(timestamp_from=data['oldest_unprocessed_timestamp'],
                       timestamp_to=data['payload']['data']['t'],
                       track_uuid=data['payload']['data']['track_uuid'],
                       new_status=Status.PROCESSED.value)

            print("SET measurement processed TRACK_UUID {track_uuid} "
                  "FROM {timestamp_from} TO {timestamp_to}".format(track_uuid=data['payload']['data']['track_uuid'],
                                                                   timestamp_from=data['oldest_unprocessed_timestamp'],
                                                                   timestamp_to=data['payload']['data']['t']))

            con = connect_db()
            cursor = con.cursor()
            cursor.execute(query)
            con.commit()
