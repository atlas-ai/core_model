import os
import json
import pandas as pd
import frame as ffc
import cleaning as fin

from multiprocessing import Process
from sqlalchemy import create_engine
from Main import detection_model


class Worker(Process):
    def __init__(self, queue):
        super(Worker, self).__init__()
        self.queue = queue
        self.engine = create_engine(os.environ.get('DB_CONNECTION_STRING'))

    def run(self):
        print('Worker started')

        for data in iter(self.queue.get, None):
            data = json.loads(data)

            print(data)

            # Query needed measurements data (add 15 seconds (15000 millis) more for overlap data)
            query = """
                        SELECT *
                        FROM measurement
                        WHERE (data->>'t')::bigint >= ('{timestamp_from}'::bigint - 15000)
                            AND (data->>'t')::bigint <= '{timestamp_to}'
                            AND (data->>'track_uuid')::uuid = '{track_uuid}'::uuid
                    """.format(timestamp_from=data['oldest_unprocessed_timestamp'],
                               timestamp_to=data['payload']['data']['t'],
                               track_uuid=data['payload']['data']['track_uuid'])

            df = pd.read_sql_query(query, con=self.engine)

            # Emulating Main.write_acc()
            df_data = df['data'].apply(lambda x: pd.Series(x))
            df_data['t'] = df_data['t'].apply(lambda x: x/1000)

            gps_data = df_data[['t', 'lat', 'long', 'alt', 'course', 'speed']]
            imu_data = df_data[['t', 'att_pitch', 'att_roll', 'att_yaw', 'rot_rate_x', 'rot_rate_y', 'rot_rate_z',
                                'g_x', 'g_y', 'g_z', 'user_a_x', 'user_a_y', 'user_a_z', 'm_x', 'm_y', 'm_z']]

            gps = fin.gps_data(gps_data)
            print('\n\nGPS\n', gps)
            imu = fin.imu_data(imu_data)
            print('\n\nIMU\n', imu)

            acc = ffc.car_acceleration(imu['rot_rate_x'], imu['rot_rate_y'], imu['rot_rate_z'], imu['user_a_x'],
                                       imu['user_a_y'], imu['user_a_z'], imu['g_x'], imu['g_y'], imu['g_z'], imu['m_x'],
                                       imu['m_y'], imu['m_z'], gps['course'], gps['speed'])

            n_smooth = 100
            acc_smooth = acc.rolling(n_smooth).mean()
            print('\n\nACC SMOOTH\n', acc_smooth)

            # Calling Main.detection_model()
            df_evt = detection_model(acc_smooth)
            print('\n\nDF EVENT\n', df_evt)

            if not df_evt.empty:
                df_evt.to_sql(name='detected_events', con=self.engine, if_exists='append')
