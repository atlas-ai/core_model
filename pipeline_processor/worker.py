import json
import pandas as pd
import cleaning as fin

from sqlalchemy import Float
from work_flow import convert_frame
from multiprocessing import Process
from pipeline_processor.models import Measurement
from work_flow import apply_filter, event_detection_model, acc_detection_model, evt_evaluation_model, \
    acc_evaluation_model, evaluation_summary


class Worker(Process):
    def __init__(self, queue, session):
        super(Worker, self).__init__()
        self.queue = queue
        self.session = session

    def run(self):
        print('Worker started')

        for data in iter(self.queue.get, None):
            data = json.loads(data)

            print('\n\n', data['payload']['data']['track_uuid'])
            print(data['payload']['data']['t'])

            timestamp_from = float(data['oldest_unprocessed_timestamp'])
            timestamp_to = float(data['payload']['data']['t'])
            track_uuid = data['payload']['data']['track_uuid']

            # Query needed measurements data (add 15 seconds (15000 millis) more for overlap data)
            measurements = self.session.query(Measurement)\
                .filter(Measurement.data['t'].astext.cast(Float) >= (timestamp_from - 15))\
                .filter(Measurement.data['t'].astext.cast(Float) <= timestamp_to)\
                .filter(Measurement.data['track_uuid'].astext == track_uuid)

            df = pd.read_sql(measurements.order_by(Measurement.data['t']).statement,
                             measurements.session.bind)

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

            print(df_sum.columns)
            print(df_sum.head())

            if not df_sum.empty:
                # Check if data has been processed already
                already_processed = self.session.query(Measurement)\
                    .filter(Measurement.data['t'].astext.cast(Float) == timestamp_from)\
                    .filter(Measurement.data['track_uuid'].astext == track_uuid).first()

                if not already_processed:
                    # Store results
                    df_sum.to_sql(name='detected_events', con=self.session.get_bind(), if_exists='append')

                    # Update measurements data and set it as processed
                    measurements.update({Measurement.processed: True}, synchronize_session='fetch')
                    self.session.commit()
