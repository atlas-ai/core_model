import json
import logging
import settings
import pandas as pd
import cleaning as fin

from enum import Enum
from multiprocessing import Process
from sqlalchemy import create_engine
from pipeline_processor import replay
from work_flow import execute_algorithm, track_summary, read_param
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

        cali_param, evt_param, acc_param, param_rtt, param_ltt, param_utn, param_lcr, param_lcl, param_acc, code_sys = \
            read_param(cali_file=settings.CALI_FILE, evt_det_file=settings.EVT_DET_FILE,
                       acc_det_file=settings.ACC_DET_FILE, rtt_eva_file=settings.RTT_EVA_FILE,
                       ltt_eva_file=settings.LTT_EVA_FILE, utn_eva_file=settings.UTN_EVA_FILE,
                       lcr_eva_file=settings.LCR_EVA_FILE, lcl_eva_file=settings.LCL_EVA_FILE,
                       acc_eva_file=settings.ACC_EVA_FILE, code_file=settings.CODE_FILE)

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

            # Query needed measurements data (add 15 seconds (15000 millis) more for overlap data)
            df = get_measurements(timestamp_from=float(data['oldest_unprocessed_timestamp']) - 15,
                                  timestamp_to=payload_data['t'], track_uuid=payload_data['track_uuid'],
                                  engine=self.engine)

            # Emulating Main.write_acc()
            df_data = df['data'].apply(lambda x: pd.Series(x))

            try:
                df_sum = execute_algorithm(raw_data=df_data, cali_param=cali_param, evt_param=evt_param,
                                           acc_param=acc_param, param_rtt=param_rtt, param_ltt=param_ltt,
                                           param_utn=param_utn, param_lcr=param_lcr, param_lcl=param_lcl,
                                           param_acc=param_acc, samp_rate=settings.SAMP_RATE, n_smooth=settings.N_SMOOTH,
                                           tn_thr=settings.TN_THR, lc_thr=settings.LC_THR, acc_thr=settings.ACC_THR,
                                           l1_thr=settings.L1_THR, l2_thr=settings.L2_THR, l3_thr=settings.L3_THR,
                                           l4_thr=settings.L4_THR, device_id=settings.DEVICE_ID,
                                           track_id=payload_data['track_uuid'])

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
            query = """
                    UPDATE measurement
                    SET status = '{new_status}'
                    WHERE (data->>'t')::numeric >= '{timestamp_from}'::numeric
                        AND (data->>'t')::numeric <= '{timestamp_to}'
                        AND (data->>'track_uuid')::uuid = '{track_uuid}'::uuid
                    """.format(timestamp_from=data['oldest_unprocessed_timestamp'],
                       timestamp_to=payload_data['t'],
                       track_uuid=payload_data['track_uuid'],
                       new_status=Status.PROCESSED.value)

            print("SET measurement processed TRACK_UUID {track_uuid} "
                  "FROM {timestamp_from} TO {timestamp_to}".format(track_uuid=payload_data['track_uuid'],
                                                                   timestamp_from=data['oldest_unprocessed_timestamp'],
                                                                   timestamp_to=payload_data['t']))

            con = connect_db()
            cursor = con.cursor()
            cursor.execute(query)
            con.commit()

            # Check if the data coming is a "track_finished" event and if so, call the track cleanup function
            if 'name' in payload_data and payload_data['name'] == Events.TRACK_FINISHED.value:
                print('track_finished EVENT RECEIVED')
                df_detected_events = get_detected_events_for_track(track_uuid=payload_data['track_uuid'],
                                                                   engine=self.engine)

                df_cleaned_detected_events = track_summary(df_eva=df_detected_events, code_sys=settings.CODE_FILE,
                              track_id=payload_data['track_uuid'], l1_thr=settings.L1_THR, l2_thr=settings.L2_THR,
                              l3_thr=settings.L3_THR, l4_thr=settings.L4_THR, acc_fac=settings.ACC_FAC)

                df_cleaned_detected_events.to_sql(name='cleaned_events', con=self.engine, if_exists='append', index=False)
                print('CLEANED EVENTS SAVED, UUID: {0}'.format(payload_data['track_uuid']))
