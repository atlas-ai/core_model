import os
import json
import pandas as pd

from multiprocessing import Process
from sqlalchemy import create_engine


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
            print(df.head())

            # Call detection algorithm
            pass
