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

            # Query needed measurements data (add 15 seconds more for overlap data)
            query = """
                        SELECT *
                        FROM measurement
                        WHERE (data->>'timestamp')::timestamp >= ('{timestamp_from}'::timestamp - '15 seconds'::interval)
                            AND (data->>'timestamp')::timestamp <= '{timestamp_to}'
                            AND (data->>'track_id')::int = {track_id}
                    """.format(timestamp_from=data['oldest_unprocessed_timestamp'],
                               timestamp_to=data['payload']['data']['timestamp'],
                               track_id=data['payload']['data']['track_id'])

            df = pd.read_sql_query(query, con=self.engine)
            print(df.head())

            # Call detection algorithm
            pass
