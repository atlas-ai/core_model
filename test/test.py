import json
import uuid
import time
import pandas as pd
from pipeline_processor.utils import connect_db

TIME_INTERVAL = 1

if __name__ == '__main__':
    filename = "test/test_data/atlas_ai_data_18561c11-dd73-4ec9-9ca0-42b13c28048f.csv"

    # Test real data
    df = pd.read_csv(filename, sep=';')
    conn = connect_db()
    curs = conn.cursor()

    uuid_str = str(uuid.uuid4())
    df['track_uuid'] = uuid_str

    df = df.sort_values(by='t')

    min_t = df['t'].min()
    max_t = df['t'].max()
    tmp_t = min_t

    print('\nTOTAL SECONDS IN TRACK:', int(max_t - min_t), '\n')

    while tmp_t < max_t:
        df_filter = df[(df['t'] >= tmp_t) & (df['t'] < tmp_t + TIME_INTERVAL)]

        if not df_filter.empty:
            df_json = df_filter.to_json(orient="records")
            df_json = json.loads(df_json)

            insert_query = "INSERT INTO measurement (data) VALUES "

            for item in df_json:
                insert_query += """('{dict_res}'::jsonb),""".format(dict_res=json.dumps(item))

            insert_query = insert_query[0:-1]
            insert_query += ';'

            print('SECONDS LEFT:', int(max_t - tmp_t), 'MIN t:', df_filter['t'].min(), 'MAX t:', df_filter['t'].max(), 'COUNT:', df_filter['t'].count())
            curs.execute(insert_query)
            conn.commit()

            time.sleep(TIME_INTERVAL)

        tmp_t += TIME_INTERVAL

    curs.close()
    conn.close()
