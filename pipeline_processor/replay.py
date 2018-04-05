import json
import time

from pipeline_processor.utils import connect_db, get_measurements_for_replay


def run(track_uuid, new_track_uuid, engine):
    df = get_measurements_for_replay(track_uuid=track_uuid, engine=engine)
    df['track_uuid'] = new_track_uuid
    df = df.sort_values(by='t')

    if not df.empty:
        conn = connect_db()
        curs = conn.cursor()

        print('START REPLAYING ORIGINAL TRACK_UUID:', track_uuid, ', NEW TRACK_UUID:', new_track_uuid)
        print('LENGHT OF THE REPLAY MEASUREMENTS:', len(df))

        # Iterate through all the measurements and insert them again in the measurements table using the new track_uuid
        current_row = df.iloc[[0]]

        for i in range(1, df.shape[0]+1):
            previous_json = current_row.to_json(orient="records")
            previous_json = json.loads(previous_json)

            insert_query = "INSERT INTO measurement_incoming (data) VALUES ('{dict_res}'::jsonb);".format(dict_res=str(json.dumps(previous_json))[1:-1])
            curs.execute(insert_query)
            conn.commit()

            if i < df.shape[0]:
                next_row_timestamp = df.iloc[i]['t']
                current_row_timestamp = current_row.iloc[0]['t']

                # To emulate the phone sending events at the original speed, we sleep for the time difference between
                # the current row and the following one
                time.sleep(next_row_timestamp - current_row_timestamp)
                current_row = df.iloc[[i]]

        curs.close()
        conn.close()

        print('FINISH REPLAYING ORIGINAL TRACK_UUID:', track_uuid, '\nNEW TRACK_UUID:', new_track_uuid, '\n')
    else:
        print('NO MEASUREMENTS FOUND FOR THAT UUID:', track_uuid, '\n')
