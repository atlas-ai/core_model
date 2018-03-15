CREATE SCHEMA IF NOT EXISTS backend;
CREATE SCHEMA IF NOT EXISTS api;

CREATE EXTENSION IF NOT EXISTS "uuid-ossp" WITH SCHEMA public;

SET SEARCH_PATH = backend;
CREATE TABLE IF NOT EXISTS measurement_incoming (
	id uuid NOT NULL DEFAULT public.uuid_generate_v1mc(),
	data jsonb,
	insert_date timestamp default current_timestamp
);


CREATE INDEX IF NOT EXISTS track_uuid_incoming_index
	ON measurement_incoming ((data->>'track_uuid'));

---------------------------

CREATE TABLE IF NOT EXISTS measurement_processed (
	id uuid NOT NULL DEFAULT public.uuid_generate_v1mc(),
	data jsonb,
	insert_date timestamp default current_timestamp
);

CREATE INDEX IF NOT EXISTS track_uuid_processed_index
	ON measurement_processed ((data->>'track_uuid'));

------------------------------------

CREATE TABLE IF NOT EXISTS current_users (
	recent_users_id SERIAL PRIMARY KEY,
	track_uuid varchar UNIQUE,
	insert_date float
);

----------------------------------

CREATE OR REPLACE FUNCTION table_insert_notify_incoming()
RETURNS trigger AS $$
DECLARE
    insert_row_timestamp numeric;
    insert_row_track_uuid text;
    oldest_unprocessed_timestamp numeric;
    notification json;
BEGIN
		SET SEARCH_PATH =public,backend;
    insert_row_timestamp = NEW.data->>'t';
    insert_row_track_uuid = NEW.data->>'track_uuid';

    IF insert_row_track_uuid IS NOT NULL AND insert_row_timestamp IS NOT NULL
    THEN
	oldest_unprocessed_timestamp =
	(
		SELECT insert_date
		FROM current_users
		WHERE track_uuid = insert_row_track_uuid
	);

	IF oldest_unprocessed_timestamp IS NULL
	THEN
		INSERT INTO current_users (track_uuid, insert_date)
		SELECT insert_row_track_uuid, insert_row_timestamp;
	END IF;



        IF (insert_row_timestamp - oldest_unprocessed_timestamp >= 60) OR
           ((NEW.data->'name') IS NOT NULL AND (NEW.data->>'name' = 'track_finished'))
        THEN
            notification = json_build_object('table', TG_TABLE_NAME,
                                             'action', TG_OP,
                                             'oldest_unprocessed_timestamp', oldest_unprocessed_timestamp,
                                             'payload', row_to_json(NEW));

            -- Send notification to channel_
            PERFORM pg_notify('notifications', notification::text);

            -- Logging
            IF (insert_row_timestamp - oldest_unprocessed_timestamp >= 60)
            THEN
                RAISE NOTICE 'ENOUGH UNPROCESSED MEASUREMENTS, NOTIFICATION SENT TO CHANNEL, UUID:%, FROM:%, TO:%', insert_row_track_uuid, oldest_unprocessed_timestamp, insert_row_timestamp;
            ELSIF (NEW.data->'name') IS NOT NULL AND (NEW.data->>'name' = 'track_finished')
            THEN
                RAISE NOTICE 'TRACK FINISHED EVENT RECEIVED, NOTIFICATION SENT TO CHANNEL, UUID:%, FROM:%, TO:%', insert_row_track_uuid, oldest_unprocessed_timestamp, insert_row_timestamp;
            END IF;
	END IF;

   ELSIF (NEW.data->'name') IS NOT NULL AND (NEW.data->>'name' = 'replay')
    THEN
        notification = json_build_object('table', TG_TABLE_NAME,
                                         'action', TG_OP,
                                         'payload', row_to_json(NEW));

        -- Send notification to channel
        PERFORM pg_notify('notifications', notification::text);

        -- Logging

RAISE NOTICE 'REPLAY EVENT RECEIVED, NOTIFICATION SENT TO CHANNEL, UUID:%, FROM:%, TO:%', insert_row_track_uuid, oldest_unprocessed_timestamp, insert_row_timestamp;

    END IF;


    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS table_insert_notify_incoming ON measurement_incoming ;
CREATE TRIGGER table_insert_notify_incoming AFTER INSERT ON measurement_incoming FOR EACH ROW EXECUTE PROCEDURE backend.table_insert_notify_incoming();

-- TO TEST
--INSERT INTO measurement_incoming (data) values('{"id": 1,"data": "adsadasdsa"}')
--INSERT INTO backend.measurement_incoming (data) values('{"t": 1510198785.434, "alt": 8.079975546112394, "g_x": -0.31675586104393005, "g_y": -0.15031655132770538, "g_z": -0.936520516872406, "lat": 31.03829833984375, "m_x": -32.47932434082031, "m_y": -54.84709167480469, "m_z": -33.0770263671875, "long": 121.12390950520833, "speed": 4.619999885559082, "course": 352.265625, "att_yaw": 0.8920412274502266, "att_roll": -0.32614773835451566, "user_a_x": -0.032853513956069946, "user_a_y": 0.03547890856862068, "user_a_z": 0.005703854840248823, "att_pitch": 0.15088845843776233, "rot_rate_x": 0.1616847708364547, "rot_rate_y": -0.00778228933742955, "rot_rate_z": 0.0006609965825576669, "track_uuid": "f06afb3b-8283-4b84-a342-76bf715cdbf9"}')
