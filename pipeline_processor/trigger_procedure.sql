
CREATE OR REPLACE FUNCTION table_insert_notify_incoming() RETURNS trigger AS $$
DECLARE
    insert_row_timestamp numeric;
    insert_row_track_uuid text;
    oldest_unprocessed_timestamp numeric;
    notification json;
BEGIN
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
	    -- Keep last 15 seconds of data
      	    UPDATE current_users
            SET insert_date = insert_row_timestamp-15
      	    WHERE track_uuid = insert_row_track_uuid;

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


DROP TRIGGER IF EXISTS table_insert_notify_incoming ON measurement_incoming;
CREATE TRIGGER table_insert_notify_incoming AFTER INSERT ON measurement_incoming FOR EACH ROW EXECUTE PROCEDURE table_insert_notify_incoming();
