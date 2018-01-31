CREATE OR REPLACE FUNCTION table_insert_notify() RETURNS trigger AS $$
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
        -- Select the oldest unprocessed row timestamp
        SELECT MIN((data->>'t')::numeric)
        INTO oldest_unprocessed_timestamp
        FROM measurement
        WHERE status = 'unprocessed'
            AND (data->>'track_uuid') = insert_row_track_uuid;

        -- Check if there is more than 45 seconds of unprocessed data or if the new row is a "track_finished" event,
        -- send a notification in such cases and update the measurements that are going to be processed.
        IF (insert_row_timestamp - oldest_unprocessed_timestamp >= 45) OR
           ((NEW.data->'name') IS NOT NULL AND (NEW.data->>'name' = 'track_finished'))
        THEN
            notification = json_build_object('table', TG_TABLE_NAME,
                                             'action', TG_OP,
                                             'oldest_unprocessed_timestamp', oldest_unprocessed_timestamp,
                                             'payload', row_to_json(NEW));

            -- Update the measurements that are going to be processed
            UPDATE measurement
            SET status = 'processing'
            WHERE status = 'unprocessed'
                AND (data->>'t')::numeric >= oldest_unprocessed_timestamp
                AND (data->>'t')::numeric <= insert_row_timestamp
                AND (data->>'track_uuid') = insert_row_track_uuid;

            -- Send notification to channel
            PERFORM pg_notify('notifications', notification::text);

            -- Logging
            IF (insert_row_timestamp - oldest_unprocessed_timestamp >= 45)
            THEN
                RAISE NOTICE 'ENOUGH UNPROCESSED MEASUREMENTS, NOTIFICATION SENT TO CHANNEL, UUID:%, FROM:%, TO:%', insert_row_track_uuid, oldest_unprocessed_timestamp, insert_row_timestamp;
            ELSIF (NEW.data->'name') IS NOT NULL AND (NEW.data->>'name' = 'track_finished')
            THEN
                RAISE NOTICE 'TRACK FINISHED EVENT RECEIVED, NOTIFICATION SENT TO CHANNEL, UUID:%, FROM:%, TO:%', insert_row_track_uuid, oldest_unprocessed_timestamp, insert_row_timestamp;
            END IF;
        END IF;
    -- Check for "replay" event
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


DROP TRIGGER IF EXISTS table_insert_notify ON measurement;
CREATE TRIGGER table_insert_notify AFTER INSERT ON measurement FOR EACH ROW EXECUTE PROCEDURE table_insert_notify();