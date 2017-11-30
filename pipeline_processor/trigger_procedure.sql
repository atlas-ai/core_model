CREATE OR REPLACE FUNCTION table_insert_notify() RETURNS trigger AS $$
DECLARE
    insert_row_timestamp numeric;
    insert_row_track_uuid text;
    oldest_unprocessed_timestamp numeric;
    notification json;
BEGIN
    insert_row_timestamp = NEW.data->>'t';
    insert_row_track_uuid = NEW.data->>'track_uuid';

    -- Select the oldest unprocessed row timestamp
    SELECT MIN((data->>'t')::numeric)
    INTO oldest_unprocessed_timestamp
    FROM measurement
    WHERE status = 'unprocessed'
        AND (data->>'track_uuid') = insert_row_track_uuid;

    -- Check if there is more than 45 seconds of unprocessed data or if the new row is a "track_finished" event and
    -- send a notification in such cases
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
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;


DROP TRIGGER IF EXISTS table_insert_notify ON measurement;
CREATE TRIGGER table_insert_notify AFTER INSERT ON measurement FOR EACH ROW EXECUTE PROCEDURE table_insert_notify();