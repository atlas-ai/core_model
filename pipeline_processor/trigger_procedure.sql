CREATE OR REPLACE FUNCTION table_insert_notify() RETURNS trigger AS $$
DECLARE
    insert_row_timestamp numeric;
    insert_row_track_uuid text;
    oldest_unprocessed_timestamp numeric;
    notification json;
BEGIN
    insert_row_timestamp = NEW.data->>'t';
    insert_row_track_uuid = NEW.data->>'track_uuid';

    -- Check if the new row is a "track_finished" event and if so, trigger the track cleanup
    IF (NEW.data->'name') IS NOT NULL AND (NEW.data->>'name' = 'track_finished')
    THEN
        notification = json_build_object('table', TG_TABLE_NAME,
                                         'action', TG_OP,
                                         'payload', row_to_json(NEW));

        -- Send notification to channel
        PERFORM pg_notify('notifications', notification::text);
    ELSE
        -- Select the oldest unprocessed row timestamp and check if more than 45 seconds have passed (45000 millis)
        SELECT MIN((data->>'t')::numeric)
        INTO oldest_unprocessed_timestamp
        FROM measurement
        WHERE status = 'unprocessed'
            AND (data->>'track_uuid') = insert_row_track_uuid;

        -- Send notification in case there is enough unprocessed data
        IF (insert_row_timestamp - oldest_unprocessed_timestamp >= 45)
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
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;


DROP TRIGGER IF EXISTS table_insert_notify ON measurement;
CREATE TRIGGER table_insert_notify AFTER INSERT ON measurement FOR EACH ROW EXECUTE PROCEDURE table_insert_notify();