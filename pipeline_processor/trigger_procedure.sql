CREATE OR REPLACE FUNCTION table_insert_notify() RETURNS trigger AS $$
DECLARE
    insert_row_timestamp bigint;
    insert_row_track_uuid text;
    oldest_unprocessed_timestamp bigint;
    notification json;
BEGIN
    insert_row_timestamp = NEW.data->>'t';
    insert_row_track_uuid = NEW.data->>'track_uuid';

    -- Select the oldest unprocessed row timestamp and check if more than 45 seconds have passed (45000 millis)
    SELECT MIN((data->>'t')::bigint)
    INTO oldest_unprocessed_timestamp
    FROM measurement
    WHERE NOT processed
        AND (data->>'track_uuid') = insert_row_track_uuid;

    IF (insert_row_timestamp - oldest_unprocessed_timestamp >= 45000)
    THEN
        -- Send notification in case there is enough unprocessed data
        notification = json_build_object('table', TG_TABLE_NAME,
                                         'action', TG_OP,
                                         'oldest_unprocessed_timestamp', oldest_unprocessed_timestamp,
                                         'payload', row_to_json(NEW));

        PERFORM pg_notify('new_measurements', notification::text);
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;


DROP TRIGGER IF EXISTS table_insert_notify ON measurement;
CREATE TRIGGER table_insert_notify AFTER INSERT ON measurement FOR EACH ROW EXECUTE PROCEDURE table_insert_notify();