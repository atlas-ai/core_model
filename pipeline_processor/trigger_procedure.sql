CREATE OR REPLACE FUNCTION table_insert_notify() RETURNS trigger AS $$
DECLARE
    insert_row_timestamp timestamp;
    insert_row_track_id bigint;
    oldest_unprocessed_timestamp timestamp;
    notification json;
BEGIN
    insert_row_timestamp = NEW.data->>'timestamp';
    insert_row_track_id = NEW.data->>'track_id';

    -- Select the oldest unprocessed row timestamp and check if more than 60 seconds have passed
    SELECT MIN((data->>'timestamp')::timestamp)
    INTO oldest_unprocessed_timestamp
    FROM measurement
    WHERE NOT processed
        AND (data->>'track_id')::int = insert_row_track_id;

    IF (insert_row_timestamp - oldest_unprocessed_timestamp >= '45 seconds'::interval)
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