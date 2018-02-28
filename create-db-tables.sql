-- For every user, 50 times per second, data are inserted in this table

CREATE TABLE IF NOT EXISTS measurement_incoming (
	id uuid NOT NULL DEFAULT uuid_generate_v1mc(),
	data jsonb, 
	insertdate timestamp
);

CREATE INDEX track_uuid_incoming_index ON measurement_incoming ((data->>'track_uuid'));

-- Once processed, data will tranfered into "measurement_processed"

CREATE TABLE IF NOT EXISTS measurement_processed (
	id uuid NOT NULL DEFAULT uuid_generate_v1mc(), 
	data jsonb, 
	insertdate timestamp
);

CREATE INDEX track_uuid_processed_index ON measurement_processed ((data->>'track_uuid'));


-- For every data inserted, a trigger reads this table to check if the quantity data for a trackid 
-- is greated than decided interval (ex. 60 seconds of data)

CREATE TABLE public.current_users (
	recent_users_id SERIAL PRIMARY KEY,
	track_uuid varchar UNIQUE,
	insert_date numeric
);


