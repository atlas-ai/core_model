CREATE TABLE IF NOT EXISTS measurement_incoming (id uuid NOT NULL DEFAULT uuid_generate_v1mc(), data jsonb, insertdate timestamp);

CREATE INDEX track_uuid_incoming_index ON measurement_incoming ((data->>'track_uuid'));


---------------------------

CREATE TABLE IF NOT EXISTS measurement_processed (id uuid NOT NULL DEFAULT uuid_generate_v1mc(), data jsonb, insertdate timestamp);

CREATE INDEX track_uuid_processed_index ON measurement_processed ((data->>'track_uuid'));


------------------------------------

CREATE TABLE public.current_users (
	recent_users_id SERIAL PRIMARY KEY,
	track_uuid varchar UNIQUE,
	insert_date timestamp
);


