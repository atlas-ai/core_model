CREATE SCHEMA IF NOT EXISTS backend;
CREATE SCHEMA IF NOT EXISTS api;

SET SEARCH_PATH to api;

CREATE EXTENSION IF NOT EXISTS "uuid-ossp" WITH SCHEMA api;

DROP TABLE IF EXISTS school CASCADE;
CREATE TABLE school (
id SERIAL PRIMARY KEY,
school_name varchar NOT NULL,
contact_person varchar,
address varchar,
phone int,
email varchar,
date_of_initial_contact timestamp,
date_modified timestamp
);



DROP TABLE IF EXISTS instructor CASCADE;
CREATE TABLE instructor (
id SERIAL PRIMARY KEY NOT NULL,
school_id int REFERENCES school,
name varchar,
phone int NOT NULL UNIQUE,
creation_date timestamp
);



DROP TABLE IF EXISTS schedule CASCADE;
CREATE TABLE schedule (
id SERIAL PRIMARY KEY,
school_id int REFERENCES school,
instructor_id int REFERENCES instructor,
class_date date,
class_start timestamp,
class_end timestamp
);

DROP TABLE IF EXISTS driver CASCADE;
CREATE TABLE driver (
id SERIAL PRIMARY KEY,
school_id int REFERENCES school,
name varchar,
phone int NOT NULL UNIQUE,
date_modified timestamp,
creation_date timestamp
);


DROP TABLE IF EXIST student_score CASCADE;
CREATE TABLE driver_score(
  id SERIAL PRIMARY KEY,
  driver_id int REFERENCES driver,
  total_score float,
  progression float,
  driving_time float,
  d1_focus float,
  f2_efficiency float,
  d3_anticipation float,
  d4_legality float,
  d5_progression float,
  r_turn_score float,
  r_turn_speed float,
  r_turn_acc float,
  l_turn_speed float,
  l_turn_acc float,
  l_turn_score float,
  u_turn_speed float,
  u_turn_acc float,
  u_turn_score float,
  r_lc_speed float,
  r_lc_acc float,
  r_lc_score float,
  l_lc_speed float,
  l_lc_acc float,
  l_lc_score float,
  creation_date timestamp,
  date_modified timestamp
);



DROP TABLE IF EXISTS class_subject CASCADE;
CREATE TABLE class_subject (
id SERIAL PRIMARY KEY,
subject_name varchar,
creation_date timestamp
);


DROP TABLE IF EXISTS physical_device CASCADE;
CREATE TABLE physical_device (
  id SERIAL PRIMARY KEY,
  device_name varchar,
  rot_rate_x float,
  rot_rate_y float,
  user_a_x float,
  user_a_y float,
  user_a_z float,
  cam_matrix_11 float,
  cam_matrix_12 float,
  cam_matrix_13 float,
  cam_matrix_21 float,
  cam_matrix_22 float,
  cam_matrix_23 float,
  cam_matrix_31 float,
  cam_matrix_32 float,
  cam_matrix_33 float,
  distortion_mat_1 float,
  distortion_mat_2 float,
  distortion_mat_3 float,
  distortion_mat_4 float,
  distortion_mat_5 float,
  reso_x float,
  reso_y float,
  creation_date timestamp
);


DROP TABLE IF EXISTS calibration CASCADE;
CREATE TABLE calibration(
id SERIAL PRIMARY KEY,
rot_rate_x float,
rot_rate_y float,
rot_rate_z float,
user_a_x float,
user_a_y float,
user_a_z float,
creation_date timestamp
);

DROP TABLE IF EXISTS device CASCADE;
CREATE TABLE device (
id SERIAL PRIMARY KEY,
physical_id int REFERENCES physical_device,
calibration_id int REFERENCES calibration,
calibration_date timestamp
);

DROP TABLE IF EXISTS route CASCADE;
CREATE TABLE route (
id SERIAL PRIMARY KEY,
school_id int REFERENCES school,
route_name varchar,
creation_date timestamp
);

DROP TABLE IF EXISTS waypoint_subjects CASCADE;
CREATE TABLE waypoint_subjects(
id SERIAL PRIMARY KEY,
subject_name VARCHAR
);

DROP TABLE IF EXISTS waypoints CASCADE;
CREATE TABLE waypoints (
id SERIAL PRIMARY KEY,
route_id int REFERENCES route,
wp_lat float,
wp_long float,
sequence int, -- waypoints order sequence
waypoint_subject_id int REFERENCES waypoint_subjects,
is_last bool
);

DROP TABLE IF EXISTS track CASCADE;
CREATE TABLE track (
id uuid NOT NULL DEFAULT uuid_generate_v1mc() PRIMARY KEY,
driver_id int NOT NULL REFERENCES driver,
device_id int NOT NULL REFERENCES device,
t float,
att_pitch float,
att_roll float,
att_yaw float,
rot_rate_x float,
rot_rate_y float,
rot_rate_z float,
user_a_x float,
user_a_y float,
user_a_z float,
g_x float,
g_y float,
g_z float,
m_x float,
m_y float,
m_z float,
lat float,
long float,
alt float,
speed float,
course float,
name varchar
-- device_id int NOT NULL REFERENCES device
);

DROP TABLE IF EXISTS booking CASCADE;
CREATE TABLE booking (
id SERIAL PRIMARY KEY,
schedule_id int REFERENCES schedule,
driver_id int REFERENCES driver,
subject_id int REFERENCES class_subject,
trackuuid uuid REFERENCES track, -- Trackuiid can be null
route_id int REFERENCES route,
attended boolean,
creation_date timestamp
);
