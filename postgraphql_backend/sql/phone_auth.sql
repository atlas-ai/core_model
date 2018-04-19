CREATE SCHEMA IF NOT EXISTS api;
CREATE SCHEMA IF NOT EXISTS backend;
create extension if not exists "pgcrypto";


DROP TABLE IF EXISTS api.users CASCADE;
CREATE TABLE api.users (
  ID Serial PRIMARY KEY,
  phone bigint UNIQUE,
  first_name text,
  last_name text,
  createdate timestamp DEFAULT now()
);


DROP TABLE IF EXISTS backend.user_roles CASCADE;
CREATE TABLE backend.user_roles (
	id int PRIMARY KEY,
	role_name varchar
);

INSERT INTO backend.user_roles (id, role_name) VALUES (2, 'school_admin');
INSERT INTO backend.user_roles (id, role_name) VALUES (3, 'instructor');
INSERT INTO backend.user_roles (id, role_name) VALUES (1, 'super_admin');

DROP TABLE IF EXISTS backend.user_account CASCADE;
CREATE TABLE backend.user_account (
	id int REFERENCES api.users,
	password_hash text NOT NULL,
	role_id int REFERENCES backend.user_roles NOT NULL
);


CREATE TABLE backend.identification_codes (
  id serial primary key,
  phone bigint,
  code int,
  createdate timestamp default now(),
  expiration_date timestamp
);


SELECT api.verify_phone(4242323);

DROP FUNCTION IF EXISTS api.get_code(phone bigint);
create OR REPLACE  function api.get_code(
  input_phone bigint
) returns text as $$
declare
  phone_test bigint;
  code int;
  exp_date timestamp;
--  id_code_record backend.identification_codes;
begin
from api.users as a
  select a.phone into phone_test
  where a.phone = input_phone;

  IF phone_test <>0 THEN
    RETURN 'Phone Exists';
  ELSE


--    SELECT expiration_date INTO exp_date
--    FROM backend.identification_codes
--    WHERE phone = input_phone
--    ORDER BY id desc limit 1;

    IF
      (SELECT expiration_date
      FROM backend.identification_codes
      WHERE phone = input_phone
      ORDER BY id desc limit 1) >now()
    THEN
      RETURN 'The code we have sent you is still valid. Please check your messages';

    ELSE
    -- GENERATE CODE
      SELECT (random() * (999999-100000))+100000 into code;

      INSERT INTO backend.identification_codes (phone, code, createdate, expiration_date)
       VALUES (input_phone, code, now(), now()+ interval '60 seconds');
      RETURN code::text;
    END IF;
  END IF;


end;
$$ language plpgsql strict security definer;



--DROP FUNCTION backend.table_insert_identification_codes();
CREATE OR REPLACE FUNCTION backend.table_insert_identification_codes() RETURNS trigger AS $$
DECLARE
    notification json;
BEGIN
    notification = json_build_object('table', TG_TABLE_NAME,
                                        'action', TG_OP,
                                         'payload', row_to_json(NEW));
    -- Send notification to channel_
    PERFORM pg_notify('phone_authentication', notification::text);

    INSERT INTO test_trigger VALUES (notification);

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS table_insert_identification_codes ON backend.identification_codes;
CREATE TRIGGER table_insert_identification_codes AFTER INSERT ON backend.identification_codes FOR EACH ROW EXECUTE PROCEDURE backend.table_insert_identification_codes();



DROP FUNCTION IF EXISTS api.verify_code(phone bigint, code int);
create OR REPLACE  function api.verify_code(
  input_phone bigint,
  input_code int
) returns text as $$
begin


  IF
    (select expiration_date
    from backend.identification_codes as a
    where a.phone = input_phone and a.code=input_code
    ORDER BY id desc LIMIT 1)>now()
  THEN

    return 'EXISTS';
  ELSE
    return 'DOESNT EXIST';
  END IF;


end;
$$ language plpgsql strict security definer;



SELECT api.get_code(3333);

SELECT api.verify_code(3333,185794);

SELECT api.register_user(123456, 'first','last','admin',2);
DROP FUNCTION api.register_user (  phone bigint,first_name text,  last_name text,  password text, role_id int);
create function api.register_user(
  phone bigint,
  first_name text,
  last_name text,
  password text,
  role_id int
) returns api.users as $$
declare
  users api.users;
begin
  insert into api.users (phone, first_name,last_name) values
    (phone,first_name, last_name)
    returning * into users;

  insert into backend.user_account (id,  password_hash,role_id) values
    (users.id, crypt(password, gen_salt('bf')),role_id);

  return users;
end;
$$ language plpgsql strict security definer;

create type api.jwt_token as (
  role text,
  phone bigint
);


DROP FUNCTION if EXISTS api.authenticate(phone bigint, password text);
create or replace function api.authenticate(
  input_phone bigint,
  input_password text
) returns api.jwt_token as $$
declare
  account backend.user_account;
begin
  select a.* into account
  from backend.user_account as a
  where a.id = (Select u.id FROM api.users as u WHERE u.phone=input_phone);

  if account.password_hash = crypt(input_password, account.password_hash) then
    return (
      (SELECT role_name FROM backend.user_roles WHERE id =account.role_id)
      , input_phone)::api.jwt_token;
  else
    return null;
  end if;
end;
$$ language plpgsql strict security definer;


SELECT api.authenticate(123456,'admin');


select a.*
from backend.user_account as a
where a.id = (Select u.id FROM api.users as u WHERE u.phone=18616721479);

comment on function forum_example.register_person(text, text, text, text) is 'Registers a single user and creates an account in our forum.';
```
  
