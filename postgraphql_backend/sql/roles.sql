alter default privileges revoke execute on functions from public;





CREATE ROLE atlas_postgraphile;
CREATE ROLE super_admin;
CREATE ROLE school_admin;
CREATE ROLE instructor;
CREATE ROLE student;
CREATE ROLE anonymous;

grant usage on schema api to atlas_postgraphile,
  super_admin,
  school_admin,
  instructor,
  student,
  anonymous;


grant uper_admin to atlas_postgraphile;
grant school_admin to atlas_postgraphile;
grant instructor to atlas_postgraphile;
grant student to atlas_postgraphile;
grant anonymous to atlas_postgraphile;


ALTER TABLE api.users enable ROW LEVEL SECURITY;

grant select on table api.users to super_admin,
school_admin,
instructor,
student;


DROP POLICY selectusers_studentrole ON api.users;
CREATE POLICY selectUsers_studentRole ON api.users FOR SELECT TO student, instructor, school_admin
USING (id=3);




grant select on table api.jwt_user_id_v  to super_admin,
school_admin,
instructor,
student;


CREATE VIEW api.jwt_user_id_v as
(
SELECT u.id
FROM api.users as u
WHERE u.phone=current_setting('jwt.claims.phone')::bigint
);


DROP POLICY selectusers_studentrole ON api.users;
CREATE POLICY selectUsers_studentRole ON api.users FOR SELECT TO student, instructor, school_admin
USING (
  id= (SELECT id FROM api.jwt_user_id_v)); -- Without a view causes infinite recursion error



alter table forum_example.person enable row level security;


CREATE POLICY booking_SCHOOL_ADMIN
  ON api.schedule FOR insert,select,update TO school_admin
  USING (
    instructor_id = current_setting('jwt.claims.user_id')::integer
    and school_id = current_setting('jwt.claims.school_id')::integer);



CREATE POLICY testPolicy ON api.booking FOR SELECT TO student
USING (id=2);

ALTER table api.testPolicies enable ROW LEVEL SECURITY;


SET ROLE=student;
SELECT * FROM api.testPolicies;

SET ROLE=postgres;
grant select on table api.testPolicies TO student;
CREATE POLICY testPolicies ON api.testPolicies FOR SELECT TO student USING (id=2);


DECLARE
  jwt_phone bigint;

  INTO jwt_phone;

  RETURN jwt_phone
  S SELECT current_setting('jwt.claims.phone')::bigint

DROP FUNCTION api.get_token_phone;

create or replace function api.get_token_phone() returns bigint as $$
  DECLARE
    jwt_phone bigint;
BEGIN
  SELECT 1 into jwt_phone;
  RETURN jwt_phone;
END;
$$ language plpgsql  stable;



create or replace function api.get_token_phone() returns bigint as $$
  DECLARE
    jwt_phone bigint;
BEGIN
  SELECT current_setting('jwt.claims.phone')::bigint into jwt_phone;
  RETURN jwt_phone;
END;
$$ language plpgsql stable;
