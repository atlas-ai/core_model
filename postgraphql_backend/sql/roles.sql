alter default privileges revoke execute on functions from public;


CREATE OLE atlas_postgraphile;
CREATE ROLE atlas_super_admin;
CREATE ROLE school_admin;
CREATE ROLE instructor;
CREATE ROLE student;



grant atlas_super_admin to atlas_postgraphile;
grant atlas_school_admin to atlas_postgraphile;
grant atlas_instructor to atlas_postgraphile;
grant atlas_student to atlas_postgraphile;
grant atlas_student to atlas_postgraphile;
grant atlas_anonymous to atlas_postgraphile;




CREATE POLICY booking_SCHOOL_ADMIN
  ON api.schedule FOR insert,select,update TO school_admin
  USING (
    instructor_id = current_setting('jwt.claims.user_id')::integer
    and school_id = current_setting('jwt.claims.school_id')::integer);
