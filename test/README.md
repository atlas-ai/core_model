# HOW TO RERUN OLD DATA

 1. Create a local environment so that you can run the pipeline locally. For this you need to create a local database 
 with the same schema as the one in production. You can use the template `create-db-tables.sql` in the project root 
 folder for the measurements tables.

2. Create the trigger in your local database. This trigger is in charge of launching the algorithm once there is enough 
 unprocessed measurements.
`psql -U [database_user] -h localhost -p 5432 -f pipeline_processor/trigger_procedure.sql [local_database_name]`

3. Make sure your `DB_CONNECTION_STRING` in your settings is pointing to test database, and you also have `PYTHONPATH` (full system path to your core_model project) with `NUM_WORKERS`. 
You can add following to your `~/.bashrc`
```
export DB_CONNECTION_STRING='postgres://{{db_user}}:{{db_pw}}@localhost:5432/{{db_name}}'
export PYTHONPATH='/ilya/Documents/gams/data_report_fork'
export NUM_WORKERS='4'
```
 
4. From the project root folder run `python test/test.py --file [test_data_file_path]`. Where the parameter `--file` is
 a path to a CSV file containing measurements from a previous track. The test script will read the data, replace the 
 `track_uuid` with a different one and start inserting the data in the database.
  
5. Right when you  execute 4., open another terminal tab and execute `python pipeline_processor/notification_listener.py`, 
this will launch the worker processes that will hold the algorithm execution once they are triggered.

<br>

# HOW TO EXPORT DATA FROM STAGING DB TO LOCAL CSV

1. Connect to staging server: 
```
ssh -i [path-to-your-private-key] -L 6543:localhost:5432 [user]@dev.atlasaitech.com
```

2. Execute the following query substituting the track uuid that you want to export the data for (it will prompt you for the password)
```
psql -U atlas -d atlas -h localhost -p 6543 -c "
COPY (
        SELECT data->>'track_uuid' AS track_uuid,
               (data->>'t')::numeric AS t,
               (data->>'alt')::numeric AS alt,
               (data->>'g_x')::numeric AS g_x,
               (data->>'g_y')::numeric AS g_y,
               (data->>'g_z')::numeric AS g_z,
               (data->>'lat')::numeric AS lat,
               (data->>'m_x')::numeric AS m_x,
               (data->>'m_y')::numeric AS m_y,
               (data->>'m_z')::numeric AS m_z,
               (data->>'long')::numeric AS long,
               (data->>'speed')::numeric AS speed,
               (data->>'course')::numeric AS course,
               (data->>'att_yaw')::numeric AS att_yaw,
               data->>'heading' AS heading,
               (data->>'att_roll')::numeric AS att_roll,
               (data->>'user_a_x')::numeric AS user_a_x,
               (data->>'user_a_y')::numeric AS user_a_y,
               (data->>'user_a_z')::numeric AS user_a_z,
               (data->>'att_pitch')::numeric AS att_pitch,
               (data->>'rot_rate_x')::numeric AS rot_rate_x,
               (data->>'rot_rate_y')::numeric AS rot_rate_y,
               (data->>'rot_rate_z')::numeric AS rot_rate_z,
               data->>'name' AS name
        FROM measurement_processed
        WHERE data->>'track_uuid' IN ('YOUR UUID HERE')
        AND ((data->'name') IS NULL OR data->>'name' NOT LIKE 'replay')
        ORDER BY data->>'track_uuid', data->>'t'
) TO STDOUT WITH CSV HEADER DELIMITER ';';
" > test/test_data/track_data.csv
```
Now the data will be downloaded into test/test_data/track_data.csv

