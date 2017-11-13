# HOW TO RERUN OLD DATA

 1. Create a local environment so that you can run the pipeline locally. For this you need to create a local database 
 with the same schema as the one in production. You can use the following template for the measurements table:
```
CREATE TYPE status AS ENUM ('unprocessed', 'processing', 'processed');
CREATE TABLE measurement (
        id UUID DEFAULT gen_random_uuid() NOT NULL, 
        data JSONB, 
        status status DEFAULT 'unprocessed'::status, 
        PRIMARY KEY (id)
);
```

2. Create the trigger in your local database. This trigger is in charge of launching the algorithm once there is enough 
 unprocessed measurements.
`psql -U [database_user] -h localhost -p 5432 -f pipeline_processor/trigger_procedure.sql [local_database_name]`

3. Make sure your `DB_CONNECTION_STRING` in your settings is pointing to this local database
 
4. From the project root folder run `python test/test.py --file [test_data_file_path]`. Where the parameter `--file` is
 a path to a CSV file containing measurements from a previous track. The test script will read the data, replace the 
 `track_uuid` with a different one and start inserting the data in the database.
  
5. Right when you  execute 4., open another terminal tab and execute `python pipeline_processor/notification_listener.py`, 
this will launch the worker processes that will hold the algorithm execution once they are triggered.

