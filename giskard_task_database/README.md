# giskard_task_database

## Configure the database with new task models
Copy the new pickle-file in the subdirectory ```config```, and it a corresponding entry to the file ```learned_db.yaml```.

## Running the actual server
```roslaunch giskard_task_database task_database.launch dummy_mode:=false```

## Testing
Start the database in ```dummy_mode```:
```roslaunnch giskard_task_database task_database.launch dummy_mode:=true```

In another shell, start the dummy client which sends out a request to test the database:
```rosrun giskard_task_database dummy_query.py```

