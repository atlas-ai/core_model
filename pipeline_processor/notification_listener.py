import os
import select
import psycopg2
import psycopg2.extensions

from multiprocessing import Queue
from urllib.parse import urlparse
from pipeline_processor.worker import Worker

NUM_WORKERS = 3


def connect_db():
    result = urlparse(os.environ.get('DB_CONNECTION_STRING'))
    username = result.username
    password = result.password
    database = result.path[1:]
    hostname = result.hostname
    port = result.port

    connection = psycopg2.connect(
        database=database,
        user=username,
        password=password,
        host=hostname,
        port=port)
    return connection


if __name__ == '__main__':
    # Initialize queue and workers
    queue = Queue()
    for i in range(NUM_WORKERS):
        Worker(queue).start()

    conn = connect_db()
    conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)

    curs = conn.cursor()
    curs.execute("LISTEN new_measurements;")

    print("Waiting for notifications on channel 'new_measurements'")

    while True:
        if select.select([conn], [], [], 5) == ([], [], []):
            print("Timeout")
        else:
            conn.poll()
            while conn.notifies:
                notify = conn.notifies.pop(0)
                print("Got NOTIFY:", notify.pid, notify.channel)

                # Send data to worker
                queue.put(notify.payload)
