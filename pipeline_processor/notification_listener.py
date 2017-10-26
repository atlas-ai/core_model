import select
import psycopg2
import psycopg2.extensions

from multiprocessing import Queue
from pipeline_processor.utils import connect_db
from pipeline_processor.worker import Worker

NUM_WORKERS = 3


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

                # Send data to worker
                queue.put(notify.payload)
