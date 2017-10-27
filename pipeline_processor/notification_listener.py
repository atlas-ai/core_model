import select
import psycopg2
import psycopg2.extensions

from multiprocessing import Queue
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from pipeline_processor.worker import Worker
from pipeline_processor.utils import connect_db
from settings import NUM_WORKERS, DB_CONNECTION_STRING


if __name__ == '__main__':
    Session = sessionmaker()
    engine = create_engine(DB_CONNECTION_STRING)
    Session.configure(bind=engine)

    # Initialize queue and workers
    queue = Queue()
    for i in range(NUM_WORKERS):
        Worker(queue, Session()).start()

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
