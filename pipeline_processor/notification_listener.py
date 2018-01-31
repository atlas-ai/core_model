import select
import psycopg2
import psycopg2.extensions

from multiprocessing import Queue
from pipeline_processor.utils import connect_db
from pipeline_processor.worker import Worker

from settings import NUM_WORKERS

if __name__ == '__main__':
    # Initialize queue and workers
    queue = Queue()
    for i in range(NUM_WORKERS):
        Worker(queue).start()

    conn = connect_db()
    conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)

    channel_name = 'notifications'

    curs = conn.cursor()
    curs.execute("LISTEN {0};".format(channel_name))
    print("\nNOTIFICATION LISTENER LAUNCHED\nWAITING FOR NOTIFICATIONS ON CHANNEL '{0}'".format(channel_name))

    while True:
        if select.select([conn], [], [], 5) == ([], [], []):
            pass
        else:
            conn.poll()
            while conn.notifies:
                notify = conn.notifies.pop(0)

                print("\nNOTIFICATION RECEIVED: '{0}'".format(notify.payload))

                # Send data to worker
                queue.put(notify.payload)
