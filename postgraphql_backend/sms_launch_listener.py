import select
import psycopg2
import psycopg2.extensions

from multiprocessing import Queue
from multiprocessing import Process

from sqlalchemy import create_engine
from urllib.parse import urlparse
import requests
import json
import os

import settings



NUM_WORKERS=10


result = urlparse(settings.DB_CONNECTION_STRING)
username = result.username
password = result.password
database = result.path[1:]
hostname = result.hostname
port = result.port



class Worker(Process):
    def __init__(self, queue):
        super(Worker, self).__init__()
        self.queue = queue
        self.engine = create_engine(settings.DB_CONNECTION_STRING)

    def terminate(self):
        print('\nWORKER TERMINATED\n')

    def run(self):
        print('WORKER STARTED')

        for data in iter(self.queue.get, None):
            data = json.loads(data)
            payload_data = data['payload']['data']

        print(payload_data)

if __name__ == '__main__':
    # Initialize queue and workers
    queue = Queue()
    for i in range(NUM_WORKERS):
        Worker(queue).start()

#    conn = connect_db()
    conn=psycopg2.connect(
            database=database,
            user=username,
            password=password,
            host=hostname,
            port=port)
    conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)

    channel_name = 'phone_authentication'

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

#                print("\nNOTIFICATION RECEIVED: '{0}'".format(notify.payload))
                payload=json.loads (notify.payload)
                print(payload['payload'])
                phone=payload['payload']['phone']
                code=payload['payload']['code']
                exp_date=payload['payload']['expiration_date']
                    # Send data to worker
                message='Your authentication code is {0}'.format(code)

                request_url='http://mt.10690404.com/send.do?Account={0}&Password={1}&Mobile={2}&Content={3}&Exno=0&Fmt=json'.format(settings.SMS_API_ACCOUNT,settings.SMS_API_PASSWORD,phone,message)
                print(request_url)
                requests.get(request_url)
