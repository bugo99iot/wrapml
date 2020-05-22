import psycopg2
import datetime
from typing import List
from wrapml.imports.vanilla import logger
import sys


def postgres_query(query: str,
                   database_url: str) -> List:

    logger.info('Making Postgres query.')
    start = datetime.datetime.now()

    with psycopg2.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            data = cur.fetchall()

    logger.info('Time taken for Postgres query: {}s'.format((datetime.datetime.now()-start).seconds))
    logger.info('Data size parsed from Postgres query: {}mb'.format(sys.getsizeof(data) / 1000000))

    return data
