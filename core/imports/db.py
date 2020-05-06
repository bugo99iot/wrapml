import psycopg2
import datetime
from typing import List
from core.utils.logging import logger
import sys


def postgres_query(query: str,
                   database_url: str) -> List:

    start = datetime.datetime.now()

    with psycopg2.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            data = cur.fetchall()

    logger.info('Time taken: {}s'.format((datetime.datetime.now()-start).seconds))
    logger.info('Data size: {}mb'.format(sys.getsizeof(data)))

    return data
