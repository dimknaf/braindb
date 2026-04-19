from contextlib import contextmanager

import psycopg2
import psycopg2.extras

from braindb.config import settings


@contextmanager
def get_conn():
    conn = psycopg2.connect(settings.database_url)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
