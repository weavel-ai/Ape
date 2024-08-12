import threading
from sqlalchemy import create_engine, text
import optuna


class OptunaSingletonStorage:
    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        raise RuntimeError("Call get_instance() instead")

    @classmethod
    def get_instance(cls, db_url):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls._create_storage(db_url)
        return cls._instance

    @staticmethod
    def _create_storage(db_url):
        engine = create_engine(db_url)
        with engine.begin() as conn:
            conn.execute(text("CREATE SCHEMA IF NOT EXISTS optuna;"))
            conn.execute(text("SET search_path TO optuna;"))

        return optuna.storages.RDBStorage(
            url=db_url,
            engine_kwargs={
                "pool_size": 20,
                "max_overflow": 0,
            },
        )
