import threading
import sqlalchemy
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
        engine = sqlalchemy.create_engine(db_url)

        with engine.connect() as connection:
            connection.execute(sqlalchemy.text("CREATE SCHEMA IF NOT EXISTS optuna;"))
            connection.commit()

        # Create Optuna RDB storage with schema specified
        storage_url = f"{db_url}?options=-c%20search_path=optuna"
        return optuna.storages.RDBStorage(
            url=storage_url,
            engine_kwargs={
                "pool_size": 20,
                "max_overflow": 0,
                "connect_args": {"options": "-c search_path=optuna"},
            },
        )
        # engine = create_engine(db_url)
        # with engine.begin() as conn:
        #     conn.execute(text("CREATE SCHEMA IF NOT EXISTS optuna;"))
        #     # conn.execute(text("SET search_path TO optuna;"))
        #     conn.commit()

        # return optuna.storages.RDBStorage(
        #     url=f"{db_url}?options=-c%20search_path=optuna",
        #     # url=db_url,
        #     engine_kwargs={
        #         "pool_size": 20,
        #         "max_overflow": 0,
        #         "connect_args": {"options": "-c search_path=optuna"},
        #     },
        # )
