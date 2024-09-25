import threading
import sqlalchemy
import optuna


class OptunaSingletonStorage:
    """
    A singleton class for managing Optuna storage.

    This class ensures that only one instance of the Optuna storage is created
    and reused across the application, following the singleton pattern.
    """

    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        """
        Private constructor to prevent direct instantiation.
        Raises a RuntimeError if called directly.
        """
        raise RuntimeError("Call get_instance() instead")

    @classmethod
    def get_instance(cls, db_url):
        """
        Get or create the singleton instance of OptunaSingletonStorage.

        Args:
            db_url (str): The database URL for creating the storage.

        Returns:
            optuna.storages.RDBStorage: The singleton instance of Optuna RDB storage.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls._create_storage(db_url)
        return cls._instance

    @staticmethod
    def _create_storage(db_url):
        """
        Create and configure the Optuna RDB storage.

        This method creates a SQLAlchemy engine, ensures the 'optuna' schema exists,
        and sets up the Optuna RDB storage with specific configuration.

        Args:
            db_url (str): The database URL for creating the storage.

        Returns:
            optuna.storages.RDBStorage: Configured Optuna RDB storage instance.
        """
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
