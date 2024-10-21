import os
import pickle
import hashlib
from datetime import datetime
from typing import Dict, Any, Optional
from threading import Lock

class MetricCache:
    _instance: Optional['MetricCache'] = None
    _lock: Lock = Lock()

    def __new__(cls, cache_dir: Optional[str] = None):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialize(cache_dir)
        return cls._instance

    def _initialize(self, cache_dir: Optional[str] = None):
        if cache_dir is None:
            cache_dir = os.path.join(os.getcwd(), '.cache', 'metric_cache')
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache: Dict[str, Any] = {}
        self._load_cache()

    @classmethod
    def get_instance(cls) -> 'MetricCache':
        return cls._instance

    def _load_cache(self):
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.pkl'):
                with open(os.path.join(self.cache_dir, filename), 'rb') as f:
                    data = pickle.load(f)
                    self.cache[data['hash']] = data

    def _hash_input(self, dataset_item: Dict[str, Any], pred: Any) -> str:
        input_data = {'dataset_item': dataset_item, 'pred': pred}
        sorted_items = sorted(input_data.items())
        input_str = pickle.dumps(sorted_items)
        return hashlib.md5(input_str).hexdigest()

    def get(self, dataset_item: Dict[str, Any], pred: Any) -> Optional[Any]:
        hash_key = self._hash_input(dataset_item, pred)
        return self.cache.get(hash_key, {}).get('output')

    def set(self, dataset_item: Dict[str, Any], pred: Any, output: Any):
        hash_key = self._hash_input(dataset_item, pred)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"{timestamp}-{hash_key}.pkl"
        data = {
            'dataset_item': dataset_item,
            'pred': pred,
            'hash': hash_key,
            'output': output
        }
        self.cache[hash_key] = data
        with open(os.path.join(self.cache_dir, filename), 'wb') as f:
            pickle.dump(data, f)
