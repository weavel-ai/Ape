import os
import pickle
import hashlib
from datetime import datetime
from typing import Dict, Any, Optional, List
from threading import Lock

class PromptCache:
    _instance: Optional['PromptCache'] = None
    _lock: Lock = Lock()
    
    def __new__(cls, cache_dir: Optional[str] = None):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialize(cache_dir)
        return cls._instance

    def _initialize(self, cache_dir: Optional[str] = None):
        if cache_dir is None:
            cache_dir = os.path.join(os.getcwd(), '.cache', 'prompt_cache')
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache: Dict[str, Any] = {}
        self._load_cache()

    @classmethod
    def get_instance(cls) -> 'PromptCache':
        return cls._instance

    def _load_cache(self):
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.pkl'):
                with open(os.path.join(self.cache_dir, filename), 'rb') as f:
                    data = pickle.load(f)
                    self.cache[data['hash']] = data
                    
    def _hash_input(
        self,
        messages: List[Dict[str, str]],
        lm_config: Dict[str, Any],
        kwargs: Dict[str, Any],
        parallel_task_id: int
    ) -> str:
        combined_data = {
            'messages': messages,
            'lm_config': lm_config,
            'kwargs': kwargs,
            'parallel_task_id': parallel_task_id
        }
        sorted_items = sorted(combined_data.items())
        input_str = pickle.dumps(sorted_items)
        return hashlib.md5(input_str).hexdigest()

    def get(
        self,
        messages: List[Dict[str, str]],
        lm_config: Dict[str, Any],
        kwargs: Dict[str, Any],
        parallel_task_id: int
    ) -> Optional[Any]:
        hash_key = self._hash_input(
            messages,
            lm_config,
            kwargs,
            parallel_task_id
        )
        return self.cache.get(hash_key, {}).get('output')

    def set(
        self,
        messages: List[Dict[str, str]],
        lm_config: Dict[str, Any],
        kwargs: Dict[str, Any],
        output: Any,
        parallel_task_id: int
    ):
        hash_key = self._hash_input(
            messages,
            lm_config,
            kwargs,
            parallel_task_id
        )
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"{timestamp}-{hash_key}.pkl"
        data = {
            'messages': messages,
            'lm_config': lm_config,
            'kwargs': kwargs,
            'parallel_task_id': parallel_task_id,
            'hash': hash_key,
            'output': output
        }
        self.cache[hash_key] = data
        with open(os.path.join(self.cache_dir, filename), 'wb') as f:
            pickle.dump(data, f)
