import time
import threading
import logging
import gc
import torch
from typing import Callable, Optional, Any

logger = logging.getLogger(__name__)

class SmartModel:
    """
    A smart wrapper that manages model lifecycle with lazy loading and auto-unloading.
    
    Args:
        load_func: Function to load the model.
        timeout_seconds: 
            - n > 0: Unload after n seconds of inactivity.
            - n = 0: Unload immediately after next check (aggressive).
            - n = -1: Never unload automatically (infinite TTL).
        check_interval: Interval in seconds to check for inactivity (default: 30s).
    """
    def __init__(self, load_func: Callable[[], Any], timeout_seconds: int = 7200, check_interval: int = 30):
        self._load_func = load_func
        self._timeout = timeout_seconds
        self._check_interval = check_interval
        self._model: Optional[Any] = None
        self._last_access = 0.0
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        
        # Only start monitor if timeout >= 0
        if self._timeout >= 0:
            self._thread = threading.Thread(target=self._monitor_loop, daemon=True, name="SmartModelMonitor")
            self._thread.start()
        else:
            self._thread = None
            logger.info("SmartModel: Auto-unload disabled (timeout=-1).")

    def get(self):
        """
        Returns the model instance. Loads it if it's not currently loaded.
        Updates the last access time.
        """
        with self._lock:
            self._last_access = time.time()
            if self._model is None:
                logger.info("SmartModel: Loading model...")
                try:
                    self._model = self._load_func()
                    logger.info("SmartModel: Model loaded successfully.")
                except Exception as e:
                    logger.error(f"SmartModel: Failed to load model: {e}")
                    raise e
            return self._model

    def _monitor_loop(self):
        logger.info(f"SmartModel: Monitor started (timeout={self._timeout}s)")
        while not self._stop_event.is_set():
            time.sleep(self._check_interval)
            try:
                self._check_idle()
            except Exception as e:
                logger.error(f"SmartModel: Error in monitor loop: {e}")

    def _check_idle(self):
        with self._lock:
            if self._model is not None:
                idle_time = time.time() - self._last_access
                if idle_time > self._timeout:
                    logger.info(f"SmartModel: Model idle for {idle_time:.1f}s. Unloading...")
                    self._unload()

    def unload(self):
        """Manually unload the model."""
        with self._lock:
            self._unload()

    def _unload(self):
        # Assumes self._lock is held
        if self._model is None:
            return

        # Optional: try to move to cpu first to assist cleanup
        if hasattr(self._model, "cpu"):
            try:
                self._model.cpu()
            except Exception:
                pass

        # Remove reference from manager
        self._model = None
        
        # Trigger GC and CUDA cleanup
        # Note: If the user is still holding a reference (e.g. inside a request),
        # the object won't be destroyed immediately, which is safe.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                 torch.cuda.ipc_collect()
        logger.info("SmartModel: Model unloaded and GPU cache cleared.")

    def stop(self):
        """Stops the background monitoring thread."""
        self._stop_event.set()