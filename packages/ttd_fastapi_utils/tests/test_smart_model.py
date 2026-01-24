import time
import pytest
from ttd_fastapi_utils.model_lifecycle import SmartModel

class MockModel:
    def __init__(self):
        self.alive = True
    
    def cpu(self):
        pass

def load_model():
    return MockModel()

def test_smart_model_lazy_loading():
    """Test that model is loaded only when get() is called."""
    manager = SmartModel(load_model)
    assert manager._model is None
    
    model = manager.get()
    assert isinstance(model, MockModel)
    assert manager._model is not None
    manager.stop()

def test_smart_model_auto_unload():
    """Test that model unloads after timeout."""
    # Timeout 1 second, check interval 0.5 second
    manager = SmartModel(load_model, timeout_seconds=1, check_interval=0.5)
    
    model = manager.get()
    assert manager._model is not None
    
    # Wait for timeout (1s) + buffer
    time.sleep(2)
    
    # Should be unloaded
    assert manager._model is None
    manager.stop()

def test_smart_model_unload_immediate():
    """Test aggressive unload (timeout=0)."""
    manager = SmartModel(load_model, timeout_seconds=0, check_interval=0.5)
    model = manager.get()
    assert manager._model is not None
    
    time.sleep(1)
    assert manager._model is None
    manager.stop()

def test_smart_model_no_unload():
    """Test disabled unload (timeout=-1)."""
    manager = SmartModel(load_model, timeout_seconds=-1)
    # The monitoring thread should not be started
    assert getattr(manager, "_thread", None) is None
    
    model = manager.get()
    assert manager._model is not None
    
    # Even after waiting, it should stay
    time.sleep(1)
    assert manager._model is not None
