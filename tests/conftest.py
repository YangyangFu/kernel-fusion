import pytest

def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "cuda: mark test as requiring CUDA"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "benchmark: mark test as performance benchmark"
    )

def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on their names or requirements."""
    import torch
    
    for item in items:
        # Mark CUDA tests
        if "cuda" in item.name.lower() or any("cuda" in marker.name for marker in item.iter_markers()):
            if not torch.cuda.is_available():
                item.add_marker(pytest.mark.skip(reason="CUDA not available"))
        
        # Mark slow tests
        if "performance" in item.name.lower() or "benchmark" in item.name.lower():
            item.add_marker(pytest.mark.slow)
            item.add_marker(pytest.mark.benchmark)
