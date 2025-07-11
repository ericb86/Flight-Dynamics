import pytest

@pytest.fixture
def common_data():
    return {
        'aircraft': {
            'model': 'TestModel',
            'performance': {
                'speed': 250,
                'altitude': 10000
            }
        },
        'control_system': {
            'gain': 1.5,
            'response_time': 0.1
        }
    }