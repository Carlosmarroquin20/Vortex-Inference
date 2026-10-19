from prometheus_client import REGISTRY


def metric(name: str, **labels: str) -> float:
    """
    Current value of a sample in the default registry, 0.0 if absent.

    Metrics are process-global and accumulate across tests, so assertions
    should compare values before and after the action under test.
    """
    value = REGISTRY.get_sample_value(name, labels)
    return value if value is not None else 0.0
