"""
A registry for eval metric.
"""

METRIC_REGISTRY = {}

def register_metric(name):
    """
    A decorator to register a metric class with a specified name in the global METRIC_REGISTRY.

    Args:
        name (str): The name under which the metric class should be registered.

    Returns:
        decorator (function): A function that takes a class `cls` as an argument,
                              registers it in the METRIC_REGISTRY, and returns the class.
    """

    def decorator(cls):
        METRIC_REGISTRY[name] = cls
        return cls
    return decorator

def get_metric(metric_name, vllm_port):
    """
    Retrieves a metric class instance from the METRIC_REGISTRY by name.

    Args:
        metric_name (str): The name of the metric to retrieve.
        vllm_port (int): The port number of the VLLM server.

    Returns:
        metric (Metrics): An instance of the metric class with the specified name.

    Raises:
        ValueError: If the metric name is not found in the METRIC_REGISTRY.
    """
    # format name
    registry_metric_name = metric_name.replace("-", "_").lower()
    if registry_metric_name not in METRIC_REGISTRY:
        raise ValueError(f"Metric {metric_name} not found in registry")
    return METRIC_REGISTRY[registry_metric_name](metric_name, vllm_port)
