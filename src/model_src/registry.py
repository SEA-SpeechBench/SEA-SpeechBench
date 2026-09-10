"""
A registry for model classes.
"""

MODEL_REGISTRY = {}

def register_model(name):
    """
    A decorator to register a model class with a specified name in the global MODEL_REGISTRY.

    Args:
        name (str): The name under which the model class should be registered.

    Returns:
        decorator (function): A function that takes a class `cls` as an argument,
                              registers it in the MODEL_REGISTRY, and returns the class.
    """

    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator

def get_model(model_name):
    """
    Retrieves a model class instance from the MODEL_REGISTRY by name.

    Args:
        model_name (str): The name of the model to retrieve.

    Returns:
        model (Model): An instance of the model class with the specified name.

    Raises:
        ValueError: If the model name is not found in the MODEL_REGISTRY.
    """
    # format name
    registry_model_name = model_name.replace("-", "_").lower()
    if registry_model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model {model_name} not found in registry")
    return MODEL_REGISTRY[registry_model_name](model_name)
