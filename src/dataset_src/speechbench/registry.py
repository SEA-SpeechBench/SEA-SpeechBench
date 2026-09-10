"""
A registry for dataset classes.
"""

DATASET_REGISTRY = {}

def register_dataset(name):
    """
    A decorator to register a dataset class with a specified name in the global DATASET_REGISTRY.

    Args:
        name (str): The name under which the dataset class should be registered.

    Returns:
        decorator (function): A function that takes a class `cls` as an argument,
                              registers it in the DATASET_REGISTRY, and returns the class.
    """

    def decorator(cls):
        DATASET_REGISTRY[name] = cls
        return cls
    return decorator

def get_dataset(name, split, num_sample, num_worker=4, use_diverse_prompts=False, use_sea_prompts=False):
    """
    Retrieves a dataset class instance from the DATASET_REGISTRY by name.

    Args:
        name (str): The name of the dataset to retrieve.
        split (str): The split of the dataset.
        num_sample (int): The number of samples to be loaded in the dataset.
        num_worker (int, optional): The number of workers to use for data loading. Defaults to 4.
        use_diverse_prompts (bool, optional): Whether to use diverse prompts. Defaults to False.
        use_sea_prompts (bool, optional): Whether to use SEA prompts. Defaults to False (i.e. English prompts).

    Returns:
        dataset (Dataset): An instance of the dataset class with the specified name.

    Raises:
        ValueError: If the dataset name is not found in the DATASET_REGISTRY.
    """
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Dataset {name} not found in registry")
    return DATASET_REGISTRY[name](split, num_sample, num_worker, use_diverse_prompts, use_sea_prompts)
