# deploy a serving endpoint for the trained model using MLflow

import mlflow


# register the model in the MLflow Model Registry with tags
def register_model(model_uri: str, model_name: str, tags: dict):
    """
    Register a model in the MLflow Model Registry.

    Args:
        model_uri (str): URI of the model in MLflow.
        model_name (str): Name of the model.
        tags (dict): Tags to be added to the model.

    Returns:
        str: Model version.
    """
    # Register the model
    result = mlflow.register_model(model_uri, model_name)

    # Add tags to the registered model
    mlflow.set_tags(tags)

    return result.version
