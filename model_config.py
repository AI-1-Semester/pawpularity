from pred_models.models import LinearRegressionModel, LogisticRegressionModel


class ModelConfig:
    # Maps model types to their corresponding class implementations
    model_factory = {
        'linear_regression': LinearRegressionModel,
        'logistic_regression': LogisticRegressionModel
    }

    # Maps use cases to the models that can be used for them
    use_case_models = {
        'human_prediction': ['logistic_regression'],
        'pawpularity_score': ['linear_regression'],
        'occlusion_detection': ['logistic_regression']
    }

    # Specifies the default model for each use case
    default_models = {
        'human_prediction': 'logistic_regression',
        'pawpularity_score': 'linear_regression',
        'occlusion_detection': 'logistic_regression'
    }

    # Defines which use cases are allowed
    allowed_use_cases = {'human_prediction', 'pawpularity_score', 'occlusion_detection'}
