from pred_models.models import LinearRegressionModel, LogisticRegressionModel, NN_PawpularityModel, NN_HumanModel, AdaBoostModel




class ModelConfig:
    # Maps model types to their corresponding class implementations
    model_factory = {
        'linear_regression': LinearRegressionModel,
        'logistic_regression': LogisticRegressionModel,
        'nn_pawpularity': NN_PawpularityModel,
        'nn_human': NN_HumanModel,
        'ada_boost': AdaBoostModel
    }

    # Maps use cases to the models that can be used for them
    use_case_models = {
        'human_prediction': ['logistic_regression','nn_human', 'ada_boost'],
        'pawpularity_score': ['linear_regression', 'nn_pawpularity'],
        'occlusion_detection': ['logistic_regression']
    }

    # Specifies the default model for each use case
    default_models = {
        'human_prediction': 'logistic_regression',
        'pawpularity_score': 'linear_regression',
        'occlusion_detection': 'logistic_regression'
    }

    # Defines which use cases are allowed
    allowed_use_cases = {
        'human_prediction', 
        'pawpularity_score', 
        'occlusion_detection'
        }

     # Maps model types to their corresponding class implementations
    model_class_to_name = {
        LinearRegressionModel: 'linear_regression',
        LogisticRegressionModel: 'logistic_regression',
        AdaBoostModel: 'ada_boost',
        NN_PawpularityModel: 'nn_pawpularity',
        NN_HumanModel: 'nn_human'
    }