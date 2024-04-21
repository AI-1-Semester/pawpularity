from pred_models.models import LinearRegressionModel, LogisticRegressionModel
from data.load_data import load_pawpularity_data,load_occlusion_data, load_humanpred_data

class ModelManager:
    _instance = None
    model_factory = {
        'linear_regression': LinearRegressionModel,
        'logistic_regression': LogisticRegressionModel
        # Add new models here as needed
    }

    #Singleton pattern
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.models = {
                'human_prediction': None,
                'pawpularity_score': None,
                'occlusion_detection': None
            }
        return cls._instance

    @classmethod
    def add_model(cls, use_case_name, model_type):
        if use_case_name not in cls._instance.models:
            raise ValueError(f"No such use case: {use_case_name}")
        
        model = cls.model_factory.get(model_type)
        if model is None:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Load data once and pass to the model
        x_train, x_test, y_train, y_test = cls.get_training_data(use_case_name)
        model.set_data(x_train, x_test, y_train, y_test)
        model.train()
        cls._instance.models[use_case_name] = model

    @classmethod
    def predict(cls, use_case_name, x_test):
        model = cls._instance.models.get(use_case_name)
        if model is None:
            raise ValueError(f"No model available for {use_case_name}")
        return model.predict(x_test)

    @classmethod
    def evaluate_model(cls, use_case_name, X, y, y_test, model_name):
        model = cls._instance.models.get(use_case_name)
        if model is None:
            raise ValueError(f"No model to evaluate for {use_case_name}")
        predictions = model.predict(X)
        return model.evaluate_model(X, y, y_test, predictions, model_name, use_case_name)

    @staticmethod
    def get_training_data(use_case_name):
        usecase_factory = {
        'pawpularity_score': load_pawpularity_data,
        'occllusion_detection': load_occlusion_data,
        'human_prediction' : load_humanpred_data
        # Add new models here as needed
        }

        training_data = usecase_factory.get(use_case_name)
        if training_data is None:
            raise ValueError(f"No data for usecase: {use_case_name}")
        return training_data