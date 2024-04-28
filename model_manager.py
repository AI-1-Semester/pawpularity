from pred_models.models import LinearRegressionModel, LogisticRegressionModel
from data.load_data import load_pawpularity_data,load_occlusion_data, load_humanpred_data,load_train_data
from model_config import ModelConfig

class ModelManager:
    _instance = None

    last_predictions = {}

    #Singleton pattern
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.models = { }
            cls.init_default_models()
        return cls._instance

    #Initialize the default models and add them to the manager
    @classmethod
    def init_default_models(cls):
        for use_case, model_type in ModelConfig.default_models.items():
            cls.add_model(use_case, model_type)

    #Retrieve available models for each use case
    @classmethod
    def get_use_case_models(cls):
        return {use_case: models for use_case, models in ModelConfig.use_case_models.items()}
    
    #Add a new model for a specific use case to the manager
    @classmethod
    def add_model(cls, use_case_name, model_type):
        if use_case_name not in ModelConfig.allowed_use_cases:
            raise ValueError(f"No such use case: {use_case_name}")
        
        model_class = ModelConfig.model_factory.get(model_type)
        if model_class is None:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Load data once and pass to the model
        x_train, x_test, y_train, y_test = cls.get_training_data(use_case_name)
        model = model_class()
        model.set_data(x_train, x_test, y_train, y_test)
        model.train()
        cls._instance.models[use_case_name] = model
        print(f"New model added for {use_case_name}")

    #Fetch prediction for a specific use case -
    #Either with x_test parameter or the current test set if None set
    @classmethod
    def predict(cls, use_case_name, x_test = None):
        model = ModelConfig.model_factory.get(use_case_name)
        if model is None:
            raise ValueError(f"No model available for {use_case_name}")
        if x_test is None:
            prediction = model.predict()
        else:
            prediction = model.predict(x_test)
        
        # Store the last prediction for use case
        cls._instance.last_predictions[use_case_name] = prediction
        return model.predict(x_test)

    #Fetch metrics for performance evaluation of a specific use case's model 
    @classmethod
    def evaluate_model(cls, use_case_name, X, y, y_test, model_name):
        model = ModelConfig.model_factory.get(use_case_name)
        if model is None:
            raise ValueError(f"No model to evaluate for {use_case_name}")
        predictions = model.predict(X)
        return model.evaluate_model(X, y, y_test, predictions, model_name, use_case_name)

    @staticmethod
    def get_training_data(use_case_name):
        usecase_factory = {
            'pawpularity_score': load_pawpularity_data,
            'occlusion_detection': load_occlusion_data,  # Ensure this is spelled correctly
            'human_prediction': load_humanpred_data,
            'train_data': load_train_data
            # Add new models here as needed
        }

        # Retrieve the function corresponding to the use case
        load_data_function = usecase_factory.get(use_case_name)
        if load_data_function is None:
            raise ValueError(f"No data function for usecase: {use_case_name}")

        # Call the function to get the actual data
        training_data = load_data_function()
        if training_data is None:
            raise ValueError(f"No data returned for usecase: {use_case_name}")

        return training_data['x_train'], training_data['x_test'], training_data['y_train'], training_data['y_test']
    
    @classmethod
    def get_last_prediction(cls, use_case_name):
        return cls._instance.last_predictions.get(use_case_name)
    

    def predict_with_custom_handling(self, use_case_name, **kwargs):
        if use_case_name == 'human_prediction':
            required_keys = ['imageId', 'o_pred']
            for key in required_keys:
                if key not in kwargs:
                    raise ValueError(f"Missing required parameter: {key}")
            return self._predict_human(**kwargs)
        else:
            return self.predict(use_case_name, kwargs.get('x_test'))
    
    ### Custom prediction handlers
    def _predict_human(self, imageId, o_pred):
        train_data = self.get_training_data('human_prediction')[0]  #Returns x_train
        row = train_data[train_data['Id'] == imageId]
        row = row.drop(['Id', 'Pawpularity', 'Action', 'Accessory', 'Near', 'Collage', 'Eyes', 'Face', 'Info', 'Subject Focus', 'Blur', 'Human', 'Group'], axis=1)
        row['Occlusion'] = o_pred  # Set the occlusion value to the predicted result
        return self.predict('human_prediction', row)