# model_manager.py
from pred_models.models import LinearRegressionModel, LogisticRegressionModel
from data.load_data import load_pawpularity_data, load_occlusion_data, load_humanpred_data, load_train_data
from model_config import ModelConfig

class ModelManager:
    _instance = None
    last_predictions = {}

    # Singleton pattern
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.models = {}
            cls.init_default_models()
        return cls._instance

    @classmethod
    def init_default_models(cls):
        for use_case, model_name in ModelConfig.default_models.items():
            cls.add_model(use_case, model_name)

    @classmethod
    def add_model(cls, use_case_name, model_name):
        if use_case_name not in ModelConfig.allowed_use_cases:
            raise ValueError(f"No such use case: {use_case_name}")

        model_class = ModelConfig.model_factory.get(model_name)
        if model_class is None:
            raise ValueError(f"Unsupported model type for {model_name}")

        x_train, x_test, y_train, y_test = cls.get_training_data(use_case_name)
        model = model_class()
        model.set_data(x_train, x_test, y_train, y_test)
        model.train()
        cls._instance.models[use_case_name] = model
        print(f"New model added for {use_case_name}")

    @classmethod
    def predict(cls, use_case_name, x_test=None, evaluate = True):
        model = cls._instance.models.get(use_case_name)
        if model is None:
            raise ValueError(f"No model available for {use_case_name}")

        isNew_x_test = False if x_test is None else True

        if x_test == None:
            x_test = model.data['x_test']

        # Dictionary to hold prediction results
        prediction_results = {
            "predictions": model.predict(x_test),
            "proba_predictions": None
        }

        print(type(model.model))
        # Obtain probability predictions if the model supports it
        if hasattr(model.model, 'predict_proba'):
            proba_output = model.predict_proba(x_test)
            # Check the shape of the output and adjust indexing accordingly
            if proba_output.ndim == 1 or proba_output.shape[1] == 1:
                # If the output is 1D or has only one column, use it as is
                prediction_results["proba_predictions"] = proba_output.flatten()
            else:
                # Otherwise, use the second column for the probability of the positive class
                prediction_results["proba_predictions"] = proba_output[:, 1]

        # Store last predictions
        cls._instance.last_predictions[use_case_name] = prediction_results

                # Evaluate the model if requested
        if evaluate and isNew_x_test == False:
            evaluation_results = model.evaluate_model(prediction_results,
                ModelConfig.default_models.get(use_case_name), #modelname
                use_case_name
            )
            model.evaluation_results = evaluation_results
            print("Evaluation Results:", evaluation_results)

        return prediction_results

    @staticmethod
    def get_training_data(use_case_name):
        usecase_factory = {
            'pawpularity_score': load_pawpularity_data,
            'occlusion_detection': load_occlusion_data,
            'human_prediction': load_humanpred_data
        }

        load_data_function = usecase_factory.get(use_case_name)
        if load_data_function is None:
            raise ValueError(f"No data function for use case: {use_case_name}")

        training_data = load_data_function()
        if training_data is None:
            raise ValueError(f"No data returned for use case: {use_case_name}")

        return training_data['x_train'], training_data['x_test'], training_data['y_train'], training_data['y_test']


    @classmethod
    def get_evaluation_results(cls, use_case_name):
        model = cls._instance.models.get(use_case_name)
        if model is None:
            raise ValueError(f"No model available for use case: {use_case_name}")
        # Optionally, you could trigger an evaluation here or just return stored results
        return model.evaluation_results

    @classmethod
    def get_last_prediction(cls, use_case_name):
        return cls._instance.last_predictions.get(use_case_name)


    def predict_with_custom_handling(self, use_case_name, **kwargs):
        print('use case name: ', use_case_name)
        if use_case_name == 'human_prediction':
            
            return self._predict_human(**kwargs)
        elif use_case_name == 'occlusion_detection':
            return self._predict_occlusion(**kwargs)
        else:
            return self.predict(use_case_name, kwargs.get('x_test'))

    def _predict_human(self, imageId, o_pred):
        train_data = self.get_training_data('human_prediction')[0]  # Ensure this returns expected DataFrame
        row = train_data[train_data['Id'] == imageId]
        if row.empty:
            raise ValueError(f"No data found for ID {imageId}")
        print("Data row before dropping columns:", row) 
        row = row.drop(['Id', 'Pawpularity', 'Action', 'Accessory', 'Near', 'Collage', 'Eyes', 'Face', 'Info', 'Subject Focus', 'Blur', 'Human', 'Group'], axis=1)
        row['Occlusion'] = o_pred
        print("Data row after setting occlusion:", row)
        prediction = self.predict('human_prediction', row)
        print("Prediction:", prediction) 
        return prediction

    def _predict_occlusion(self, x_test):
        # Implement any special preprocessing or parameter adjustments for occlusion detection here
        return self.predict('occlusion_detection', x_test)
