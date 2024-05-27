# model_manager.py
from pred_models.models import LinearRegressionModel, LogisticRegressionModel
from data.load_data import load_pawpularity_data, load_occlusion_data, load_humanpred_data, load_train_data, load_clustering_data
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
    def init_singleton(cls):
        # This method explicitly initializes the singleton instance
        return cls.__new__(cls)

    @classmethod
    def init_default_models(cls):
        for use_case, model_name in ModelConfig.default_models.items():
            cls.add_model(use_case, model_name)

    @classmethod
    def add_model(cls, use_case_name, model_name):
        if use_case_name not in ModelConfig.allowed_use_cases:
            raise ValueError(f"No such use case: {use_case_name}")
        
        # Check if the model already exists and is of the same type
        existing_model = cls._instance.models.get(use_case_name)
        if existing_model and isinstance(existing_model, ModelConfig.model_factory[model_name]):
            print(f"Model for {use_case_name} is already set to {model_name}. No action taken.")
            return

        model_class = ModelConfig.model_factory.get(model_name)
        if model_class is None:
            raise ValueError(f"Unsupported model type for {model_name}")

        x_train, x_test, y_train, y_test = cls.get_training_data(use_case_name)
        model = model_class()
        
        # Clustering
        if use_case_name == 'data_clustering':
        # Load and set data for clustering
            clustering_data = load_clustering_data()
            x_train, x_test = clustering_data['x_train'], clustering_data['x_test']
            model = model_class(n_clusters=3)  # Initialize with the desired number of clusters
            model.set_data(x_train, x_test)
        else:
            x_train, x_test, y_train, y_test = cls.get_training_data(use_case_name)
            model = model_class()
            model.set_data(x_train, x_test, y_train, y_test)
            
        model.set_data(x_train, x_test, y_train, y_test)
        model.train()
        cls._instance.models[use_case_name] = model
        print(f"New model added for {use_case_name}")
       
        cls._instance.predict(use_case_name,evaluate=True)

    @classmethod
    def predict(cls, use_case_name, x_test=None, evaluate=False):
        # Validate if the use case name is allowed before proceeding
        if use_case_name not in ModelConfig.allowed_use_cases:
            raise ValueError(f"No such use case: {use_case_name}")

        model = cls._instance.models.get(use_case_name)
        if model is None:
            raise ValueError(f"No model available for {use_case_name}")
        
        isNew_x_test = x_test is not None

        if x_test is None:
            x_test = model.data['x_test']

        predictions = model.predict(x_test)
        prediction_results = {
            "predictions": predictions,
            "proba_predictions": None
        }

        # Obtain probability predictions if the model supports it
        if hasattr(model.model, 'predict_proba'):
            proba_output = model.predict_proba(x_test)
            # Check the shape of the output and adjust indexing accordingly
            if proba_output.ndim == 1 or (proba_output.ndim == 2 and proba_output.shape[1] == 1):
                # If the output is 1D or has only one column, use it as is
                prediction_results["proba_predictions"] = proba_output.flatten()
            else:
                # Otherwise, use the second column for the probability of the positive class
                prediction_results["proba_predictions"] = proba_output[:, 1]

        # Store last predictions
        cls._instance.last_predictions[use_case_name] = prediction_results

        # Evaluate the model if requested and if the test data hasn't been newly provided
        if evaluate and not isNew_x_test:
            evaluation_results = cls._instance.evaluate_model(prediction_results, use_case_name)
            model.evaluation_results = evaluation_results
            print("Evaluation Results:", evaluation_results)

        return prediction_results

    def evaluate_model(cls, prediction_results, use_case_name):
        model = cls._instance.models.get(use_case_name)
        if model is None:
            raise ValueError(f"No model available for {use_case_name}")

        try:
            # Directly call the model's evaluate method with all required parameters
            evaluation_results = model.evaluate_model(
                prediction_results,
                ModelConfig.model_class_to_name.get(use_case_name),  # Pass the default model name for the ROC curve
                use_case_name
            )
            model.evaluation_results = evaluation_results
            print(f"Evaluation Results Updated for {use_case_name}: ", evaluation_results)
        except Exception as e:
            print(f"Error during evaluation for {use_case_name}: {str(e)}")
            evaluation_results = {}

        return evaluation_results

    @staticmethod
    def get_training_data(use_case_name):
        usecase_factory = {
            'pawpularity_score': load_pawpularity_data,
            'occlusion_detection': load_occlusion_data,
            'human_prediction': load_humanpred_data,
            'data_clustering': load_clustering_data
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

    @classmethod
    def predict_with_custom_handling(self, use_case_name, **kwargs):
        print('use case name: ', use_case_name)
        if use_case_name == 'human_prediction':
            
            return self._predict_human(**kwargs)
        elif use_case_name == 'occlusion_detection':
            return self._predict_occlusion(**kwargs)
        else:
            return self.predict(use_case_name, kwargs.get('x_test'))

    @classmethod
    def _predict_human(cls, imageId, o_pred):
        train_data = cls._instance.get_training_data('human_prediction')[0]  # Ensure this returns expected DataFrame
        row = train_data[train_data['Id'] == imageId]
        if row.empty:
            raise ValueError(f"No data found for ID {imageId}")
        print("Data row before dropping columns:", row) 
        row = row.drop(['Id', 'Pawpularity', 'Action', 'Accessory', 'Near', 'Collage', 'Eyes', 'Face', 'Info', 'Subject Focus', 'Blur', 'Human', 'Group'], axis=1)
        row['Occlusion'] = o_pred
        print("Data row after setting occlusion:", row)
        prediction = cls._instance.predict('human_prediction', row)
        print("Prediction:", prediction) 
        return prediction

    @classmethod
    def _predict_occlusion(cls, x_test):
        transformed_x_test = x_test.drop(['Occlusion'], axis=1)
        # Implement any special preprocessing or parameter adjustments for occlusion detection here
        return cls._instance.predict('occlusion_detection', transformed_x_test)
