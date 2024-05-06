from pred_models.models import LinearRegressionModel, LogisticRegressionModel
from model_manager import  ModelManager
import numpy as np
from data.load_data import load_humanpred_data,load_occlusion_data,load_train_data,load_pawpularity_data

# Instantiate the ModelManager
manager = ModelManager()

# Scenario 1: Using internal 'x_test'
# Predict using internal x_test data for each use case
try:
     human_prediction = manager.predict('human_prediction')
     pawpularity_score = manager.predict('pawpularity_score')
     occlusion_detection = manager.predict('occlusion_detection')
     

     print("Prediction for human_prediction:", human_prediction['predictions'])
     print("Prediction for human_prediction:", human_prediction['proba_predictions'])
     print("Prediction for pawpularity_score:", pawpularity_score['predictions'])
     print("Prediction probability for pawpularity_score:", pawpularity_score['proba_predictions'])
     print("Prediction for occlusion_detection:", occlusion_detection['predictions'])
     print("Prediction probability for occlusion_detection:", occlusion_detection['proba_predictions'])
     
except ValueError as e:
     print(e)

try:
    human_eval_results = ModelManager.get_evaluation_results('human_prediction')
    pawpularity_eval_results = ModelManager.get_evaluation_results('pawpularity_score')
    occlusion_eval_results = ModelManager.get_evaluation_results('occlusion_detection')

    print("Human Prediction Evaluation:", human_eval_results)
    print("Pawpularity Score Evaluation:", pawpularity_eval_results)
    print("Occlusion Detection Evaluation:", occlusion_eval_results)

except ValueError as e:
    print(e)
