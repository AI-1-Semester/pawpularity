from pred_models.models import LinearRegressionModel, LogisticRegressionModel
from model_manager import  ModelManager
import numpy as np

# Instantiate the ModelManager
manager = ModelManager()

# Scenario 1: Using internal 'x_test'
# Predict using internal x_test data for each use case
try:
    human_prediction = manager.predict('human_prediction')
    pawpularity_score = manager.predict('pawpularity_score')
    occlusion_detection = manager.predict('occlusion_detection')

    print("Prediction for human_prediction:", human_prediction)
    print("Prediction for pawpularity_score:", pawpularity_score)
    print("Prediction for occlusion_detection:", occlusion_detection)
except ValueError as e:
    print(e)


# Scenario 2: Providing 'x_test' as a parameter
# Generate a sample x_test or extract from existing data
# For demonstration, let's assume you extract a sample from the loaded test data
# First, ensure you have access to the loaded test data:

# Load test data (make sure your data loading functions are accessible)
x_test_human_pred, _, _, _ = manager.get_training_data('human_prediction')
x_test_pawpularity, _, _, _ = manager.get_training_data('pawpularity_score')
x_test_occlusion, _, _, _ = manager.get_training_data('occlusion_detection')

# Use a subset for testing
x_test_human_pred_sample = x_test_human_pred[:10]  # Modify as needed
x_test_pawpularity_sample = x_test_pawpularity[:10]  # Modify as needed
x_test_occlusion_sample = x_test_occlusion[:10]  # Modify as needed

# Predict using provided x_test
try:
    human_prediction_custom = manager.predict('human_prediction', x_test_human_pred_sample)
    pawpularity_score_custom = manager.predict('pawpularity_score', x_test_pawpularity_sample)
    occlusion_detection_custom = manager.predict('occlusion_detection', x_test_occlusion_sample)

    print("Custom prediction for human_prediction:", human_prediction_custom)
    print("Custom prediction for pawpularity_score:", pawpularity_score_custom)
    print("Custom prediction for occlusion_detection:", occlusion_detection_custom)
except ValueError as e:
    print(e)

# Add a linear regression model to the manager
# Assuming ModelManager and LinearRegressionModel have been properly defined and imported
# manager.add_model('pawpularity_score', 'linear_regression')

# #Fetch data
# x_train,y_train,x_test,y_test = manager.get_training_data('pawpularity_score')

# Retrieve the model
# model = manager.models['pawpularity_score']

# # Set data and train
# model.set_data(x_train, x_test, y_train, y_test)
# model.train()