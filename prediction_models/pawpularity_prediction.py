## Uge 10 - Opgave 2
# Skriv en applikation, hvor man kan afprøve din model, 
# så applikationen giver mulighed for at man kan inputte data og returnerer det billede,
# der svarer til samt pawpularity score for billedet.

from data.load_data import load_pawpularity_data
from data.load_data import load_train_data
from prediction_models.models import train_linear_regression_model, train_stacked_model

# Load clean data
train_data = load_train_data()

# Load the processed data
loaded_data = load_pawpularity_data()
x_train = loaded_data['x_train']
x_test = loaded_data['x_test']
y_train = loaded_data['y_train']
y_test = loaded_data['y_test']

# use linear regression model
linear_model = train_linear_regression_model(x_train, x_test, y_train, y_test , 'pawpularity_score')
# stacked_model = train_stacked_model(x_train, x_test, y_train, y_test, 'pawpularity_score')

# method to process the selection from the GUI
def process_pawpularity(input):
  print("\n User input data:", input)

  pawpularity_result = linear_model['model'].predict(input)

  # print to console
  # print("\n Pawpularity result: ", pawpularity_result)
  return pawpularity_result

# method to find the imageId
def find_imageId(pawpularity_result):
  # Calculate the absolute difference between the pawpularity_result and all pawpularity scores in the train_data
  differences = abs(y_train - pawpularity_result)
  # Find the index of the minimum difference
  min_difference_index = differences.idxmin()
  # Return the Id of the image with the closest pawpularity score
  return train_data.loc[min_difference_index, 'Id']

# method to create the image path
def create_image_path(imageId):
  return f"./data/train/train_images/{imageId}.jpg"