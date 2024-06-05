## Uge 10 - Opgave 2
# Skriv en applikation, hvor man kan afprøve din model, 
# så applikationen giver mulighed for at man kan inputte data og returnerer det billede,
# der svarer til samt pawpularity score for billedet.

from data.load_data import load_pawpularity_data
from data.load_data import load_train_data

# Load clean data
train_data = load_train_data()

# Load the processed data
loaded_data = load_pawpularity_data()
y_train = loaded_data['y_train']


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