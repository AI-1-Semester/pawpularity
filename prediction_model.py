## Uge 10 - Opgave 2
# Skriv en applikation, hvor man kan afprøve din model, 
# så applikationen giver mulighed for at man kan inputte data og returnerer det billede,
# der svarer til samt pawpularity score for billedet.

import pandas as pd
import sklearn as sk
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

train_data = pd.read_csv("./data/train/train.csv")
test_data = pd.read_csv("./data/test/test.csv")
sample_submission = pd.read_csv("./data/test/sample_submission.csv")

# Prepare the data
X_train = train_data.drop(['Id', 'Pawpularity', 'Subject Focus'], axis=1)
X_test = test_data.drop(['Id', 'Subject Focus'], axis=1)
y_train = train_data['Pawpularity']
y_test = sample_submission['Pawpularity']

# Create a Linear Regression model
model = LinearRegression()

# Fit the model with the training data
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

print(f'\n prediction model initialized')

# method to process the selection from the GUI
def process_selection(input):
  print("\n User input data:", input)

  pawpularity_result = model.predict(input)

  print("\n Pawpularity result: ", pawpularity_result)
  return pawpularity_result

def find_imageId(pawpularity_result):
  # Calculate the absolute difference between the pawpularity_result and all pawpularity scores in the train_data
  differences = abs(train_data['Pawpularity'] - pawpularity_result)
  # Find the index of the minimum difference
  min_difference_index = differences.idxmin()
  # Return the Id of the image with the closest pawpularity score
  return train_data.loc[min_difference_index, 'Id']

def create_image_path(imageId):
  return f"./data/train/train_images/{imageId}.jpg"