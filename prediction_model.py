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
X_train = train_data.drop(['Id', 'Pawpularity'], axis=1)
X_test = test_data.drop(['Id'], axis=1)
y_train = train_data['Pawpularity']
y_test = sample_submission['Pawpularity']

# Create a Linear Regression model
model = LinearRegression()

# Fit the model with the training data
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

def process_selection(selection):
  print("selection:", selection)

  model.predict(selection)
