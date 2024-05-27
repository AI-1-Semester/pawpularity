from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score

from data.load_data import load_occlusion_data

# Load the processed data
loaded_data = load_occlusion_data()
x_train = loaded_data['x_train']
x_test = loaded_data['x_test']
y_train = loaded_data['y_train']
y_test = loaded_data['y_test']

# Create a Gaussian Naive Bayes model
gnb = GaussianNB()

# Create an AdaBoost Classifier model (with hyperparameters tuning)
model = AdaBoostClassifier(gnb, n_estimators=100)

# Fit the model with the training data
model.fit(x_train, y_train)

# Make predictions on the test set
y_pred = model.predict(x_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)

# Print to console
# print("\n Occlusion accuracy: ", accuracy)

# method to process the selection from the GUI
def process_boosting_occlusion(input):

  transformed_input = input.drop(['Occlusion'], axis=1)
  occlusion_result = model.predict_proba(transformed_input)

  result = (occlusion_result[0 , 1] * 100)
  o_pred = model.predict(transformed_input)[0]

  return {"occlusion_probability": result, "o_pred": o_pred}