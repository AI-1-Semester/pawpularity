from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import StackingClassifier

from metrics.performance_measure import calculate_performance_measure

# method to train the linear regression model
def train_linear_regression_model(x_train, x_test, y_train, y_test, use_case_name):
   # Create a Linear Regression model
  model = LinearRegression()

  # Fit the model with the training data
  model.fit(x_train, y_train)

  # Make predictions on the test set
  predictions = model.predict(x_test)

  # calculate performance measure
  performance_measure = calculate_performance_measure(y_test, predictions)

  # print to console
  print(f'\n linear prediction model initialized')
  print(f'\n Performance measure for {use_case_name}:\n Mean Squared Error: {performance_measure["mse"]}\n Mean Absolute Error: {performance_measure["mae"]}\n R2 Score: {performance_measure["r2"]}')

  # return the model and the predictions
  return {"model": model, "predictions": predictions}

# method to train the logistic regression model
def train_logistic_regression_model(x_train, x_test, y_train):
  # Create a logistic regression model
  model = LogisticRegression()

  # Train the model
  model.fit(x_train, y_train)

  # Make predictions
  predictions = model.predict(x_test)

  # print to console
  print(f'\n logistic prediction model initialized')

  # return the model return the model and the predictions
  return {"model": model, "predictions": predictions}

# Method to train stacked model
def train_stacked_model(x_train, x_test, y_train, y_test, use_case_name):
  
  # List of tuples containing name and classifier (default hyperparameters)
  models = [('Logistic Regression',LogisticRegression(max_iter=1000)),
 ('Nearest Neighbors',KNeighborsClassifier()),
 ('Decision Tree',DecisionTreeClassifier()),
 ('Support Vector Classifier',SVC()),
 ('Naive Bayes',GaussianNB()),
 ('SVC Linear', SVC(kernel='linear'))]
  
  # create stacking model
  model = StackingClassifier(estimators=models, 
  final_estimator=LogisticRegression(), cv=3)

  # Train the model
  model.fit(x_train, y_train)

  # Make predictions
  predictions = model.predict(x_test)

  # calculate performance measure (activate accuracy_score in performance_measure.py to calculate accuracy)
  #performance_measure = calculate_performance_measure(y_test, predictions)

  # print to console
  print(f'\n stacked prediction model initialized')
  # print(f'\n Performance measure for {use_case_name}:\n Mean Squared Error: {performance_measure["mse"]}\n Mean Absolute Error: {performance_measure["mae"]}\n R2 Score: {performance_measure["r2"]} \n Accuracy: {performance_measure["accuracy"]}')

  # return the model and the predictions
  return {"model": model, "predictions": predictions}