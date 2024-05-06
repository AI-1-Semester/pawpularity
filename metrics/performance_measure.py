from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score

def calculate_performance_measure(y_test, predictions):
    # Calculate the mean squared error
    mse = mean_squared_error(y_test, predictions)

    # Calculate the mean absolute error
    mae = mean_absolute_error(y_test, predictions)

    # Calculate the R2 score
    r2 = r2_score(y_test, predictions)

    return {"mse": mse, "mae": mae, "r2": r2}

def calculate_accuracy(y_test, predictions):
    # Calculate the accuracy of predictions
    acc = accuracy_score(y_test, predictions)
    return acc