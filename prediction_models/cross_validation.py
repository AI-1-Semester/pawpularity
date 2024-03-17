def CrossValidation(train_data, test_data):
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import cross_validate, KFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    import pandas as pd

    X = train_data.drop(['Id', 'Pawpularity'], axis=1)
    y = train_data['Pawpularity']
    X_test = test_data.drop(['Id'], axis=1)

    # Create a list of models
    models = [
        ('Logistic Regression', LogisticRegression(max_iter=200)),
        ('Linear Regression', LinearRegression()),
        ('Random Forest', RandomForestClassifier(n_estimators=100)),
        ('SVM', SVC())
    ]

    # Define a K-Fold cross-validation splitter
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # Define multiple scoring metrics
    scoring = ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']

    # Dictionary to hold the cross-validation results
    cv_results = {}

    # Iterate over models and perform cross-validation
    for name, model in models:
        results = cross_validate(model, X, y, cv=cv, scoring=scoring)
        # Store results in the dictionary
        cv_results[name] = {
            'R^2': results['test_r2'],
            'Mean Squared Error': -results['test_neg_mean_squared_error'], # Negate to make positive
            'Mean Absolute Error': -results['test_neg_mean_absolute_error'] # Negate to make positive
        }

    return cv_results

# To call the function, use the following line of code (uncomment it and run in your script):
# cv_results = CrossValidation(train_data, test_data)
# print(cv_results)