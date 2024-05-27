def CrossValidation(X, y, models, n_splits=5, random_state=42):
    from sklearn.model_selection import cross_validate, KFold
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scoring = ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']
    cv_results = {}

    for name, model in models:
        results = cross_validate(model, X, y, cv=cv, scoring=scoring)
        cv_results[name] = {
            'R^2': results['test_r2'],
            'Mean Squared Error': -results['test_neg_mean_squared_error'],
            'Mean Absolute Error': -results['test_neg_mean_absolute_error']
        }

    return cv_results

# To call the function, use the following line of code (uncomment it and run in your script):
# cv_results = CrossValidation(train_data, test_data, models, n_splits(default is 5), random_state(default is 42))
# print(cv_results)