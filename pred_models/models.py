from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression, LogisticRegression
from metrics.performance_measure import calculate_performance_measure
from metrics.cross_validation  import CrossValidation
from metrics.roc_curve import create_roc_curve

class BaseModel(ABC):

    def __init__(self):
        self.evaluation_results = {}
        self.data = {}

    @abstractmethod
    def _train(self, x_train, y_train):
        pass
    
    def predict(self, x_test=None):
        # Use internal data if no external x_test is provided
        x_to_predict = x_test if x_test is not None else self.data.get('x_test')
        if x_to_predict is None:
            raise ValueError("No test data available for prediction")
        return self.model.predict(x_to_predict) if self.model else None

    def set_data(self, x_train, x_test, y_train, y_test):
        self.data['x_train'] = x_train
        self.data['x_test'] = x_test
        self.data['y_train'] = y_train
        self.data['y_test'] = y_test

    def train(self):
        if not self.data:
            raise ValueError("Data has not been set")
        self._train()

    def evaluate_model(self, X, y, y_test, predictions, model_name, use_case_name, n_splits=5, random_state=42):
        # Calculate performance metrics
        performance_metrics = calculate_performance_measure(y_test, predictions)
        self.evaluation_results['performance_metrics'] = performance_metrics

        # Create ROC curve and store the figure object
        roc_curve_fig = create_roc_curve(y_test, predictions, model_name, use_case_name)
        self.evaluation_results['roc_curve'] = roc_curve_fig

        # Perform cross-validation
        cross_val_results = CrossValidation(X, y, self, n_splits, random_state)
        self.evaluation_results['cross_validation'] = cross_val_results

        return self.evaluation_results

class LinearRegressionModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = LinearRegression()
        print('Linear Regression Model Initialized.\n')

    def _train(self):
        self.model.fit(self.data['x_train'], self.data['y_train'])
        print('\nLinear regression model trained.')

class LogisticRegressionModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = LogisticRegression()
        print('Logistic Regression Model Initialized.\n')

    def _train(self):
        self.model.fit(self.data['x_train'], self.data['y_train'])
        print('\nLogistic regression model trained.')

# Add more classes for other models following the same structure :-)