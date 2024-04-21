from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression, LogisticRegression
from metrics.performance_measure import calculate_performance_measure
from metrics.cross_validation  import CrossValidation
from metrics.roc_curve import create_roc_curve

class BaseModel(ABC):

    def __init__(self):
        self.evaluation_results = {}

    @abstractmethod
    def train(self, x_train, y_train):
        pass
    
    @abstractmethod
    def predict(self, x_test):
        pass

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
        self.model = None

    def train(self, x_train, y_train):
        self.model = LinearRegressionModel()
        self.model.fit(x_train, y_train)
        print('\nLinear regression model trained.')

    def predict(self, x_test):
        return self.model.predict(x_test) if self.model else None

class LogisticRegressionModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = None

    def train(self, x_train, y_train):
        self.model = LogisticRegression()
        self.model.fit(x_train, y_train)
        print('\nLinear regression model trained.')

    def predict(self, x_test):
        return self.model.predict(x_test) if self.model else None

# Add more classes for other models following the same structure...