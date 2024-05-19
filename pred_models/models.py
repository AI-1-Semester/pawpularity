from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression, LogisticRegression
from metrics.performance_measure import calculate_performance_measure, calculate_accuracy
from metrics.cross_validation  import CrossValidation
from metrics.roc_curve import create_roc_curve
from torch import nn, optim
import torch
from models.neural_network import Net


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

    def predict_proba(self, x_test=None):
        x_to_predict = x_test if x_test is not None else self.data.get('x_test')
        if x_to_predict is None:
            raise ValueError("No test data available for prediction")
        if hasattr(self.model, 'predict_proba'):
            proba_output = self.model.predict_proba(x_to_predict)
            # Check the shape of the output and adjust indexing accordingly
            if proba_output.ndim == 1 or proba_output.shape[1] == 1:
                # If the output is 1D or has only one column, use it as is
                return proba_output.flatten()
            else:
                # Otherwise, use the second column for the probability of the positive class
                return proba_output[:, 1]
        else:
            raise NotImplementedError("This model does not support probability predictions.")

    def set_data(self, x_train, x_test, y_train, y_test):
        self.data['x_train'] = x_train
        self.data['x_test'] = x_test
        self.data['y_train'] = y_train
        self.data['y_test'] = y_test

    def train(self):
        if not self.data:
            raise ValueError("Data has not been set")
        self._train()

    def evaluate_model(self, prediction_results, model_name, use_case_name, n_splits=5, random_state=42):
        predictions = prediction_results["predictions"]
        probabilities = prediction_results["proba_predictions"]

        # Calculate standard performance metrics (for both regression and classification)
        self.evaluation_results['performance_metrics'] = calculate_performance_measure(self.data['y_test'], predictions)

        # Calculate accuracy and ROC curve if probabilities are available (typically for classification tasks)
        if probabilities is not None:
            self.evaluation_results['accuracy'] = calculate_accuracy(self.data['y_test'], predictions)
            try:
                self.evaluation_results['roc_curve'] = create_roc_curve(self.data['y_test'], probabilities, model_name, use_case_name)
            except Exception as e:
                print(f"Error creating ROC curve for {use_case_name}: {str(e)}")
        else:
            print(f"No probabilities available, skipping accuracy and ROC curve for {use_case_name}.")

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

class NN_PawpularityModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = None
        self.criterion = nn.MSELoss()
        self.optimizer = None
        print('Neural Network Pawpularity Model Initialized.\n')

    def set_data(self, x_train, x_test, y_train, y_test):
        super().set_data(x_train, x_test, y_train, y_test)
        input_size = self.data['x_train'].shape[1]
        self.model = Net(input_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)

    def _train(self):
        x_train_tensor = torch.tensor(self.data['x_train'].values).float()
        y_train_tensor = torch.tensor(self.data['y_train'].values).float()

        for epoch in range(1000):
            self.model.train()
            self.optimizer.zero_grad()
            y_pred = self.model(x_train_tensor)
            loss = self.criterion(y_pred.squeeze(), y_train_tensor)
            loss.backward()
            self.optimizer.step()
        print('\nNeural network pawpularity model trained.')

    def predict(self, x_test=None):
        self.model.eval()
        x_to_predict = torch.tensor(x_test.values if x_test is not None else self.data['x_test'].values).float()
        with torch.no_grad():
            predictions = self.model(x_to_predict).numpy()
        return predictions.squeeze()