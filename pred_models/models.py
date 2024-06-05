from abc import ABC, abstractmethod
from sklearn.ensemble import AdaBoostClassifier, StackingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from metrics.performance_measure import calculate_performance_measure, calculate_accuracy
from metrics.cross_validation  import CrossValidation
from metrics.roc_curve import create_roc_curve
from torch import nn, optim
import torch
from prediction_models.nn_human_prediction import HumanPredictionNN
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
import seaborn as sns

from prediction_models.nn_pawpularity_prediction import Net

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

class StackedModel(BaseModel):

    def __init__(self):
        # List of tuples containing name and classifier (default hyperparameters)
        models = [('Logistic Regression',LogisticRegression(max_iter=1000)),
        ('Nearest Neighbors',KNeighborsClassifier()),
        ('Decision Tree',DecisionTreeClassifier()),
        ('Support Vector Classifier',SVC()),
        ('Naive Bayes',GaussianNB()),
        ('SVC Linear', SVC(kernel='linear'))]

         # create stacking model
        model = StackingClassifier(estimators=models, final_estimator=LogisticRegression(), cv=3)

        super().__init__()
        self.model = model
        print('Stacked Model Initialized.\n')
    
    def _train(self):
        self.model.fit(self.data['x_train'], self.data['y_train'])

    def predict(self, x_test=None):
        return super().predict(x_test)

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
    
class AdaBoostModel(BaseModel):
     def __init__(self):
        super().__init__()
        self.model = None
        print('AdaBoost Model Initialized.\n')

     def set_data(self, x_train, x_test, y_train, y_test):
        super().set_data(x_train, x_test, y_train, y_test)
        gnb = GaussianNB()
        self.model = AdaBoostClassifier(gnb, n_estimators=100)

     def _train(self):
        self.model.fit(self.data['x_train'], self.data['y_train'])
        print('\nAdaBoost model trained.')

     def process_boosting_occlusion(self, input):
        transformed_input = input.drop(['Occlusion'], axis=1)
        occlusion_result = self.model.predict_proba(transformed_input)

        result = (occlusion_result[0 , 1] * 100)
        o_pred = self.model.predict(transformed_input)[0]

        return {"occlusion_probability": result, "o_pred": o_pred}
     
class NN_HumanModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = None
        self.criterion = nn.BCELoss()
        self.optimizer = None
        print('Neural Network Human Prediction Model Initialized.\n')

    def set_data(self, x_train, x_test, y_train, y_test):
        super().set_data(x_train, x_test, y_train, y_test)
        input_size = self.data['x_train'].shape[1]
        self.model = HumanPredictionNN(input_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def _train(self):
        x_train_tensor = torch.tensor(self.data['x_train'].values).float()
        y_train_tensor = torch.tensor(self.data['y_train'].values).float().unsqueeze(1)

        for epoch in range(1000):
            self.model.train()
            self.optimizer.zero_grad()
            outputs = self.model(x_train_tensor)
            loss = self.criterion(outputs, y_train_tensor)
            loss.backward()
            self.optimizer.step()
        print('\nNeural network human prediction model trained.')

    def predict(self, x_test=None):
        self.model.eval()
        x_to_predict = torch.tensor(x_test.values if x_test is not None else self.data['x_test'].values).float()
        with torch.no_grad():
            predictions = self.model(x_to_predict).numpy().squeeze()
        return (predictions >= 0.5).astype(float)

# K-Means Clustering Model
class KMeansModel(BaseModel):
    def __init__(self, n_clusters=3):
        super().__init__()
        self.n_clusters = n_clusters
        self.model = KMeans(n_clusters=n_clusters)
        self.pca = PCA(n_components=2)  # Reduce to 2 components for visualization
        print('KMeans Clustering Model Initialized.\n')

    def _train(self):
        self.model.fit(self.data['x_train'])
        self.pca.fit(self.data['x_train'])
        print('\nKMeans clustering model trained.')

    def predict(self, x_test=None):
        x_to_predict = x_test if x_test is not None else self.data.get('x_test')
        if x_to_predict is None:
            raise ValueError("No test data available for prediction")
        return self.model.predict(x_to_predict)

    def set_data(self, x_train, x_test, y_train=None, y_test=None):
        self.data['x_train'] = x_train
        self.data['x_test'] = x_test
        # Fit PCA during the set_data step
        self.pca.fit(x_train)

    def plot_clusters(self):
        x_test = self.data['x_test']
        labels = self.predict(x_test)
        x_test_reduced = self.pca.transform(x_test)  # Reduce dimensions
        fig, ax = plt.subplots(figsize=(8, 6))
        
        scatter = ax.scatter(x_test_reduced[:, 0], x_test_reduced[:, 1], c=labels, cmap='viridis', s=50, alpha=0.7)
        ax.set_title("K-Means Clustering Results")
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        plt.colorbar(scatter, ax=ax, label='Cluster')
        ax.grid(True)

        # Annotate points with original indices
        for i, (x, y) in enumerate(x_test_reduced):
            ax.text(x, y, f"{i}", fontsize=8, ha='right', color='black')

        return fig

    def feature_analysis(self):
        # Analyze cluster centers in the reduced PCA space
        cluster_centers = self.model.cluster_centers_
        cluster_centers_reduced = self.pca.transform(cluster_centers)
        cluster_features = pd.DataFrame(cluster_centers_reduced, columns=['PC1', 'PC2'])
        cluster_features['Cluster'] = range(self.n_clusters)
        return cluster_features

    def plot_feature_distributions(self):
        labels = self.predict(self.data['x_test'])
        df = self.data['x_test'].copy()
        df['Cluster'] = labels
        for feature in df.columns[:-1]:  # Skip the 'Cluster' column
            plt.figure(figsize=(10, 6))
            sns.boxplot(x='Cluster', y=feature, data=df)
            plt.title(f'Distribution of {feature} by Cluster')
            plt.show()

    def plot_elbow_method(self):
        x_train = self.data['x_train']
        sse = []
        for k in range(1, 11):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(x_train)
            sse.append(kmeans.inertia_)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(range(1, 11), sse, marker='o')
        ax.set_title('Elbow Method for Optimal k')
        ax.set_xlabel('Number of clusters (k)')
        ax.set_ylabel('Sum of squared errors (SSE)')
        ax.grid(True)

        return fig
# Add more classes for other models following the same structure :-)

class ConvolutionalNeuralNetworkModel(nn.Module):
    def __init__(self):
        super(ConvolutionalNeuralNetworkModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.fc1 = nn.Linear(128 * 6 * 6, 512)
        self.fc2 = nn.Linear(512, 2)
        print('Convolutional Neural Network Initialized.\n')

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(-1, 128 * 6 * 6)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x