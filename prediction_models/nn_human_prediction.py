import pandas as pd
import torch
from torch import nn, optim
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

# Load the data
train_data = pd.read_csv("./data/train/train.csv")

# Transform the data
correlated_data = train_data.drop(columns=['Id', 'Pawpularity', 'Action', 'Accessory', 'Near', 'Collage', 'Eyes', 'Face', 'Info', 'Subject Focus', 'Blur', 'Group'], axis=1)
df = pd.DataFrame(correlated_data)

# Split the data into training and testing sets
X = df.drop('Human', axis=1)
y = df['Human']
X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train).float()
X_test_tensor = torch.tensor(X_test).float()
y_train_tensor = torch.tensor(y_train).float()
y_test_tensor = torch.tensor(y_test).float()

# Define the neural network model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(X_train.shape[1], 50)  # Change the number of neurons if necessary
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  # Using sigmoid for the binary classification output
        return x

# Initialize the model, loss function, and optimizer
model = Net()
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss for binary classification
optimizer = optim.Adam(model.parameters(), lr=0.01)

# DataLoader for batching
train_data = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_data, batch_size=10, shuffle=True)

# Train the model
num_epochs = 50  # You can adjust the number of epochs
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()

# Evaluate the model
model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor).squeeze()
    predictions = (outputs > 0.5).float()  # Convert probabilities to binary output
    accuracy = (predictions == y_test_tensor).float().mean()

print(f'Neural Network Human prediction model - Accuracy: {accuracy.item()}')

def predict_human(imageId, o_pred):
    row = train_data[train_data['Id'] == imageId]
    row = row.drop(['Id', 'Pawpularity', 'Action', 'Accessory', 'Near', 'Collage', 'Eyes', 'Face', 'Info', 'Subject Focus', 'Blur', 'Human', 'Group'], axis=1)
    row['Occlusion'] = o_pred
    row_tensor = torch.tensor(row.values).float()
    prediction = model(row_tensor)
    return prediction.item() > 0.5
