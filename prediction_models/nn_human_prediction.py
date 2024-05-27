import pandas as pd
import torch
from torch import nn, optim
from sklearn.preprocessing import StandardScaler
from data.load_data import load_train_data, load_humanpred_data

# Load the data
train_data = load_train_data()
loaded_data = load_humanpred_data()
x_train = loaded_data['x_train']
x_test = loaded_data['x_test']
y_train = loaded_data['y_train']
y_test = loaded_data['y_test']

# Standardize the data
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Convert to PyTorch tensors
x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

# Define the neural network model
class HumanPredictionNN(nn.Module):
    def __init__(self, input_size):
        super(HumanPredictionNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return self.sigmoid(x)

# Initialize the model
input_size = x_train_tensor.shape[1]
model = HumanPredictionNN(input_size)

# Define loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
for epoch in range(1000):
    model.train()
    optimizer.zero_grad()
    outputs = model(x_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{epoch}], Loss: {loss.item():.4f}')

print('Training complete.')

# Save the model
torch.save(model.state_dict(), 'human_prediction_model.pth')

# Evaluate the model
model.eval()
with torch.no_grad():
    y_pred = model(x_test_tensor)
    y_pred_class = (y_pred >= 0.5).float()
    accuracy = (y_pred_class == y_test_tensor).float().mean()
    print(f'Accuracy: {accuracy.item():.4f}')

# Prediction function
def predict_human(imageId, o_pred):
    print(f"Predicting human for imageId: {imageId} with occlusion: {o_pred}")
    
    # Find the row in the dataframe that matches the imageId
    row = train_data[train_data['Id'] == imageId]

    if row.empty:
        raise ValueError(f"No data found for imageId: {imageId}")

    # Prepare the row for prediction by dropping the 'Human' column
    row = row.drop(['Id', 'Pawpularity', 'Action', 'Accessory', 'Near', 'Collage', 'Eyes', 'Face', 'Info', 'Subject Focus', 'Blur', 'Human', 'Group'], axis=1)

    # Set the 'Occlusion' column to the provided o_pred value
    row['Occlusion'] = o_pred
    
    # Standardize the row using the same scaler used during training
    row_scaled = scaler.transform(row)
    
    # Convert to PyTorch tensor
    row_tensor = torch.tensor(row_scaled, dtype=torch.float32)
    
    # Make a prediction
    with torch.no_grad():
        prediction = model(row_tensor).item()

    print(f"Prediction score: {prediction}")
    
    # If the prediction is 1 (or above a threshold, e.g., 0.5), there is a human in the image
    return prediction >= 0.5