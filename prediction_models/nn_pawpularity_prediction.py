from data.load_data import load_pawpularity_data
from data.load_data import load_train_data
import torch
from torch import nn, optim

train_data = load_train_data()

loaded_data = load_pawpularity_data()
x_train = loaded_data['x_train']
x_test = loaded_data['x_test']
y_train = loaded_data['y_train']
y_test = loaded_data['y_test']

class Net(nn.Module):
    def __init__(self, input_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 100)
        self.fc2 = nn.Linear(100, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

x_train_tensor = torch.tensor(x_train.values).float()
y_train_tensor = torch.tensor(y_train.values).float()

model = Net(input_size=x_train.shape[1])
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train the model
for epoch in range(1000):
    model.train()
    optimizer.zero_grad()
    y_pred = model(x_train_tensor)
    loss = criterion(y_pred.squeeze(), y_train_tensor)
    loss.backward()
    optimizer.step()

def process_pawpularity(input):
    print("\n User input data:", input)
    input_tensor = torch.tensor(input.values).float()
    pawpularity_result = model(input_tensor).detach().numpy()
    return pawpularity_result

def find_imageId(pawpularity_result):
    # Calculate the absolute difference between the pawpularity_result and all pawpularity scores in the train_data
    differences = abs(y_train - pawpularity_result)
    # Find the index of the minimum difference
    min_difference_index = differences.idxmin()
    # Return the Id of the image with the closest pawpularity score
    return train_data.loc[min_difference_index, 'Id']

def create_image_path(imageId):
    return f"./data/train/train_images/{imageId}.jpg"