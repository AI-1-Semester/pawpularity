## Uge 10 - Opgave 2
# Skriv en applikation, hvor man kan afprøve din model, 
# Hvis der er et menneske på billedet du viser, skal billedet fjernes fra visningen.

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

train_data = pd.read_csv("./data/train/train.csv")

##  Transform the data

# Drop the columns not to be used
correlated_data = train_data.drop(columns=['Id', 'Pawpularity', 'Action', 'Accessory', 'Near', 'Collage', 'Eyes', 'Face', 'Info', 'Subject Focus', 'Blur', 'Group'], axis=1)

df = pd.DataFrame(correlated_data)

# Split the data into training and testing sets (test_size=0.2 means that 20% of the data will be used for testing)
X = df.drop('Human', axis=1)
y = df['Human']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a logistic regression model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Check the accuracy of the model
accuracy = accuracy_score(y_test, predictions)

# Print to console
# print(f'\nHuman prediction model:  = Accuracy: {accuracy}')

def predict_human(imageId, o_pred):
    # Find the row in the dataframe that matches the imageId
    row = train_data[train_data['Id'] == imageId]

    # Prepare the row for prediction by dropping the 'Human' column
    row = row.drop(['Id', 'Pawpularity', 'Action', 'Accessory', 'Near', 'Collage', 'Eyes', 'Face', 'Info', 'Subject Focus', 'Blur', 'Human', 'Group'], axis=1)

    # swap the values of the prediction and the 'Occlusion' column
    row['Occlusion'] = o_pred

    # Make a prediction
    prediction = model.predict(row)

    print(row)

    # If the prediction is 1, there is a human in the image
    if prediction == 1:
        return True
    else:
        return False