
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

# load the data
train_data = pd.read_csv("./data/train/train.csv")
test_data = pd.read_csv("./data/test/test.csv")

# Prepare the data
df = pd.DataFrame(train_data)

X = df[['Eyes', 'Face', 'Near', 'Action', 'Accessory', 'Group', 'Collage', 'Human', 'Info', 'Blur']]
y = df['Occlusion']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Gaussian Naive Bayes model
model = GaussianNB()
model.fit(X_train, y_train)

# Make predictions on the test set
y_prob = model.predict_proba(X_test)

# Extract the probability of 'Occlusion'
occlusion_prob = y_prob[:, 1]  # Assuming 'Occlusion' is the second class

# print(occlusion_prob)

# Create a DataFrame from the probabilities
prob_df = pd.DataFrame(occlusion_prob, columns=['Occlusion_Prob'])

# Reset index of your test data
X_test_reset = X_test.reset_index(drop=True)

# Concatenate the probabilities with the original data
result_df = pd.concat([X_test_reset, prob_df], axis=1)

# print(result_df)

# method to process the selection from the GUI
def process_occlusion(input):

  transformed_input = input.drop(['Occlusion'], axis=1)
  occlusion_result = model.predict_proba(transformed_input)

  result = (occlusion_result[0 , 0] * 100)

  return result