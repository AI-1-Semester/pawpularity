import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score

# load the data
train_data = pd.read_csv("./data/train/train.csv")
test_data = pd.read_csv("./data/test/test.csv")

# Prepare the data
df = pd.DataFrame(train_data)

X = df[['Eyes', 'Face', 'Near', 'Action', 'Accessory', 'Group', 'Collage', 'Human', 'Info', 'Blur']]
y = df['Occlusion']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

gnb = GaussianNB()

model = BaggingClassifier(gnb, n_estimators=100, max_samples=0.7)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)

print(accuracy)

# method to process the selection from the GUI
def process_occlusion(input):

  transformed_input = input.drop(['Occlusion'], axis=1)
  occlusion_result = model.predict_proba(transformed_input)

  result = (occlusion_result[0 , 1] * 100)
  o_pred = model.predict(transformed_input)[0]

  return {"occlusion_probability": result, "o_pred": o_pred}