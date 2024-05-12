import pandas as pd
from data.load_data import  load_train_data
from model_manager import ModelManager

def predict_human(imageId, o_pred):
    train_data = load_train_data()
    # Ensure `train_data` is either passed to this function or available globally
    row = train_data[train_data['Id'] == imageId]

    # Prepare the row for prediction by dropping unnecessary columns
    row = row.drop(['Id', 'Pawpularity', 'Action', 'Accessory', 'Near', 'Collage', 'Eyes', 'Face', 'Info', 'Subject Focus', 'Blur', 'Human', 'Group'], axis=1)

    # Swap the values of the prediction and the 'Occlusion' column
    row['Occlusion'] = o_pred

    # Fetch the model from the ModelManager and make a prediction
    model = ModelManager.get_model('human_prediction')
    prediction = model.predict(row)

    # Print the row for debugging
    print(row)

    # Return whether there is a human in the image
    return prediction[0] == 1

# Example usage:
# train_data = pd.read_csv('path_to_train_data.csv')
# imageId = 'specific_image_id'
# o_pred = 'predicted_occlusion_value'
# print(predict_human(imageId, o_pred, train_data))