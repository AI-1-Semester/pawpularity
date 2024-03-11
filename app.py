## Uge 10 - Opgave 2
# Skriv en applikation, hvor man kan afprøve din model, 
# så applikationen giver mulighed for at man kan inputte data og returnerer det billede,
# der svarer til samt pawpularity score for billedet.

import tkinter as tk
import numpy as np
from tkinter import Label, Entry, Frame, Button, Checkbutton
from prediction_models.bayes_model import process_occlusion
from prediction_models.logical.human_prediction import predict_human
from models.paw_picture import PawPicture
from prediction_models.linear.pawpularity_prediction import create_image_path, find_imageId, process_pawpularity
import pandas as pd
from PIL import Image, ImageTk
 
# tkinter application class
class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Machine Learning Application")
        self.geometry("1200x900")
        
        # Initial View
        self.main_frame = Frame(self)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        self.initialize_main_view()

    def initialize_main_view(self):
        for widget in self.main_frame.winfo_children():
            widget.destroy()

        Label(self.main_frame, text="Machine Learning", font=('Arial', 24)).pack(pady=20)
        
        Button(self.main_frame, text="Open Form", command=self.open_form).pack(pady=10)

    # method to open the image view
    def open_image_view(self, image_path, isHuman, pawpularity_score, occlusion_probability):
        for widget in self.main_frame.winfo_children():
            widget.destroy()
    
        if isHuman and image_path == "":
            print("isHuman = True")
            Label(self.main_frame, text=f"Is human {isHuman}", font=('Arial', 24)).pack(side=tk.LEFT, pady=20)
        else:
            print("isHuman = false")
            Label(self.main_frame, text="Image:", width=40, height=10).pack(pady=20)
            image = Image.open(image_path)
            displayImage = ImageTk.PhotoImage(image)
            self.label = Label(self.main_frame, image=displayImage, width=500, height=500)
            self.label.image = displayImage
            self.label.pack(pady=10)
            Label(self.main_frame, text=f"Is human {isHuman}", font=('Arial', 24)).pack(side=tk.LEFT, pady=20)

        if pawpularity_score:
            Label(self.main_frame, text=f"Pawpularity Score: {pawpularity_score}", font=('Arial', 24)).pack(side=tk.LEFT, pady=20)
        if occlusion_probability:
            Label(self.main_frame, text=f"Occlusion probability: {occlusion_probability}", font=('Arial', 24)).pack(side=tk.LEFT, pady=20)

        Button(self.main_frame, text="Open Form", command=self.close_image_and_open_form).pack(pady=10)
        Button(self.main_frame, text="Back", command=self.close_image_and_initialize_main_view).pack(pady=10)

    # method to close the image and open the form
    def close_image_and_open_form(self):
        if hasattr(self, 'label'):
            self.label.destroy()
        self.open_form()

    # method to close the image and initialize the main view
    def close_image_and_initialize_main_view(self):
        if hasattr(self, 'label'):
            self.label.destroy()
        self.initialize_main_view()

    # method to open the form, to input the picture data
    def open_form(self):
        for widget in self.main_frame.winfo_children():
            widget.destroy()
    
        arr = []

        Label(self.main_frame, text="Enter Data Below:").grid(row=0, columnspan=2, pady=10)

        # You create a list to hold the IntVar() references and the corresponding labels
        entries = [("Eyes", 1), ("Face", 2), ("Near", 3), ("Action", 4),
               ("Accessory", 5), ("Group", 6), ("Collage", 7), 
               ("Human", 8), ("Occlusion", 9), ("Info", 10), ("Blur", 11)]

    # Now loop through the entries and create a checkbutton for each
        for label_text, row in entries:
            Label(self.main_frame, text=f"{label_text}:").grid(row=row, column=0, sticky="e", pady=2)
            input = tk.IntVar()
            entry = Checkbutton(self.main_frame, variable=input)
            entry.grid(row=row, column=1, sticky="w", pady=2)
            arr.append(input)

    # Place Cancel and OK buttons
        Button(self.main_frame, text="Cancel", command=self.open_image_view).grid(row=12, column=0, padx=(20, 10), pady=20, sticky="e")
        Button(self.main_frame, text="OK", command=lambda: self.submit_data(arr)).grid(row=12, column=1, padx=(10, 20), pady=20, sticky="w")

    # method to submit the data, when the form is filled and ok is clicked
    def submit_data(self, arr):
        
        # create a PawPicture object with the input data
        createdPicture = PawPicture(arr[0].get(), arr[1].get(), arr[2].get(), arr[3].get(), arr[4].get(), arr[5].get(), arr[6].get(), arr[7].get(), arr[8].get(), arr[9].get(), arr[10].get())
        
        # transform to 2d array
        createdPictureList = [list(vars(createdPicture).keys()), list(vars(createdPicture).values())]
        
        # Create a DataFrame from the list
        df = pd.DataFrame([createdPictureList[1]], columns=createdPictureList[0])

        # call the method from the prediction_model.py to process the selection
        pawpularity_score = process_pawpularity(df)

        occlusion_probability = process_occlusion(df)
        print("\n Occlusion probability: ", occlusion_probability, "%")

        # call the method from the prediction_model.py to find the imageId
        imageId = find_imageId(pawpularity_score)

        isHuman = predict_human(imageId)

        if isHuman:
            self.open_image_view("", isHuman, pawpularity_score[0], occlusion_probability)
        else:
            # call the method from the prediction_model.py to create the image path
            imagepath = create_image_path(imageId)

            #print(imageId)

            # Return to image view or wherever you want after submission    
            self.open_image_view(imagepath, isHuman, pawpularity_score[0], occlusion_probability)

        

if __name__ == "__main__":
    app = Application()
    app.mainloop()