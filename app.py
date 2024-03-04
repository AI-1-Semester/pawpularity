import tkinter as tk
import numpy as np
from tkinter import Label, Entry, Frame, Button, Checkbutton
from paw_picture import PawPicture
from prediction_model import process_selection
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
    def open_image_view(self):
      for widget in self.main_frame.winfo_children():
        widget.destroy()

      Label(self.main_frame, text="Image will be displayed here", width=40, height=10).pack(pady=20)

      image = Image.open("./data/train/train_images/0007de18844b0dbbb5e1f607da0606e0.jpg")
      displayImage = ImageTk.PhotoImage(image)
      label = Label(image=displayImage, width=500, height=500)
      label.image = displayImage
      label.pack(pady=10)

      Button(self.main_frame, text="Open Form", command=self.open_form).pack(pady=10)
      Button(self.main_frame, text="Back", command=self.initialize_main_view).pack(pady=10)

    # method to open the form, to input the picture data
    def open_form(self):
        for widget in self.main_frame.winfo_children():
            widget.destroy()
        
        # emty array to store the input data
        arr = []

        Label(self.main_frame, text="Enter Data Below:").pack(pady=10)
        
        Label(self.main_frame, text="Eyes:").pack()
        input = tk.IntVar()
        entry1 = Checkbutton(self.main_frame, variable=input)
        entry1.pack()
        arr.append(input)

        Label(self.main_frame, text="Face:").pack()
        input = tk.IntVar()
        entry1 = Checkbutton(self.main_frame, variable=input)
        entry1.pack()
        arr.append(input)
        
        Label(self.main_frame, text="Near:").pack()
        input = tk.IntVar()
        entry1 = Checkbutton(self.main_frame, variable=input)
        entry1.pack()
        arr.append(input)

        Label(self.main_frame, text="Action:").pack()
        input = tk.IntVar()
        entry1 = Checkbutton(self.main_frame, variable=input)
        entry1.pack()
        arr.append(input)
        
        Label(self.main_frame, text="Accessory:").pack()
        input = tk.IntVar()
        entry1 = Checkbutton(self.main_frame, variable=input)
        entry1.pack()
        arr.append(input)

        Label(self.main_frame, text="Group:").pack()
        input = tk.IntVar()
        entry1 = Checkbutton(self.main_frame, variable=input)
        entry1.pack()
        arr.append(input)
        
        Label(self.main_frame, text="Collage:").pack()
        input = tk.IntVar()
        entry1 = Checkbutton(self.main_frame, variable=input)
        entry1.pack()
        arr.append(input)

        Label(self.main_frame, text="Human:").pack()
        input = tk.IntVar()
        entry1 = Checkbutton(self.main_frame, variable=input)
        entry1.pack()
        arr.append(input)

        Label(self.main_frame, text="Occlusion:").pack()
        input = tk.IntVar()
        entry1 = Checkbutton(self.main_frame, variable=input)
        entry1.pack()
        arr.append(input)

        Label(self.main_frame, text="Info:").pack()
        input = tk.IntVar()
        entry1 = Checkbutton(self.main_frame, variable=input)
        entry1.pack()
        arr.append(input)

        Label(self.main_frame, text="Blur:").pack()
        input = tk.IntVar()
        entry1 = Checkbutton(self.main_frame, variable=input)
        entry1.pack()
        arr.append(input)
        
        Button(self.main_frame, text="Cancel", command=self.open_image_view).pack(side=tk.LEFT, padx=(20, 10), pady=20)
        Button(self.main_frame, text="OK", command=lambda: self.submit_data(arr)).pack(side=tk.RIGHT, padx=(10, 20), pady=20)

    # method to submit the data, when the form is filled and ok is clicked
    def submit_data(self, arr):
        
        # create a PawPicture object with the input data
        createdPicture = PawPicture(arr[0].get(), arr[1].get(), arr[2].get(), arr[3].get(), arr[4].get(), arr[5].get(), arr[6].get(), arr[7].get(), arr[8].get(), arr[9].get(), arr[10].get())
        
        # transform to 2d array
        createdPictureList = [list(vars(createdPicture).keys()), list(vars(createdPicture).values())]
        
        # Create a DataFrame from the list
        df = pd.DataFrame([createdPictureList[1]], columns=createdPictureList[0])

        # call the method from the prediction_model.py to process the selection
        process_selection(df)
            
        self.open_image_view()  # Return to image view or wherever you want after submission

if __name__ == "__main__":
    app = Application()
    app.mainloop()