import tkinter as tk
import numpy as np
from tkinter import Label, Entry, Frame, Button, Checkbutton

from paw_picture import PawPicture
from prediction_model import process_selection

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

    def open_image_view(self):
        for widget in self.main_frame.winfo_children():
            widget.destroy()
        
        Label(self.main_frame, text="Image will be displayed here", width=40, height=10).pack(pady=20)
        
        Button(self.main_frame, text="Open Form", command=self.open_form).pack(pady=10)
        Button(self.main_frame, text="Back", command=self.initialize_main_view).pack(pady=10)

    def open_form(self):
        for widget in self.main_frame.winfo_children():
            widget.destroy()

        arr = []

        Label(self.main_frame, text="Enter Data Below:").pack(pady=10)
        
        Label(self.main_frame, text="Subject Focus:").pack()
        subjectFocus = tk.IntVar()
        entry1 = Checkbutton(self.main_frame, variable=subjectFocus)
        entry1.pack()
        arr.append(subjectFocus)
        
        Label(self.main_frame, text="Eyes:").pack()
        subjectFocus = tk.IntVar()
        entry1 = Checkbutton(self.main_frame, variable=subjectFocus)
        entry1.pack()
        arr.append(subjectFocus)

        Label(self.main_frame, text="Face:").pack()
        subjectFocus = tk.IntVar()
        entry1 = Checkbutton(self.main_frame, variable=subjectFocus)
        entry1.pack()
        arr.append(subjectFocus)
        
        Label(self.main_frame, text="Near:").pack()
        subjectFocus = tk.IntVar()
        entry1 = Checkbutton(self.main_frame, variable=subjectFocus)
        entry1.pack()
        arr.append(subjectFocus)

        Label(self.main_frame, text="Action:").pack()
        subjectFocus = tk.IntVar()
        entry1 = Checkbutton(self.main_frame, variable=subjectFocus)
        entry1.pack()
        arr.append(subjectFocus)
        
        Label(self.main_frame, text="Accessory:").pack()
        subjectFocus = tk.IntVar()
        entry1 = Checkbutton(self.main_frame, variable=subjectFocus)
        entry1.pack()
        arr.append(subjectFocus)

        Label(self.main_frame, text="Group:").pack()
        subjectFocus = tk.IntVar()
        entry1 = Checkbutton(self.main_frame, variable=subjectFocus)
        entry1.pack()
        arr.append(subjectFocus)
        
        Label(self.main_frame, text="Collage:").pack()
        subjectFocus = tk.IntVar()
        entry1 = Checkbutton(self.main_frame, variable=subjectFocus)
        entry1.pack()
        arr.append(subjectFocus)

        Label(self.main_frame, text="Human:").pack()
        subjectFocus = tk.IntVar()
        entry1 = Checkbutton(self.main_frame, variable=subjectFocus)
        entry1.pack()
        arr.append(subjectFocus)

        Label(self.main_frame, text="Occlusion:").pack()
        subjectFocus = tk.IntVar()
        entry1 = Checkbutton(self.main_frame, variable=subjectFocus)
        entry1.pack()
        arr.append(subjectFocus)

        Label(self.main_frame, text="Info:").pack()
        subjectFocus = tk.IntVar()
        entry1 = Checkbutton(self.main_frame, variable=subjectFocus)
        entry1.pack()
        arr.append(subjectFocus)

        Label(self.main_frame, text="Blur:").pack()
        subjectFocus = tk.IntVar()
        entry1 = Checkbutton(self.main_frame, variable=subjectFocus)
        entry1.pack()
        arr.append(subjectFocus)
        
        Button(self.main_frame, text="Cancel", command=self.open_image_view).pack(side=tk.LEFT, padx=(20, 10), pady=20)
        Button(self.main_frame, text="OK", command=lambda: self.submit_data(arr)).pack(side=tk.RIGHT, padx=(10, 20), pady=20)

    def submit_data(self, arr):
        # Placeholder function for data submission logic
        createdPicture = PawPicture(arr[0].get(), arr[1].get(), arr[2].get(), arr[3].get(), arr[4].get(), arr[5].get(), arr[6].get(), arr[7].get(), arr[8].get(), arr[9].get(), arr[10].get(), arr[11].get())
        print(createdPicture)
        
        createdPictureList = [list(vars(createdPicture).keys()), list(vars(createdPicture).values())]
        print(createdPictureList)

        # newArr = np.array(createdPictureList).reshape(1, -1)
        # print(newArr)
        #print(list(vars(createdPicture).values()))

        process_selection(createdPictureList)
            
        self.open_image_view()  # Return to image view or wherever you want after submission

if __name__ == "__main__":
    app = Application()
    app.mainloop()