import tkinter as tk
from tkinter import Frame
import pandas as pd
from PIL import Image, ImageTk
from models.paw_picture import PawPicture
from prediction_models.occlusion_bagging_bayes import process_occlusion
from prediction_models.occlusion_adaboost_bayes import process_boosting_occlusion
from prediction_models.human_prediction import predict_human
from prediction_models.pawpularity_prediction import create_image_path, find_imageId, process_pawpularity
from uiHelper import GridManager
from model_manager import ModelManager
from model_config import ModelConfig
from tkinter import Toplevel, Scrollbar, Canvas
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Assuming GridManager is in the same file or imported appropriately
# from grid_manager import GridManager

class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Machine Learning Application")
        self.geometry("1200x900")
        
        # Initialize the grid manager for the main frame
        self.main_frame = Frame(self)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        self.grid_manager = GridManager(self.main_frame, rows=15, columns=2)  # Adjust rows and columns as needed
        ModelManager.init_singleton()

        self.initialize_main_view()

    def initialize_main_view(self):
        # Clear the grid
        for widget in self.main_frame.winfo_children():
            widget.destroy()

        # Initialize the grid manager
        self.grid_manager.create_grid()

        # Add 'Machine Learning' label and 'Open Form' button
        self.grid_manager.add_label(row=0, column=0, text="Machine Learning", font=('Arial', 24))
        select_models_button = self.grid_manager.add_button(row=1, column=0, text="Select Models")
        select_models_button.configure(command=self.create_model_selection_form)
        open_form_button = self.grid_manager.add_button(row=1, column=1, text="Open Form")
        open_form_button.configure(command=self.open_form)

        # You might need to adjust the rowspan and columnspan
        # Merge cells for the label
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.grid_manager.add_label(row=0, column=0, text="Machine Learning").grid(rowspan=1, columnspan=2)

    # ... (other methods remain the same but replace pack with grid using grid_manager)
    def open_image_view(self, image_path, isHuman, pawpularity_score, occlusion_probability):
        for widget in self.main_frame.winfo_children():
            widget.destroy()

        # Set up the grid manager for the main frame
        grid_manager = GridManager(self.main_frame, rows=4, columns=2)

        # Add the image or the "Is human" label to the left side (column 0)
        if isHuman and image_path == "":
            grid_manager.add_label(0, 0, f"Is human {isHuman}", font=('Arial', 24))
        else:
            image = Image.open(image_path)
            displayImage = ImageTk.PhotoImage(image)
            # Keep a reference to the image to avoid garbage collection issues
            self.displayImage = displayImage
            self.label = tk.Label(self.main_frame, image=self.displayImage)
            self.label.grid(row=0, column=0, rowspan=4, sticky="nsew")  # Image spans multiple rows

        # Add the Pawpularity Score and Occlusion probability labels to the right side (column 1)
        if pawpularity_score is not None:
            grid_manager.add_label(0, 1, f"Pawpularity Score: {pawpularity_score}", font=('Arial', 24))
        if occlusion_probability is not None:
            grid_manager.add_label(1, 1, f"Occlusion probability: {occlusion_probability}", font=('Arial', 24))
        
        # Add "Open Form" and "Back" buttons to the bottom
        grid_manager.add_button(3, 0, "Open Form").configure(command=self.close_image_and_open_form)
        grid_manager.add_button(3, 1, "Back").configure(command=self.close_image_and_initialize_main_view)
        grid_manager.add_button(2, 0, "Metrics").configure(command=self.open_metrics)

        # Configure column widths to give more space to the image
        self.main_frame.grid_columnconfigure(0, weight=3)  # Image column gets more space
        self.main_frame.grid_columnconfigure(1, weight=1)  # Info column gets less space


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


    # open_form and open_image_view need to be updated to use grid_manager as well
    # I'll show the updated open_form method:

    def open_form(self):
        # Clear the current view
        for widget in self.main_frame.winfo_children():
            widget.destroy()

        # Create a GridManager for the main_frame with a desired number of rows and columns
        grid_manager = GridManager(self.main_frame, rows=13, columns=2)  # Adjust rows and columns as needed

        # Add title label
        grid_manager.add_label(0, 0, "Enter Data Below:")
        grid_manager.window.grid_columnconfigure(1, weight=1)  # Make the second column expand

        # Initialize an array to hold the input data variables
        arr = []

        # Define the entries for the form
        entries = [("Eyes", 1), ("Face", 2), ("Near", 3), ("Action", 4),
                   ("Accessory", 5), ("Group", 6), ("Collage", 7), 
                   ("Human", 8), ("Occlusion", 9), ("Info", 10), ("Blur", 11)]

        # Loop through the entries and add labels and checkbuttons using the grid manager
        for label_text, row in entries:
            grid_manager.add_label(row, 0, f"{label_text}:")
            input_var = tk.IntVar()
            checkbutton = tk.Checkbutton(self.main_frame, variable=input_var)
            checkbutton.grid(row=row, column=1, sticky="w")
            arr.append(input_var)

        # Add Cancel and OK buttons using the grid manager
        grid_manager.add_button(12, 0, "Cancel").configure(command=self.open_image_view)
        grid_manager.add_button(12, 1, "OK").configure(command=lambda: self.submit_data(arr))

    # ... (update open_image_view similarly to use grid_manager)

    def create_model_selection_form(self):
         # Clear the current view
        for widget in self.main_frame.winfo_children():
            widget.destroy()

        # Using GridManager to manage layout
        grid_manager = GridManager(self.main_frame, rows=20, columns=2)

        # Assuming ModelManager has a method get_available_models()
        use_cases = ModelConfig.use_case_models
        grid_manager.add_label(0,0,"Use Cases", font=('Helvetica', 12)).grid(sticky='W')
        grid_manager.add_label(0,1,"Model selection", font=('Helvetica', 12)).grid(sticky='W')

        self.model_vars = {}
        self.comboboxes = {}
        row = 1
        for use_case, models in use_cases.items():
            grid_manager.add_label(row, 0, f"{use_case}  :")
            cb = grid_manager.add_combobox(row, 1, models)
            self.comboboxes[use_case] = cb 
            row += 1

        # Buttons for Cancel and OK
        grid_manager.add_button(row, 0, "Cancel", borderwidth=5, command=self.initialize_main_view)
        grid_manager.add_button(row, 1, "OK", borderwidth=5, command=self.save_model_selections)

    def submit_data(self, arr):
            
            # create a PawPicture object with the input data
            createdPicture = PawPicture(arr[0].get(), arr[1].get(), arr[2].get(), arr[3].get(), arr[4].get(), arr[5].get(), arr[6].get(), arr[7].get(), arr[8].get(), arr[9].get(), arr[10].get())
            
            # transform to 2d array
            createdPictureList = [list(vars(createdPicture).keys()), list(vars(createdPicture).values())]
            
            # Create a DataFrame from the list
            df = pd.DataFrame([createdPictureList[1]], columns=createdPictureList[0])

            # call the method from the Modelmanage to process the selection
            ModelManager.predict("pawpularity_score", x_test=df)
            paw_pred_results = ModelManager.last_predictions['pawpularity_score']
            paw_predictions = paw_pred_results['predictions']
            print(f"paw pred - {paw_predictions}")

            ModelManager._predict_occlusion(x_test=df)
            occlu_pred_results = ModelManager.last_predictions['occlusion_detection']
            occlusion_predictions = occlu_pred_results['proba_predictions']
            print(f"oclu pred - {occlusion_predictions}")
            # occlusion_result = process_occlusion(df)
            
            # print to console
            # print("\n Occlusion probability: ", occlusion_result['occlusion_probability'], "%")

            # call the method from the prediction_model.py to find the imageId
            imageId = find_imageId(paw_predictions)

            isHuman = predict_human(imageId, occlusion_predictions)
            if isHuman:
                self.open_image_view("", isHuman, paw_predictions, occlusion_predictions)
            else:
                # call the method from the prediction_model.py to create the image path
                imagepath = create_image_path(imageId)

                #print(imageId)

                # Return to image view or wherever you want after submission    
                self.open_image_view(imagepath, isHuman, paw_predictions, occlusion_predictions)


    def save_model_selections(self):
        for use_case, combobox in self.comboboxes.items():
            selected_model = combobox.get()
            if selected_model:  # Ensure there's a selection
                try:
                    ModelManager.add_model(use_case, selected_model)
                    print(f"Model for {use_case} set to {selected_model}")
                except ValueError as e:
                    print(e)
        self.initialize_main_view() 

    def open_metrics(self):
         # Clear the current view
        for widget in self.main_frame.winfo_children():
            widget.destroy()

        # Create a new top-level window
        metrics_window = Toplevel(self.main_frame)
        metrics_window.title("Model Metrics")

        # Create a canvas with a scrollbar
        canvas = Canvas(metrics_window)
        scrollbar = Scrollbar(metrics_window, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)

        # Configure the canvas to be scrollable
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Add the canvas and scrollbar to the window
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Call a function to populate the metrics and plots
        self.populate_metrics(scrollable_frame)

        btn_frame = tk.Frame(self.main_frame)
        btn_frame.pack(fill=tk.X, pady=10)
        back_btn = tk.Button(btn_frame, text="Back to Main View", command=self.initialize_main_view)
        back_btn.pack(side=tk.LEFT, padx=10)

    def populate_metrics(self, frame):
        # Example use cases
        use_cases = ['pawpularity_score', 'occlusion_detection', 'human_prediction']
        for use_case in use_cases:
            self.add_metrics_for_use_case(frame, use_case)

    def add_metrics_for_use_case(self, frame, use_case_name):
        # Retrieve the evaluation results for a given use case
        evaluation_results = ModelManager.get_evaluation_results(use_case_name)

        # Display performance metrics
        if 'performance_metrics' in evaluation_results:
            perf_metrics = evaluation_results['performance_metrics']
            metrics_info = f"Performance Metrics for {use_case_name}: MSE={perf_metrics.get('mse', 'N/A')}, MAE={perf_metrics.get('mae', 'N/A')}, R2={perf_metrics.get('r2', 'N/A')}"
            perf_label = tk.Label(frame, text=metrics_info)
            perf_label.pack(pady=10)

        # Display accuracy if available
        if 'accuracy' in evaluation_results:
            accuracy_info = f"Accuracy for {use_case_name}: {evaluation_results['accuracy']}"
            accuracy_label = tk.Label(frame, text=accuracy_info)
            accuracy_label.pack(pady=10)

        # Handle and display the ROC curve if it's part of the results
        if 'roc_curve' in evaluation_results and evaluation_results['roc_curve'] is not None:
            fig = evaluation_results['roc_curve']
            canvas = FigureCanvasTkAgg(fig, master=frame)  # Create a Tkinter.DrawingArea.
            canvas.draw()
            widget = canvas.get_tk_widget()
            widget.pack(fill=tk.BOTH, expand=True)

if __name__ == "__main__":
    app = Application()
    app.mainloop()
