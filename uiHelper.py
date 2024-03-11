import tkinter as tk

class GridManager:
    def __init__(self, window, rows, columns):
        self.window = window
        self.rows = rows
        self.columns = columns
        self.create_grid()

    def create_grid(self):
        # Make the grid cells expandable
        for i in range(self.rows):
            self.window.grid_rowconfigure(i, weight=1)
        for j in range(self.columns):
            self.window.grid_columnconfigure(j, weight=1)

    def add_button(self, row, column, text='', command=None):
        button = tk.Button(self.window, text=text, command=command)
        button.grid(row=row, column=column, sticky="nsew")
        return button

    def add_label(self, row, column, text='', font=None):
        label = tk.Label(self.window, text=text, font=font if font else None)
        label.grid(row=row, column=column, sticky="nsew")
        return label

    def add_checkbutton(self, row, column, variable):
        checkbutton = tk.Checkbutton(self.window, variable=variable)
        checkbutton.grid(row=row, column=column, sticky="nsew")
        return checkbutton
    

