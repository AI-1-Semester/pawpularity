import tkinter as tk
from tkinter import ttk

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

    def add_button(self, row, column, text='', command=None, borderwidth=None, relief=None, bg=None, fg=None, font=None):
        button = tk.Button(self.window, text=text, command=command)

          # Apply optional styling if provided
        if borderwidth is not None:
            button.config(borderwidth=borderwidth)
        if relief is not None:
            button.config(relief=relief)
        if bg is not None:
            button.config(bg=bg)
        if fg is not None:
            button.config(fg=fg)
        if font is not None:
            button.config(font=font)

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

    def add_combobox(self, row, column, values):
        var = tk.StringVar(self.window)
        combobox = ttk.Combobox(self.window, textvariable=var, values=values, state=["readonly"])
        combobox.grid(row=row, column=column, sticky="nsew")
        combobox.set(values[0])  # Set default value as the first in the list
        return var
    

