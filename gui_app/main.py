"""
Точка входа в GUI-приложение
"""
import tkinter as tk
from .app import population_algorithmApp


if __name__ == '__main__':
    root = tk.Tk()
    app = population_algorithmApp(root)
    root.mainloop()