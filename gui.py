import tkinter as tk
from tkinter import filedialog, messagebox
from main import processImages
from Notation import ImageEditor
import os

class ImageProcessingApp:
    def __init__(self, master):
        self.master = master
        master.title("Image Processing App")

        self.welcome_label = tk.Label(master, text="Welcome! Create your Map")
        self.welcome_label.pack(pady=10)

        self.process_button = tk.Button(master, text="Select Dataset Folder", command=self.select_dataset)
        self.process_button.pack(pady=20)

        self.algorithm_frame = tk.Frame(master)
        self.algorithm_label = tk.Label(self.algorithm_frame, text="Select Algorithm:")
        self.algorithm_label.pack(pady=5)

        self.algorithm_var = tk.StringVar()
        algorithms = ["Random", "Greedy", "Sequentially", "Cluster"]
        self.algorithm_menu = tk.OptionMenu(self.algorithm_frame, self.algorithm_var, *algorithms)
        self.algorithm_menu.pack(pady=5)

        self.algorithm_button = tk.Button(self.algorithm_frame, text="Process Images", command=self.process_images)
        self.algorithm_button.pack(pady=10)

        # Initialize dataset_path as None
        self.dataset_path = None

    def select_dataset(self):
        self.dataset_path = filedialog.askdirectory(title="Select Dataset Folder")
        if self.dataset_path:
            # if selected, switch to the algorithm selection frame
            self.process_button.pack_forget()
            self.welcome_label.pack_forget()

            self.algorithm_frame.pack(pady=20)

    def process_images(self):
        selected_algorithm = self.algorithm_var.get()

        if not selected_algorithm:
            messagebox.showwarning("Warning", "Please select an algorithm.")
            return

        if not self.dataset_path:
            messagebox.showwarning("Warning", "Please select a dataset folder.")
            return
        
        processImages(self.dataset_path, selected_algorithm)
        messagebox.showinfo("Success", "Image processing completed successfully!")

        # if images are merged successfully update the text inside the window
        self.algorithm_frame.pack_forget()

        self.welcome_label.config(text="Click on location to add text.")
        self.welcome_label.pack(pady=10)

        outputpath = "./output"
        image_path = f"{outputpath}/{selected_algorithm.lower()}/final.jpeg"
        marker_path = "hiclipart.png"  # pointer icon

        editor = ImageEditor(self.master, image_path, marker_path)
        editor.run_editor()

    def run_app(self):
        self.master.mainloop()

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessingApp(root)
    app.run_app()
