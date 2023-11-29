import tkinter as tk
from tkinter import simpledialog, filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont

class StyledDialog(simpledialog.Dialog):
    def __init__(self, parent, title, prompt):
        self.prompt = prompt
        super().__init__(parent, title=title)

    def body(self, master):
        tk.Label(master, text=self.prompt, wraplength=300).pack(pady=10)
        self.entry = tk.Entry(master, width=30)
        self.entry.pack(pady=10)
        return self.entry  # Focus on entry widget

    def apply(self):
        result = self.entry.get()
        self.result = result

class ImageEditor:
    def __init__(self, master, image_path, marker_path):
        self.image_path = image_path
        self.marker_path = marker_path

        self.root = master
        self.root.title("Image Editor")

        self.image = Image.open(self.image_path)
        self.original_image = self.image.copy()
        self.tk_image = ImageTk.PhotoImage(self.image)

        self.marker_icon = Image.open(self.marker_path)

        self.canvas = tk.Canvas(
            self.root, width=self.image.width, height=self.image.height)
        self.canvas.pack()
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

        self.canvas.bind("<Button-1>", self.on_canvas_click)

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.changes_made = False

    def on_canvas_click(self, event):
        x, y = event.x, event.y
        text = StyledDialog(self.root, "Enter location name",
                            "Enter text to be placed:").result
        if text:
            self.draw_point_with_text(x, y,text)
            self.changes_made = True

    def draw_point_with_text(self, x, y, text):
        draw = ImageDraw.Draw(self.image)

        font_size = 20
        font = ImageFont.truetype("arial.ttf", font_size)

        self.image.paste(self.marker_icon, (x-15, y-20))

        draw.text((x+10, y-22), text, fill="#dbd2c9", font=font)

        self.tk_image = ImageTk.PhotoImage(self.image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

    def save_image(self):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png", filetypes=[("PNG files", "*.png")])
        if file_path:
            self.image.save(file_path)
            print(f"Image saved to {file_path}")
            self.changes_made = False

    def on_close(self):
        if self.changes_made:
            response = messagebox.askquestion(
                "Save Changes", "Do you want to save changes before closing?")
            if response == 'yes':
                self.save_image()
        self.root.destroy()

    def run_editor(self):
        self.root.mainloop()