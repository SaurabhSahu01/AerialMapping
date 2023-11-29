import tkinter as tk
from tkinter import filedialog, messagebox, Canvas, Scrollbar, simpledialog
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

class ScrollableImage(tk.Frame):
    def __init__(self, master=None, **kwargs):
        tk.Frame.__init__(self, master, **kwargs)
        self.image = None
        self.canvas = Canvas(self, bg="white", highlightthickness=0)
        self.scroll_y = Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scroll_y.pack(side="right", fill="y")
        self.scroll_x = Scrollbar(self, orient="horizontal", command=self.canvas.xview)
        self.scroll_x.pack(side="bottom", fill="x")
        self.canvas.configure(yscrollcommand=self.scroll_y.set, xscrollcommand=self.scroll_x.set)
        self.canvas.pack(side="left", fill="both", expand=True)

        self.canvas.bind("<Configure>", self.resize)
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)
        self.canvas.bind("<Left>", lambda e: self.canvas.xview_scroll(-1, "units"))
        self.canvas.bind("<Right>", lambda e: self.canvas.xview_scroll(1, "units"))
        self.canvas.bind("<Up>", lambda e: self.canvas.yview_scroll(-1, "units"))
        self.canvas.bind("<Down>", lambda e: self.canvas.yview_scroll(1, "units"))

    def on_mousewheel(self, event):
        if event.delta:
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def load_image(self, image_path):
        image = Image.open(image_path)
        self.image = ImageTk.PhotoImage(image)
        self.canvas.create_image(0, 0, anchor="nw", image=self.image)

    def resize(self, event):
        if self.image:
            self.canvas.config(scrollregion=self.canvas.bbox("all"))

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

        self.scrollable_image = ScrollableImage(self.root)
        self.scrollable_image.pack(expand=tk.YES, fill=tk.BOTH)
        self.scrollable_image.load_image(self.image_path)

        self.scrollable_image.canvas.bind("<Button-1>", self.on_canvas_click)

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.changes_made = False

    def on_canvas_click(self, event):
        x, y = self.scrollable_image.canvas.canvasx(event.x), self.scrollable_image.canvas.canvasy(event.y)
        text = StyledDialog(self.root, "Enter location name",
                            "Enter text to be placed:").result
        if text:
            self.draw_point_with_text(x, y, text)
            self.changes_made = True

    def draw_point_with_text(self, x, y, text):
        draw = ImageDraw.Draw(self.image)

        font_size = 30
        font = ImageFont.truetype("arial.ttf", font_size)

        # Convert x and y to integers
        x, y = int(x), int(y)

        self.image.paste(self.marker_icon, (x - 15, y - 20))

        draw.text((x + 10, y - 22), text, fill="#dbd2c9", font=font)

        self.tk_image = ImageTk.PhotoImage(self.image)
        self.scrollable_image.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)


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
        self.image.close()  # Close the image explicitly
        self.root.destroy()


    def run_editor(self):
        self.root.mainloop()

# Example usage:
if __name__ == "__main__":
    root = tk.Tk()
    editor = ImageEditor(root, "output/sequentially/final.jpeg", "hiclipart.png")
    editor.run_editor()
