import argparse
import numpy as np
from tkinter import *
from tkinter import ttk
from tkinter import colorchooser
from tkinter.filedialog import *
from tkmacosx import Button
from PIL import Image, ImageTk
from algorithm import ImagePalette

class UI():
    def __init__(self, root, args):
        self.args = args
        self.centers = np.zeros((args.k, 3))

        root.title("Palette")
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)

        # Create a frame, split to grid
        self.mainframe = ttk.Frame(root, padding="12 12 12 12")
        self.mainframe.grid(column=0, row=0, sticky=(N, W, E, S))

        self.mainframe.columnconfigure(0, weight=1)
        self.mainframe.rowconfigure(0, weight=1)
        
        # Create a canvas to display image
        self.canvas = Canvas(self.mainframe, width=760, height=505, background='gray75')
        self.canvas.grid(column=0, row=0, rowspan=args.k + 1, sticky=(N, W, E, S))

        # Palette
        self.palettes = []
        for i in range(args.k):
            self.palettes.append(Button(self.mainframe, width=80, height=80, background='gray75', focuscolor='gray75', command=self.choose_color(i)))
            self.palettes[i].grid(column=1, row=i+1, sticky=S)
            self.palettes[i].columnconfigure(0, weight=1)
            self.palettes[i].rowconfigure(0, weight=1)

        # Choose image button
        self.button_frame = ttk.Frame(self.mainframe, padding="15 20 0 0")
        self.button_frame.grid(column=1, row=0, sticky=N+E+W)
        self.choose_button = ttk.Button(self.button_frame, text="Choose Image", command=self.choose_image)
        self.choose_button.grid(column=0, row=0, sticky=N)

        # Reset image button
        self.reset_button = ttk.Button(self.button_frame, text="Reset Image", command=self.reset_image)
        self.reset_button.grid(column=0, row=1, sticky=N)
        
        # Save image button
        self.save_button = ttk.Button(self.button_frame, text="Save Image", command=self.save_image)
        self.save_button.grid(column=0, row=2, sticky=N)

        # Init image
        image = Image.open(args.image)
        self.show_image(image)

    def from_rgb(self, rgb):
        """
        translates an rgb tuple of int to a tkinter friendly color code
        """
        return f'#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}'

    def choose_image(self):
        path = askopenfile()
        if path:
            image = Image.open(path.name)
            # resize image
            image.thumbnail((760, 505))
            self.show_image(image)
    
    def reset_image(self):
        self.image_palette.reset_palette()
        image = self.image_palette.get_img()
        self.image = ImageTk.PhotoImage(Image.fromarray(image))
        self.canvas.create_image(2.5, 2.5, anchor=NW, image=self.image)
        self.centers = self.image_palette.get_center_color()
        self.update_palette()

    def save_image(self):
        path = asksaveasfilename(defaultextension=".png")
        if path:
            with open(path, 'wb') as f:
                ImageTk.getimage(self.image).save(path)

    def choose_color(self, i):
        def callback():
            color = colorchooser.askcolor(title="Choose color")
            if color:
                color = np.asarray(color[0], dtype=np.uint8)
                self.centers[i] = color
                self.update_palette()
                self.update_image(color, i)
                self.centers = self.image_palette.get_center_color()
                self.update_palette()
        return callback

    def update_image(self, new_color, pid):
        edited_image = self.image_palette.update_image(new_color, pid)
        self.image = ImageTk.PhotoImage(Image.fromarray(edited_image))
        self.canvas.create_image(2.5, 2.5, anchor=NW, image=self.image)

    def show_image(self, image):
        self.image = ImageTk.PhotoImage(image)
        self.canvas.create_image(2.5, 2.5, anchor=NW, image=self.image)
        # fit canvas to image
        self.canvas.config(width=self.image.width(), height=self.image.height())
        # Refresh canvas
        self.canvas.update()
        # Calculate centers
        self.image_palette = ImagePalette(self.args, image)
        self.centers = self.image_palette.kmeans()
        # Update palette
        self.update_palette()

    def update_palette(self):
        for i in range(self.args.k):
            palette_i = self.from_rgb(self.centers[i].tolist())
            self.palettes[i].config(background=palette_i, focuscolor=palette_i)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="K-means clustering")
    parser.add_argument(
        '--image',
        type=str, 
        help='Path to the image',
        default='../data/src3.png'
    )
    parser.add_argument(
        '--k', 
        type=int, 
        help='Number of clusters',
        default=5
    )
    args = parser.parse_args()

    src = np.asarray(Image.open(args.image))[...,:3]
    # display image on tkinter window
    
    root = Tk()
    UI(root, args)
    root.mainloop()