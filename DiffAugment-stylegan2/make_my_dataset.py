import os
import argparse
import tkinter as tk
from PIL import Image, ImageTk, ImageDraw
import numpy as np
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data-dir', required=True)
parser.add_argument('-o', '--output-dir', required=True)
parser.add_argument('-w', '--window-size', type=int, default=512)
parser.add_argument('-r', '--resolution', type=int, default=256)
args = parser.parse_args()

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.state = -1
        self.canvas = tk.Canvas(self, bg='gray', height=args.window_size, width=args.window_size)
        self.canvas.bind("<Button-1>", self.L_press)
        self.canvas.bind("<ButtonRelease-1>", self.L_release)
        self.canvas.bind("<B1-Motion>", self.L_move)
        self.canvas.bind("<Button-3>", self.R_press)
        self.canvas.bind("<ButtonRelease-3>", self.R_release)
        self.canvas.bind("<B3-Motion>", self.R_move)
        self.canvas.bind("<MouseWheel>", self.mouse_wheel)
        self.canvas.bind("<Key>", self.key_down)
        self.canvas.bind("<KeyRelease>", self.key_up)
        self.canvas.pack()

        self.canvas.focus_set()
        self.canvas_image = self.canvas.create_image(0, 0, anchor='nw')

        self.flist = os.listdir(args.data_dir)
        assert len(self.flist) > 0

        try:
            os.makedirs(args.output_dir)
        except:
            pass
        
        self.image_id = -1
        self.is_moving = False
        self.is_scaling = False
        
        self.new_image()
        self.display()
    
    def generate(self, idx=0):
        self.cur_idx = idx
        latent = np.random.randn(1, *self.networks[idx].input_shape[1:])
        real = misc.adjust_dynamic_range(self.real_image, [0, 255], [-1, 1])
        if self.truncations[idx] is not None:
            fake = self.networks[idx].run(latent, self.label, real, self.mask, truncation_psi=self.truncations[idx])
        else:
            fake = self.networks[idx].run(latent, self.label, real, self.mask, is_training=True, style_mixing_prob=None)
        self.fake_image = misc.adjust_dynamic_range(fake, [-1, 1], [0, 255]).clip(0, 255).astype(np.uint8)
    
    def new_image(self):
        self.image_id += 1
        if self.image_id >= len(self.flist):
            exit()
        self.fname = self.flist[self.image_id]
        self.image = cv2.imread(os.path.join(args.data_dir, self.fname))[..., ::-1]
        self.shape = self.image.shape
        self.size = min(*self.shape[:2])
        self.begin = [(self.shape[0] - self.size) // 2, (self.shape[1] - self.size) // 2]

    def display(self, state=0):
        self.image_for_display = self.image[self.begin[0]: self.begin[0] + self.size, self.begin[1]: self.begin[1] + self.size]
        self.image_for_display_resized = cv2.resize(self.image_for_display, (args.window_size, args.window_size))
        self.tkimage = ImageTk.PhotoImage(image=Image.fromarray(self.image_for_display_resized))
        self.canvas.itemconfig(self.canvas_image, image=self.tkimage)
    
    def save_image(self):
        img = cv2.resize(self.image_for_display, (args.resolution, args.resolution))[..., ::-1]
        cv2.imwrite(os.path.join(args.output_dir, self.fname), img)
    
    def get_pos(self, event):
        return (-int(event.y), -int(event.x))
    
    def L_press(self, event):
        self.last_pos = self.get_pos(event)
        self.last_begin = self.begin
        self.is_moving = True
    
    def L_move(self, event):
        if self.is_moving:
            a = self.last_pos
            b = self.get_pos(event)
            self.begin = [self.last_begin[0] + (b[0] - a[0]) * self.size // 512, self.last_begin[1] + (b[1] - a[1]) * self.size // 512]
            self.begin = [max(self.begin[0], 0), max(self.begin[1], 0)]
            self.begin = [min(self.begin[0], self.shape[0] - self.size), min(self.begin[1], self.shape[1] - self.size)]
            self.display()
    
    def L_release(self, event):
        self.L_move(event)
        self.is_moving = False
    
    def mouse_wheel(self, event):
        center = [self.begin[0] + self.size // 2, self.begin[1] + self.size // 2]
        if event.num == 5 or event.delta == -120:
            self.size = min(int(self.size / 0.9), *self.shape[:2])
        elif event.num == 4 or event.delta == 120:
            self.size = int(self.size * 0.9)
        self.begin = [center[0] - self.size // 2, center[1] - self.size // 2]
        self.begin = [max(self.begin[0], 0), max(self.begin[1], 0)]
        self.begin = [min(self.begin[0], self.shape[0] - self.size), min(self.begin[1], self.shape[1] - self.size)]
        self.display()
    
    def R_press(self, event):
        self.last_pos = self.get_pos(event)
        self.is_scaling = True
    
    def R_move(self, event):
        a = self.last_pos
        b = self.get_pos(event)
        self.display()
    
    def R_release(self, event):
        self.R_move(event)
        self.is_scaling = False
    
    def key_down(self, event):
        if event.keysym == 'space':
            self.new_image()
            self.display()
        elif event.keysym == 'Return':
            self.save_image()
            self.new_image()
            self.display()
    
    def key_up(self, event):
        pass

def main():
    app = App()
    app.mainloop()

if __name__ == "__main__":
    main()