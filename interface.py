import tkinter as  tk
from tkinter.filedialog import *
from tkinter import ttk;
import function
import cv2
from PIL import Image, ImageTk
import threading
import time


class interface(ttk.Frame):
    pic_path = ""
    viewHigh = 600
    viewWide = 600
    updataTime = 0
    thread = None
    threadRun = False
    camera = None
    colorTransform = {"green": "绿色","yellow": "黄色", "blue":"蓝色","black":"黑色"}

    def __init__(self, win):
        ttk.Frame.__init__(self, win)
        frame_left = ttk.Frame(self)
        frame_right1 = ttk.Frame(self)
        frame_right2 = ttk.Frame(self)
        win.title("车牌识别")
        #win.state("normal")
        self.pack(fill=tk.BOTH, expand=tk.YES, padx="5", pady="5")
        frame_left.pack(side=LEFT, expand=1, fill=BOTH)
        frame_right1.pack(side=TOP, expand=1, fill=tk.Y)
        frame_right2.pack(side=RIGHT, expand=0)
        ttk.Label(frame_left, text='原图：').pack(anchor="nw")
        ttk.Label(frame_right1, text='车牌位置：').grid(column=0, row=0, sticky=tk.W)

        from_pic_ctl = ttk.Button(frame_right2, text="打开图片", width=20, command=self.from_pic)

        self.image_ctl = ttk.Label(frame_left)
        self.image_ctl.pack(anchor="nw")

        self.roi_ctl = ttk.Label(frame_right1)
        self.roi_ctl.grid(column=0, row=1, sticky=tk.W)
        ttk.Label(frame_right1, text='识别结果：').grid(column=0, row=2, sticky=tk.W)
        self.r_ctl = ttk.Label(frame_right1, text="")
        self.r_ctl.grid(column=0, row=3, sticky=tk.W)
        self.color_ctl = ttk.Label(frame_right1, text="", width="20")
        self.color_ctl.grid(column=0, row=4, sticky=tk.W)
        from_pic_ctl.pack(anchor="se", pady="5")
        self.predictor = function.Predictor()
        self.predictor.train_svm()

    def from_pic(self):
        self.threadRun = False
        self.pic_path = askopenfilename(title="选择识别图片", filetypes=[("jpg图片", "*.jpg"),("png图片", "*.png")])
        if self.pic_path:
            img_bgr = function.imreadex(self.pic_path)
            self.imgtk = self.get_imgtk(img_bgr)
            self.image_ctl.configure(image=self.imgtk)
            text, plate, color = function.all(img_bgr)
            self.show(text, plate, color)

    def show(self, text, plate, color):
        if text:
            plate = cv2.cvtColor(plate, cv2.COLOR_BGR2RGB)
            plate = Image.fromarray(plate)
            self.imgtk_roi = ImageTk.PhotoImage(image=plate)
            self.roi_ctl.configure(image=self.imgtk_roi, state='enable')
            self.r_ctl.configure(text=str(text))
            self.update_time = time.time()

            c = self.colorTransform[color]
            print(c)
            if(color == 'black'):
                self.color_ctl.configure(text=c, background=color,foreground='red', state='enable')
            else:
                self.color_ctl.configure(text=c, background=color,foreground='white', state='enable')
            #self.color_ctl=ttk.Label(ttk.Frame(self),text=c, background=color, state='enable').grid(column=0, row=4, sticky=tk.W)

            #except:
                # self.color_ctl.configure(state='disabled')
        elif self.update_time + 8 < time.time():
            self.roi_ctl.configure(state='disabled')
            self.r_ctl.configure(text="")
            #self.color_ctl.configure(state='disabled')

    def get_imgtk(self, img_bgr):
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=im)
        wide = imgtk.width()
        high = imgtk.height()
        if wide > self.viewWide or high > self.viewHigh:
            wide_factor = self.viewWide / wide
            high_factor = self.viewHigh / high
            factor = min(wide_factor, high_factor)
            wide = int(wide * factor)
            if wide <= 0: wide = 1
            high = int(high * factor)
            if high <= 0: high = 1
            im = im.resize((wide, high), Image.ANTIALIAS)
            imgtk = ImageTk.PhotoImage(image=im)
        return imgtk

def close_window():
    print("destroy")
    if interface.threadRun:
        interface.threadRun = False
        interface.thread.join(2.0)
    win.destroy()

if __name__ == '__main__':
    win = tk.Tk()

    interface = interface(win)
    win.protocol('WM_DELETE_WINDOW', close_window)
    win.mainloop()