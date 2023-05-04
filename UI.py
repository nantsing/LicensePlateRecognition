# -*- coding:utf-8 -*-
# author: DuanshengLiu
import sys
import time
import numpy as np
import cv2 as cv
from tkinter import *
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk
from recognize import *


class Window:
    def __init__(self, win, ww, wh):
        self.win = win
        self.ww = ww
        self.wh = wh
        self.win.geometry("%dx%d+%d+%d" % (ww, wh, 200, 50))  # 界面启动时的初始位置
        self.win.title("基于SVM的车牌识别系统")
        self.img_src_path = None

        self.label_src = Label(self.win, text='原图:', font=('微软雅黑', 13)).place(x=0, y=0)
        self.label_time = Label(self.win, text='识别总耗时:', font=('微软雅黑', 13)).place(x=50, y=wh - 55)
        self.label_lic1 = Label(self.win, text='车牌区域1:', font=('微软雅黑', 13)).place(x=615, y=0)
        self.label_pred1 = Label(self.win, text='识别结果1:', font=('微软雅黑', 13)).place(x=615, y=85)
        self.label_lic2 = Label(self.win, text='车牌区域2:', font=('微软雅黑', 13)).place(x=615, y=180)
        self.label_pred2 = Label(self.win, text='识别结果2:', font=('微软雅黑', 13)).place(x=615, y=265)
        self.label_lic3 = Label(self.win, text='车牌区域3:', font=('微软雅黑', 13)).place(x=615, y=360)
        self.label_pred3 = Label(self.win, text='识别结果3:', font=('微软雅黑', 13)).place(x=615, y=445)
        self.label_note = Label(self.win, text='注意：若传入单张车牌，请选择简单识别；其他请选择一般识别！:', \
                                font=('微软雅黑', 13)).place(x=615, y=wh - 50)

        self.can_src = Canvas(self.win, width=512, height=512, bg='white', relief='solid', borderwidth=1)  # 原图画布
        self.can_src.place(x=50, y=0)
        self.can_time = Canvas(self.win, width=200, height=35, bg='white', relief='solid', borderwidth=1) 
        self.can_time.place(x=130, y=wh - 60)
        self.can_lic1 = Canvas(self.win, width=245, height=85, bg='white', relief='solid', borderwidth=1)  # 车牌区域1画布
        self.can_lic1.place(x=710, y=0)
        self.can_pred1 = Canvas(self.win, width=245, height=65, bg='white', relief='solid', borderwidth=1)  # 车牌识别1画布
        self.can_pred1.place(x=710, y=90)
        self.can_lic2 = Canvas(self.win, width=245, height=85, bg='white', relief='solid', borderwidth=1)  # 车牌区域2画布
        self.can_lic2.place(x=710, y=175)
        self.can_pred2 = Canvas(self.win, width=245, height=65, bg='white', relief='solid', borderwidth=1)  # 车牌识别2画布
        self.can_pred2.place(x=710, y=265)
        self.can_lic3 = Canvas(self.win, width=245, height=85, bg='white', relief='solid', borderwidth=1)  # 车牌区域3画布
        self.can_lic3.place(x=710, y=350)
        self.can_pred3 = Canvas(self.win, width=245, height=65, bg='white', relief='solid', borderwidth=1)  # 车牌识别3画布
        self.can_pred3.place(x=710, y=440)

        self.button1 = Button(self.win, text='选择文件', width=10, height=1, command=self.load_show_img)  # 选择文件按钮
        self.button1.place(x=500, y=wh - 30)
        
        self.button2 = Button(self.win, text='简单识别', width=10, height=1, command=self.display_easy)  # 识别车牌按钮
        self.button2.place(x=625, y=wh - 30)
        
        self.button2 = Button(self.win, text='一般识别', width=10, height=1, command=self.display)  # 识别车牌按钮
        self.button2.place(x=750, y=wh - 30)
        
        self.button3 = Button(self.win, text='清空所有', width=10, height=1, command=self.clear)  # 清空所有按钮
        self.button3.place(x=875, y=wh - 30)

        print("已启动,开始识别吧！")


    def load_show_img(self):
        self.clear()
        sv = StringVar()
        sv.set(askopenfilename())
        self.img_src_path = Entry(self.win, state='readonly', text=sv).get()  # 获取到所打开的图片
        img_open = Image.open(self.img_src_path)
        if img_open.size[0] * img_open.size[1] > 240 * 80:
            img_open = img_open.resize((512, 512), Image.LANCZOS)
        self.img_Tk = ImageTk.PhotoImage(img_open)
        self.can_src.create_image(258, 258, image=self.img_Tk, anchor='center')
        
    def display_easy(self):
        if self.img_src_path == None:  # 还没选择图片就进行预测
            self.can_pred1.create_text(32, 15, text='请选择图片', anchor='nw', font=('黑体', 28))
        else:
            start = time.time()
            img_src = cv.imread(self.img_src_path)
            h, w = img_src.shape[0], img_src.shape[1]
            strings = RecognizePlates([img_src], log= False)
            
            if len(strings) != 0 and strings[0] != '':
                end = time.time()
                self.can_lic1.delete('all')
                self.can_pred1.delete('all')
                self.can_time.delete('all')
                self.can_time.create_text(5, 5, text=str(round(end - start, 6)) + ' s', anchor='nw', font=('黑体', 28))
                image = Image.fromarray(img_src[:, :, ::-1])
                image = image.resize((245, 65), Image.LANCZOS)
                self.lic_Tk1 = ImageTk.PhotoImage(image)
                self.can_lic1.create_image(5, 5, image=self.lic_Tk1, anchor='nw')
                self.can_pred1.create_text(35, 15, text=strings[0], anchor='nw', font=('黑体', 28))
            else:  
                self.can_pred1.create_text(47, 15, text='未能识别', anchor='nw', font=('黑体', 27))
            

    def display(self):
        if self.img_src_path == None:  # 还没选择图片就进行预测
            self.can_pred1.create_text(32, 15, text='请选择图片', anchor='nw', font=('黑体', 28))
        else:
            start = time.time()
            img_src = cv.imread(self.img_src_path)
            h, w = img_src.shape[0], img_src.shape[1]
            imageCopy, plates, strings = Recognize(self.img_src_path, log= True)

            if len(strings) != 0 and strings[0] != '':
                end = time.time()
                img = Image.fromarray(imageCopy[:, :, ::-1])  # 将BGR转为RGB
                if img.size[0] * img.size[1] > 240 * 80:
                    img = img.resize((512, 512), Image.LANCZOS)
                self.img_Tk = ImageTk.PhotoImage(img)
                self.can_src.delete('all')  # 显示前,先清空画板
                self.can_time.delete('all')
                self.can_src.create_image(258, 258, image=self.img_Tk,
                                          anchor='center')  # 绘制出了定位的车牌轮廓,将其显示在画板上
                self.can_time.create_text(5, 5, text=str(round(end - start, 6)) + ' s', anchor='nw', font=('黑体', 28))
                for i, plate in enumerate(plates):
                    if i == 0:
                        image = Image.fromarray(plate[:, :, ::-1])
                        image = image.resize((245, 65), Image.LANCZOS)
                        self.lic_Tk1 = ImageTk.PhotoImage(image)
                        self.can_lic1.delete('all')
                        self.can_pred1.delete('all')
                        self.can_lic1.create_image(5, 5, image=self.lic_Tk1, anchor='nw')
                        self.can_pred1.create_text(35, 15, text=strings[0], anchor='nw', font=('黑体', 28))
                    elif i == 1:
                        image = Image.fromarray(plate[:, :, ::-1])
                        image = image.resize((245, 65), Image.LANCZOS)
                        self.lic_Tk2 = ImageTk.PhotoImage(image)
                        self.can_lic2.delete('all')
                        self.can_pred2.delete('all')
                        self.can_lic2.create_image(5, 5, image=self.lic_Tk2, anchor='nw')
                        self.can_pred2.create_text(35, 15, text=strings[1], anchor='nw', font=('黑体', 28))
                    elif i == 2:
                        image = Image.fromarray(plate[:, :, ::-1])
                        image = image.resize((245, 65), Image.LANCZOS)
                        self.lic_Tk3 = ImageTk.PhotoImage(image)
                        self.can_lic3.delete('all')
                        self.can_pred3.delete('all')
                        self.can_lic3.create_image(5, 5, image=self.lic_Tk3, anchor='nw')
                        self.can_pred3.create_text(35, 15, text=strings[2], anchor='nw', font=('黑体', 28))

            else:  
                self.can_pred1.create_text(47, 15, text='未能识别', anchor='nw', font=('黑体', 27))

    def clear(self):
        self.can_src.delete('all')
        self.can_time.delete('all')
        self.can_lic1.delete('all')
        self.can_lic2.delete('all')
        self.can_lic3.delete('all')
        self.can_pred1.delete('all')
        self.can_pred2.delete('all')
        self.can_pred3.delete('all')
        self.img_src_path = None

    def closeEvent():  # 关闭前清除session(),防止'NoneType' object is not callable
        # keras.backend.clear_session()
        sys.exit()


if __name__ == '__main__':
    win = Tk()
    ww = 1000  # 窗口宽设定1000
    wh = 600  # 窗口高设定600
    Window(win, ww, wh)
    win.protocol("WM_DELETE_WINDOW", Window.closeEvent)
    win.mainloop()
