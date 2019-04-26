import wx
import os
import cv2
import numpy as np

from scipy.ndimage.filters import convolve


class DeHaze():
    def __init__(self):
        self.win = wx.Frame(None, title="DeHaze", size=(700, 500))
        # self.win.EnableCloseButton(True)
        self.setLayout()
        self.win.Show()

    def togray(self, src):
        img = np.array(src, np.float)
        res = (img[:, :, 0]*0.299+img[:, :, 1]*0.587+img[:, :, 2]*0.144)
        return res.astype(np.uint8)

    def mean(self, src, size):
        filter_ = np.zeros(size)+1.0/(size[0]*size[1])
        return convolve(src, filter_)

    def guidefilter(self, I, P, r, exps):
        w = (2*r+1, 2*r+1)
        meani = self.mean(I, w)
        meanp = self.mean(P, w)

        corri = self.mean(I*I, w)
        corrip = self.mean(I*P, w)

        vari = corri-meani*meani
        covip = corrip-meani*meanp

        a = covip/(vari+exps)
        b = meanp-a*meani

        meana = self.mean(a, w)
        meanb = self.mean(b, w)
        q = meana*I+meanb

        return q

    def _setLayout(self):
        bkg = wx.Panel(self.win)
        button = wx.Button(bkg, label="open")
        button.Bind(wx.EVT_BUTTON, self.open)
        h = wx.BoxSizer()
        h.Add(button, flag=wx.CENTER)
        bkg.SetSizer(h)
        pass

    def open(self, e):
        # img=cv2.imread("/Users/kaneiki/Desktop/Image_Processing/imgs/haze.png")
        # self.pic.SetBitmap(wx.Bitmap.FromBuffer(img.shape[1],img.shape[0],img))
        wildcard = "Image Files (*.jpg)|*.jpg|(*.png)|*.png|(*.jpeg)|*.jpeg"
        dlg = wx.FileDialog(self.win, "Choose a file",
                            os.getcwd(), "", wildcard, wx.FD_OPEN)

        if dlg.ShowModal() == wx.ID_OK:
            self.path = dlg.GetPath()
            self.img = cv2.imread(self.path)
            # print(self.path)
            # print(self.img)
            A = self.sl1.GetValue()
            size = self.sl2.GetValue()
            w = (self.sl3.GetValue())/100.0
            img = self.dehaze(self.img, 2*size+1, w, A)
            self.pic.SetBitmap(wx.Bitmap.FromBuffer(
                img.shape[1], img.shape[0], cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
        dlg.Destroy()

    def change(self, e):
        A = self.sl1.GetValue()
        size = self.sl2.GetValue()
        w = (self.sl3.GetValue())/100.0
        img = self.dehaze(self.img, 2*size+1, w, A)
        self.pic.SetBitmap(wx.Bitmap.FromBuffer(
            img.shape[1], img.shape[0], cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))

    def get_dark(self, src, r=1):
        t = np.min(src, axis=2)
        w, h = t.shape
        res = t.copy()
        for i in range(r+1):
            res[0:w-i, 0:h -
                i] = np.min(np.stack([res[0:w-i, 0:h-i], t[i:w, i:h]], axis=2), axis=2)
            res[i:w, i:h] = np.min(
                np.stack([res[i:w, i:h], t[0:w-i, 0:h-i]], axis=2), axis=2)
        res = np.array(cv2.bilateralFilter(
            res.astype(np.uint8), 2, 0.6, 0.8), np.uint8)
        return res

    def dehaze(self, src, size, w=0.95, A=255):
        img = np.array(src, dtype=np.float)
        print("img", img)

        # A = np.max(img)
        dark = self.get_dark(img, size//2)
        print("dark", dark)
        A0 = np.max(img[:,:,0])
        A1 = np.max(img[:,:,1])
        A2 = np.max(img[:,:,2])
        
        # g = self.togray(src)
        # dark = self.guidefilter(dark, g, 2, 100)
        ia=img.copy()
        ia[:,:,0]=ia[:,:,0]/A0
        ia[:,:,1]=ia[:,:,1]/A1
        ia[:,:,1]=ia[:,:,2]/A2

        t = 1 - w*self.get_dark(ia,size//2)
        t = np.where(t < 0.1, 0.1, t)
        print(t.shape)
        print("t", t)

        res = np.zeros(img.shape)
        res[:, :, 0] = (img[:, :, 0]-A)/t+A
        res[:, :, 1] = (img[:, :, 1]-A)/t+A
        res[:, :, 2] = (img[:, :, 2]-A)/t+A
        print("res", res)
        res = res.astype(np.uint8)
        return res

    def setLayout(self):
        bkg = wx.Panel(self.win)
        layout = wx.BoxSizer()
        v1 = wx.BoxSizer(wx.VERTICAL)
        # b=wx.Bitmap("/Users/kaneiki/Desktop/Image_Processing/imgs/haze1.jpg", wx.BITMAP_TYPE_ANY)
        self.img = cv2.imread(
            "/Users/kaneiki/Desktop/Image_Processing/imgs/haze1.jpg")
        img = self.dehaze(self.img, 3, 0.85, 209)
        bmp = wx.Bitmap.FromBuffer(img.shape[1], img.shape[0], cv2.cvtColor(
            np.uint8(img), cv2.COLOR_BGR2RGB))
        self.pic = wx.StaticBitmap(bkg, bitmap=bmp)
        v1.Add(self.pic)

        v2 = wx.BoxSizer(wx.VERTICAL)
        bu = wx.Button(bkg, label="open")
        bu.Bind(wx.EVT_BUTTON, self.open)

        self.sl1 = wx.Slider(bkg, value=209, minValue=0, maxValue=255,
                             style=wx.SL_HORIZONTAL | wx.SL_LABELS)
        self.sl1.Bind(wx.EVT_SLIDER, self.change)
        self.sl2 = wx.Slider(bkg, value=1, minValue=0, maxValue=20,
                             style=wx.SL_HORIZONTAL | wx.SL_LABELS)
        self.sl2.Bind(wx.EVT_SLIDER, self.change)
        self.sl3 = wx.Slider(bkg, value=85, minValue=0, maxValue=100,
                             style=wx.SL_HORIZONTAL | wx.SL_LABELS)
        self.sl3.Bind(wx.EVT_SLIDER, self.change)

        v2.Add(bu, flag=wx.TOP, border=5)
        v2.Add(wx.StaticText(bkg, label="A:"), flag=wx.LEFT, border=5)
        v2.Add(self.sl1, flag=wx.TOP, border=5)
        v2.Add(wx.StaticText(bkg, label="size:"), flag=wx.LEFT, border=5)
        v2.Add(self.sl2, flag=wx.TOP, border=5)
        v2.Add(wx.StaticText(bkg, label="w:"), flag=wx.LEFT, border=5)
        v2.Add(self.sl3, flag=wx.TOP, border=5)

        layout.Add(v1, proportion=1, flag=wx.EXPAND |
                   wx.LEFT | wx.BOTTOM | wx.RIGHT, border=5)
        layout.Add(v2, proportion=1, flag=wx.EXPAND | wx.LEFT, border=5)

        bkg.SetSizer(layout)


if __name__ == "__main__":
    app = wx.App()
    DeHaze()
    app.MainLoop()
