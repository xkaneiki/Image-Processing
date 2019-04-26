import cv2
import wx
import numpy as np
import os


class Contrast:
    def __init__(self):
        self.win=wx.Frame(None,title="Contrast",size=(800,600))
        self.bkg=wx.Panel(self.win)
        self.setlayout()
        self.win.Show()

    def setlayout(self):
        self.button=wx.Button(self.bkg,label="open")
        self.button.Bind(wx.EVT_BUTTON,self.open)
        layout=wx.BoxSizer()
        self.img=np.zeros((500,600),np.float)
        bmp=wx.Bitmap.FromBuffer(self.img.shape[1],self.img.shape[0],cv2.cvtColor(np.uint8(self.img),cv2.COLOR_BGR2RGB))
        self.pic=wx.StaticBitmap(self.bkg,bitmap=bmp,size=(500,600))
        p=wx.BoxSizer()
        p.Add(self.pic)
        
        layout.Add(p)

        v=wx.BoxSizer(wx.VERTICAL)
        v.Add(self.button,flag=wx.CENTER,border=5)
        
        self.sl=wx.Slider(self.bkg,value=100,minValue=0,maxValue=200,style=wx.SL_HORIZONTAL|wx.SL_LABELS)
        self.sl.Bind(wx.EVT_SLIDER,self.change)
        v.Add(self.sl,flag=wx.CENTER,border=5)
        
        layout.Add(v)
        
        self.bkg.SetSizer(layout)

    def contrast(self,src,ct):
        return ((src-0.5)*ct+0.5)*255
    
    def change(self,e):
        ct=self.sl.GetValue()/255.0
        src=np.array(self.img,np.float)/255.0
        tmp=self.contrast(src,ct)
        tmp=np.where(tmp<0,0,tmp)
        tmp=np.where(tmp>255,255,tmp)
        src=tmp.astype(np.uint8)
        src=src[0:min(500,src.shape[0]),0:min(600,src.shape[1]),:]
        print(src.shape)
        self.pic.SetBitmap(wx.Bitmap.FromBuffer(src.shape[1],src.shape[0],cv2.cvtColor(src, cv2.COLOR_BGR2RGB)))

    def open(self,e):
        wildcard = "Image Files (*.jpg)|*.jpg|(*.png)|*.png|(*.jpeg)|*.jpeg"
        dlg = wx.FileDialog(self.win, "Choose a file",
                            os.getcwd(), "", wildcard, wx.FD_OPEN)

        if dlg.ShowModal() == wx.ID_OK:
            path=dlg.GetPath()
            self.img=cv2.imread(path)
            ct=self.sl.GetValue()/100
            src=np.array(self.img,np.float)/255.0
            tmp=self.contrast(src,ct)
            tmp=np.where(tmp<0,0,tmp)
            tmp=np.where(tmp>255,255,tmp)
            src=tmp.astype(np.uint8)
            src=src[0:min(500,src.shape[0]),0:min(600,src.shape[1]),:]
            print(src.shape)
            self.pic.SetBitmap(wx.Bitmap.FromBuffer(src.shape[1],src.shape[0],cv2.cvtColor(src, cv2.COLOR_BGR2RGB)))
        dlg.Destroy()


if __name__=="__main__":
    app=wx.App()

    Contrast()
    
    app.MainLoop()