import wx  
#创建类 
class ChangeFrame(wx.Frame): 
    #初始化
    def __init__(self):
        #继承父类的__init__()函数
        wx.Frame.__init__(self, None, -1, 'Change Picture',size=(400, 360))
        self.InitUI() 
        self.Centre() 
        self.Show()

    def InitUI(self):
        #创建面板
        panel = wx.Panel(self, -1)
        #利用wxpython的GridBagSizer()进行页面布局
        sizer = wx.GridBagSizer(10, 20) #列间隔为10，行间隔为20      
        #添加上海字段，并加入页面布局，为第一行，第一列
        text1 = wx.StaticText(panel, label="上海") 
        sizer.Add(text1, pos = (0, 0), flag=wx.ALL, border=5)
        #获取shanghai.png图片，转化为Bitmap形式，添加到第一行，第二列
        image1 = wx.Image('/Users/kaneiki/Desktop/Image_Processing/imgs/haze1.jpg', wx.BITMAP_TYPE_ANY).Rescale(320, 120).ConvertToBitmap() 
        bmp1 = wx.StaticBitmap(panel, -1, image1) #转化为wx.StaticBitmap()形式
        sizer.Add(bmp1, pos=(0,1), flag=wx.ALL, border=5)
        #添加北京字段，并加入页面布局，为第二行，第一列
        self.text2 = wx.StaticText(panel, label="北京") 
        sizer.Add(self.text2, pos=(1,0), flag=wx.ALL, border=5) 
        #获取beijing.png图片，转化为Bitmap形式，添加到第二行，第二列
        image2 = wx.Image('/Users/kaneiki/Desktop/Image_Processing/imgs/haze.png', wx.BITMAP_TYPE_ANY).Rescale(320, 120).ConvertToBitmap() 
        self.bmp2 = wx.StaticBitmap(panel, -1, image2) #转化为wx.StaticBitmap()形式
        sizer.Add(self.bmp2, pos=(1,1), flag=wx.ALL, border=5)
        #添加登录按钮，并加入页面布局，为第四行，第2列
        btn = wx.Button(panel, -1, "将北京换成广州")
        sizer.Add(btn, pos=(2,1), flag=wx.ALL, border=5)
        #为登录按钮绑定change_picture事件
        self.Bind(wx.EVT_BUTTON, self.change_picture, btn)
        #将Panmel适应GridBagSizer()放置
        panel.SetSizerAndFit(sizer)

    #定义文字及图片转换函数
    def change_picture(self, event):
        #将文字“北京”换成“广州”
        self.text2.SetLabel("广州")
        #获取广州的图片，转化为Bitmap形式
        image = wx.Image('/Users/kaneiki/Desktop/Image_Processing/imgs/haze.png', wx.BITMAP_TYPE_ANY).Rescale(320, 120).ConvertToBitmap()
        #更新GridBagSizer()的self.bmp2
        self.bmp2.SetBitmap(wx.BitmapFromImage(image))
#主函数        
if __name__ == '__main__':  
    app = wx.App()  
    frame = ChangeFrame()   
    app.MainLoop()