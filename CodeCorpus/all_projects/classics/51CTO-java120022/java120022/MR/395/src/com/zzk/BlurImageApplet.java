package com.zzk;

import java.applet.Applet;
import java.awt.Graphics;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.awt.image.ConvolveOp;
import java.awt.image.Kernel;

public class BlurImageApplet extends Applet {
    private BufferedImage image;// 声明缓冲图像对象
    public void paint(Graphics g) {
        Image img = null;// 声明创建图像对象
        String value = "模糊图像";
        img = getImage(getCodeBase(), "com/zzk/PPD.jpg");// 获得图片信息
        int a = img.getWidth(this); // 获得图片宽度赋给变量a
        int b = img.getHeight(this);// 获得图片高度赋给变量b
        if (a >= 0 || b >= 0) {
            image = new BufferedImage(img.getWidth(this), img.getHeight(this),
                    BufferedImage.TYPE_INT_RGB);// 创建缓冲图像对象
            image.getGraphics().drawImage(img, 0, 0, null);// 在缓冲图像对象上绘制图像
            float[] data = { 0.0532f, 0.132f, 0.0532f, 0.132f, 0.19f, 0.138f,
                    0.0532f, 0.132f, 0.0532f };// 声明表示像素分量的数组
            Kernel kernel = new Kernel(3, 3, data); // 创建 Kernel对象
            ConvolveOp op = new ConvolveOp(kernel);// 创建ConvolveOp对象
            image = op.filter(image, null); // 过滤缓冲图像对象
            g.drawImage(img, 25, 35, this);// 绘制缓冲图像对象
            g.drawImage(image, 215, 35, this);// 绘制缓冲图像对象
            g.drawString(value, 268, 186);// 绘制文本
        }
    }
}