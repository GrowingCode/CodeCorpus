package com.zzk;

import java.applet.Applet;
import java.awt.Graphics;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.awt.image.RescaleOp;

public class SetImageLightenessApplet extends Applet {
    private BufferedImage image;// 声明缓冲图像对象
    public void paint(Graphics g) {
        Image img = null;// 声明图像对象
        String value = "调整图片亮度";
        img = getImage(getCodeBase(), "com/zzk/PPD.jpg");// 获得图像对象
        int a = img.getWidth(this); // 获得图片宽度赋给变量a
        int b = img.getHeight(this);// 获得图片高度赋给变量b
        if (a >= 0 || b >= 0) {
            image = new BufferedImage(img.getWidth(this), img.getHeight(this),
                    BufferedImage.TYPE_INT_RGB);// 创建缓冲图像对象
            image.getGraphics().drawImage(img, 0, 0, null);// 在缓冲图像对象上绘制图像
            float fa = 2.0f;// 声明表示像素分量
            float fb = -30.0f;// 声明表示像素分量
            RescaleOp op = new RescaleOp(fa, fb, null);// 创建RescaleOp对象
            image = op.filter(image, null); // 过滤缓冲图像对象，实现调整图像亮度的功能
            g.drawImage(img, 30, 40, this);// 绘制原图像对象
            g.drawImage(image, 220, 40, this);// 绘制调整亮度后的缓冲图像对象
            g.drawString(value, 265, 188);// 绘制文本
        }
    }
}
