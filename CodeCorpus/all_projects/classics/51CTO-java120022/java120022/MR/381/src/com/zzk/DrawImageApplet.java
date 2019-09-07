package com.zzk;

import java.applet.Applet;
import java.awt.Graphics;
import java.awt.Image;

public class DrawImageApplet extends Applet {
    public void paint(Graphics g) {// 初始化方法
        String value = "绘制图片";
        Image image = null;// 声明图像对象
        image = getImage(getCodeBase(), "com/zzk/PD2.jpg");// 获得图片信息
        int x = 10;// 图像位置的横坐标
        int y = 20;// 图像位置的纵坐标
        int width = image.getWidth(this);// 获得图像的宽度
        int height = image.getHeight(this);// 获取图像的高度
        g.drawImage(image, x + 150, y + 30, width / 5, height / 5, this);// 绘制图像
        g.drawImage(image, x + 25, y + 10, (int) (width * 0.2),
                (int) (height * 0.3), this);// 绘制图像
        g.drawString(value, 140, 170);// 绘制文本
    }
}
