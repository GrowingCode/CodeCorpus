package com.zzk;

import java.applet.Applet;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.geom.AffineTransform;

public class TurnImageApplet extends Applet {
    public void paint(Graphics g) {
        String value = "翻转图像";
        Image img = null;// 声明图像对象
        img = getImage(getCodeBase(), "com/zzk/PD4.jpg");// 获得图片信息
        int w = img.getWidth(this);// 设置图像的宽度
        int h = img.getHeight(this);// 设置图像的高度
        Graphics2D g2d = (Graphics2D) g;// 将g转换为可以利用的Graphics2D
        g2d.drawString(value, 100, 130);// 绘制文本
        AffineTransform tr = new AffineTransform(-1, 0, 0, 1, w, 0);// 创建变换对象并横向翻转
        AffineTransform tr2 = new AffineTransform(1, 0, 0, -1, 0, h);// 创建变换对象并垂直翻转
        tr.translate(-20, 40);// 图像位置的平移
        tr2.translate(120, -40);// 图像位置的平移
        g2d.drawImage(img, tr, this);// 绘制图像
        g2d.drawImage(img, tr2, this);// 绘制图像
    }
}
