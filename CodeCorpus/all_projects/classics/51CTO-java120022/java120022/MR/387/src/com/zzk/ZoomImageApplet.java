package com.zzk;

import java.applet.Applet;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.geom.AffineTransform;

public class ZoomImageApplet extends Applet {
    public void paint(Graphics g) {
        String value = "缩放图像";
        Image img = null;// 声明图像对象
        img = getImage(getCodeBase(), "com/zzk/PD2.jpg");// 获得图片信息
        AffineTransform tr = AffineTransform.getScaleInstance(0.25, 0.25);// 创建变形以获得缩放对象
        tr.translate(120, 100);// 设置对象平移
        AffineTransform tr2 = AffineTransform.getScaleInstance(0.15, 0.15);// 创建变形以获得缩放对象
        tr2.translate(900, 950);// 设置对象平移
        Graphics2D g2d = (Graphics2D) g;// 将g转换成一个可用的Graphics2D对象
        g2d.drawImage(img, tr, this);// 绘制图像
        g2d.drawImage(img, tr2, this);// 绘制图像
        g2d.drawString(value, 60, 150);// 绘制文本
    }
}
