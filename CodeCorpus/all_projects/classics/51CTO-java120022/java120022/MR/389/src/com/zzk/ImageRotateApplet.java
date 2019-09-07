package com.zzk;

import java.applet.Applet;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.geom.AffineTransform;
import java.io.File;
import java.io.IOException;
import javax.imageio.ImageIO;

public class ImageRotateApplet extends Applet {
    public void paint(Graphics g) {
        String value = "旋转图像";
        Image img = null;// 声明图像对象
        img = getImage(getCodeBase(), "com/zzk/PD2.jpg");// 获得图片信息
        Graphics2D g2d = (Graphics2D) g; // 强制转换为Graphics2D对象
        g2d.drawString(value, 180, 150);// 绘制文本
        int x = 50;// 图像位置的横坐标
        int y = -10;// 图像位置的纵坐标
        int w = img.getWidth(this);// 获得图片的宽度
        int h = img.getHeight(this);// 获得图片的高度
        g2d.drawImage(img, x, y + 50, w / 5, h / 5, this);// 绘制图形
        g2d.drawImage(img, x + 150, y + 50, w / 5, h / 5, this);// 绘制图形
        AffineTransform tr = new AffineTransform();// 创建变形对象
        tr.rotate(90, 15, 15, 65);// 设置旋转角度
        g2d.setTransform(tr);// 执行旋转
        g2d.drawImage(img, x + 150, y + 20, w / 5, h / 5, this);// 绘制图形
        tr.rotate(35, 15, 30, 65);// 设置旋转角度
        g2d.setTransform(tr);// 执行旋转
        g2d.drawImage(img, x + 120, y - 60, w / 5, h / 5, this);// 绘制图形
    }
}