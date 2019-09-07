package com.zzk;

import java.applet.Applet;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.geom.AffineTransform;

public class SlantImageApplet extends Applet {
    public void paint(Graphics g) {
        String value = "倾斜图像";
        Image img = null;// 声明图像对象
        img = getImage(getCodeBase(), "com/zzk/PD5.jpg");// 获得图片信息
        Graphics2D g2d = (Graphics2D) g;// 强制转换为Graphics2D对象
        g2d.drawString(value, 209, 170);// 绘制文本
        AffineTransform tr = new AffineTransform();// 创建AffineTransform对象
        tr.translate(210, 32);// 图像位置的平移
        tr.shear(3, 3);// 倾斜图像
        g2d.drawImage(img, tr, this);// 绘制图像
        AffineTransform tr1 = AffineTransform.getScaleInstance(3.5, 3.5);// 获得AffineTransform对象
        tr1.translate(15, 13);// 图像位置的平移
        g2d.drawImage(img, tr1, this);// 绘制图像
    }
}
