package com.zzk;

import java.applet.Applet;
import java.awt.Color;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Rectangle;
import java.awt.TexturePaint;
import java.awt.geom.Ellipse2D;
import java.awt.image.BufferedImage;

public class DrawGrainApplet extends Applet {
    private BufferedImage img;// 声明图像对象
    public void init() { // 初始化方法
        img = new BufferedImage(50, 50, BufferedImage.TYPE_INT_RGB);// 创建缓冲对象
        Graphics2D g = img.createGraphics();// 创建Graphics2D对象
        g.setPaint(Color.yellow);// 指定颜色
        g.draw(new Rectangle(0, 0, 25, 25));// 绘制矩形
        g.setPaint(Color.red);// 指定颜色
        g.fill(new Rectangle(25, 0, 25, 25));// 填充矩形
        g.setPaint(Color.green);// 指定颜色
        g.fill(new Rectangle(0, 0, 25, 25));// 填充矩形
    }
    public void paint(Graphics g) {
        Graphics2D g2d = (Graphics2D) g;// 强制转换为Graphics2D对象
        g2d.setPaint(new TexturePaint(img, new Rectangle(0, 0, 10, 10)));// 创建纹理并把它加到Graphics中
        g2d.fill(new Ellipse2D.Double(10, 10, 260, 145));// 填充图形
    }
}
