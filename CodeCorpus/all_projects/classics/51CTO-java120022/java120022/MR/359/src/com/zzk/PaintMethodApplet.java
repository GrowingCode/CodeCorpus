package com.zzk;

import java.applet.Applet;
import java.awt.Color;
import java.awt.Graphics;

public class PaintMethodApplet extends Applet {
    public void paint(Graphics g) {
        g.setColor(Color.BLUE);// 设置颜色
        g.drawString("这是使用paint()方法绘制的文本。", 30, 40);// 绘制文本内容
        g.setColor(Color.RED);// 设置颜色
        g.drawString("下面是使用paint()方法绘制的图形。", 30, 80);// 绘制文本内容
        g.drawRect(30, 100, 50, 40);// 绘制图形
    }
}
