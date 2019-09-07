package com.zzk;

import java.applet.Applet;
import java.awt.Color;
import java.awt.Graphics;

public class RepaintMethodApplet extends Applet {
    static int iFlag = 0;// 定义标记变量
    public void start() {
        iFlag++;// 调整标记变量的值
        repaint();// 重新调用paint()方法
    }
    public void paint(Graphics g) {
        if (iFlag == 1) {
            g.drawString("这是使用paint()方法绘制的文本。", 30, 60);// 绘制文本
        } else if (iFlag == 2) {
            g.setColor(Color.RED);// 设置颜色
            g.drawString("下面是使用paint()方法绘制的图形。", 30, 60);// 绘制文本
            g.drawRect(30, 80, 50, 40);// 绘制矩形
        } else if (iFlag == 3) {
            g.setColor(Color.BLUE);// 设置颜色
            g.drawString("下面是使用paint()方法绘制的图形。", 30, 60);// 绘制文本
            g.drawRect(30, 80, 50, 40);// 绘制矩形
        } else {
            g.setColor(Color.GREEN);// 设置颜色
            g.drawString("下面是使用paint()方法绘制的图形。", 30, 60);// 绘制文本
            g.drawRect(30, 80, 50, 40);// 绘制矩形
        }
    }
}
