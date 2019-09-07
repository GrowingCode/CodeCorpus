package com.zzk;

import java.applet.Applet;
import java.awt.Color;
import java.awt.Graphics;

public class UpdateMethodApplet extends Applet {
    boolean flag = false;// 定义标记变量
    public void start() {
        repaint();// 重新调用paint方法
    }
    public void paint(Graphics g) {
        g.setColor(Color.RED);// 设置颜色
        g.drawString("这是使用paint()方法绘制的文本。", 30, 120);// 绘制文本
        g.setColor(Color.BLUE);// 设置颜色
        g.drawString("下面是使用paint()方法绘制的图形。", 30, 140);// 绘制文本
        g.drawRect(30, 150, 50, 40);// 绘制矩形
        update(g);// 调用updatae方法
    }
    public void update(Graphics g) {
        if (flag) {
            g.clearRect(0, 0, 300, 220);// 标记变量为true时,清除内容
            flag = false;// 设置标记变量为false
        } else {
            flag = true;// 设置标记变量为true
        }
        g.setColor(Color.BLUE);// 设置颜色
        g.drawString("这是使用updatePaint()方法绘制的文本。", 30, 20);// 绘制文本
        g.setColor(Color.RED);// 设置颜色
        g.drawString("下面是使用updatePaint()方法绘制的图形。", 30, 40);// 绘制文本
        g.drawRect(30, 50, 50, 40);// 绘制矩形
    }
}
