package com.zzk;

import java.applet.Applet;
import java.awt.Color;
import java.awt.Graphics;

public class DrawLineApplet extends Applet {
    public void paint(Graphics g) {
        String value = "画直线";
        int x = 215;// 直线的横坐标(右)
        int y = 45;// 直线的纵坐标(右)
        int x1 = 15;// 直线的横坐标(左)
        int y1 = 45;// 直线的纵坐标(左)
        int x2 = 300;// 直线的横坐标(右)
        int y2 = 100;// 直线的纵坐标(右)
        int x3 = 60;// 直线的横坐标(左)
        int y3 = 100;// 直线的纵坐标(左)
        g.setColor(Color.blue);// 设置颜色红色
        g.drawLine(x, y, x1, y1);// 绘制直线
        g.drawLine(x2, y2, x3, y3);// 绘制直线
        g.drawString(value, 120, 75);// 绘制文本
    }
}
