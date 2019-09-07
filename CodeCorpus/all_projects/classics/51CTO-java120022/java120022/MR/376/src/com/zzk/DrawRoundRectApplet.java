package com.zzk;

import java.applet.Applet;
import java.awt.Color;
import java.awt.Graphics;

public class DrawRoundRectApplet extends Applet {
    public void paint(Graphics g) {
        String value = "画圆角矩形";
        int x = 20;// 圆角矩形位置的横坐标
        int y = 20;// 圆角矩形位置的纵坐标
        int width = 70;// 圆角矩形宽度
        int height = 129;// 圆角矩形高度
        int xr = 5;                         // 圆角矩形圆角的水平弧度
        int yr = 7;                         // 圆角矩形圆角的垂直弧度

        g.setColor(Color.blue);// 设置颜色
        g.drawRoundRect(x, y, width, height, xr, yr); // 绘制图形
        
        int x0 = 35; // 圆角矩形位置的横坐标
        int y0 = 35; // 圆角矩形位置的纵坐标
        int width0 = 82; // 圆角矩形宽度
        int height0 = 139; // 圆角矩形高度
        int xr0 = 10; // 圆角矩形圆角的水平弧度
        int yr0 = 12; // 圆角矩形圆角的垂直弧度
        g.setColor(Color.blue);// 设置颜色
        g.drawRoundRect(x0, y0, width0, height0, xr0, yr0);// 绘制图形
        
        int x1 = 59;// 圆角矩形位置的横坐标
        int y1 = 59;// 圆角矩形位置的纵坐标
        int width1 = 92;// 圆角矩形宽度
        int height1 = 151;// 圆角矩形高度
        int xr1 = 20;// 圆角矩形圆角的水平弧度
        int yr1 = 22;// 圆角矩形圆角的垂直弧度
        g.setColor(Color.blue);// 设置颜色
        g.drawRoundRect(x1, y1, width1, height1, xr1, yr1);// 绘制图形
        g.drawString(value, 165, 30);// 绘制文本
    }
}
