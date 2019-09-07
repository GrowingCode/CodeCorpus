package com.zzk;

import java.applet.Applet;
import java.awt.Color;
import java.awt.Graphics;

public class DrawRectApplet extends Applet {
    public void paint(Graphics g) {
        String value = "画矩形";
        int x = 42;// 矩形的横坐标
        int y = 27;// 矩形的纵坐标
        int width = 122;// 矩形的宽度
        int height = 64;// 矩形的高度
        g.setColor(Color.BLUE);// 设置颜色蓝色
        g.drawRect(x, y, width, height);// 绘制矩形
        int x0 = 54;// 矩形的横坐标
        int y0 = 37;// 矩形的纵坐标
        int width0 = 130;// 矩形的宽度
        int height0 = 69;// 矩形的高度
        g.setColor(Color.BLUE);// 设置颜色蓝色
        g.drawRect(x0, y0, width0, height0);// 绘制矩形
        int x1 = 67;// 矩形的横坐标
        int y1 = 48;// 矩形的纵坐标
        int width1 = 137;// 矩形的宽度
        int height1 = 73;// 矩形的高度
        g.setColor(Color.BLUE);// 设置颜色蓝色
        g.drawRect(x1, y1, width1, height1);// 绘制矩形
        g.drawString(value, 185, 143);// 绘制文本
    }
}
