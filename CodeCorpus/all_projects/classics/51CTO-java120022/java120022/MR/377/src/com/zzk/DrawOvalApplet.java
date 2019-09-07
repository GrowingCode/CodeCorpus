package com.zzk;

import java.applet.Applet;
import java.awt.Color;
import java.awt.Graphics;

public class DrawOvalApplet extends Applet {
    public void paint(Graphics g) {
        String value = "画椭圆";
        int x = 25;// 椭圆位置横坐标
        int y = 40;// 椭圆位置
        int xr = 150;// 椭圆的横坐标半径
        int yr = 150;// 椭圆的纵坐标半径
        g.drawOval(x, y, xr, yr);// 绘制椭圆
        int x0 = 67;// 椭圆位置横坐标
        int y0 = 40;// 椭圆位置
        int xr0 = 65;// 椭圆的横坐标半径
        int yr0 = 150;// 椭圆的纵坐标半径
        g.setColor(Color.blue);// 设置颜色
        g.drawOval(x0, y0, xr0, yr0);// 绘制椭圆
        g.drawString(value, 180, 190);// 绘制文本
    }
}
