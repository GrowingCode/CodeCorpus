package com.zzk;

import java.applet.Applet;
import java.awt.Color;
import java.awt.Graphics;

public class DrawPolygonApplet extends Applet {
    public void paint(Graphics g) {
        String value = "绘制多角形";
        int x[] = { 60, 103, 170, 150, 120 };// 声明多角形的横坐标数组
        int y[] = { 80, 180, 140, 80, 120 };// 声明多角形的纵坐标数组
        int num = x.length;// 取得多角形x,y坐标对数组的长度
        g.setColor(Color.blue);// 设置颜色
        g.drawPolygon(x, y, num);// // 绘制多角形
        g.drawString(value, 120, 70);// 绘制文本
    }
}
