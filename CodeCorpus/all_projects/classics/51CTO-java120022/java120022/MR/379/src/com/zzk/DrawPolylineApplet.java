package com.zzk;

import java.applet.Applet;
import java.awt.Color;
import java.awt.Graphics;

public class DrawPolylineApplet extends Applet {
    public void paint(Graphics g) {
        String value = "折线";
        int[] x = { 30, 60, 60, 20 };// 声明表示折线横坐标的数组
        int[] y = { 30, 70, 150, 80 };// 声明表示折线纵坐标的数组
        int num1 = x.length;// 提取x,y坐标对数组的长度
        g.setColor(Color.blue);// 设置颜色
        g.drawPolyline(x, y, num1);// 绘制折线
        int[] x0 = { 80, 110, 65, 80, 200 };// 声明表示折线横坐标的数组
        int[] y0 = { 30, 70, 100, 120, 150 };// 声明表示折线纵坐标的数组
        int num2 = x0.length;// 提取x0,y0坐标对数组的长度
        g.drawPolyline(x0, y0, num2);// 绘制折线
        g.drawString(value, 160, 130);// 绘制文本
    }
}
