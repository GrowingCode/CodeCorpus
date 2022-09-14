package com.zzk;

import java.applet.Applet;
import java.awt.Color;
import java.awt.Font;
import java.awt.GradientPaint;
import java.awt.Graphics;
import java.awt.Graphics2D;

public class GradientTextApplet extends Applet {
    public void paint(Graphics g) { // 重写paint()方法
        Graphics2D g2 = (Graphics2D) g;// 转换为Graphics2D类型
        String value = "渐变色文字";// 绘制的文本
        int x = 15; // 文本位置的横坐标
        int y = 120; // 文本位置的纵坐标
        Font font = new Font("楷体", Font.BOLD, 60); // 创建字体对象
        g2.setFont(font); // 设置字体
        GradientPaint paint = new GradientPaint(20, 20, Color.BLUE, 100, 120,
                Color.RED, true);// 创建循环渐变的GradientPaint对象
        g2.setPaint(paint);// 设置渐变
        g2.drawString(value, x, y); // 绘制文本
    }
}
