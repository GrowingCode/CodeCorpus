package com.zzk;

import java.applet.Applet;
import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics;

public class ShadowTextApplet extends Applet {
    public void paint(Graphics g) { // 重写paint()方法
        String value = "阴影文字";
        int x = 10; // 文本位置的横坐标
        int y = 120; // 文本位置的纵坐标
        Font font = new Font("华文行楷", Font.BOLD, 80); // 创建字体对象
        g.setFont(font); // 设置字体
        g.setColor(Color.GRAY);// 设置颜色为灰色
        g.drawString(value, x, y); // 绘制文本
        x += 3;// 调整绘制点的横坐标值
        y += 3;// 调整绘制点的纵坐标值
        g.setColor(Color.BLACK);// 设置颜色黑色
        g.drawString(value, x, y); // 绘制文本
    }
}
