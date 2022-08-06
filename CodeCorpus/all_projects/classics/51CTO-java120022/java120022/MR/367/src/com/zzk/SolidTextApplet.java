package com.zzk;

import java.applet.Applet;
import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics;

public class SolidTextApplet extends Applet {
    public void paint(Graphics g) { // 重写paint()方法
        String value = "立体文字效果";
        int x = 10; // 文本位置的横坐标
        int y = 120; // 文本位置的纵坐标
        Font font = new Font("宋体", Font.BOLD, 72); // 创建字体对象
        g.setFont(font); // 设置字体
        g.setColor(Color.GRAY);// 设置颜色为灰色
        int i = 0;// 循环变量
        while (i < 8) {
            g.drawString(value, x, y); // 绘制文本
            x += 1;// 调整绘制点的横坐标值
            y += 1;// 调整绘制点的纵坐标值
            i++;// 调整循环变量的值
        }
        g.setColor(Color.BLACK);// 设置颜色黑色
        g.drawString(value, x, y); // 绘制文本
    }
}
