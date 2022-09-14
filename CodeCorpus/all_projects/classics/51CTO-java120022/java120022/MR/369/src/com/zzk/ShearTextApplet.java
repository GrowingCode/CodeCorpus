package com.zzk;

import java.applet.Applet;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.Graphics2D;

public class ShearTextApplet extends Applet {
    public void paint(Graphics g) { // 重写paint()方法
        Graphics2D g2 = (Graphics2D) g;// 转换为Graphics2D类型
        String value = "倾斜文字";// 绘制的文本
        int x = 10; // 文本位置的横坐标
        int y = 190; // 文本位置的纵坐标
        Font font = new Font("华文行楷", Font.BOLD, 72); // 创建字体对象
        g2.setFont(font); // 设置字体
        g2.shear(0.1, -0.4);// 倾斜画布
        g2.drawString(value, x, y); // 绘制文本
    }
}
