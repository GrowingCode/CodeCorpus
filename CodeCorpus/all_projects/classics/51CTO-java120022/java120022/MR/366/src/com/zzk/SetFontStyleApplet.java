package com.zzk;

import java.applet.Applet;
import java.awt.Font;
import java.awt.Graphics;

public class SetFontStyleApplet extends Applet {
    public void paint(Graphics g) {
        Font font = new Font("宋体", Font.PLAIN, 16);// 创建字体对象，字体样式为普通字体
        g.setFont(font);// 设置字体
        g.drawString("普通字体", 30, 20);// 绘制文本
        font = new Font("宋体", Font.ITALIC, 22);// 创建字体对象，字体样式为倾斜字体
        g.setFont(font);// 设置字体
        g.drawString("倾斜字体", 30, 60);// 绘制文本
        font = new Font("宋体", Font.BOLD, 28);// 创建字体对象，字体样式为加粗字体
        g.setFont(font);// 设置字体
        g.drawString("加粗字体", 30, 120);// 绘制文本
    }
}
