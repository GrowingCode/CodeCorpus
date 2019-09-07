package com.zzk;

import java.applet.Applet;
import java.awt.Font;
import java.awt.Graphics;

public class SetFontSizeApplet extends Applet {
    public void paint(Graphics g) {
        Font font = new Font("宋体", Font.PLAIN, 16);// 创建字体对象，字体大小为16
        g.setFont(font);// 设置字体
        g.drawString("字体大小为16", 30, 20);// 绘制文本
        font = new Font("宋体", Font.PLAIN, 22);// 创建字体对象，字体大小为22
        g.setFont(font);// 设置字体
        g.drawString("字体大小为22", 30, 60);// 绘制文本
        font = new Font("宋体", Font.PLAIN, 36);// 创建字体对象，字体大小为36
        g.setFont(font);// 设置字体
        g.drawString("字体大小为36", 30, 120);// 绘制文本
    }
}
