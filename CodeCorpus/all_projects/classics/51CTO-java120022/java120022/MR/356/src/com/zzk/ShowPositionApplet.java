package com.zzk;

import java.applet.Applet;
import java.awt.Color;
import java.awt.Graphics;
public class ShowPositionApplet extends Applet {
    String author;// 声明成员变量
    public void init() {
        author = "ZhenKun Zhang";// 初始化成员变量
    }
    public void paint(Graphics g){
        g.setColor(Color.blue);// 设置颜色
        g.drawString(author, 50, 30);// 绘制文本
    }
}
