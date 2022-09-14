package com.zzk;

import java.applet.Applet;
import java.awt.Font;
import java.awt.Graphics;

public class SetTextPositionApplet extends Applet {
    public void paint(Graphics g) {
        Font font = new Font("宋体", Font.PLAIN, 32);// 创建字体对象
        g.setFont(font);// 设置字体
        g.drawString("标题", 65, 40);// 绘制文本，其位置坐标为(65，40)
        font = new Font("宋体", Font.PLAIN, 16);// 创建字体对象
        g.setFont(font);// 设置字体
        g.drawString("正文一的内容", 50, 80);// 绘制文本，其位置坐标为(50，80)
        g.drawString("正文二的内容", 50, 120);// 绘制文本，其位置坐标为(50，120)
        g.drawString("正文三的内容", 50, 160);// 绘制文本，其位置坐标为(50，160)
    }
}
