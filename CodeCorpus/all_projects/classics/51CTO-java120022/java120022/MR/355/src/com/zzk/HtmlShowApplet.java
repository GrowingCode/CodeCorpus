package com.zzk;

import java.applet.Applet;
import java.awt.Graphics;
public class HtmlShowApplet extends Applet {
    public void paint(Graphics g){
        g.drawString("html文件已经运行", 50, 50);// 绘制文本
        g.drawString("在html中显示了Applet程序", 50, 80);// 绘制文本
    }
}
