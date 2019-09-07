package com.zzk;

import java.applet.Applet;
import java.awt.Color;
import java.awt.Graphics;
import java.net.URL;

public class GainAddressBarPathApplet extends Applet {
    URL url;// 声明URL对象
    public void start() {
        url = this.getDocumentBase();// 获得地址栏上路径的URL对象
    }
    public void paint(Graphics g) {
        g.setColor(Color.blue);// 设置颜色
        g.drawString(url.getFile(), 30, 20);// 绘制地址栏上的路径
    }
}
