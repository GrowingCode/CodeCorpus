package com.zzk;

import java.applet.Applet;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.geom.Area;
import java.awt.geom.Rectangle2D;

public class AddOperationApplet extends Applet {
    public void paint(Graphics g) {
        String value = "图形相加运算";
        Graphics2D g2d = (Graphics2D) g;// p转换为可用的Graphics2D对象
        Rectangle2D.Float rect1 = new Rectangle2D.Float(20, 70, 185, 60);// 创建矩形对象
        Rectangle2D.Float rect2 = new Rectangle2D.Float(120, 20, 65, 160);// 创建矩形对象
        Area area1 = new Area(rect1);// 创建区域矩形
        Area area2 = new Area(rect2);// 创建区域矩形
        area1.add(area2);// 两个区域进行相加
        g2d.draw(area1);// 绘制相加后的区域矩形
        Rectangle2D.Float rect3 = new Rectangle2D.Float(230, 70, 185, 60);// 创建矩形对象
        Rectangle2D.Float rect4 = new Rectangle2D.Float(290, 20, 65, 160);// 创建矩形对象
        Area area3 = new Area(rect3);// 创建区域矩形
        Area area4 = new Area(rect4);// 创建区域矩形
        area3.add(area4);// 两个区域进行相加
        g2d.draw(area3);// 绘制相加后的区域矩形
        g2d.drawString(value, 25, 56);// 绘制文本
    }
}
