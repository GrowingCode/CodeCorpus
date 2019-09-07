package com.zzk;

import java.applet.Applet;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.geom.Area;
import java.awt.geom.Ellipse2D;
import java.awt.geom.Rectangle2D;

public class IntersectOperationApplet extends Applet {
    public void paint(Graphics g) {
        Graphics2D g2d = (Graphics2D) g;// 强制转换为Graphics2D对象
        Rectangle2D.Float rect1 = new Rectangle2D.Float(0, 40, 140, 140);// 创建矩形对象
        Ellipse2D.Float ellipse1 = new Ellipse2D.Float(80, 40, 140, 140);// 创建椭圆对象
        Area area1 = new Area(rect1);// 创建矩形区域
        Area area2 = new Area(ellipse1);// 创建椭圆区域
        area1.intersect(area2);// 两个区域相交
        g2d.fill(area1);// 绘制相交后的区域图形
        Rectangle2D.Float rect2 = new Rectangle2D.Float(240, 0, 160, 160);// 创建矩形对象
        Ellipse2D.Float ellipse2 = new Ellipse2D.Float(190, 60, 140, 140);// 创建椭圆对象
        Area area3 = new Area(rect2);// 创建矩形区域
        Area area4 = new Area(ellipse2);// 创建椭圆区域
        area3.intersect(area4);// 两个区域相交
        g2d.fill(area3);// 绘制相交后的区域图形
    }
}
