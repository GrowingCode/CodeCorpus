package com.zzk;

import java.applet.Applet;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.geom.Area;
import java.awt.geom.Ellipse2D;

public class ExclusiveOrOperationApplet extends Applet {
    public void paint(Graphics g) { // 重写paint()方法
        Graphics2D g2d = (Graphics2D) g; // 强制转换为Graphics2D对象
        Ellipse2D.Float ellipse1 = new Ellipse2D.Float(30, 80, 180, 80);// 创建椭圆对象
        Ellipse2D.Float ellipse2 = new Ellipse2D.Float(80, 30, 80, 180);// 创建椭圆对象
        Area area1 = new Area(ellipse1);// 创建椭圆区域
        Area area2 = new Area(ellipse2);// 创建椭圆区域
        area1.exclusiveOr(area2);// 两个区域图形进行异或运算
        g2d.fill(area1);// 绘制异或运算后的区域图形
        Ellipse2D.Float ellipse3 = new Ellipse2D.Float(270, 80, 180, 80);// 创建椭圆对象
        Ellipse2D.Float ellipse4 = new Ellipse2D.Float(270, 30, 80, 180);// 创建椭圆对象
        Area area3 = new Area(ellipse3);// 创建椭圆区域
        Area area4 = new Area(ellipse4);// 创建椭圆区域
        area3.exclusiveOr(area4);// 绘制异或运算后的区域图形
        g2d.fill(area3);// 绘制异或运算后的区域图形
    }
}
