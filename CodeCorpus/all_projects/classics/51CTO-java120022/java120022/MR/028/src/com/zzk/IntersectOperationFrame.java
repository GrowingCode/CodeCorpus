package com.zzk;

import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.geom.Area;
import java.awt.geom.Ellipse2D;
import java.awt.geom.Rectangle2D;
import javax.swing.JFrame;
import javax.swing.JPanel;

public class IntersectOperationFrame extends JFrame {
    IntersectOperationPanel intersectOperationPanel = new IntersectOperationPanel(); // 创建面板类的实例
    public static void main(String args[]) { // 主方法
        IntersectOperationFrame frame = new IntersectOperationFrame(); // 创建窗体类的实例
        frame.setVisible(true); // 显示窗体
    }
    
    public IntersectOperationFrame() {
        super(); // 调用超类的构造方法
        setTitle("图形的交运算"); // 窗体标题
        setBounds(100, 100, 395, 240); // 窗体的显示位置和大小
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE); // 窗体关闭方式
        add(intersectOperationPanel); // 将面板类的实例添加到窗体容器中
    }
    
    class IntersectOperationPanel extends JPanel { // 创建内部面板类
        public void paint(Graphics g) {    // 重写paint()方法
            Graphics2D g2 = (Graphics2D)g; // 获得Graphics2D对象
            Rectangle2D.Float rect = new Rectangle2D.Float(30, 30, 160, 120);// 创建矩形对象
            Ellipse2D.Float ellipse = new Ellipse2D.Float(20, 30, 180, 180);// 创建圆对象
            Area area1 = new Area(rect);   // 创建区域矩形
            Area area2 = new Area(ellipse);   // 创建区域圆
            area1.intersect(area2);// 两个区域图形进行交运算
            g2.draw(area1);  // 绘制交运算后的区域图形
            Ellipse2D.Float ellipse1 = new Ellipse2D.Float(190, 20, 100, 140);// 创建椭圆对象
            Ellipse2D.Float ellipse2 = new Ellipse2D.Float(240, 20, 100, 140);// 创建椭圆对象
            Area area3 = new Area(ellipse1);// 创建区域椭圆
            Area area4 = new Area(ellipse2);// 创建区域椭圆
            area3.intersect(area4);// 两个区域椭圆进行交运算
            g2.fill(area3);  // 绘制交运算后的区域椭圆
        }
    }
}
