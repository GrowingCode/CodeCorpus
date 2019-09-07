package com.zzk;

import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.geom.Area;
import java.awt.geom.Ellipse2D;
import javax.swing.JFrame;
import javax.swing.JPanel;

public class PlusOperationFrame extends JFrame {
    PlusOperationPanel plusOperationPanel = new PlusOperationPanel(); // 创建面板类的实例
    public static void main(String args[]) { // 主方法
        PlusOperationFrame frame = new PlusOperationFrame(); // 创建窗体类的实例
        frame.setVisible(true); // 显示窗体
    }
    
    public PlusOperationFrame() {
        super(); // 调用超类的构造方法
        setTitle("图形的加运算"); // 窗体标题
        setBounds(100, 100, 395, 240); // 窗体的显示位置和大小
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE); // 窗体关闭方式
        add(plusOperationPanel); // 将面板类的实例添加到窗体容器中
    }
    
    class PlusOperationPanel extends JPanel { // 创建内部面板类
        public void paint(Graphics g) {    // 重写paint()方法
            Graphics2D g2 = (Graphics2D)g; // 获得Graphics2D对象
            Ellipse2D.Float ellipse1 = new Ellipse2D.Float(20, 70, 160, 60);// 创建椭圆对象
            Ellipse2D.Float ellipse2 = new Ellipse2D.Float(120, 20, 60, 160);// 创建椭圆对象
            Area area1 = new Area(ellipse1);   // 创建区域椭圆
            Area area2 = new Area(ellipse2);   // 创建区域椭圆
            area1.add(area2);// 两个区域椭圆进行加运算
            g2.draw(area1);  // 绘制加运算后的区域椭圆
            Ellipse2D.Float ellipse3 = new Ellipse2D.Float(200, 70, 160, 60);// 创建椭圆对象
            Ellipse2D.Float ellipse4 = new Ellipse2D.Float(250, 20, 60, 160);// 创建椭圆对象
            Area area3 = new Area(ellipse3);// 创建区域椭圆
            Area area4 = new Area(ellipse4);// 创建区域椭圆
            area3.add(area4);// 两个区域椭圆进行加运算
            g2.draw(area3);  // 绘制加运算后的区域椭圆
        }
    }
}
