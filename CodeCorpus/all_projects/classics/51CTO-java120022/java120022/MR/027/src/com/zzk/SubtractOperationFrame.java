package com.zzk;

import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.geom.Area;
import java.awt.geom.Ellipse2D;
import javax.swing.JFrame;
import javax.swing.JPanel;

public class SubtractOperationFrame extends JFrame {
    SubtractOperationPanel subtractOperationPanel = new SubtractOperationPanel(); // 创建面板类的实例
    public static void main(String args[]) { // 主方法
        SubtractOperationFrame frame = new SubtractOperationFrame(); // 创建窗体类的实例
        frame.setVisible(true); // 显示窗体
    }
    
    public SubtractOperationFrame() {
        super(); // 调用超类的构造方法
        setTitle("图形的减运算"); // 窗体标题
        setBounds(100, 100, 395, 240); // 窗体的显示位置和大小
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE); // 窗体关闭方式
        add(subtractOperationPanel); // 将面板类的实例添加到窗体容器中
    }
    
    class SubtractOperationPanel extends JPanel { // 创建内部面板类
        public void paint(Graphics g) {    // 重写paint()方法
            Graphics2D g2 = (Graphics2D)g; // 获得Graphics2D对象
            Ellipse2D.Float ellipse1 = new Ellipse2D.Float(20, 20, 160, 160);// 创建圆对象
            Ellipse2D.Float ellipse2 = new Ellipse2D.Float(90, 20, 160, 160);// 创建圆对象
            Area area1 = new Area(ellipse1);   // 创建区域圆
            Area area2 = new Area(ellipse2);   // 创建区域圆
            area1.subtract(area2);// 两个区域圆进行减运算
            g2.fill(area1);  // 绘制减运算后的区域圆
            Ellipse2D.Float ellipse3 = new Ellipse2D.Float(200, 70, 160, 60);// 创建椭圆对象
            Ellipse2D.Float ellipse4 = new Ellipse2D.Float(250, 40, 60, 60);// 创建圆对象
            Area area3 = new Area(ellipse3);// 创建区域椭圆
            Area area4 = new Area(ellipse4);// 创建区域圆
            area3.subtract(area4);// 两个区域图形进行减运算
            g2.draw(area3);  // 绘制减运算后的区域图形
        }
    }
}
