package com.zzk;

import java.awt.Graphics;
import javax.swing.JFrame;
import javax.swing.JPanel;

public class DrawEllipseFrame extends JFrame {
    DrawEllipsePanel ellipsePanel = new DrawEllipsePanel(); // 创建面板类的实例
    
    public static void main(String args[]) { // 主方法
        DrawEllipseFrame frame = new DrawEllipseFrame(); // 创建窗体类的实例
        frame.setVisible(true); // 显示窗体
    }
    
    public DrawEllipseFrame() {
        super(); // 调用超类的构造方法
        setTitle("绘制椭圆"); // 窗体标题
        setBounds(100, 100, 269, 222); // 窗体的显示位置和大小
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE); // 窗体关闭方式
        add(ellipsePanel); // 将面板类的实例添加到窗体容器中
    }
    
    class DrawEllipsePanel extends JPanel { // 创建内部面板类
        public void paint(Graphics g) {     // 重写paint()方法
            g.drawOval(30, 20, 80, 50);     // 绘制空心椭圆
            g.drawOval(150, 10, 50, 80);    // 绘制空心椭圆
            g.fillOval(40, 90, 50, 80);     // 绘制实心椭圆
            g.fillOval(140, 110, 80, 50);   // 绘制实心椭圆
        }
    }
}
