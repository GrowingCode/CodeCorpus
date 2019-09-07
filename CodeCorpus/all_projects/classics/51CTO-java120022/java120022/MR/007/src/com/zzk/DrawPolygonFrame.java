package com.zzk;

import java.awt.Graphics;
import javax.swing.JFrame;
import javax.swing.JPanel;

public class DrawPolygonFrame extends JFrame {
    DrawPolygonPanel polygonPanel = new DrawPolygonPanel(); // 创建面板类的实例
    
    public static void main(String args[]) { // 主方法
        DrawPolygonFrame frame = new DrawPolygonFrame(); // 创建窗体类的实例
        frame.setVisible(true); // 显示窗体
    }
    
    public DrawPolygonFrame() {
        super(); // 调用超类的构造方法
        setTitle("绘制多边形"); // 窗体标题
        setBounds(100, 100, 352, 259); // 窗体的显示位置和大小
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE); // 窗体关闭方式
        add(polygonPanel); // 将面板类的实例添加到窗体容器中
    }
    
    class DrawPolygonPanel extends JPanel { // 创建内部面板类
        public void paint(Graphics g) { // 重写paint()方法
            int[] x1 = { 100,120,180,140,150,100,50,60,20,80 }; // 多边形的横坐标
            int[] y1 = { 20,85,90,120,180,140,180,120,90,85 }; // 多边形的纵坐标
            int n1 = 10;// 多边形的边数
            g.fillPolygon(x1, y1, n1);// 绘制多边形
            int[] x2 = { 210, 270, 310, 270, 210, 170 }; // 多边形的横坐标
            int[] y2 = { 20, 20, 65, 110, 110, 65 }; // 多边形的纵坐标
            int n2 = 6;// 多边形的边数
            g.fillPolygon(x2, y2, n2);// 绘制实心多边形
            int[] x3 = { 180, 220, 260, 240, 260, 220, 180, 200 }; // 多边形的横坐标
            int[] y3 = { 120, 140, 120, 160, 200, 180, 200, 160 }; // 多边形的纵坐标
            int n3 = 8;// 多边形的边数
            g.drawPolygon(x3, y3, n3);// 绘制多边形
        }
    }
}
