package com.zzk;

import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.geom.QuadCurve2D;

import javax.swing.JFrame;
import javax.swing.JPanel;

public class DrawQuadCurveFrame extends JFrame {
    DrawQuadCurvePanel quadCurvePanel = new DrawQuadCurvePanel(); // 创建面板类的实例
    
    public static void main(String args[]) { // 主方法
        DrawQuadCurveFrame frame = new DrawQuadCurveFrame(); // 创建窗体类的实例
        frame.setVisible(true); // 显示窗体
    }
    
    public DrawQuadCurveFrame() {
        super(); // 调用超类的构造方法
        setTitle("绘制二次曲线"); // 窗体标题
        setBounds(100, 100, 269, 175); // 窗体的显示位置和大小
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE); // 窗体关闭方式
        add(quadCurvePanel); // 将面板类的实例添加到窗体容器中
    }
    
    class DrawQuadCurvePanel extends JPanel { // 创建内部面板类
        public void paint(Graphics g) { // 重写paint()方法
            Graphics2D g2=(Graphics2D)g;// 获得Graphics2D对象
            // 创建二次曲线，其中点120,100是控制点，点60,20是起始点坐标，点180,20是终点坐标
            QuadCurve2D.Double quadCurve1 = new QuadCurve2D.Double(60,20,120,100,180,20);
            g2.draw(quadCurve1); // 绘制二次曲线
            // 创建二次曲线，其中点120,40是控制点，点60,120是起始点坐标，点180,120是终点坐标
            QuadCurve2D.Double quadCurve2 = new QuadCurve2D.Double(60,120,120,40,180,120);
            g2.draw(quadCurve2); // 绘制二次曲线
        }
    }
}
