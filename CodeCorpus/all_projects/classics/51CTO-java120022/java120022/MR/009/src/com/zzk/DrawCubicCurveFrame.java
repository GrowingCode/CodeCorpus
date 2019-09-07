package com.zzk;

import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.geom.CubicCurve2D;
import javax.swing.JFrame;
import javax.swing.JPanel;

public class DrawCubicCurveFrame extends JFrame {
    DrawCubicCurvePanel cubicCurvePanel = new DrawCubicCurvePanel(); // 创建面板类的实例
    
    public static void main(String args[]) { // 主方法
        DrawCubicCurveFrame frame = new DrawCubicCurveFrame(); // 创建窗体类的实例
        frame.setVisible(true); // 显示窗体
    }
    
    public DrawCubicCurveFrame() {
        super(); // 调用超类的构造方法
        setTitle("绘制三次曲线"); // 窗体标题
        setBounds(100, 100, 297, 199); // 窗体的显示位置和大小
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE); // 窗体关闭方式
        add(cubicCurvePanel); // 将面板类的实例添加到窗体容器中
    }
    
    class DrawCubicCurvePanel extends JPanel { // 创建内部面板类
        public void paint(Graphics g) { // 重写paint()方法
            Graphics2D g2=(Graphics2D)g;// 获得Graphics2D对象
            // 创建三次曲线，其中点140,-140和点140,300是控制点，点20,80是起始点坐标，点260,80是终点坐标
            CubicCurve2D.Double cubicCurve = new CubicCurve2D.Double(20,80,140,-140,140,300,260,80);
            g2.draw(cubicCurve); // 绘制三次曲线
        }
    }
}
