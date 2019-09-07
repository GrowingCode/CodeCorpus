package com.zzk;

import java.awt.Color;
import java.awt.GradientPaint;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.geom.Rectangle2D;
import javax.swing.JFrame;
import javax.swing.JPanel;

public class FillGradientFrame extends JFrame {
    FillGradientPanel fillGradientPanel = new FillGradientPanel(); // 创建面板类的实例
    public static void main(String args[]) { // 主方法
        FillGradientFrame frame = new FillGradientFrame(); // 创建窗体类的实例
        frame.setVisible(true); // 显示窗体
    }
    
    public FillGradientFrame() {
        super(); // 调用超类的构造方法
        setTitle("为图形填充渐变色"); // 窗体标题
        setBounds(100, 100, 338, 220); // 窗体的显示位置和大小
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE); // 窗体关闭方式
        add(fillGradientPanel); // 将面板类的实例添加到窗体容器中
    }
    
    class FillGradientPanel extends JPanel { // 创建内部面板类
        public void paint(Graphics g) { // 重写paint()方法
            Graphics2D g2 = (Graphics2D) g; // 获得Graphics2D对象
            Rectangle2D.Float rect = new Rectangle2D.Float(20, 20, 280, 140);// 创建矩形对象
            // 创建循环渐变的GraphientPaint对象
            GradientPaint paint = new GradientPaint(20,20,Color.BLUE,100,80,Color.RED,true);
            g2.setPaint(paint);// 设置渐变
            g2.fill(rect);// 绘制矩形
        }
    }
}
