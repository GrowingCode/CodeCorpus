package com.zzk;

import java.awt.BasicStroke;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.geom.Ellipse2D;
import javax.swing.JFrame;
import javax.swing.JPanel;

public class StrokeWidthFrame extends JFrame {
    ChangeStrokeWidthPanel strokeWidthPanel = new ChangeStrokeWidthPanel(); // 创建面板类的实例
    
    public static void main(String args[]) { // 主方法
        StrokeWidthFrame frame = new StrokeWidthFrame(); // 创建窗体类的实例
        frame.setVisible(true); // 显示窗体
    }
    
    public StrokeWidthFrame() {
        super(); // 调用超类的构造方法
        setTitle("设置笔画的粗细"); // 窗体标题
        setBounds(100, 100, 300, 220); // 窗体的显示位置和大小
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE); // 窗体关闭方式
        add(strokeWidthPanel); // 将面板类的实例添加到窗体容器中
    }
    
    class ChangeStrokeWidthPanel extends JPanel { // 创建内部面板类
        public void paint(Graphics g) { // 重写paint()方法
            Graphics2D g2 = (Graphics2D)g; // 获得Graphics2D对象
            BasicStroke stroke = new BasicStroke(1); // 创建宽度是1的笔画对象
            g2.setStroke(stroke);// 设置笔画对象
            Ellipse2D.Float ellipse = new Ellipse2D.Float(20,20,100,60);// 创建椭圆对象
            g2.draw(ellipse);// 绘制椭圆
            stroke = new BasicStroke(4);// 创建宽度是4的笔画对象
            g2.setStroke(stroke);// 设置笔画对象
            ellipse = new Ellipse2D.Float(160,20,100,60);// 创建椭圆对象
            g2.draw(ellipse);// 绘制椭圆
            stroke = new BasicStroke(6);// 创建宽度是6的笔画对象
            g2.setStroke(stroke);// 设置笔画对象
            ellipse = new Ellipse2D.Float(20,100,100,60);// 创建椭圆对象
            g2.draw(ellipse);// 绘制椭圆
            stroke = new BasicStroke(8);// 创建宽度是8的笔画对象
            g2.setStroke(stroke);// 设置笔画对象
            ellipse = new Ellipse2D.Float(160,100,100,60);// 创建椭圆对象
            g2.draw(ellipse);// 绘制椭圆
        }
    }
}
