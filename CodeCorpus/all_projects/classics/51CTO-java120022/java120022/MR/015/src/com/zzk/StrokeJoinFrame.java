package com.zzk;

import java.awt.BasicStroke;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.geom.Rectangle2D;
import javax.swing.JFrame;
import javax.swing.JPanel;

public class StrokeJoinFrame extends JFrame {
    ChangeStrokeJoinPanel strokeJoinPanel = new ChangeStrokeJoinPanel(); // 创建面板类的实例
    
    public static void main(String args[]) { // 主方法
        StrokeJoinFrame frame = new StrokeJoinFrame(); // 创建窗体类的实例
        frame.setVisible(true); // 显示窗体
    }
    
    public StrokeJoinFrame() {
        super(); // 调用超类的构造方法
        setTitle("设置连接方式"); // 窗体标题
        setBounds(100, 100, 338, 198); // 窗体的显示位置和大小
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE); // 窗体关闭方式
        add(strokeJoinPanel); // 将面板类的实例添加到窗体容器中
    }
    
    class ChangeStrokeJoinPanel extends JPanel { // 创建内部面板类
        public void paint(Graphics g) { // 重写paint()方法
            Graphics2D g2 = (Graphics2D)g; // 获得Graphics2D对象
            BasicStroke stroke = new BasicStroke(10,BasicStroke.CAP_BUTT,BasicStroke.JOIN_BEVEL); // 创建宽度是10的平头斜角连接笔画对象
            g2.setStroke(stroke);// 设置笔画对象
            Rectangle2D.Float rect = new Rectangle2D.Float(20,60,80,50);// 创建矩形对象
            g2.drawString("斜角连接", 35, 50);  // 绘制文本
            g2.draw(rect);// 绘制矩形
            stroke = new BasicStroke(10,BasicStroke.CAP_BUTT,BasicStroke.JOIN_MITER); // 创建宽度是10的平头尖角连接笔画对象
            g2.setStroke(stroke);// 设置笔画对象
            rect = new Rectangle2D.Float(120,60,80,50);// 创建矩形对象
            g2.drawString("尖角连接", 135, 50);  // 绘制文本
            g2.draw(rect);// 绘制矩形
            stroke = new BasicStroke(10,BasicStroke.CAP_BUTT,BasicStroke.JOIN_ROUND); // 创建宽度是10的平头圆角连接笔画对象
            g2.setStroke(stroke);// 设置笔画对象
            rect = new Rectangle2D.Float(220,60,80,50);// 创建矩形对象
            g2.drawString("圆角连接", 235, 50);  // 绘制文本
            g2.draw(rect);// 绘制矩形
        }
    }
}
