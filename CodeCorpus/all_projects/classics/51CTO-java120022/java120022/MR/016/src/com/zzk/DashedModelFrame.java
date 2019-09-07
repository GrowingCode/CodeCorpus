package com.zzk;

import java.awt.BasicStroke;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.geom.Line2D;
import javax.swing.JFrame;
import javax.swing.JPanel;

public class DashedModelFrame extends JFrame {
    DashedModelPanel dashedModelPanel = new DashedModelPanel(); // 创建面板类的实例
    
    public static void main(String args[]) { // 主方法
        DashedModelFrame frame = new DashedModelFrame(); // 创建窗体类的实例
        frame.setVisible(true); // 显示窗体
    }
    
    public DashedModelFrame() {
        super(); // 调用超类的构造方法
        setTitle("设置虚线模式"); // 窗体标题
        setBounds(100, 100, 326, 220); // 窗体的显示位置和大小
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE); // 窗体关闭方式
        add(dashedModelPanel); // 将面板类的实例添加到窗体容器中
    }
    
    class DashedModelPanel extends JPanel { // 创建内部面板类
        public void paint(Graphics g) { // 重写paint()方法
            Graphics2D g2 = (Graphics2D)g; // 获得Graphics2D对象
            float[] arr = {30.0f,30.0f};  // 创建虚线模式的数组
            BasicStroke stroke = new BasicStroke(10,BasicStroke.CAP_BUTT,BasicStroke.JOIN_BEVEL,1.0f,arr,0); // 创建宽度是10的平头虚线笔画对象
            g2.setStroke(stroke);// 设置笔画对象
            Line2D.Float line = new Line2D.Float(20,50,300,50);// 创建直线对象
            g2.drawString("平头样式虚线模式", 110, 40);  // 绘制文本
            g2.draw(line);// 绘制直线
            stroke = new BasicStroke(10,BasicStroke.CAP_ROUND,BasicStroke.JOIN_BEVEL,1.0f,arr,0); // 创建宽度是10的圆头虚线笔画对象
            g2.setStroke(stroke);// 设置笔画对象
            line = new Line2D.Float(20,90,300,90);// 创建直线对象
            g2.drawString("圆头样式虚线模式", 110, 80);  // 绘制文本
            g2.draw(line);// 绘制直线
            stroke = new BasicStroke(10,BasicStroke.CAP_SQUARE,BasicStroke.JOIN_BEVEL,1.0f,arr,0); // 创建宽度是10的方头虚线笔画对象
            g2.setStroke(stroke);// 设置笔画对象
            line = new Line2D.Float(20,130,300,130);// 创建直线对象
            g2.drawString("方头样式虚线模式", 110, 120);  // 绘制文本
            g2.draw(line);// 绘制直线
        }
    }
}
