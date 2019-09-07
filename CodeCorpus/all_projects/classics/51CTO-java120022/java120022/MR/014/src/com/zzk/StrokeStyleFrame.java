package com.zzk;

import java.awt.BasicStroke;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.geom.Line2D;
import javax.swing.JFrame;
import javax.swing.JPanel;

public class StrokeStyleFrame extends JFrame {
    ChangeStrokeStylePanel strokeStylePanel = new ChangeStrokeStylePanel(); // 创建面板类的实例
    
    public static void main(String args[]) { // 主方法
        StrokeStyleFrame frame = new StrokeStyleFrame(); // 创建窗体类的实例
        frame.setVisible(true); // 显示窗体
    }
    
    public StrokeStyleFrame() {
        super(); // 调用超类的构造方法
        setTitle("设置笔画样式"); // 窗体标题
        setBounds(100, 100, 306, 220); // 窗体的显示位置和大小
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE); // 窗体关闭方式
        add(strokeStylePanel); // 将面板类的实例添加到窗体容器中
    }
    
    class ChangeStrokeStylePanel extends JPanel { // 创建内部面板类
        public void paint(Graphics g) { // 重写paint()方法
            Graphics2D g2 = (Graphics2D)g; // 获得Graphics2D对象
            BasicStroke stroke = new BasicStroke(10,BasicStroke.CAP_BUTT,BasicStroke.JOIN_BEVEL); // 创建宽度是10的平头笔画对象
            g2.setStroke(stroke);// 设置笔画对象
            Line2D.Float line = new Line2D.Float(50,50,240,50);// 创建直线对象
            g2.drawString("平头样式", 120, 40);  // 绘制文本
            g2.draw(line);// 绘制直线
            stroke = new BasicStroke(10,BasicStroke.CAP_ROUND,BasicStroke.JOIN_BEVEL); // 创建宽度是10的圆头笔画对象
            g2.setStroke(stroke);// 设置笔画对象
            line = new Line2D.Float(50,90,240,90);// 创建直线对象
            g2.drawString("圆头样式", 120, 80);  // 绘制文本
            g2.draw(line);// 绘制直线
            stroke = new BasicStroke(10,BasicStroke.CAP_SQUARE,BasicStroke.JOIN_BEVEL); // 创建宽度是10的方头笔画对象
            g2.setStroke(stroke);// 设置笔画对象
            line = new Line2D.Float(50,130,240,130);// 创建直线对象
            g2.drawString("方头样式", 120, 120);  // 绘制文本
            g2.draw(line);// 绘制直线
        }
    }
}
