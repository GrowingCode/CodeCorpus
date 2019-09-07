package com.zzk;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Graphics;
import java.awt.Graphics2D;

import javax.swing.JFrame;
import javax.swing.JPanel;

public class FiveDaisyChainFrame extends JFrame {
    FiveDaisyChainPanel fivePanel = new FiveDaisyChainPanel(); // 创建面板类的实例
    
    public static void main(String args[]) { // 主方法
        FiveDaisyChainFrame frame = new FiveDaisyChainFrame(); // 创建窗体类的实例
        frame.setVisible(true); // 显示窗体
    }
    
    public FiveDaisyChainFrame() {
        super(); // 调用超类的构造方法
        setTitle("绘制五环图案"); // 窗体标题
        setBounds(100, 100, 269, 222); // 窗体的显示位置和大小
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE); // 窗体关闭方式
        add(fivePanel); // 将面板类的实例添加到窗体容器中
    }
    
    class FiveDaisyChainPanel extends JPanel { // 创建内部面板类
        public void paint(Graphics g) {     // 重写paint()方法
            Graphics2D g2 = (Graphics2D)g; // 获得Graphics2D对象
            BasicStroke stroke = new BasicStroke(3); // 创建宽度是3的笔画对象
            g2.setStroke(stroke);// 设置笔画对象
            Color color = new Color(0,162,232);// 创建颜色对象
            g2.setColor(color);// 设置颜色
            g2.drawOval(30, 40, 60, 60);     // 绘制第一个圆
            color = new Color(233,123,16);   // 创建新的颜色对象
            g2.setColor(color);// 设置颜色
            g2.drawOval(60, 70, 60, 60);     // 绘制第二个圆
            color = new Color(28,20,100);// 创建新的颜色对象
            g2.setColor(color);// 设置颜色
            g2.drawOval(92, 40, 60, 60);     // 绘制第三个圆
            color = new Color(0,255,0);// 创建新的颜色对象
            g2.setColor(color);// 设置颜色
            g2.drawOval(122, 70, 60, 60);    // 绘制第四个圆
            color = new Color(255,0,0);// 创建新的颜色对象
            g2.setColor(color);// 设置颜色
            g2.drawOval(154, 40, 60, 60);    // 绘制第五个圆
        }
    }
}
