package com.zzk;

import java.awt.Graphics;
import javax.swing.JFrame;
import javax.swing.JPanel;

public class DrawLineFrame extends JFrame {
    DrawLinePanel linePanel = new DrawLinePanel(); // 创建面板类的实例
    
    public static void main(String args[]) { // 主方法
        DrawLineFrame frame = new DrawLineFrame(); // 创建窗体类的实例
        frame.setVisible(true); // 显示窗体
    }
    
    public DrawLineFrame() {
        super(); // 调用超类的构造方法
        setTitle("绘制直线"); // 窗体标题
        setBounds(100, 100, 273, 167); // 窗体的显示位置和大小
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE); // 窗体关闭方式
        add(linePanel); // 将面板类的实例添加到窗体容器中
    }
    
    class DrawLinePanel extends JPanel {   // 创建内部面板类
        public void paint(Graphics g) {    // 重写paint()方法
            g.drawLine(70, 50, 180, 50);   // 绘制第一条水平线
            g.drawLine(70, 80, 180, 80);   // 绘制第二条水平线
            g.drawLine(110, 10, 140, 120); // 绘制斜线
        }
    }
}
