package com.zzk;

import java.awt.Graphics;
import javax.swing.JFrame;
import javax.swing.JPanel;

public class DrawRectangleFrame extends JFrame {
    DrawRectanglePanel rectPanel = new DrawRectanglePanel(); // 创建面板类的实例
    
    public static void main(String args[]) { // 主方法
        DrawRectangleFrame frame = new DrawRectangleFrame(); // 创建窗体类的实例
        frame.setVisible(true); // 显示窗体
    }
    
    public DrawRectangleFrame() {
        super(); // 调用超类的构造方法
        setTitle("绘制矩形"); // 窗体标题
        setBounds(100, 100, 269, 184); // 窗体的显示位置和大小
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE); // 窗体关闭方式
        add(rectPanel); // 将面板类的实例添加到窗体容器中
    }
    
    class DrawRectanglePanel extends JPanel { // 创建内部面板类
        public void paint(Graphics g) {       // 重写paint()方法
            g.drawRect(30, 40, 80, 60);       // 绘制空心矩形
            g.fillRect(140, 40, 80, 60);      // 绘制实心矩形
        }
    }
}
