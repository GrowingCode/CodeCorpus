package com.zzk;

import java.awt.Graphics;
import javax.swing.JFrame;
import javax.swing.JPanel;

public class DrawSquareFrame extends JFrame {
    DrawSquarePanel squarePanel = new DrawSquarePanel(); // 创建面板类的实例
    
    public static void main(String args[]) { // 主方法
        DrawSquareFrame frame = new DrawSquareFrame(); // 创建窗体类的实例
        frame.setVisible(true); // 显示窗体
    }
    
    public DrawSquareFrame() {
        super(); // 调用超类的构造方法
        setTitle("绘制正方形"); // 窗体标题
        setBounds(100, 100, 280, 180); // 窗体的显示位置和大小
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE); // 窗体关闭方式
        add(squarePanel); // 将面板类的实例添加到窗体容器中
    }
    
    class DrawSquarePanel extends JPanel {// 创建内部面板类
        public void paint(Graphics g) {   // 重写paint()方法
            g.drawRect(20, 20, 100, 100); // 绘制空心正方形
            g.drawRect(40, 40, 60, 60);   // 绘制空心正方形
            g.drawRect(140, 20, 100, 100);   // 绘制空心正方形
            g.fillRect(160, 40, 60, 60);  // 绘制实心正方形
        }
    }
}
