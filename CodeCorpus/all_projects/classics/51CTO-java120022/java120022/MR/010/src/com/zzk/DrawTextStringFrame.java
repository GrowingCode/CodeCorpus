package com.zzk;

import java.awt.Graphics;
import javax.swing.JFrame;
import javax.swing.JPanel;

public class DrawTextStringFrame extends JFrame {
    DrawTextStringPanel textStringPanel = new DrawTextStringPanel(); // 创建面板类的实例
    
    public static void main(String args[]) { // 主方法
        DrawTextStringFrame frame = new DrawTextStringFrame(); // 创建窗体类的实例
        frame.setVisible(true); // 显示窗体
    }
    
    public DrawTextStringFrame() {
        super(); // 调用超类的构造方法
        setTitle("绘制文本"); // 窗体标题
        setBounds(100, 100, 308, 186); // 窗体的显示位置和大小
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE); // 窗体关闭方式
        add(textStringPanel); // 将面板类的实例添加到窗体容器中
    }
    
    class DrawTextStringPanel extends JPanel { // 创建内部面板类
        public void paint(Graphics g) { // 重写paint()方法
            String value = "静夜思";
            int x = 120;  // 文本位置的横坐标
            int y = 30;  // 文本位置的纵坐标
            g.drawString(value, x, y);   // 绘制文本
            value = "李白";
            x = 170;  // 文本位置的横坐标
            y = 50;  // 文本位置的纵坐标
            g.drawString(value, x, y);   // 绘制文本
            value = "床前明月光，";
            x = 70;  // 文本位置的横坐标
            y = 80;  // 文本位置的纵坐标
            g.drawString(value, x, y);   // 绘制文本
            value = "疑是地上霜。";
            x = 145;  // 文本位置的横坐标
            y = 80;  // 文本位置的纵坐标
            g.drawString(value, x, y);   // 绘制文本
            value = "举头望明月，";
            x = 70;  // 文本位置的横坐标
            y = 110;  // 文本位置的纵坐标
            g.drawString(value, x, y);   // 绘制文本
            value = "低头思故乡。";
            x = 145;  // 文本位置的横坐标
            y = 110;  // 文本位置的纵坐标
            g.drawString(value, x, y);   // 绘制文本
        }
    }
}
