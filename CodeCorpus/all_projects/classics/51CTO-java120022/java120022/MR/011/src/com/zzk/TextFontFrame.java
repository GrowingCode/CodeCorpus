package com.zzk;

import java.awt.Font;
import java.awt.Graphics;
import javax.swing.JFrame;
import javax.swing.JPanel;
public class TextFontFrame extends JFrame {
    ChangeTextFontPanel changeTextFontPanel = new ChangeTextFontPanel(); // 创建面板类的实例
    
    public static void main(String args[]) { // 主方法
        TextFontFrame frame = new TextFontFrame(); // 创建窗体类的实例
        frame.setVisible(true); // 显示窗体
    }
    public TextFontFrame() {
        super(); // 调用超类的构造方法
        setTitle("设置文本的字体"); // 窗体标题
        setBounds(100, 100, 333, 199); // 窗体的显示位置和大小
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE); // 窗体关闭方式
        add(changeTextFontPanel); // 将面板类的实例添加到窗体容器中
    }
    class ChangeTextFontPanel extends JPanel { // 创建内部面板类
        public void paint(Graphics g) { // 重写paint()方法
            String value = "明日编程词典社区";
            int x = 40; // 文本位置的横坐标
            int y = 50; // 文本位置的纵坐标
            Font font = new Font("华文行楷", Font.BOLD + Font.ITALIC, 26); // 创建字体对象
            g.setFont(font); // 设置字体
            g.drawString(value, x, y); // 绘制文本
            value = "http://community.mrbccd.com";
            x = 10; // 文本位置的横坐标
            y = 100; // 文本位置的纵坐标
            font = new Font("宋体", Font.BOLD, 20); // 创建字体对象
            g.setFont(font); // 设置字体
            g.drawString(value, x, y); // 绘制文本
        }
    }
}
