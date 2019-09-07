package com.zzk;

import java.awt.Color;
import java.awt.Graphics;
import javax.swing.JFrame;
import javax.swing.JPanel;

public class TextAndShapeColorFrame extends JFrame {
    TextAndShapeColorPanel colorPanel = new TextAndShapeColorPanel(); // 创建面板类的实例
    
    public static void main(String args[]) { // 主方法
        TextAndShapeColorFrame frame = new TextAndShapeColorFrame(); // 创建窗体类的实例
        frame.setVisible(true); // 显示窗体
    }
    
    public TextAndShapeColorFrame() {
        super(); // 调用超类的构造方法
        setTitle("设置文本和图形的颜色"); // 窗体标题
        setBounds(100, 100, 300, 193); // 窗体的显示位置和大小
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE); // 窗体关闭方式
        add(colorPanel); // 将面板类的实例添加到窗体容器中
    }
    
    class TextAndShapeColorPanel extends JPanel { // 创建内部面板类
        public void paint(Graphics g) { // 重写paint()方法
            String value = "只要努力————";
            int x = 60;    // 文本位置的横坐标
            int y = 60;     // 文本位置的纵坐标
            Color color = new Color(255,0,0); // 创建颜色对象
            g.setColor(color); // 设置颜色
            g.drawString(value, x, y);   // 绘制文本
            value = "一切皆有可能";
            x = 140;    // 文本位置的横坐标
            y = 100;     // 文本位置的纵坐标
            color = new Color(0,0,255);// 创建颜色对象
            g.setColor(color);// 设置颜色
            g.drawString(value, x, y);   // 绘制文本
            color = Color.ORANGE; // 通过Color类的字段获得颜色对象
            g.setColor(color);// 设置颜色
            g.drawRoundRect(40,30,200,100,40,30);// 绘制圆角矩形
            g.drawRoundRect(45,35,190,90,36,26);// 绘制圆角矩形
        }
    }
}
