package com.zzk;

import java.awt.Color;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.geom.Ellipse2D;
import javax.swing.JFrame;
import javax.swing.JPanel;

public class DrawFlowerFrame extends JFrame {
    DrawFlowerPanel drawFlowerPanel = new DrawFlowerPanel(); // 创建面板类的实例
    public static void main(String args[]) { // 主方法
        DrawFlowerFrame frame = new DrawFlowerFrame(); // 创建窗体类的实例
        frame.setVisible(true); // 显示窗体
    }
    
    public DrawFlowerFrame() {
        super(); // 调用超类的构造方法
        setTitle("绘制花瓣"); // 窗体标题
        setBounds(100, 100, 338, 230); // 窗体的显示位置和大小
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE); // 窗体关闭方式
        add(drawFlowerPanel); // 将面板类的实例添加到窗体容器中
    }
    
    class DrawFlowerPanel extends JPanel { // 创建内部面板类
        public void paint(Graphics g) {     // 重写paint()方法
            Graphics2D g2 = (Graphics2D)g; // 获得Graphics2D对象
            g2.translate(drawFlowerPanel.getWidth() / 2, drawFlowerPanel.getHeight() / 2);// 平移坐标轴
            // 绘制绿色花瓣
            Ellipse2D.Float ellipse = new Ellipse2D.Float(30, 0, 70, 20);// 创建椭圆对象
            Color color = new Color(0,255,0);//创建颜色对象
            g2.setColor(color);//指定颜色
            g2.fill(ellipse);// 绘制椭圆
            int i=0;
            while (i<8){
                g2.rotate(30);// 旋转画布
                g2.fill(ellipse);// 绘制椭圆
                i++;
            }
            // 绘制红色花瓣
            ellipse = new Ellipse2D.Float(20, 0, 60, 15);// 创建椭圆对象
            color = new Color(255,0,0);//创建颜色对象
            g2.setColor(color);//指定颜色
            g2.fill(ellipse);// 绘制椭圆
            i=0;
            while (i<15){
                g2.rotate(75);// 旋转画布
                g2.fill(ellipse);// 绘制椭圆
                i++;
            }
            // 绘制黄色花瓣
            ellipse = new Ellipse2D.Float(10, 0, 50, 15);// 创建椭圆对象
            color = new Color(255,255,0);//创建颜色对象
            g2.setColor(color);//指定颜色
            g2.fill(ellipse);// 绘制椭圆
            i=0;
            while (i<8){
                g2.rotate(30);// 旋转画布
                g2.fill(ellipse);// 绘制椭圆
                i++;
            }
            // 绘制红色中心点
            color = new Color(255, 0, 0);// 创建颜色对象
            g2.setColor(color);// 指定颜色
            ellipse = new Ellipse2D.Float(-10, -10, 20, 20);// 创建椭圆对象
            g2.fill(ellipse);// 绘制椭圆
        }
    }
}
