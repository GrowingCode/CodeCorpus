package com.zzk;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.geom.Ellipse2D;
import javax.swing.JFrame;
import javax.swing.JPanel;

public class DrawCachetFrame extends JFrame {
    DrawCachetPanel drawCachetPanel = new DrawCachetPanel(); // 创建面板类的实例
    
    public static void main(String args[]) { // 主方法
        DrawCachetFrame frame = new DrawCachetFrame(); // 创建窗体类的实例
        frame.setVisible(true); // 显示窗体
    }
    
    public DrawCachetFrame() {
        super(); // 调用超类的构造方法
        setTitle("绘制公章"); // 窗体标题
        setBounds(100, 100, 340, 240); // 窗体的显示位置和大小
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE); // 窗体关闭方式
        add(drawCachetPanel); // 将面板类的实例添加到窗体容器中
    }
    
    class DrawCachetPanel extends JPanel { // 创建内部面板类
        public void paint(Graphics g) { // 重写paint()方法
            Graphics2D g2 = (Graphics2D) g; // 获得Graphics2D对象
            g2.translate(170, 100);// 平移坐标轴
            BasicStroke stroke = new BasicStroke(6); // 创建宽度是6的笔画对象
            g2.setStroke(stroke);// 设置笔画对象
            // 绘制圆
            Ellipse2D.Float ellipse = new Ellipse2D.Float(-80, -80, 160, 160);// 创建圆对象
            Color color = new Color(255, 0, 0);// 创建颜色对象
            g2.setColor(color);// 指定颜色
            g2.draw(ellipse);// 绘制圆
            // 绘制五星
            int[] x1 = { 0, 8, 30, 16, 25, 0, -25, -16, -30, -8 }; // 多边形的横坐标
            int[] y1 = { -35, -10, -15, 5, 28, 10, 28, 5, -15, -10 }; // 多边形的纵坐标
            int n1 = 10;// 多边形的边数
            g2.fillPolygon(x1, y1, n1);// 绘制多边形
            // 绘制文本
            g2.scale(1.8, 1.8);// 放大
            Font font = new Font("宋体", Font.BOLD, 12);// 创建字体
            g2.setFont(font);// 设置字体
            g2.drawString("专 用 章", -25, 30);// 绘制文本
            int width = getWidth();// 获得面板宽度
            int height = getHeight();// 获得面板高度
            char[] array = " 明日科技有限公司         ".toCharArray();// 把字符串转换为字符数组
            int len = array.length * 2;// 定义半径
            font = new Font("宋体", Font.BOLD, 10);// 创建新字体
            g2.setFont(font);// 设置字体
            double angle = 0;// 初始角度
            for (int i = 0; i < array.length; i++) {// 遍历字符串中的字符
                int x = (int) (len * Math.sin(Math.toRadians(angle + 270)));// 计算每个文字的位置
                int y = (int) (len * Math.cos(Math.toRadians(angle + 270)));// 计算每个文字的位置
                g2.drawString(array[i] + "", width / 2 + x - 168, height / 2
                        - y - 95);// 绘制每个字符，其中168和95是坐标平移值
                angle = angle + 360d / array.length;// 改变角度
            }
        }
    }
}
