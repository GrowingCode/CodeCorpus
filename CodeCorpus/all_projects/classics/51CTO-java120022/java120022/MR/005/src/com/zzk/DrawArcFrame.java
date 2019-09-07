package com.zzk;
import java.awt.Graphics;
import javax.swing.JFrame;
import javax.swing.JPanel;
public class DrawArcFrame extends JFrame {
    DrawArcPanel arcPanel = new DrawArcPanel(); // 创建面板类的实例
    public static void main(String args[]) { // 主方法
        DrawArcFrame frame = new DrawArcFrame(); // 创建窗体类的实例
        frame.setVisible(true); // 显示窗体
    }
    public DrawArcFrame() {
        super(); // 调用超类的构造方法
        setTitle("绘制圆弧"); // 窗体标题
        setBounds(100, 100, 269, 184); // 窗体的显示位置和大小
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE); // 窗体关闭方式
        add(arcPanel); // 将面板类的实例添加到窗体容器中
    }
    class DrawArcPanel extends JPanel { // 创建内部面板类
        public void paint(Graphics g) { // 重写paint()方法
            g.drawArc(20, 20, 80, 80, 0, 120);    // 绘制圆弧
            g.drawArc(20, 40, 80, 80, 0, -120);   // 绘制圆弧
            g.drawArc(150, 20, 80, 80, 180, -120);// 绘制圆弧
            g.drawArc(150, 40, 80, 80, 180, 120); // 绘制圆弧
        }
    }
}
