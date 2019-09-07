package com.zzk;
import java.awt.Graphics;
import javax.swing.JFrame;
import javax.swing.JPanel;
public class DrawSectorFrame extends JFrame {
    DrawSectorPanel sectorPanel = new DrawSectorPanel(); // 创建面板类的实例
    public static void main(String args[]) { // 主方法
        DrawSectorFrame frame = new DrawSectorFrame(); // 创建窗体类的实例
        frame.setVisible(true); // 显示窗体
    }
    public DrawSectorFrame() {
        super(); // 调用超类的构造方法
        setTitle("绘制填充扇形"); // 窗体标题
        setBounds(100, 100, 278, 184); // 窗体的显示位置和大小
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE); // 窗体关闭方式
        add(sectorPanel); // 将面板类的实例添加到窗体容器中
    }
    class DrawSectorPanel extends JPanel { // 创建内部面板类
        public void paint(Graphics g) { // 重写paint()方法
            g.fillArc(40, 20, 80, 80, 0, 150);    // 绘制填充扇形
            g.fillArc(140, 20, 80, 80, 180, -150);// 绘制填充扇形
            g.fillArc(40, 40, 80, 80, 0, -110);   // 绘制填充扇形
            g.fillArc(140, 40, 80, 80, 180, 110); // 绘制填充扇形
        }
    }
}
