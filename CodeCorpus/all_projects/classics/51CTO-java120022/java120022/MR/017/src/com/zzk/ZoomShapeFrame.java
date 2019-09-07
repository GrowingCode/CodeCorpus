package com.zzk;

import java.awt.BasicStroke;
import java.awt.BorderLayout;
import java.awt.FlowLayout;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.geom.Rectangle2D;
import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JPanel;

public class ZoomShapeFrame extends JFrame {
    int flag = 0; // 为0时显示原大小，为1时放大图形，为2时缩小图形
    ZoomShapePanel zoomPanel = new ZoomShapePanel(); // 创建面板类的实例
    
    public static void main(String args[]) { // 主方法
        ZoomShapeFrame frame = new ZoomShapeFrame(); // 创建窗体类的实例
        frame.setVisible(true); // 显示窗体
    }
    
    public ZoomShapeFrame() {
        super(); // 调用超类的构造方法
        setTitle("缩放图形"); // 窗体标题
        setBounds(100, 100, 338, 220); // 窗体的显示位置和大小
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE); // 窗体关闭方式
        add(zoomPanel); // 将面板类的实例添加到窗体容器中
        final JPanel panel = new JPanel();
        final FlowLayout flowLayout = new FlowLayout();
        flowLayout.setHgap(30);
        panel.setLayout(flowLayout);
        getContentPane().add(panel, BorderLayout.SOUTH);
        
        final JButton btn_zoomin = new JButton();
        btn_zoomin.addActionListener(new ActionListener() {
            public void actionPerformed(final ActionEvent e) {
                flag = 1;             // 放大图形标记
                zoomPanel.repaint();  // 重新调用面板的paint()方法
            }
        });
        btn_zoomin.setText("放    大");
        panel.add(btn_zoomin);
        
        final JButton btn_zoomout = new JButton();
        btn_zoomout.addActionListener(new ActionListener() {
            public void actionPerformed(final ActionEvent e) {
                flag = 2;           // 缩小图形标记
                zoomPanel.repaint();// 重新调用面板的paint()方法
            }
        });
        btn_zoomout.setText("缩    小");
        panel.add(btn_zoomout);

        final JButton btn_restore = new JButton();
        btn_restore.addActionListener(new ActionListener() {
            public void actionPerformed(final ActionEvent e) {
                flag = 0;           // 还原图形
                zoomPanel.repaint();// 重新调用面板的paint()方法
            }
        });
        btn_restore.setText("还    原");
        panel.add(btn_restore);
    }
    
    class ZoomShapePanel extends JPanel { // 创建内部面板类
        public void paint(Graphics g) { // 重写paint()方法
            Graphics2D g2 = (Graphics2D) g; // 获得Graphics2D对象
            Rectangle2D.Float rect = new Rectangle2D.Float(120, 50, 80, 50);// 创建矩形对象
            BasicStroke stroke = new BasicStroke(10); // 创建宽度是10的笔画对象
            g2.setStroke(stroke);// 设置笔画对象
            g2.clearRect(0, 0, 338, 220);  // 清除原有内容
            if (flag == 0) {
                g2.draw(rect);// 绘制原矩形
            } else if (flag == 1) {
                g2.scale(1.3, 1.3);// 放大1.3倍
                g2.draw(rect);// 绘制矩形
            } else if (flag == 2) {
                g2.scale(0.5, 0.5);// 缩小0.5倍
                g2.draw(rect);// 绘制矩形
            }
        }
    }
}
