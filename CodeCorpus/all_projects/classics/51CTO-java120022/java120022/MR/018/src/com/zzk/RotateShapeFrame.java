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

public class RotateShapeFrame extends JFrame {
    int flag = 0; // 为0时显示原图形，为1时顺时针旋转，为2时逆时针旋转
    double rotateValue = 0.0;  // 值变大顺时针旋转，值变小逆时针旋转
    RotateShapePanel rotateShapePanel = new RotateShapePanel(); // 创建面板类的实例
    
    public static void main(String args[]) { // 主方法
        RotateShapeFrame frame = new RotateShapeFrame(); // 创建窗体类的实例
        frame.setVisible(true); // 显示窗体
    }
    
    public RotateShapeFrame() {
        super(); // 调用超类的构造方法
        setTitle("旋转图形"); // 窗体标题
        setBounds(100, 100, 338, 220); // 窗体的显示位置和大小
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE); // 窗体关闭方式
        add(rotateShapePanel); // 将面板类的实例添加到窗体容器中
        final JPanel panel = new JPanel();
        final FlowLayout flowLayout = new FlowLayout();
        flowLayout.setHgap(30);
        panel.setLayout(flowLayout);
        getContentPane().add(panel, BorderLayout.SOUTH);
        
        final JButton btn_deasil = new JButton();
        btn_deasil.addActionListener(new ActionListener() {
            public void actionPerformed(final ActionEvent e) {
                flag = 1;             // 顺时针标记
                rotateValue += 0.1;   // 顺时针旋转值
                rotateShapePanel.repaint();  // 重新调用面板的paint()方法
            }
        });
        btn_deasil.setText("顺时针");
        panel.add(btn_deasil);
        
        final JButton btn_widdershins = new JButton();
        btn_widdershins.addActionListener(new ActionListener() {
            public void actionPerformed(final ActionEvent e) {
                flag = 2;           // 逆时针标记
                rotateValue -= 0.1; // 逆时针旋转值
                rotateShapePanel.repaint();// 重新调用面板的paint()方法
            }
        });
        btn_widdershins.setText("逆时针");
        panel.add(btn_widdershins);

        final JButton btn_restore = new JButton();
        btn_restore.addActionListener(new ActionListener() {
            public void actionPerformed(final ActionEvent e) {
                flag = 0;           // 还原图形
                rotateValue = 0;
                rotateShapePanel.repaint();// 重新调用面板的paint()方法
            }
        });
        btn_restore.setText("还    原");
        panel.add(btn_restore);
    }
    
    class RotateShapePanel extends JPanel { // 创建内部面板类
        public void paint(Graphics g) { // 重写paint()方法
            Graphics2D g2 = (Graphics2D) g; // 获得Graphics2D对象
            Rectangle2D.Float rect = new Rectangle2D.Float(40, 40, 80, 50);// 创建矩形对象
            BasicStroke stroke = new BasicStroke(10); // 创建宽度是10的笔画对象
            g2.setStroke(stroke);// 设置笔画对象
            g2.clearRect(0, 0, 338, 220);  // 清除原有内容
            if (flag == 0) {
                g2.draw(rect);// 绘制原矩形
            } else if (flag == 1) {
                g2.rotate(rotateValue);// 顺时针旋转
                g2.draw(rect);// 绘制矩形
            } else if (flag == 2) {
                g2.rotate(rotateValue);// 逆时针旋转
                g2.draw(rect);// 绘制矩形
            }
        }
    }
}
