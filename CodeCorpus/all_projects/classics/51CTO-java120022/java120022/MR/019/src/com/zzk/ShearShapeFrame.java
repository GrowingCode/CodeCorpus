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

public class ShearShapeFrame extends JFrame {
    int flag = 0; // 为0时显示原图形，为1时向下斜切，为2时向上斜切
    ShearShapePanel shearShapePanel = new ShearShapePanel(); // 创建面板类的实例
    public static void main(String args[]) { // 主方法
        ShearShapeFrame frame = new ShearShapeFrame(); // 创建窗体类的实例
        frame.setVisible(true); // 显示窗体
    }
    
    public ShearShapeFrame() {
        super(); // 调用超类的构造方法
        setTitle("斜切图形"); // 窗体标题
        setBounds(100, 100, 338, 230); // 窗体的显示位置和大小
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE); // 窗体关闭方式
        add(shearShapePanel); // 将面板类的实例添加到窗体容器中
        final JPanel panel = new JPanel();
        final FlowLayout flowLayout = new FlowLayout();
        flowLayout.setHgap(30);
        panel.setLayout(flowLayout);
        getContentPane().add(panel, BorderLayout.SOUTH);
        
        final JButton btn_deasil = new JButton();
        btn_deasil.addActionListener(new ActionListener() {
            public void actionPerformed(final ActionEvent e) {
                flag = 1;             // 下斜切标记
                shearShapePanel.repaint();  // 重新调用面板的paint()方法
            }
        });
        btn_deasil.setText("下斜切");
        panel.add(btn_deasil);
        
        final JButton btn_widdershins = new JButton();
        btn_widdershins.addActionListener(new ActionListener() {
            public void actionPerformed(final ActionEvent e) {
                flag = 2;           // 上斜切标记
                shearShapePanel.repaint();// 重新调用面板的paint()方法
            }
        });
        btn_widdershins.setText("上斜切");
        panel.add(btn_widdershins);

        final JButton btn_restore = new JButton();
        btn_restore.addActionListener(new ActionListener() {
            public void actionPerformed(final ActionEvent e) {
                flag = 0;           // 还原图形
                shearShapePanel.repaint();// 重新调用面板的paint()方法
            }
        });
        btn_restore.setText("还    原");
        panel.add(btn_restore);
    }
    
    class ShearShapePanel extends JPanel { // 创建内部面板类
        public void paint(Graphics g) { // 重写paint()方法
            Graphics2D g2 = (Graphics2D) g; // 获得Graphics2D对象
            Rectangle2D.Float rect = new Rectangle2D.Float(120, 50, 80, 50);// 创建矩形对象
            BasicStroke stroke = new BasicStroke(10); // 创建宽度是10的笔画对象
            g2.setStroke(stroke);// 设置笔画对象
            g2.clearRect(0, 0, 338, 230);  // 清除原有内容
            if (flag == 0) {
                g2.draw(rect);// 绘制原矩形
            } else if (flag == 1) {
                g2.shear(0.2,0.2);// 向下斜切
                g2.draw(rect);// 绘制矩形
            } else if (flag == 2) {
                g2.shear(-0.2,-0.2);// 向上斜切
                g2.draw(rect);// 绘制矩形
            }
        }
    }
}
