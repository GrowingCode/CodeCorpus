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

public class TranslationAxisFrame extends JFrame {
    int flag = 0; // 为0时还原坐标轴，为1时平移坐标轴
    TranslationAxisPanel axisPanel = new TranslationAxisPanel(); // 创建面板类的实例
    public static void main(String args[]) { // 主方法
        TranslationAxisFrame frame = new TranslationAxisFrame(); // 创建窗体类的实例
        frame.setVisible(true); // 显示窗体
    }
    
    public TranslationAxisFrame() {
        super(); // 调用超类的构造方法
        setTitle("平移坐标轴"); // 窗体标题
        setBounds(100, 100, 338, 230); // 窗体的显示位置和大小
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE); // 窗体关闭方式
        add(axisPanel); // 将面板类的实例添加到窗体容器中
        final JPanel panel = new JPanel();
        final FlowLayout flowLayout = new FlowLayout();
        flowLayout.setHgap(30);
        panel.setLayout(flowLayout);
        getContentPane().add(panel, BorderLayout.SOUTH);
        
        final JButton btn_deasil = new JButton();
        btn_deasil.addActionListener(new ActionListener() {
            public void actionPerformed(final ActionEvent e) {
                flag = 1;             // 平移坐标轴标记
                axisPanel.repaint();  // 重新调用面板的paint()方法
            }
        });
        btn_deasil.setText("平移坐标轴");
        panel.add(btn_deasil);
        
        final JButton btn_restore = new JButton();
        btn_restore.addActionListener(new ActionListener() {
            public void actionPerformed(final ActionEvent e) {
                flag = 0;           // 还原坐标轴标记
                axisPanel.repaint();// 重新调用面板的paint()方法
            }
        });
        btn_restore.setText("还原坐标轴");
        panel.add(btn_restore);
    }
    
    class TranslationAxisPanel extends JPanel { // 创建内部面板类
        public void paint(Graphics g) { // 重写paint()方法
            Graphics2D g2 = (Graphics2D) g; // 获得Graphics2D对象
            Rectangle2D.Float rect = new Rectangle2D.Float(10, 10, 80, 50);// 创建矩形对象
            BasicStroke stroke = new BasicStroke(10); // 创建宽度是10的笔画对象
            g2.setStroke(stroke);// 设置笔画对象
            g2.clearRect(0, 0, 338, 230);  // 清除原有内容
            if (flag == 0) {
                g2.translate(0, 0);// 平移坐标轴
                g2.draw(rect);// 绘制矩形
            } else if (flag == 1) {
                g2.translate(120, 60);// 平移坐标轴
                g2.draw(rect);// 绘制矩形
            }
        }
    }
}
