package com.zzk;

import java.awt.AWTException;
import java.awt.Color;
import java.awt.EventQueue;
import java.awt.MouseInfo;
import java.awt.Point;
import java.awt.PointerInfo;
import java.awt.Robot;

import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JTextField;

public class GainAnyPointColorFrame extends JFrame {
    
    private JTextField tf_blue;
    private JTextField tf_green;
    private JTextField tf_red;
    private JTextField tf_y;
    private JTextField tf_x;
    private static final long serialVersionUID = -486745172657329259L;
    
    /**
     * Launch the application
     * 
     * @param args
     */
    public static void main(String args[]) {
        EventQueue.invokeLater(new Runnable() {
            public void run() {
                try {
                    GainAnyPointColorFrame frame = new GainAnyPointColorFrame();
                    frame.setVisible(true);
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        });
    }
    
    /**
     * Create the frame
     */
    public GainAnyPointColorFrame() {
        super();
        GainColor gc = new GainColor();// 创建GainColor对象
        Thread thread = new Thread(gc);// 创建线程对象
        thread.start();// 启动线程对象
        getContentPane().setLayout(null);
        setTitle("获取鼠标在任意位置的颜色值");
        setBounds(100, 100, 300, 207);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        
        final JLabel label = new JLabel();
        label.setText("鼠标所在点的X坐标：");
        label.setBounds(26, 21, 126, 25);
        getContentPane().add(label);
        
        final JLabel label_1 = new JLabel();
        label_1.setText("鼠标所在点的Y坐标：");
        label_1.setBounds(26, 45, 126, 25);
        getContentPane().add(label_1);
        
        tf_x = new JTextField();
        tf_x.setBounds(155, 22, 103, 22);
        getContentPane().add(tf_x);
        
        tf_y = new JTextField();
        tf_y.setBounds(155, 46, 103, 22);
        getContentPane().add(tf_y);
        
        final JLabel label_2 = new JLabel();
        label_2.setText("颜色的Red值：");
        label_2.setBounds(26, 79, 97, 18);
        getContentPane().add(label_2);
        
        final JLabel label_3 = new JLabel();
        label_3.setText("颜色的Green值：");
        label_3.setBounds(26, 103, 111, 18);
        getContentPane().add(label_3);
        
        final JLabel label_4 = new JLabel();
        label_4.setText("颜色的Blue值：");
        label_4.setBounds(26, 127, 109, 18);
        getContentPane().add(label_4);
        
        tf_red = new JTextField();
        tf_red.setBounds(155, 77, 103, 22);
        getContentPane().add(tf_red);
        
        tf_green = new JTextField();
        tf_green.setBounds(155, 101, 103, 22);
        getContentPane().add(tf_green);
        
        tf_blue = new JTextField();
        tf_blue.setBounds(155, 125, 103, 22);
        getContentPane().add(tf_blue);
        
    }
    
    class GainColor implements Runnable {
        @Override
        public void run() {
            while (true) {
                PointerInfo mi = MouseInfo.getPointerInfo();// 鼠标指针当前位置的PointerInfo对象
                Point p = mi.getLocation();// 获得屏幕上表示指针坐标的Point对象
                int x = p.x;// 获得X坐标
                int y = p.y;// 获得Y坐标
                try {
                    Robot robot = new Robot();// 创建Robot对象
                    Color color = robot.getPixelColor(x, y);// 获得屏幕指定位置的颜色对象
                    int r = color.getRed();// 获得颜色的R值
                    int g = color.getGreen();// 获得颜色的G值
                    int b = color.getBlue();// 获得颜色的B值
                    tf_x.setText(String.valueOf(x));// 显示X坐标值
                    tf_y.setText(String.valueOf(y));// 显示Y坐标值
                    tf_red.setText(String.valueOf(r));// 显示颜色的R值
                    tf_green.setText(String.valueOf(g));// 显示颜色的G值
                    tf_blue.setText(String.valueOf(b));// 显示颜色的B值
                    Thread.sleep(10);// 线程休眠10毫秒
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        }
        
    }
}
