package com.zzk;

import java.awt.AWTException;
import java.awt.BasicStroke;
import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Rectangle;
import java.awt.Robot;
import java.awt.Toolkit;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.event.MouseMotionAdapter;
import java.awt.image.BufferedImage;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import javax.imageio.ImageIO;
import javax.swing.JFrame;
import javax.swing.JPanel;
import com.sun.awt.AWTUtilities;

@SuppressWarnings("serial")
public class CaptureScreenImageFrame extends JFrame {
    private PartZoomInPanel partZoomInPanel = null;// 声明图像面板对象
    private int pressPanelX = 0, pressPanelY = 0;// 鼠标按下点的X、Y坐标
    private int pressX = 0, pressY = 0;// 鼠标按下点在屏幕上的X、Y坐标
    private int releaseX = 0, releaseY = 0;// 鼠标释放点在屏幕上的X、Y坐标
    private Robot robot = null;// 声明Robot对象
    private BufferedImage buffImage = null;// 声明缓冲图像对象
    private boolean flag = false;// 声明标记变量，为true时显示选择区域的矩形，否则不显示
    private Rectangle rect = null;
    
    public static void main(String args[]) {
        CaptureScreenImageFrame frame;
        try {
            frame = new CaptureScreenImageFrame("屏幕抓图程序");
            frame.setVisible(true);
        } catch (AWTException e) {
            e.printStackTrace();
        }
    }
    
    public CaptureScreenImageFrame(String title) throws AWTException {
        super(title);
        setAlwaysOnTop(true);
        partZoomInPanel = new PartZoomInPanel(); // 创建图像面板对象
        Toolkit toolkit = getToolkit();
        Dimension dim = toolkit.getScreenSize();
        setBounds(0, 0, dim.width, dim.height - 30); // 设置窗体大小和位置
        setUndecorated(true);// 取消窗体修饰
        AWTUtilities.setWindowOpacity(this, 0.01f);// 设置窗体透明
        getContentPane().add(partZoomInPanel, BorderLayout.CENTER);
        robot = new Robot();// 创建Robot对象
        rect = new Rectangle(0, 0, dim.width, dim.height);// 创建Rectangle对象
        buffImage = robot.createScreenCapture(rect);// 获得缓冲图像对象
        partZoomInPanel.addMouseListener(new MouseAdapter() {
            public void mousePressed(final MouseEvent e) { // 鼠标键按下事件
                buffImage = robot.createScreenCapture(rect);// 获得缓冲图像对象
                partZoomInPanel.repaint();
                AWTUtilities.setWindowOpacity(CaptureScreenImageFrame.this, 1f);// 设置窗体为不透明
                pressPanelX = e.getX(); // 获得鼠标按下点的X坐标
                pressPanelY = e.getY();// 获得鼠标按下点的Y坐标
                pressX = e.getXOnScreen() + 1;// 鼠标按下点在屏幕上的X坐标加1，即去除选择线
                pressY = e.getYOnScreen() + 1;// 鼠标按下点在屏幕上的Y坐标加1，即去除选择线
                flag = true;// 为标记变量赋值为true
            }
            
            public void mouseReleased(final MouseEvent e) { // 鼠标键释放事件
                releaseX = e.getXOnScreen() - 1;// 鼠标释放点在屏幕上的X坐标减1，即去除选择线
                releaseY = e.getYOnScreen() - 1;// 鼠标释放点在屏幕上的Y坐标减1，即去除选择线
                try {
                    if (releaseX - pressX > 0 && releaseY - pressY > 0) {
                        Rectangle rect = new Rectangle(pressX, pressY, releaseX
                                - pressX, releaseY - pressY);// 创建Rectangle对象
                        buffImage = robot.createScreenCapture(rect);// 获得缓冲图像对象
                        FileOutputStream out = new FileOutputStream("c:/zzkkee.jpg");// 保存位置的输出流对象
                        ImageIO.write(buffImage, "jpg", out);// 写入磁盘
                        out.flush();
                        out.close();
                    }
                } catch (FileNotFoundException e2) {
                    e2.printStackTrace();
                } catch (IOException e3) {
                    e3.printStackTrace();
                }
                flag = false;   // 为标记变量赋值为false
                if (e.getButton() == MouseEvent.BUTTON1) {// 单击鼠标右键
                    CaptureScreenImageFrame.this.dispose();
                    CaptureScreenImageFrame frame;
                    try {
                        frame = new CaptureScreenImageFrame("屏幕抓图程序");
                        AWTUtilities.setWindowOpacity(frame, 0.01f);
                        frame.setVisible(true);
                    } catch (AWTException e1) {
                        e1.printStackTrace();
                    }
                }
            }
            
            public void mouseClicked(final MouseEvent e) { // 鼠标键单击事件
                if (e.getButton() == MouseEvent.BUTTON3) {// 单击鼠标右键
                    System.exit(0);// 退出系统
                }
            }
        });
        partZoomInPanel.addMouseMotionListener(new MouseMotionAdapter() {
            public void mouseDragged(final MouseEvent e) {// 鼠标拖动事件
                if (flag) {
                    releaseX = e.getXOnScreen();// 获得鼠标释放点在屏幕上的X坐标
                    releaseY = e.getYOnScreen();// 获得鼠标释放点在屏幕上的Y坐标
                    partZoomInPanel.repaint();// 调用PartZoomInPanel面板的paint()方法
                }
            }
        });
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE); // 设置窗体关闭模式
    }
    
    class PartZoomInPanel extends JPanel {// 创建绘制原图像的面板类
        public void paint(Graphics g) {
            Graphics2D g2 = (Graphics2D) g;
            g2.drawImage(buffImage, 0, 0, PartZoomInPanel.this);
            g2.setColor(Color.BLACK);
            if (flag) {
                float[] arr = { 5.0f }; // 创建虚线模式的数组
                BasicStroke stroke = new BasicStroke(1, BasicStroke.CAP_BUTT,
                        BasicStroke.JOIN_BEVEL, 1.0f, arr, 0); // 创建宽度是1的平头虚线笔画对象
                g2.setStroke(stroke);// 设置笔画对象
                g2.drawRect(pressPanelX, pressPanelY, releaseX - pressX,
                        releaseY - pressY);// 绘制矩形选区
            }
        }
    }
    
}
