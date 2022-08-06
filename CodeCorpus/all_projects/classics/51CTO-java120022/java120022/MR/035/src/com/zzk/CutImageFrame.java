package com.zzk;

import java.awt.AWTException;
import java.awt.BasicStroke;
import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.Rectangle;
import java.awt.Robot;
import java.awt.Toolkit;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.event.MouseMotionAdapter;
import java.awt.image.BufferedImage;
import java.net.URL;
import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.JSplitPane;
public class CutImageFrame extends JFrame {
    private Image img = null; // 声明图像对象
    private OldImagePanel oldImagePanel = null; // 声明图像面板对象
    private int pressPanelX = 0, pressPanelY = 0;// 鼠标按下点的X、Y坐标 
    private int pressX = 0, pressY = 0;// 鼠标按下点在屏幕上的X、Y坐标
    private int releaseX = 0, releaseY = 0;// 鼠标释放点在屏幕上的X、Y坐标
    private Robot robot = null;  // 声明Robot对象
    private BufferedImage buffImage = null; // 声明缓冲图像对象
    private CutImagePanel cutImagePanel = new CutImagePanel(); // 创建绘制裁剪结果的面板
    private boolean flag = false;  // 声明标记变量，为true时显示选择区域的矩形，否则不显示
    public static void main(String args[]) {
        CutImageFrame frame = new CutImageFrame();
        frame.setVisible(true);
    }
    public CutImageFrame() {
        super();
        URL imgUrl = CutImageFrame.class.getResource("/img/image.jpg");// 获取图片资源的路径
        img = Toolkit.getDefaultToolkit().getImage(imgUrl); // 获取图像资源
        oldImagePanel = new OldImagePanel(); // 创建图像面板对象
        this.setBounds(200, 160, 355, 276); // 设置窗体大小和位置
        final JSplitPane splitPane = new JSplitPane();
        splitPane.setDividerLocation((this.getWidth() / 2) - 10);
        getContentPane().add(splitPane, BorderLayout.CENTER);
        splitPane.setLeftComponent(oldImagePanel);
        splitPane.setRightComponent(cutImagePanel);
        oldImagePanel.addMouseListener(new MouseAdapter() {
            public void mousePressed(final MouseEvent e) {  // 鼠标键按下事件
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
                    robot = new Robot();// 创建Robot对象
                    if (releaseX - pressX > 0 && releaseY - pressY > 0) {
                        Rectangle rect = new Rectangle(pressX, pressY, releaseX
                                - pressX, releaseY - pressY);// 创建Rectangle对象
                        buffImage = robot.createScreenCapture(rect);// 获得缓冲图像对象
                        cutImagePanel.repaint(); // 调用CutImagePanel面板的paint()方法
                    }
                } catch (AWTException e1) {
                    e1.printStackTrace();
                }
                flag = false;// 为标记变量赋值为false
            }
        });
        oldImagePanel.addMouseMotionListener(new MouseMotionAdapter() {
            public void mouseDragged(final MouseEvent e) {// 鼠标拖动事件
                if (flag) {
                    releaseX = e.getXOnScreen();// 获得鼠标释放点在屏幕上的X坐标
                    releaseY = e.getYOnScreen();// 获得鼠标释放点在屏幕上的Y坐标
                    oldImagePanel.repaint();// 调用OldImagePanel面板的paint()方法
                }
            }
        });
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE); // 设置窗体关闭模式
        this.setTitle("裁剪图片"); // 设置窗体标题
    }
    
    class OldImagePanel extends JPanel {// 创建绘制原图像的面板类
        public void paint(Graphics g) {
            Graphics2D g2 = (Graphics2D) g;
            g2.drawImage(img, 0, 0, this.getWidth(), this.getHeight(), this);// 绘制图像
            g2.setColor(Color.WHITE);
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
    
    class CutImagePanel extends JPanel {// 创建绘制裁剪结果的面板类
        public void paint(Graphics g) {
            g.clearRect(0, 0, this.getWidth(), this.getHeight());// 清除绘图上下文的内容
            g.drawImage(buffImage, 0, 0, releaseX - pressX, releaseY - pressY,
                    this);// 绘制图像
        }
    }
}
