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
import java.awt.event.InputEvent;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.event.MouseMotionAdapter;
import java.awt.event.WindowEvent;
import java.awt.event.WindowFocusListener;
import java.awt.image.BufferedImage;
import javax.swing.JFrame;
import javax.swing.JPanel;
import com.sun.awt.AWTUtilities;

@SuppressWarnings("serial")
public class ScreenZoomInFrame extends JFrame {
    private PartZoomInPanel partZoomInPanel = null; // 声明图像面板对象
    private int pressPanelX = 0, pressPanelY = 0;// 鼠标按下点的X、Y坐标
    private int pressX = 0, pressY = 0;// 鼠标按下点在屏幕上的X、Y坐标
    private int releaseX = 0, releaseY = 0;// 鼠标释放点在屏幕上的X、Y坐标
    private Robot robot = null; // 声明Robot对象
    private BufferedImage buffImage = null; // 声明缓冲图像对象
    private BufferedImage zoomBuffImage = null; // 声明缓冲图像对象
    private boolean flag = false; // 声明标记变量，为true时显示选择区域的矩形，否则不显示
    private boolean mouseFlag = false;// 进行放大图像的标记变量，为true时进行放大，否则不放大
    private Rectangle rect = null;
    private Dimension dim = null;
    
    public static void main(String args[]) {
        ScreenZoomInFrame frame;
        try {
            frame = new ScreenZoomInFrame("屏幕放大镜");
            frame.setVisible(true);
        } catch (AWTException e) {
            e.printStackTrace();
        }
        
    }
    
    public ScreenZoomInFrame(String title) throws AWTException {
        super(title);
        getContentPane().addMouseListener(new MouseAdapter() {
        	public void mouseClicked(final MouseEvent e) {
        		if (e.getButton()==MouseEvent.BUTTON3) {
        			System.exit(0);// 单击鼠标右键退出
        		}
        	}
        });
        setAlwaysOnTop(true);
        partZoomInPanel = new PartZoomInPanel(); // 创建图像面板对象
        Toolkit toolkit = getToolkit();
        dim = toolkit.getScreenSize();
        setBounds(0, 0, dim.width, dim.height - 30); // 设置窗体大小和位置
        setUndecorated(true);
        getContentPane().add(partZoomInPanel, BorderLayout.CENTER);
        AWTUtilities.setWindowOpacity(this, 0.01f);
        robot = new Robot();// 创建Robot对象
        rect = new Rectangle(0, 0, dim.width, dim.height);// 创建Rectangle对象
        buffImage = robot.createScreenCapture(rect);// 获得缓冲图像对象
        addWindowFocusListener(new WindowFocusListener() {
            public void windowGainedFocus(final WindowEvent e) {
            }
            
            public void windowLostFocus(final WindowEvent e) {
                flag = false;// 为标记变量赋值为true
                mouseFlag = false;// 将标记设置为false
                AWTUtilities.setWindowOpacity(ScreenZoomInFrame.this, 0.01f);
                buffImage = robot.createScreenCapture(rect);// 获得缓冲图像对象
                partZoomInPanel.repaint();
                AWTUtilities.setWindowOpacity(ScreenZoomInFrame.this, 1f);
                robot.mouseMove(200, 730);
                robot.mousePress(InputEvent.BUTTON1_MASK);
                robot.mouseRelease(InputEvent.BUTTON1_MASK);
            }
        });
        //
        partZoomInPanel.addMouseListener(new MouseAdapter() {
            public void mousePressed(final MouseEvent e) { // 鼠标键按下事件
                AWTUtilities.setWindowOpacity(ScreenZoomInFrame.this, 0.01f);
                buffImage = robot.createScreenCapture(rect);// 获得缓冲图像对象
                partZoomInPanel.repaint();
                AWTUtilities.setWindowOpacity(ScreenZoomInFrame.this, 1f);
                pressPanelX = e.getX(); // 获得鼠标按下点的X坐标
                pressPanelY = e.getY();// 获得鼠标按下点的Y坐标
                pressX = e.getXOnScreen() + 1;// 鼠标按下点在屏幕上的X坐标加1，即去除选择线
                pressY = e.getYOnScreen() + 1;// 鼠标按下点在屏幕上的Y坐标加1，即去除选择线
                flag = true;// 为标记变量赋值为true
                mouseFlag = false;// 将标记设置为false
            }
            
            public void mouseReleased(final MouseEvent e) { // 鼠标键释放事件
                releaseX = e.getXOnScreen() - 1;// 鼠标释放点在屏幕上的X坐标减1，即去除选择线
                releaseY = e.getYOnScreen() - 1;// 鼠标释放点在屏幕上的Y坐标减1，即去除选择线
                if (releaseX - pressX > 0 && releaseY - pressY > 0) {
                    Rectangle rect = new Rectangle(pressX, pressY, releaseX
                            - pressX, releaseY - pressY);// 创建Rectangle对象
                    zoomBuffImage = robot.createScreenCapture(rect);// 获得缓冲图像对象
                }
                flag = false;// 为标记变量赋值为false
                mouseFlag = true;// 标记为true，进行放大
                partZoomInPanel.repaint();// 调用paint()方法，实现放大
            }
            
            public void mouseClicked(final MouseEvent e) { // 鼠标键单击事件
                if (e.getButton() == MouseEvent.BUTTON3) {// 单击鼠标右键
                    System.exit(0);// 退出系统
                }
                mouseFlag = false;// 将标记设置为false，不放大图像
                partZoomInPanel.repaint();// 调用PartZoomInPanel面板的paint()方法
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
    
    class PartZoomInPanel extends JPanel {// 创建面板类
        public void paint(Graphics g) {
            Graphics2D g2 = (Graphics2D) g;
            g2.drawImage(buffImage, 0, 0, this);// 绘制图像
            g2.setColor(Color.BLACK);
            if (flag) {
                float[] arr = { 5.0f }; // 创建虚线模式的数组
                BasicStroke stroke = new BasicStroke(1, BasicStroke.CAP_BUTT,
                        BasicStroke.JOIN_BEVEL, 1.0f, arr, 0); // 创建宽度是1的平头虚线笔画对象
                g2.setStroke(stroke);// 设置笔画对象
                g2.drawRect(pressPanelX, pressPanelY, releaseX - pressX,
                        releaseY - pressY);// 绘制矩形选区
            }
            
            if (mouseFlag) {// 条件为真
                int zoomX = pressPanelX - (releaseX - pressX) / 4;// 放大内容绘制点的x坐标
                int zoomY = pressPanelY - (releaseY - pressY) / 4;// 放大内容绘制点的y坐标
                if (zoomX <= 0) {
                    zoomX = 0;// 坐标值小于等于0，让坐标值为0
                }
                if (zoomY <= 0) {
                    zoomY = 0;// 坐标值小于等于0，让坐标值为0
                }
                g2.drawImage(zoomBuffImage, zoomX, zoomY,
                        (int) ((releaseX - pressX) * 1.5f),
                        (int) ((releaseY - pressY) * 1.5f), this);// 绘制放大后的内容
            }
        }
    }
    
}
