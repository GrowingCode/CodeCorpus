package com.zzk;

import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
import com.sun.awt.AWTUtilities;

/**
 * @author 张振坤
 */
@SuppressWarnings("serial")
public class ClockFrame extends JDialog {
    private float opqua = 0.7f;
    private ClockPanel clockPanel;
    private Point fp; // 拖曳窗体之前的鼠标位置
    
    public static void main(String args[]) {
        EventQueue.invokeLater(new Runnable() {
            public void run() {
                try {
                    ClockFrame frame = new ClockFrame();// 创建窗体对象
                    frame.setVisible(true);// 显示窗体
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        });
    }
    
    /**
     * 布局窗体的构造方法
     */
    public ClockFrame() {
        super();
        setUndecorated(true);// 取消窗体修饰
        setAlwaysOnTop(true);// 窗体置顶
        setTitle("明日科技石英钟");// 设置窗体标题
        getContentPane().setLayout(new BorderLayout());
        setBounds(100, 30, 217, 257);
        setDefaultCloseOperation(JFrame.HIDE_ON_CLOSE);
        clockPanel = new ClockPanel();// 创建时钟面板
        getContentPane().add(clockPanel);
        // 为时钟面板添加鼠标按键事件监听器
        clockPanel.addMouseListener(new MouseAdapter() {
            public void mousePressed(final MouseEvent e) {
                fp = e.getPoint();
                if (e.getButton()==MouseEvent.BUTTON3){
                    System.exit(0);// 右键退出
                }
            }
        });
        // 在时钟面板的鼠标拖曳事件中移动窗体
        clockPanel.addMouseMotionListener(new MouseMotionAdapter() {
            public void mouseDragged(final MouseEvent e) {
                JDialog frame = (JDialog) getRootPane().getParent();
                Point point = e.getLocationOnScreen();
                frame.setLocation(point.x - fp.x, point.y - fp.y);
            }
        });
        pack();
        
        addKeyListener(new KeyAdapter() {// 为窗体添加键盘事件监听器
            public void keyPressed(final KeyEvent e) {
                int code = e.getKeyCode();
                switch (code) {// 判断按键编码
                    case KeyEvent.VK_ADD:// +符号按键会降低透明图
                        opqua += 0.05;
                        opqua = opqua > 0.95f ? 1f : opqua;
                        break;
                    case KeyEvent.VK_SUBTRACT:// -符号按键会提升透明度
                        opqua -= 0.05;
                        opqua = opqua < 0.1f ? 0.1f : opqua;
                        break;
                }
                // 如果按Ctrl+Shift+X組合鍵，将退出程序
                if (code == KeyEvent.VK_X
                        && e.getModifiersEx() == (KeyEvent.CTRL_DOWN_MASK | KeyEvent.SHIFT_DOWN_MASK))
                    System.exit(0);
                AWTUtilities.setWindowOpacity(ClockFrame.this, opqua);// 更新窗体透明度
            }
        });
        
        AWTUtilities.setWindowOpacity(this, opqua);
        Dimension screenSize = getToolkit().getScreenSize();
        double width = screenSize.getWidth();
        int x = (int) (width - getWidth() - 30);
        setLocation(x, 30);
        
        new Thread() {// 创建线程对象，更新时钟面板界面
            @Override
            public void run() {
                try {
                    while (true) {
                        sleep(1000);// 休眠1秒
                        clockPanel.repaint();// 重新绘制时钟面板界面
                    }
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
        }.start();
    }
}
