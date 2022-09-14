package com.zzk;

import java.awt.BasicStroke;
import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Cursor;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.event.MouseMotionAdapter;
import java.awt.image.BufferedImage;
import javax.swing.JButton;
import javax.swing.JColorChooser;
import javax.swing.JFrame;
import javax.swing.JMenu;
import javax.swing.JMenuBar;
import javax.swing.JMenuItem;
import javax.swing.JToolBar;
import javax.swing.WindowConstants;

public class DrawPictureFrame extends JFrame {
    // 创建图像对象
    BufferedImage image = new BufferedImage(570, 390,
            BufferedImage.TYPE_INT_BGR);
    Graphics gs = image.getGraphics(); // 获得图像的绘图上下文对象
    Graphics2D g = (Graphics2D) gs; // 将绘图上下文对象转换为Graphics2D类型
    DrawPictureCanvas canvas = new DrawPictureCanvas(); // 创建画布对象
    Color foreColor = Color.BLACK; // 定义前景色
    Color backgroundColor = Color.WHITE; // 定义背景色
    boolean rubber = false; // 橡皮标识变量
    int x = -1; // 上一次鼠标绘制点的横坐标
    int y = -1; // 上一次鼠标绘制点的纵坐标
    private JMenuItem newItemMenuItem_6 = null;
    
    public DrawPictureFrame() {
        super();
        setResizable(false);
        setTitle("画图程序");
        setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        setBounds(100, 80, 574, 397);
        g.setColor(backgroundColor); // 用背景色设置绘图上下文对象的颜色
        g.fillRect(0, 0, 570, 390); // 用背景色填充整个画布
        g.setColor(foreColor); // 用前景色设置绘图上下文对象的颜色
        canvas.setImage(image); // 设置画布的图像
        getContentPane().add(canvas); // 将画布添加到窗体容器默认布局的中部位置
        canvas.addMouseMotionListener(new MouseMotionAdapter() {
            public void mouseDragged(final MouseEvent e) {
                if (rubber) { // 橡皮标识为true，表示使用橡皮
                    if (x > 0 && y > 0) {
                        g.setColor(backgroundColor); // 用背景色设置绘图上下文对象的颜色
                        g.fillRect(x, y, 10, 10); // 擦除鼠标经过位置的图像
                    }
                    x = e.getX(); // 获得鼠标在画布上的横坐标
                    y = e.getY(); // 获得鼠标在画布上的纵坐标
                } else { // 橡皮标识为false，表示画图
                    if (x > 0 && y > 0) {
                        // 在鼠标经过处画直线
                        g.drawLine(x, y, e.getX(), e.getY());
                    }
                    x = e.getX(); // 上一次鼠标绘制点的横坐标
                    y = e.getY(); // 上一次鼠标绘制点的纵坐标
                }
                canvas.repaint(); // 更新画布
            }
            
            public void mouseMoved(final MouseEvent arg0) {
                if (rubber) {
                    // 设置鼠标指针的形状
                    setCursor(Cursor.getPredefinedCursor(Cursor.HAND_CURSOR));
                } else {
                    // 设置鼠标指针的形状
                    setCursor(Cursor
                            .getPredefinedCursor(Cursor.CROSSHAIR_CURSOR));
                }
            }
        });
        canvas.addMouseListener(new MouseAdapter() {
            public void mouseReleased(final MouseEvent arg0) {
                x = -1; // 上一次鼠标绘制点的横坐标
                y = -1; // 上一次鼠标绘制点的纵坐标
            }
        });
        final JToolBar toolBar = new JToolBar();
        toolBar.addMouseMotionListener(new MouseMotionAdapter() {
            public void mouseMoved(final MouseEvent arg0) {
                // 设置鼠标指针的形状
                setCursor(Cursor.getPredefinedCursor(Cursor.DEFAULT_CURSOR));
            }
        });
        getContentPane().add(toolBar, BorderLayout.NORTH);
        final JButton button = new JButton();
        button.addActionListener(new ActionListener() {
            public void actionPerformed(final ActionEvent arg0) {
                // 声明画笔的属性，粗细为1像素
                BasicStroke bs = new BasicStroke(1, BasicStroke.CAP_BUTT,
                        BasicStroke.JOIN_MITER);
                g.setStroke(bs); // 设置绘图上下文对象的画笔
            }
        });
        button.setText("  细  线  ");
        toolBar.add(button);
        final JButton button_1 = new JButton();
        button_1.addActionListener(new ActionListener() {
            public void actionPerformed(final ActionEvent arg0) {
                // 声明画笔的属性，粗细为2像素
                BasicStroke bs = new BasicStroke(2, BasicStroke.CAP_BUTT,
                        BasicStroke.JOIN_MITER);
                g.setStroke(bs); // 设置绘图上下文对象的画笔
            }
        });
        button_1.setText("  粗  线  ");
        toolBar.add(button_1);
        
        final JButton button_2 = new JButton();
        button_2.addActionListener(new ActionListener() {
            public void actionPerformed(final ActionEvent arg0) {
                // 声明画笔的属性，粗细为4像素
                BasicStroke bs = new BasicStroke(4, BasicStroke.CAP_BUTT,
                        BasicStroke.JOIN_MITER);
                g.setStroke(bs); // 设置绘图上下文对象的画笔
            }
        });
        button_2.setText("  较  粗  ");
        toolBar.add(button_2);
        final JButton button_3 = new JButton();
        button_3.addActionListener(new ActionListener() {
            public void actionPerformed(final ActionEvent arg0) {
                // 打开选择颜色对话框
                Color bgColor = JColorChooser.showDialog(null, "选择颜色对话框",
                        Color.CYAN);
                if (bgColor != null) {
                    backgroundColor = bgColor;
                }
                g.setColor(backgroundColor); // 设置绘图上下文对象的背景色
                g.fillRect(0, 0, 570, 390); // 用背景色填充整个画布
                g.setColor(foreColor); // 设置绘图上下文对象的前景色
                canvas.repaint(); // 更新画布
            }
        });
        button_3.setText("背景颜色");
        toolBar.add(button_3);
        final JButton button_4 = new JButton();
        button_4.addActionListener(new ActionListener() {
            public void actionPerformed(final ActionEvent arg0) {
                // 打开选择颜色对话框
                Color fColor = JColorChooser.showDialog(null, "选择颜色对话框",
                        Color.CYAN);
                if (fColor != null) {
                    foreColor = fColor;
                }
                g.setColor(foreColor); // 设置绘图上下文对象的前景色
            }
        });
        button_4.setText("前景颜色");
        toolBar.add(button_4);
        final JButton button_5 = new JButton();
        button_5.addActionListener(new ActionListener() {
            public void actionPerformed(final ActionEvent arg0) {
                g.setColor(backgroundColor); // 设置绘图上下文对象的背景色
                g.fillRect(0, 0, 570, 390); // 用背景色填充整个画布
                g.setColor(foreColor); // 设置绘图上下文对象的前景色
                canvas.repaint(); // 更新画布
            }
        });
        button_5.setText("  清  除  ");
        toolBar.add(button_5);
        
        final JButton button_6 = new JButton();
        button_6.addActionListener(new ActionListener() {
            public void actionPerformed(final ActionEvent arg0) {
                if (button_6.getText().equals("  橡  皮  ")) { // 单击工具栏上的橡皮按钮，使用橡皮
                    rubber = true; // 设置橡皮标识为true
                    newItemMenuItem_6.setText("画  图"); // 改变菜单上显示的文本为画图
                    button_6.setText("  画  图  "); // 改变按钮上显示的文本为画图
                } else { // 单击工具栏上的画图按钮，使用画笔
                    rubber = false; // 设置橡皮标识为false
                    newItemMenuItem_6.setText("橡  皮"); // 改变菜单上显示的文本为橡皮
                    button_6.setText("  橡  皮  "); // 改变按钮上显示的文本为橡皮
                    g.setColor(foreColor); // 设置绘图上下文对象的前景色
                }
            }
        });
        button_6.setText("  橡  皮  ");
        toolBar.add(button_6);
        
        final JButton button_7 = new JButton();
        button_7.addActionListener(new ActionListener() {
            public void actionPerformed(final ActionEvent arg0) {
                System.exit(0); // 退出应用程序
            }
        });
        button_7.setText("  退  出  ");
        toolBar.add(button_7);
        
        final JMenuBar menuBar = new JMenuBar();
        setJMenuBar(menuBar);
        
        final JMenu menu = new JMenu();
        menu.setText("线  型");
        menuBar.add(menu);
        
        final JMenuItem newItemMenuItem = new JMenuItem();
        newItemMenuItem.addActionListener(new ActionListener() {
            public void actionPerformed(final ActionEvent e) {
                // 声明画笔的属性，粗细为1像素
                BasicStroke bs = new BasicStroke(1, BasicStroke.CAP_BUTT,
                        BasicStroke.JOIN_MITER);
                g.setStroke(bs); // 设置绘图上下文对象的画笔
            }
        });
        newItemMenuItem.setText("细  线");
        menu.add(newItemMenuItem);
        
        final JMenuItem newItemMenuItem_1 = new JMenuItem();
        newItemMenuItem_1.addActionListener(new ActionListener() {
            public void actionPerformed(final ActionEvent e) {
                // 声明画笔的属性，粗细为2像素
                BasicStroke bs = new BasicStroke(2, BasicStroke.CAP_BUTT,
                        BasicStroke.JOIN_MITER);
                g.setStroke(bs); // 设置绘图上下文对象的画笔
            }
        });
        newItemMenuItem_1.setText("粗  线");
        menu.add(newItemMenuItem_1);
        
        final JMenuItem newItemMenuItem_2 = new JMenuItem();
        newItemMenuItem_2.addActionListener(new ActionListener() {
            public void actionPerformed(final ActionEvent e) {
                // 声明画笔的属性，粗细为4像素
                BasicStroke bs = new BasicStroke(4, BasicStroke.CAP_BUTT,
                        BasicStroke.JOIN_MITER);
                g.setStroke(bs); // 设置绘图上下文对象的画笔
            }
        });
        newItemMenuItem_2.setText("较  粗");
        menu.add(newItemMenuItem_2);
        
        final JMenu menu_1 = new JMenu();
        menu_1.setText("颜  色");
        menuBar.add(menu_1);
        
        final JMenuItem newItemMenuItem_3 = new JMenuItem();
        newItemMenuItem_3.addActionListener(new ActionListener() {
            public void actionPerformed(final ActionEvent e) {
                // 打开选择颜色对话框
                Color fColor = JColorChooser.showDialog(null, "选择颜色对话框",
                        Color.CYAN);
                if (fColor != null) {
                    foreColor = fColor;
                }
                g.setColor(foreColor); // 设置绘图上下文对象的前景色
            }
        });
        newItemMenuItem_3.setText("前景颜色");
        menu_1.add(newItemMenuItem_3);
        
        final JMenuItem newItemMenuItem_4 = new JMenuItem();
        newItemMenuItem_4.addActionListener(new ActionListener() {
            public void actionPerformed(final ActionEvent e) {
                // 打开选择颜色对话框
                Color bgColor = JColorChooser.showDialog(null, "选择颜色对话框",
                        Color.CYAN);
                if (bgColor != null) {
                    backgroundColor = bgColor;
                }
                g.setColor(backgroundColor); // 设置绘图上下文对象的背景色
                g.fillRect(0, 0, 570, 390); // 用背景色填充整个画布
                g.setColor(foreColor); // 设置绘图上下文对象的前景色
                canvas.repaint(); // 更新画布
            }
        });
        newItemMenuItem_4.setText("背景颜色");
        menu_1.add(newItemMenuItem_4);
        
        final JMenu menu_2 = new JMenu();
        menu_2.setText("编  辑");
        menuBar.add(menu_2);
        
        final JMenuItem newItemMenuItem_5 = new JMenuItem();
        newItemMenuItem_5.addActionListener(new ActionListener() {
            public void actionPerformed(final ActionEvent e) {
                g.setColor(backgroundColor); // 设置绘图上下文对象的背景色
                g.fillRect(0, 0, 570, 390); // 用背景色填充整个画布
                g.setColor(foreColor); // 设置绘图上下文对象的前景色
                canvas.repaint(); // 更新画布
            }
        });
        newItemMenuItem_5.setText("清  除");
        menu_2.add(newItemMenuItem_5);
        
        newItemMenuItem_6 = new JMenuItem();
        newItemMenuItem_6.addActionListener(new ActionListener() {
            public void actionPerformed(final ActionEvent e) {
                if (newItemMenuItem_6.getText().equals("橡  皮")) { // 单击橡皮菜单，使用橡皮
                    rubber = true; // 设置橡皮标识为true
                    newItemMenuItem_6.setText("画  图"); // 改变菜单上显示的文本为画图
                    button_6.setText("  画  图  "); // 改变按钮上显示的文本为画图
                } else { // 单击工具栏上的画图按钮，使用画笔
                    rubber = false; // 设置橡皮标识为false
                    newItemMenuItem_6.setText("橡  皮"); // 改变菜单上显示的文本为橡皮
                    button_6.setText("  橡  皮  "); // 改变按钮上显示的文本为橡皮
                    g.setColor(foreColor); // 设置绘图上下文对象的前景色
                }
            }
        });
        newItemMenuItem_6.setText("橡  皮");
        menu_2.add(newItemMenuItem_6);
        
        final JMenu menu_3 = new JMenu();
        menu_3.setText("系统");
        menuBar.add(menu_3);
        
        final JMenuItem newItemMenuItem_8 = new JMenuItem();
        newItemMenuItem_8.addActionListener(new ActionListener() {
            public void actionPerformed(final ActionEvent e) {
                System.exit(0);
            }
        });
        newItemMenuItem_8.setText("退  出");
        menu_3.add(newItemMenuItem_8);
    }
    
    public static void main(String[] args) {
        DrawPictureFrame frame = new DrawPictureFrame();
        frame.setVisible(true);
    }
}
