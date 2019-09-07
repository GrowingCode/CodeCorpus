package com.zzk;

import java.applet.Applet;
import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Image;

public class TextScaleApplet extends Applet implements Runnable {
    public void start() {
        Thread thread = new Thread(this);// 创建线程对象
        thread.start();// 启动线程对象
    }
    private Image img = null; // 声明图像对象
    boolean flag = false;// 定义标记变量,用户控制x的值
    int x = 12;// 定义表示字体大小的变量x
    Font font = new Font("华文楷体", Font.BOLD, 42);// 创建字体对象
    public void paint(Graphics g) {
        img = getImage(getCodeBase(), "com/zzk/PD3.jpg");// 获取图片资源的路径
        Graphics2D g2 = (Graphics2D) g;// 获得Graphics2D对象
        g2.drawImage(img, 0, 0, getWidth(), getHeight(), this);// 绘制图像
        g2.setFont(font);// 指定字体
        g2.setColor(Color.red);// 指定颜色
        g2.drawString("学无止境", 30, 120);// 绘制文本
    }
    public void run() {
        while (true) {
            if (flag) { // flag为true时
                x -= 2; // x进行减1计算
                if (x <= 12) {// x小于等于12时
                    x = 12; // x等于12
                    flag = false;// 为flag赋值为false
                }
            } else {// flag为false时
                x += 2;// x进行加1计算
                if (x >= 72) {// x大于等于72时
                    x = 72;// x等于72
                    flag = true;// 为flag赋值为true
                }
            }
            font = new Font("华文楷体", Font.BOLD, x);// 重新创建字体对象
            repaint();// 调用paint()方法
            try {
                Thread.sleep(50);// 休眠50毫秒
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }
//    public void update(Graphics g) {
//        paint(g);
//    }
}
