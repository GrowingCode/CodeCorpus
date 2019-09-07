package com.zzk;

import java.applet.Applet;
import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.Image;

public class RollTextApplet extends Applet implements Runnable {
    String value = "明日图书网的网址：http://www.mingribook.com";// 存放绘制的内容
    int x;// 设置横坐标
    int y;// 设置纵坐标
    public void init() { // 初始化方法
        x = 316;// 存储绘制点的x坐标
        y = 190;// 存储绘制点的y坐标
    }
    public void start() {
        Thread thread = new Thread(this);// 创建线程对象
        thread.start();// 启动线程对象
    }
    public void paint(Graphics g) {
        Image img = null;// 声明图像对象
        img = getImage(getCodeBase(), "com/zzk/PD3.jpg");// 获取图片资源路径
        g.drawImage(img, 0, 0, getWidth(), getHeight(), this);// 绘制图像
        g.clearRect(0, 0, 0, 230);// 清除绘图上下文的内容
        g.setColor(Color.BLACK);// 定义颜色
        Font font = new Font("华文楷体", Font.BOLD, 20);// 创建字体对象
        g.setFont(font);// 定义字体
        g.drawString(value, x, y);// 绘制文本
    }
    public void run() {
        try {
            while (true) { // 读取内容
                Thread.sleep(50); // 当前线程休眠1秒
                if (x <= -440) {// 该条件可以根据需要自行调整
                    x = 316;// x坐标定位到最右侧
                } else {
                    x -= 2;// x坐标左移
                }
                repaint();// 调用paint()方法
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
    public void update(Graphics g) { // 重写update()方法防止闪烁
        paint(g);
    }
}
