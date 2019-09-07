package com.zzk;

import java.applet.Applet;
import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.Image;

public class CaptionSpecificApplet extends Applet implements Runnable {
    int x = 50;// 存储绘制点的x坐标
    int y = 216;// 存储绘制点的y坐标
    String value = "明日图书网的网址";// 存储绘制的内容
    public void start() {
        Thread thread = new Thread(this);// 创建线程对象
        thread.start();// 启动线程对象
    }
    public void paint(Graphics g) {
        g.clearRect(0, 0, 316, 237);// 清除绘图上下文内容
        Image img = null;// 声明图片对象
        img = getImage(getCodeBase(), "com/zzk/PD3.jpg");
        g.drawImage(img, 0, 0, getWidth(), getHeight(), this);// 绘制图像
        Font font = new Font("华文楷体", Font.BOLD, 20);// 创建字体对象
        g.setFont(font);// 指定字体
        g.setColor(Color.RED);// 指定颜色
        g.drawString(value, x, y);// 绘制文本
    }
    public void run() {
        try {
            while (true) { // 读取内容
                Thread.sleep(100); // 当前线程休眠1秒
                if (y <= 216 - 50) {// 如果已经向上移动50像素
                    y = 216;// y坐标定位到最下方
                    if (value.equals("明日图书网的网址")) {
                        value = "http://www.mingribook.com";// 改变绘制的内容
                    } else {
                        value = "明日图书网的网址";// 改变绘制的内容
                    }
                } else {// 如果还没向上移动到50像素
                    y -= 2;// y坐标上移
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
