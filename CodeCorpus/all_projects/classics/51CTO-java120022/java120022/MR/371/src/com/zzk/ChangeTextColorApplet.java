package com.zzk;

import java.applet.Applet;
import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.util.Random;

public class ChangeTextColorApplet extends Applet implements Runnable {
    Color color = new Color(0, 0, 255);// 创建颜色对象
    public void init() {
        Thread thread = new Thread(this);// 创建线程对象
        thread.start();// 启动线程对象
    }
    public void paint(Graphics g) { // 重写paint()方法
        Graphics2D g2 = (Graphics2D) g;// 转换为Graphics2D类型
        String value = "会变色的文字";// 绘制的文本
        int x = 10; // 文本位置的横坐标
        int y = 110; // 文本位置的纵坐标
        Font font = new Font("楷体", Font.BOLD, 40); // 创建字体对象
        g2.setFont(font); // 设置字体
        g2.setColor(color);// 设置颜色
        g2.drawString(value, x, y); // 绘制文本
    }
    public void run() {
        Random random = new Random();// 创建随机数对象
        while (true) {
            int R = random.nextInt(256);// 随机产生颜色的R值
            int G = random.nextInt(256);// 随机产生颜色的G值
            int B = random.nextInt(256);// 随机产生颜色的B值
            color = new Color(R, G, B);// 创建颜色对象
            repaint();// 调用paint()方法
            try {
                Thread.sleep(300);// 休眠300毫秒
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }
}
