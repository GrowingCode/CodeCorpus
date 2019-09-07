package com.zzk;

import java.applet.Applet;
import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.Image;

public class TextFlashApplet extends Applet implements Runnable {
    boolean flag = true;// 标记变量
    String value = "";// 存放绘制内容的变量
    public void start() {
        Thread thread = new Thread(this);// 创建线程对象
        thread.start();// 启动线程对象
    }
    public void paint(Graphics g) {
        Image img = null;// 声明图像对象
        img = getImage(getCodeBase(), "com/zzk/PD3.jpg");
        g.clearRect(0, 0, 310, 230);// 清除绘图上下文的内容
        g.drawImage(img, 0, 0, getWidth(), getHeight(), this);
        Font font = new Font("华文楷体", Font.BOLD, 42);// 创建字体对象
        g.setFont(font);// 指定字体
        g.setColor(Color.red);// 指定颜色
        g.drawString(value, 30, 110);// 绘制文本
    }
    public void run() {
        try {
            while (true) { // 读取内容
                Thread.sleep(150);// 当前线程休眠150秒
                if (flag) {// flag为true
                    flag = false; // 赋值为false
                    value = "JAVA编程词典";// 为value赋值
                } else {
                    flag = true;// 赋值为true
                    value = "";// 赋值为空字符窜
                }
                repaint();// 调用paint()方法
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
