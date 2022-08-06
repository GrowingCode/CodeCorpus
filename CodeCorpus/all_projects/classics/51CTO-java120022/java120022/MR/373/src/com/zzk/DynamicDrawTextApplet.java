package com.zzk;

import java.applet.Applet;
import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics;

public class DynamicDrawTextApplet extends Applet implements Runnable {
    int x = 20;// 起始点的x坐标
    int y = 30;// 起始点的y坐标
    String textStrings = "小应用程序中缩放图像\n小应用程序中翻转图像\n小应用程序中旋转图像\n小应用程序中倾斜图像";// 需要动态绘制的字符串
    String value = "";// 存储字符串中的单个字符
    public void init() {
        Thread thread = new Thread(this);// 创建线程对象
        thread.start();// 启动线程对象
    }
    public void paint(Graphics g) {
        Font font = new Font("华文楷体", Font.BOLD, 20);// 创建字体对象
        g.setFont(font);// 指定字体
        g.setColor(Color.RED);// 指定颜色
        g.drawString(value, x, y);// 绘制文本
    }
    public void update(Graphics g) {// 重写update方法，防止无法显示绘制的所有内容
        paint(g);// 调用paint()方法
    }
    public void run() {
        try {
            for (int i = 0; i < textStrings.length(); i++) {
                Thread.sleep(400); // 当前线程休眠400毫秒
                value = textStrings.substring(i, i + 1);// 截取字符串中的一个字符
                if (value.equals("\n")) {// 是换行符
                    x = 20;// 下一行起始点的x坐标
                    y += 30;// 下一行文本的y坐标
                } else {// 不是回车或换行符
                    x += 20;// 当前行下一个字的x坐标
                }
                repaint();// 调用paint()方法
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
