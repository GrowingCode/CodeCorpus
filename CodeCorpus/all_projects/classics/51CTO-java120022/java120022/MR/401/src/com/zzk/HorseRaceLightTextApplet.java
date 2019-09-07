package com.zzk;

import java.applet.Applet;
import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.Image;

public class HorseRaceLightTextApplet extends Applet implements Runnable {
    public void start() {
        Thread thread = new Thread(this);// 创建线程
        thread.start();// 启动线程对象
    }
    String value = "拥有编程词典，学习编程真轻松。";// 存储绘制的内容
    char[] drawChar = value.toCharArray();// 将绘制内容转换为字符数组
    int[] x = new int[drawChar.length];// 存储每个字符绘制点x坐标的数组
    int y = 100;// 存储绘制点y坐标
    
    public void paint(Graphics g) {
        Image img = null;
        img = getImage(getCodeBase(), "com/zzk/PD3.jpg"); // 创建图像对象
        g.clearRect(0, 0, getWidth(), getHeight());// 清除绘图上下文内容
        g.drawImage(img, 0, 0, getWidth(), getHeight(), this);// 绘制图像
        Font font = new Font("华文楷体", Font.BOLD, 20);// 创建字体对象
        g.setFont(font);// 指定字体
        g.setColor(Color.RED);// 指定颜色
        for (int j = drawChar.length - 1; j >= 0; j--) {
            g.drawString(drawChar[drawChar.length - 1 - j] + "", x[j], y);// 绘制文本
        }
    }
    public void run() {
        boolean flag = false;// 为false时表示第一次执行,x坐标进行等比递增,否则进行等差递减
        while (true) {// 读取内容
            try {
                for (int i = drawChar.length - 1; i >= 0; i--) {
                    if (!flag) {
                        x[i] = x[i] + 20 * i;// x坐标进行等比递增
                    } else {
                        x[i] = x[i] + 20;// x坐标进行等比递减
                    }
                    if (x[i] >= 360 - 20) {// 大于窗体宽度减20的值时
                        x[i] = 0; // x坐标为0
                    }
                }
                repaint();// 调用 paint()方法
                if (!flag) {
                    flag = true;// 赋值为true;
                }
                Thread.sleep(1000);// 当前线程休眠300毫秒
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }
    public void update(Graphics g) { // 重写update()方法防止闪烁
        paint(g);
    }
}
