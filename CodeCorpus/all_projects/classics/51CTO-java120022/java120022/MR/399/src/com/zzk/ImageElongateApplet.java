package com.zzk;

import java.applet.Applet;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.geom.AffineTransform;

public class ImageElongateApplet extends Applet implements Runnable {
    private boolean flag = true;// 声明标记变量
    private static float xw;// 定义调整图像宽度的变量
    public void init(){
        xw = 0.5f;// 初始化图像宽度
    }
    public void start(){
        Thread th = new Thread(this);// 创建线程对象
        th.start();// 启动线程对象
    }
    public void paint(Graphics g) {
        Image img = null;// 声明图像对象
        img = getImage(getCodeBase(), "com/zzk/PD4.jpg");// 获得图片信息
        int w = img.getWidth(this);// 设置图像的宽度
        int h = img.getHeight(this);// 设置图像的高度
        Graphics2D g2d = (Graphics2D) g;// 将g转换为可以利用的Graphics2D
        g2d.drawImage(img, w - 50, h, this);// 绘制图像
        AffineTransform tr = new AffineTransform(xw, 0, 0, 1, 150, h);// 创建仿射变换对象进行设置变换（第一个参数）
        g2d.drawImage(img, tr, this);// 绘制图像
        
    }
    @Override
    public void run() {
        while (true) {
            if (flag) {// 标记变量为true时执行
                xw += 0.1f;// 使宽度变大
                if (xw > 2.0f) {// 宽度大于2.0时
                    flag = false;// 标记变量为false
                }
            } else {// 标记变量为false时执行
                xw -= 0.1f;// 使宽度变小
                if (xw < 0.5f) {// 宽度小于2.0时
                    flag = true;// 标记变量为true
                }
            }
            try {
                Thread.sleep(200);// 休眠200毫秒
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
            repaint();// 重新调用paint()方法
        }
    }
}
