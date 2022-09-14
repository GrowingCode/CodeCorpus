package com.zzk;

import java.awt.EventQueue;
import java.awt.Graphics;
import java.awt.Image;
import java.awt.image.ImageProducer;
import java.awt.image.MemoryImageSource;
import javax.swing.JFrame;
import javax.swing.JPanel;

public class CreateImageFrame extends JFrame {
    
    public static void main(String args[]) {
        EventQueue.invokeLater(new Runnable() {
            public void run() {
                try {
                    CreateImageFrame frame = new CreateImageFrame();
                    frame.setVisible(true);
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        });
    }
    
    public CreateImageFrame() {
        super();
        setTitle("使用像素值生成图像");
        setBounds(100, 100, 300, 220);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        getContentPane().add(new CreateImagePanel());
    }
    
    class CreateImagePanel extends JPanel {// 创建用像素值生成图像的面板类
        public void paint(Graphics g) {
            int w = 300;// 宽度
            int h = 220;// 高度
            int pix[] = new int[w * h];// 存储像素值的数组
            int index = 0;// 存储数组的索引
            for (int y = 0; y < h; y++) {// 在纵向进行调整，从黑色渐变到红色
                int red = (y * 255) / (h - 1);// 计算纵向的颜色值
                for (int x = 0; x < w; x++) {// 在横向进行调整，从黑色渐变到蓝色
                    int blue = (x * 255) / (w - 1);// 计算横向的颜色值
                    // 通过移位运算和逻辑或运算计算像素值，并赋值给像素数组
                    pix[index++] = (255 << 24) | (red << 16) | blue;
                }
            }
            // 创建使用数组为Image生成像素值的ImageProducer对象
            ImageProducer imageProducer = new MemoryImageSource(w, h, pix, 0, w);
            Image img = createImage(imageProducer);// 创建图像对象
            g.drawImage(img, 0, 0,getWidth(),getHeight(), this);// 绘制图像
        }
    }
}
