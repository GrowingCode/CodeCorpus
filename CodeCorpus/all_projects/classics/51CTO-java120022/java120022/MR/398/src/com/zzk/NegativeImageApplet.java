package com.zzk;

import java.applet.Applet;
import java.awt.Graphics;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.awt.image.LookupOp;
import java.awt.image.ShortLookupTable;

public class NegativeImageApplet extends Applet {
    private BufferedImage image;// 声明缓冲图像对象
    public void paint(Graphics g) {
        String value = "反向图像";
        Image img = null;// 声明图像对象
        img = getImage(getCodeBase(), "com/zzk/PD3.jpg");// 获得图片信息
        int a = img.getWidth(this); // 获得图片宽度赋给变量a
        int b = img.getHeight(this);// 获得图片高度赋给变量b
        if (a >= 0 || b >= 0) {
            image = new BufferedImage(img.getWidth(this), img.getHeight(this),
                    BufferedImage.TYPE_INT_RGB);// 创建缓冲图像对象
            image.getGraphics().drawImage(img, 0, 0, null);// 在缓冲图像对象上绘制图像
            short[] negative = new short[256 * 1];// 创建表示颜色反向的分量数组
            for (int i = 0; i < 256; i++) {
                negative[i] = (short) (255 - i);// 为数组赋值
            }
            ShortLookupTable table = new ShortLookupTable(0, negative);// 创建查找表对象
            LookupOp op = new LookupOp(table, null);// 创建实现从源到目标查找操作的LookupOp对象
            image = op.filter(image, null);// 调用LookupOp对象的filter()方法，实现图像反向功能
            if (image != null) {
                g.drawImage(img, 35, 40, null);// 绘制缓冲图像对象
                g.drawImage(image, 220, 40, null);// 绘制缓冲图像对象
            }
            g.drawString(value, 265, 175);// 绘制文本
        }
    }
}