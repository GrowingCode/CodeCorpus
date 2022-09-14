package com.zzk;

import java.awt.Canvas;
import java.awt.Graphics;
import java.awt.Image;

/**
 * @author 张振坤
 *         创建画布类
 */
public class DrawPictureCanvas extends Canvas {
    private Image image = null; // 定义Image对象的引用
    public DrawPictureCanvas() {
        super();
    }
    public void setImage(Image image) {
        this.image = image; // 为成员变量赋值
    }
    /*
     * 重写paint()方法,在画布上绘制图像
     */
    public void paint(Graphics g) {
        g.drawImage(image, 0, 0, null); // 在画布上绘制图像
    }
    /*
     * 重写update()方法，这样可以解决屏幕闪耀的问题
     */
    public void update(Graphics g) {
        paint(g); // 调用paint方法
    }
}
